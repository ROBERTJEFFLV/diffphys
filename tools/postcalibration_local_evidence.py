"""Read-only post-calibration evidence for the structured controller.

This stage deliberately performs no optimizer step and never installs or
changes conformal quantiles.  It reloads a frozen checkpoint, verifies its
deployment hash before and after beta-0 equilibrium/allocator/JVP probes, and
emits an evidence checkpoint carrying the unchanged weights and report.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import load_q2_policy  # noqa: E402
from env_l2f import L2FParams, L2FSimulator  # noqa: E402
from structured_checkpoint import (  # noqa: E402
    CADENCE_SEMANTICS_VERSION,
    deployment_policy_hash,
)
from structured_distillation import (  # noqa: E402
    build_dagger_scenario_bank,
    collect_dagger_episode,
    phase_a_equilibrium_gate,
)
from structured_local_distillation import (  # noqa: E402
    LOCAL_ACTION_PARITY_THRESHOLD,
    LOCAL_ALLOCATOR_RESIDUAL_THRESHOLD,
    LOCAL_JVP_P95_THRESHOLD,
    LOCAL_TAYLOR_R2_THRESHOLD,
    beta0_action_parity_diagnostics,
    build_local_derivative_batch,
    collect_equilibrium_history,
    lifted_h250_diagnostics,
    local_derivative_diagnostics,
    local_mode_diagnostics,
    projection_diagnostics,
)
from structured_rollout import load_structured_policy  # noqa: E402
from structured_rollout import (  # noqa: E402
    StructuredBoundaryCodec,
    StructuredClosedLoopState,
    make_structured_step_map,
    structured_observation,
    structured_global_yaw_basis,
)
from structured_policy import StructuredPolicyState  # noqa: E402
from structured_stability import (  # noqa: E402
    AugmentedStabilityConfig,
    matrix_free_augmented_stability_report,
)
from policy_observation import build_policy_observation, initial_observation_state  # noqa: E402


AUGMENTED_STABILITY_SCHEMA_VERSION = 2


def _slice_closed_loop(state: StructuredClosedLoopState, index: int) -> StructuredClosedLoopState:
    """Select one scenario without dropping any v2 policy-state field."""

    physical = replace(state.physical, **{
        field.name: getattr(state.physical, field.name)[index:index + 1]
        for field in fields(state.physical)
    })
    policy = replace(state.policy, **{
        field.name: (
            getattr(state.policy, field.name)[index:index + 1]
            if torch.is_tensor(getattr(state.policy, field.name))
            else getattr(state.policy, field.name)
        )
        for field in fields(state.policy)
    })
    return StructuredClosedLoopState(physical=physical, policy=policy)


@torch.no_grad()
def _post_call75_states(student, simulator, bank) -> StructuredClosedLoopState:
    """Roll the deployed student to the post-call75 complete state."""

    physical = bank.state
    batch = physical.position.shape[0]
    observation, _ = build_policy_observation(
        physical, initial_observation_state(
            batch, device=physical.position.device, dtype=physical.position.dtype
        ), mode="integral25", integral_input_frame="body", noise_max=0.0,
    )
    policy_state = student.initial_state(observation)
    closed = StructuredClosedLoopState(physical=physical, policy=policy_state)
    for _ in range(75):
        observation = structured_observation(closed)
        output = student.forward_with_aux(observation, closed.policy, simulator.params.dt)
        closed = StructuredClosedLoopState(
            physical=simulator.step(closed.physical, output.action, grad_decay=1.0),
            policy=output.next_state,
        )
    return closed


def _force_bins(bank) -> torch.Tensor:
    """Rank external-force magnitude inside each 4x4 authority cell.

    The formal bank contains four scenarios per ``(TW, log-alpha)`` cell.
    Global force quartiles do not guarantee coverage after crossing with those
    authority labels and made empty-cell rejection almost inevitable.  Local
    rank bins preserve the registered scenarios while assigning exactly one
    sample to each force rank in every complete four-sample authority cell.
    """

    magnitude = bank.state.external_force.norm(dim=-1).detach().cpu()
    tw_bin = bank.tw_bin.detach().cpu()
    alpha_bin = bank.log_alpha_bin.detach().cpu()
    bins = torch.full_like(tw_bin, -1)
    for tw in range(4):
        for alpha in range(4):
            selected = torch.nonzero(
                (tw_bin == tw) & (alpha_bin == alpha), as_tuple=False
            ).reshape(-1)
            if selected.numel() != 4:
                raise RuntimeError(
                    "augmented stability requires exactly four scenarios per authority cell"
                )
            local_order = selected[torch.argsort(
                magnitude.index_select(0, selected), stable=True
            )]
            bins[local_order] = torch.arange(4, dtype=bins.dtype)
    if bool((bins < 0).any().item()):
        raise RuntimeError("force-rank assignment did not cover every stability scenario")
    return bins


def _augmented_stability_evidence(student, simulator, bank, deployment_hash: str) -> dict[str, Any]:
    """Probe complete v2 maps on every authority/force cell.

    This deliberately uses the local 64-scenario post-call75 bank only.  It
    does not draw blind seeds, fit thresholds, or run any training.
    """

    closed = _post_call75_states(student, simulator, bank)
    force_bins = _force_bins(bank)
    config = AugmentedStabilityConfig()
    rows: list[dict[str, Any]] = []
    for index in range(bank.count):
        row: dict[str, Any] = {
            "scenario": index,
            "tw_bin": int(bank.tw_bin[index]),
            "log_alpha_bin": int(bank.log_alpha_bin[index]),
            "force_bin": int(force_bins[index]),
            "deployment_policy_hash": deployment_hash,
            "evidence_hash": deployment_hash,
            "post_call": 75,
            "poincare_horizon_steps": config.horizon_steps,
        }
        try:
            selected = _slice_closed_loop(closed, index)
            codec = StructuredBoundaryCodec(
                selected.physical, student, boot_completed=True,
            )
            packed = codec.pack(selected)[0]
            yaw_basis = structured_global_yaw_basis(codec, packed)
            discrete = torch.zeros(codec.state_dim, dtype=torch.bool, device=packed.device)
            for name in ("identification_failed", "slow_counter"):
                if name in codec.slices:
                    discrete[codec.slices[name]] = True
            step_map = make_structured_step_map(student, simulator, codec)
            result = matrix_free_augmented_stability_report(
                step_map, packed, config=config, yaw_basis=yaw_basis,
                discrete_mask=discrete,
            )
            row.update(result.as_dict())
        except (RuntimeError, ValueError, FloatingPointError) as error:
            # A malformed/nonsmooth map is evidence failure, never a missing
            # observation that can be silently excluded from the reduction.
            row.update({
                "gate_passed": False, "finite": False,
                "arnoldi_converged": False, "arnoldi_breakdown": True,
                "fixed_point_iterations": 0,
                "fixed_point_converged": False,
                "fixed_point_anchor_passed": False,
                "fixed_point_residual": float("nan"),
                "fixed_point_anchor_hash": None,
                "error": repr(error),
                "evidence_kind": "sampled_matrix_free_evidence_not_a_proof_or_certificate",
            })
        rows.append(row)

    cells = []
    for tw in range(4):
        for alpha in range(4):
            for force in range(4):
                selected = [row for row in rows if row["tw_bin"] == tw
                            and row["log_alpha_bin"] == alpha
                            and row["force_bin"] == force]
                cells.append({
                    "tw_bin": tw, "log_alpha_bin": alpha, "force_bin": force,
                    "samples": len(selected),
                    "finite": bool(selected) and all(bool(item.get("finite", False)) for item in selected),
                    "gate_passed": bool(selected) and all(bool(item.get("gate_passed", False)) for item in selected),
                    "spectral_radius_max": max((float(item.get("spectral_radius_non_yaw_estimate", float("nan")))
                                                 for item in selected), default=None),
                    "fixed_point_residual_max": max((float(item.get("fixed_point_residual", float("nan")))
                                                      for item in selected), default=None),
                    "jvp_central_relative_error_max": max((float(item.get("jvp_central_relative_error", float("nan")))
                                                           for item in selected), default=None),
                    "deployment_policy_hash": deployment_hash,
                    "evidence_hash": deployment_hash,
                })
    gate = bool(rows and all(bool(row.get("gate_passed", False)) for row in rows)
                and all(bool(cell["gate_passed"]) for cell in cells)
                and all(row.get("deployment_policy_hash") == deployment_hash for row in rows))
    return {
        "schema_version": AUGMENTED_STABILITY_SCHEMA_VERSION,
        "post_call": 75,
        "poincare_map": "complete_structured_boundary_codec_v2",
        "horizon_steps": config.horizon_steps,
        "fixed_point_anchor": {
            "method": "projected_picard_on_25_step_poincare_map",
            "max_iterations": config.fixed_point_max_iterations,
            "relaxation": config.fixed_point_relaxation,
            "residual_norm": "normalized_non_yaw_non_discrete_l2",
        },
        "yaw_quotient": "declared_global_yaw_orientation_tangent_only",
        "cadences": list(config.cadences),
        "thresholds": {
            "spectral_radius_non_yaw_max": config.spectral_radius_threshold,
            "finite_time_gain_max_by_cadence": {
                str(cadence): limit for cadence, limit in zip(
                    config.cadences, config.finite_gain_thresholds
                )
            },
            "jvp_central_relative_error_max": config.jvp_fd_tolerance,
            "arnoldi_residual_max": config.residual_tolerance,
            "fixed_point_residual_max": config.fixed_point_residual_threshold,
        },
        "evidence_kind": "sampled_matrix_free_evidence_not_a_proof_or_certificate",
        "deployment_policy_hash": deployment_hash,
        "evidence_hash": deployment_hash,
        "gate_passed": gate,
        "scenario_results": rows,
        "by_authority_force_cell": cells,
        "all_64_authority_force_cells_present": all(cell["samples"] > 0 for cell in cells),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="post-calibration beta-0 evidence")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True,
                        help="fixed Q2 checkpoint used only for near-equilibrium labels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seed", type=int, default=2718)
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _move_bank(bank, device: torch.device):
    if device.type == "cpu":
        return bank
    from env_l2f import L2FState
    state = L2FState(**{
        name: getattr(bank.state, name).to(device)
        for name in bank.state.__dataclass_fields__
    })
    return type(bank)(state, bank.tw_bin.to(device), bank.log_alpha_bin.to(device), bank.stratum)


def _registered_oracle(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable Phase-B oracle inherited by Phase C.

    Post-calibration evidence is not allowed to invent a new annulus or fit a
    threshold on its own data.  Requiring the complete nested contract here
    prevents the old flat screening fields from accidentally becoming a
    formal release gate.
    """
    oracle = report.get("structured_residual_oracle")
    if not isinstance(oracle, Mapping) or int(oracle.get("schema_version", 0)) < 1:
        raise RuntimeError("postcheck requires a schema-v1 structured residual oracle inherited from Phase C")
    if not bool(oracle.get("pre_registered", False)):
        raise RuntimeError("postcheck requires a pre-registered residual oracle")
    annulus = oracle.get("annulus")
    if not isinstance(annulus, Mapping):
        raise RuntimeError("postcheck oracle is missing its away annulus")
    r_min, r_max = float(annulus.get("r_min", 0.0)), float(annulus.get("r_max", 0.0))
    action_threshold = float(oracle.get(
        "whole_policy_action_rms_threshold",
        oracle.get("action_rms_threshold", float("nan")),
    ))
    allocator_threshold = float(oracle.get(
        "allocator_residual_p99_threshold",
        oracle.get("residual_p99_threshold", float("nan")),
    ))
    baseline = float(oracle.get("baseline_residual_p99", float("nan")))
    multiplier = float(oracle.get("relative_multiplier", float("nan")))
    if not (0.0 < r_min < r_max and torch.isfinite(torch.tensor(
            (action_threshold, allocator_threshold, baseline, multiplier))).all()
            and action_threshold > 0.0 and allocator_threshold > 0.0
            and baseline >= 0.0 and multiplier > 0.0):
        raise RuntimeError("postcheck oracle has invalid registered thresholds")
    phase_b_hash = oracle.get("phase_b_deployment_hash")
    if not isinstance(phase_b_hash, str) or not phase_b_hash:
        raise RuntimeError("postcheck oracle is missing Phase-B deployment hash")
    return {
        "schema_version": int(oracle["schema_version"]),
        "phase_b_deployment_hash": phase_b_hash,
        "r_min": r_min, "r_max": r_max,
        "action_threshold": action_threshold,
        "allocator_threshold": allocator_threshold,
        "baseline_residual_p99": baseline,
        "relative_multiplier": multiplier,
        "contract": dict(oracle),
    }


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    student, source = load_structured_policy(args.checkpoint, device=device)
    student.eval()
    if not bool(student.capability_calibration_valid.item()):
        raise RuntimeError("post-calibration evidence requires a checkpoint with valid conformal q")
    teacher, teacher_args = load_q2_policy(
        args.source_checkpoint, device=device, dtype=torch.float32
    )
    teacher.eval()
    simulator = L2FSimulator(L2FParams(dt=float(teacher_args.get("dt", student.config.dt))))
    bank = _move_bank(
        build_dagger_scenario_bank(64, seed=args.seed, dt=simulator.params.dt), device
    )

    hash_before = deployment_policy_hash(student)
    source_report = source.get("report", {})
    if not isinstance(source_report, Mapping):
        raise RuntimeError("postcheck checkpoint has no structured report")
    oracle = _registered_oracle(source_report)
    calibration_metadata = source_report.get("capability_calibration")
    if not isinstance(calibration_metadata, Mapping):
        raise RuntimeError("postcheck requires calibration metadata in the same checkpoint")
    # Calibration changes deployment behavior through q.  The evidence must
    # therefore be measured on exactly the hash emitted by calibration, not a
    # stale pre-calibration copy.
    calibration_hash = calibration_metadata.get("policy_config_hash")
    calibrated_hash = calibration_metadata.get("calibrated_parameter_hash")
    if calibration_hash != hash_before or calibrated_hash != hash_before:
        raise RuntimeError("postcheck calibration metadata hash does not match deployed checkpoint")
    q_before = student.capability_conformal_q.detach().clone()
    n_before = student.capability_calibration_n.detach().clone()
    valid_before = student.capability_calibration_valid.detach().clone()

    # beta0 means analytic equilibrium/trim is always the executed action;
    # Q2 is used only to define the fixed hidden-state near-equilibrium label.
    snapshots = collect_equilibrium_history(
        teacher, student, simulator, bank, snapshot_steps=(50, 75)
    )
    batch = build_local_derivative_batch(
        teacher, student, snapshots, radii=(0.1, 0.05, 0.025), seed=args.seed + 1
    )
    local = local_derivative_diagnostics(teacher, student, snapshots, batch)
    modes = local_mode_diagnostics(teacher, student, snapshots, batch)
    projection = projection_diagnostics(student, snapshots)
    parity = beta0_action_parity_diagnostics(
        # Include post-burn-in residual/contextual control.  H25 would measure
        # only the temporary boot controller and cannot validate Phase C.
        teacher, student, simulator, bank, horizon=125,
        annulus_radius=oracle["r_min"], annulus_r_max=oracle["r_max"],
        parity_threshold=oracle["action_threshold"],
    )
    beta0_episode = collect_dagger_episode(
        teacher, student, simulator, bank, beta=0.0, horizon=125,
        episode_seed=args.seed + 100,
    )
    phase_a_gate_report, phase_a_gate_passed = phase_a_equilibrium_gate(
        beta0_episode
    )
    h250 = lifted_h250_diagnostics(student, simulator, bank, horizon=250, segment_length=25)
    # P1-6: mandatory complete-state, post-call75 matrix-free stability probe.
    # This is intentionally local sampled evidence; it never consumes blind
    # seeds and never promotes itself to a nonlinear certificate.
    try:
        stability = _augmented_stability_evidence(student, simulator, bank, hash_before)
    except (RuntimeError, ValueError, FloatingPointError) as error:
        stability = {
            "schema_version": AUGMENTED_STABILITY_SCHEMA_VERSION,
            "post_call": 75,
            "poincare_map": "complete_structured_boundary_codec_v2",
            "fixed_point_anchor": {
                "method": "projected_picard_on_25_step_poincare_map",
                "max_iterations": AugmentedStabilityConfig.fixed_point_max_iterations,
                "residual_norm": "normalized_non_yaw_non_discrete_l2",
            },
            "evidence_kind": "sampled_matrix_free_evidence_not_a_proof_or_certificate",
            "deployment_policy_hash": hash_before,
            "evidence_hash": hash_before,
            "gate_passed": False,
            "error": repr(error),
            "scenario_results": [], "by_authority_force_cell": [],
        }
    hash_after = deployment_policy_hash(student)
    unchanged = bool(
        hash_before == hash_after
        and torch.equal(q_before, student.capability_conformal_q)
        and torch.equal(n_before, student.capability_calibration_n)
        and torch.equal(valid_before, student.capability_calibration_valid)
    )
    phase_a_gate = phase_a_gate_report
    e0_gate = bool(
        local["equilibrium_action_trim_rms"] <= 1.3e-3
        and local["equilibrium_action_trim_max"] <= 5.0e-3
    )
    jvp_cells = [
        row["coverage_at_registered_threshold"]
        for row in local["jvp_coverage_by_authority_cell"]
        if row["coverage_at_registered_threshold"] is not None
    ]
    deployment_mode_rows = modes["by_authority_cell"]
    contextual_gain_deployed = bool(
        modes["active_fraction"] >= 0.999
        and len(deployment_mode_rows) == 32
        and all(row["active_samples"] > 0 and row["inactive_samples"] == 0
                for row in deployment_mode_rows)
    )
    jvp_gate = bool(
        local["jvp_normalized_error_p95"] <= LOCAL_JVP_P95_THRESHOLD
        and len(jvp_cells) == 16 and min(jvp_cells) >= 0.95
        and contextual_gain_deployed
    )
    taylor_gate = bool(
        local["taylor_remainder_over_radius2_max"] <= LOCAL_TAYLOR_R2_THRESHOLD
    )
    parity_cells = parity["by_authority_cell"]
    parity_gate = bool(
        parity["parity_p95"] is not None
        and parity["parity_p95"] <= oracle["action_threshold"]
        and parity["parity_coverage"] is not None
        and parity["parity_coverage"] >= 0.95
        and len(parity_cells) == 16
        and all(row["away_samples"] > 0 for row in parity_cells)
        and all(row["parity_coverage"] is not None and row["parity_coverage"] >= 0.95
                and row["parity_p95"] <= oracle["action_threshold"]
                for row in parity_cells)
    )
    allocator_p99 = parity.get("allocator_residual_p99")
    allocator_baseline_ok = (
        allocator_p99 is not None
        and allocator_p99 <= oracle["allocator_threshold"]
        and allocator_p99 <= oracle["baseline_residual_p99"] * oracle["relative_multiplier"]
    )
    allocator_gate = bool(
        parity["allocator_min_headroom"] is not None
        and parity["allocator_min_headroom"] >= 0.0
        and allocator_baseline_ok
    )
    h250_gate = bool(h250["finite"] and h250["max_omega"] <= 5.0)
    stability_hash_ok = bool(
        stability.get("deployment_policy_hash") == hash_after
        and stability.get("evidence_hash") == hash_after
        and all(row.get("deployment_policy_hash") == hash_after
                for row in stability.get("scenario_results", []))
        and all(row.get("evidence_hash") == hash_after
                for row in stability.get("scenario_results", []))
        and all(row.get("deployment_policy_hash") == hash_after
                for row in stability.get("by_authority_force_cell", []))
        and all(row.get("evidence_hash") == hash_after
                for row in stability.get("by_authority_force_cell", []))
    )
    stability_gate = bool(stability.get("gate_passed", False) and stability_hash_ok)
    phase_a_gate_passed = bool(phase_a_gate_passed)
    postcheck_gate = bool(
        unchanged and phase_a_gate_passed and e0_gate and jvp_gate
        and taylor_gate and parity_gate and allocator_gate
        and bool(projection["projection_bound_passed"]) and h250_gate
        and stability_gate
    )
    for evidence in (local, modes, parity, projection, h250):
        evidence["evidence_hash"] = hash_after
    report = dict(source.get("report", {}))
    report.update({
        "cadence_semantics_version": CADENCE_SEMANTICS_VERSION,
        "cadence_semantics": {
            "call_index_completed_transitions": True,
            "call0_has_response": False,
            "publication_calls": [50, 75],
            "availability_t25": [0, 0, 0, 0, 0, 0],
            "publication_rule_after_first": "positive slow_cadence offsets from call50",
            "t50_call_index": 50,
        },
        "phase": "post-calibration-evidence",
        "active_phase": "C",
        "evidence_stage": "postcalibration",
        "active_phase_components": [
            "calibration_frozen", "beta0_equilibrium", "allocator", "near_eq_jvp",
            "projection", "augmented_closed_loop_stability_p1_6",
        ],
        "source_checkpoint": str(args.checkpoint.resolve()),
        "q2_checkpoint": str(args.source_checkpoint.resolve()),
        "scenario_count": bank.count,
        "beta": 0.0,
        "snapshot_steps": [50, 75],
        "directions_per_snapshot_radius": batch.direction_count,
        "local_metrics": local,
        "local_mode_metrics": modes,
        "beta0_action_parity": parity,
        "postcheck_action_parity": {
            "gate_passed": parity_gate,
            "deployment_policy_hash": hash_after,
            "oracle_phase_b_deployment_hash": oracle["phase_b_deployment_hash"],
            "annulus": {"r_min": oracle["r_min"], "r_max": oracle["r_max"]},
            "action_rms_threshold": oracle["action_threshold"],
            "p95": parity["parity_p95"], "coverage": parity["parity_coverage"],
            "away_samples": parity["away_samples"],
            "all_16_authority_cells_have_samples": all(
                row["away_samples"] > 0 for row in parity_cells
            ),
        },
        "postcheck_allocator": {
            "gate_passed": allocator_gate,
            "deployment_policy_hash": hash_after,
            "p99": allocator_p99,
            "absolute_threshold": oracle["allocator_threshold"],
            "baseline_p99": oracle["baseline_residual_p99"],
            "relative_multiplier": oracle["relative_multiplier"],
            "headroom_min": parity["allocator_min_headroom"],
        },
        "projection": projection,
        "h250_segments": h250,
        "augmented_closed_loop_stability": {
            **stability,
            "deployment_policy_hash": hash_after,
            "evidence_hash": hash_after,
            "hash_bound": stability_hash_ok,
            "gate_passed": stability_gate,
        },
        "deployment_policy_hash_before": hash_before,
        "deployment_policy_hash_after": hash_after,
        "weights_q_unchanged": unchanged,
        "evidence_hash": hash_after,
        "deployment_policy_hash": hash_after,
        "calibration_deployment_hash": hash_before,
        "phase_c_deployment_hash": source_report.get("phase_c_deployment_hash",
                                                       source_report.get("student_policy_deployment_hash")),
        "structured_residual_oracle": oracle["contract"],
        "phase_a_equilibrium_gate_recheck": {
            "gate": phase_a_gate,
            "recomputed": True,
            "reason": "fresh beta0 DAgger episode on post-calibration bank",
            "evidence_hash": hash_after,
        },
        # Migration consumes these explicit, hash-bound gates.  Do not leave
        # the stale Phase-A/Phase-B values inherited through calibration.
        "near_equilibrium_jvp": {
            "gate_passed": jvp_gate,
            "deployment_policy_hash": hash_after,
            "normalized_error_p95": local["jvp_normalized_error_p95"],
            "registered_threshold": LOCAL_JVP_P95_THRESHOLD,
            "contextual_gain_deployed_all_authority_cells": contextual_gain_deployed,
        },
        "equilibrium_gate": {
            "gate_passed": bool(phase_a_gate_passed and e0_gate),
            "deployment_policy_hash": hash_after,
            "phase_a_gate": phase_a_gate,
            "equilibrium_action_trim_gate_passed": e0_gate,
        },
        "postcheck_gate_passed": postcheck_gate,
        "formal_gates": {
            "calibration_frozen": unchanged,
            "phase_a_equilibrium_gate": phase_a_gate_passed,
            "action_parity": parity_gate,
            "allocator": allocator_gate,
            "allocator_residual_registered_threshold": oracle["allocator_threshold"],
            "e0_trim": e0_gate,
            "jvp_normalized": jvp_gate,
            "contextual_gain_deployed_all_authority_cells": contextual_gain_deployed,
            "taylor_remainder": taylor_gate,
            "h250_finite": h250_gate,
            "projection": bool(projection["projection_bound_passed"]),
            "near_eq_jvp": jvp_gate,
            "equilibrium_action": bool(phase_a_gate_passed and e0_gate),
            "augmented_closed_loop_stability": stability_gate,
            "augmented_stability_hash_bound": stability_hash_ok,
        },
        "migration_gate_passed": False,
        "promotion": False,
        "evidence_only": True,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "structured-recurrent-motor-policy",
        "model": student.state_dict(),
        "config": asdict(student.config),
        "fast_feedback_verified": student.fast_feedback.verified,
        "report": report,
    }, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(args.output),
        "report": str(args.report),
        "deployment_policy_hash": hash_after,
        "weights_q_unchanged": unchanged,
        "migration_gate_passed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

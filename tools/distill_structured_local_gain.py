"""CPU-smoke Phase-B local contextual-gain distillation.

This tool consumes a calibrated Phase-A student and a fixed Q2 teacher, fits
only the cap-conditioned contextual gain head, and writes a checkpoint whose
formal migration gate is intentionally false.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import load_q2_policy  # noqa: E402
from env_l2f import L2FParams, L2FSimulator  # noqa: E402
from structured_distillation import build_dagger_scenario_bank  # noqa: E402
from structured_checkpoint import CADENCE_SEMANTICS_VERSION  # noqa: E402
from structured_local_distillation import (  # noqa: E402
    LOCAL_JVP_P95_THRESHOLD,
    LOCAL_PRECAL_SURROGATE_P95_THRESHOLD,
    LOCAL_RADII,
    LOCAL_TAYLOR_R2_THRESHOLD,
    build_local_derivative_batch,
    collect_common_history,
    collect_equilibrium_history,
    fit_contextual_gain_local,
    lifted_h250_diagnostics,
    local_derivative_diagnostics,
    local_mode_diagnostics,
    projection_diagnostics,
)
from structured_rollout import load_structured_policy  # noqa: E402
from structured_checkpoint import deployment_policy_hash  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-B local contextual-gain derivative distillation")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="calibrated Phase-A student checkpoint")
    parser.add_argument("--source-checkpoint", type=Path, required=True,
                        help="Q2 teacher checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seed", type=int, default=1707)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--identifier-dim", type=int, default=64)
    parser.add_argument(
        "--allow-screening-only", action="store_true",
        help="run a non-promotable smoke when Phase-A release gates are false",
    )
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


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    torch.manual_seed(args.seed)
    student, student_source = load_structured_policy(args.checkpoint, device=device)
    previous_phase = student_source.get("report", {}).get("active_phase")
    phase_a_report = student_source.get("report", {})
    phase_a_ready = (
        previous_phase == "A2"
        and bool(phase_a_report.get("equilibrium_gate_passed", False))
        and bool(phase_a_report.get("capability_calibration_installed_in_stage", False))
    )
    if not phase_a_ready and not args.allow_screening_only:
        raise RuntimeError(f"local gain tool requires a Phase-A2 checkpoint, got {previous_phase!r}")
    teacher, teacher_args = load_q2_policy(args.source_checkpoint, device=device, dtype=torch.float32)
    teacher.eval()
    student.eval()
    # Local fitting requires no residual contribution and only updates the
    # cap-conditioned contextual gain head.
    if float(student.config.residual_scale) != 0.0:
        raise RuntimeError("local Phase-B prototype requires residual_scale=0")
    simulator = L2FSimulator(L2FParams(dt=float(teacher_args.get("dt", student.config.dt))))
    train = _move_bank(build_dagger_scenario_bank(64, seed=args.seed, dt=simulator.params.dt), device)
    heldout = _move_bank(build_dagger_scenario_bank(64, seed=args.seed + 1, dt=simulator.params.dt), device)
    # The local-JVP target is built only from a fixed analytic-equilibrium
    # recurrent history.  Moving Q2 trajectories are collected separately as
    # a distribution-shift diagnostic and never enter the fit.
    train_snapshots = collect_equilibrium_history(
        teacher, student, simulator, train, snapshot_steps=(50, 75)
    )
    heldout_snapshots = collect_equilibrium_history(
        teacher, student, simulator, heldout, snapshot_steps=(50, 75)
    )
    train_shift_snapshots = collect_common_history(
        teacher, student, simulator, train, snapshot_steps=(50, 75)
    )
    heldout_shift_snapshots = collect_common_history(
        teacher, student, simulator, heldout, snapshot_steps=(50, 75)
    )
    derivative_batch, loss_history = fit_contextual_gain_local(
        student, teacher, train_snapshots, iterations=args.iterations,
        radii=LOCAL_RADII, seed=args.seed + 10,
    )
    heldout_batch = build_local_derivative_batch(
        teacher, student, heldout_snapshots, radii=LOCAL_RADII, seed=args.seed + 11
    )
    local_metrics = local_derivative_diagnostics(
        teacher, student, heldout_snapshots, heldout_batch
    )
    mode_metrics = local_mode_diagnostics(
        teacher, student, heldout_snapshots, heldout_batch
    )
    projection = projection_diagnostics(student, heldout_snapshots)
    h250 = lifted_h250_diagnostics(
        student, simulator, heldout, horizon=250, segment_length=25
    )
    h25 = lifted_h250_diagnostics(
        student, simulator, heldout, horizon=25, segment_length=25
    )
    calibration_valid = bool(student.capability_calibration_valid.item())
    e0_gate_passed = bool(
        local_metrics["equilibrium_action_trim_rms"] <= 1.3e-3
        and local_metrics["equilibrium_action_trim_max"] <= 5.0e-3
    )
    mode_rows = mode_metrics["by_authority_cell"]
    mode_cells_covered = bool(
        len(mode_rows) == 32
        and all(row["active_samples"] + row["inactive_samples"] > 0
                for row in mode_rows)
    )
    jvp_gate_passed = bool(
        mode_metrics["selected_gate_error_p95"] is not None
        and mode_metrics["selected_gate_error_p95"]
        <= LOCAL_PRECAL_SURROGATE_P95_THRESHOLD
        and mode_cells_covered
    )
    taylor_gate_passed = bool(
        local_metrics["taylor_remainder_over_radius2_max"] <= LOCAL_TAYLOR_R2_THRESHOLD
    )
    h250_gate_passed = bool(
        h250["finite"] and h250["finite_fraction"] >= 1.0
        and h250["max_omega"] <= 5.0
    )
    phase_b_gate_passed = bool(
        e0_gate_passed and jvp_gate_passed and taylor_gate_passed
        and projection["projection_bound_passed"] and h250_gate_passed
        and h25["finite"] and phase_a_ready
    )
    # Any contextual-gain weight update invalidates the inherited calibration.
    # The diagnostics above intentionally use the input q, but the emitted
    # checkpoint must be recalibrated before postcheck can consume it.
    student.invalidate_capability_calibration()
    # Preserve the complete Phase-A provenance (especially equilibrium and
    # calibration gates) so a later phase cannot accidentally treat this
    # checkpoint as an uncalibrated fresh student.
    report = dict(student_source.get("report", {}))
    report.update({
        "phase": "structured-local-gain-phase-B-prototype",
        "cadence_semantics_version": CADENCE_SEMANTICS_VERSION,
        "cadence_semantics": {
            "call_index_completed_transitions": True,
            "call0_has_response": False,
            "publication_calls": [50, 75],
            "availability_t25": [0, 0, 0, 0, 0, 0],
            "publication_rule_after_first": "positive slow_cadence offsets from call50",
            "t50_call_index": 50,
        },
        "active_phase": "B",
        "active_phase_components": ["contextual_gain_head", "local_jvp", "projection"],
        "phase_a_equilibrium_gate": student_source.get("report", {}).get("equilibrium_gate"),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "phase_a_checkpoint": str(args.checkpoint.resolve()),
        "screening_only": bool(args.allow_screening_only and not phase_a_ready),
        "scenario_count_train": train.count,
        "scenario_count_heldout": heldout.count,
        "phase_a_calibration_valid_input": calibration_valid,
        "phase_a_calibration_n_input": int(student_source.get("report", {}).get(
            "capability_calibration", {}
        ).get("n_calibration", 0)),
        "capability_calibration_valid": bool(student.capability_calibration_valid.item()),
        "requires_final_calibration": True,
        "authority_layout": "4x4 TW/log-alpha; 4 scenarios/cell",
        "snapshot_steps": [50, 75],
        "shift_diagnostic_snapshot_steps": [50, 75],
        "shift_diagnostic_snapshot_count_train": len(train_shift_snapshots),
        "shift_diagnostic_snapshot_count_heldout": len(heldout_shift_snapshots),
        "jvp_state_contract": (
            "teacher hidden fixed; student hidden/identifier/integral fixed; "
            "motor-error columns perturb observation previous_action and "
            "student state.motor_estimate together; previous_executed_action fixed"
        ),
        "equilibrium_burnin_contract": (
            "fixed analytic equilibrium observation; student observer receives "
            "analytic motor_trim_target; Q2 action updates labels/hidden only"
        ),
        "local_fit_semantics": (
            "deployment action-JVP when contextual confidence is active; "
            "otherwise explicit pre-gate Jacobian surrogate, with gate-aware "
            "deployment diagnostics retained"
        ),
        "radii": list(LOCAL_RADII),
        "directions_per_snapshot_radius": derivative_batch.direction_count,
        "loss_history": loss_history,
        "local_metrics": local_metrics,
        "local_mode_metrics": mode_metrics,
        "projection": projection,
        "h250_segments": h250,
        "h250_gate_passed": h250_gate_passed,
        "student_policy_deployment_hash": deployment_policy_hash(student),
        "lifted_25_step_diagnostic": h25,
        "K_ref_frozen": not any(name == "K_ref" for name, _ in student.named_parameters()),
        "residual_scale": float(student.config.residual_scale),
        "migration_gate_passed": False,
        "phase_b_gate_passed": phase_b_gate_passed,
        "formal_gates": {
            "e0_trim": e0_gate_passed,
            "jvp_normalized": jvp_gate_passed,
            "jvp_gate_semantics": "precalibration deployment-or-surrogate screen",
            "jvp_registered_p95_threshold": LOCAL_PRECAL_SURROGATE_P95_THRESHOLD,
            "jvp_all_authority_cells_covered": mode_cells_covered,
            "final_deployment_jvp_p95_threshold": LOCAL_JVP_P95_THRESHOLD,
            "taylor_remainder": taylor_gate_passed,
            "projection": bool(projection["projection_bound_passed"]),
            "h250_finite": bool(h250["finite"]),
            "phase_a_calibration": False,
        },
        "unsupported_gate": False,
        "unsupported_reasons": [
            "no full-policy JVP/KKT migration proof",
            "local derivative sample is directional finite-difference, not deployment certificate",
        ],
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
    print(json.dumps({"checkpoint": str(args.output), "report": str(args.report),
                      "migration_gate_passed": False}, sort_keys=True))


if __name__ == "__main__":
    main()

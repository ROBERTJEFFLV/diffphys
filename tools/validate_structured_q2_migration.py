"""Torch-only paired migration gates for a structured Q2 policy.

This validator deliberately runs the teacher and student from identical reset
states.  It records per-scenario evidence before reducing anything to a gate;
the resulting checkpoint contains only the student state dict.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import clone_state, load_q2_policy  # noqa: E402
from env_l2f import L2FParams, L2FSimulator, L2FState  # noqa: E402
from policy_observation import (  # noqa: E402
    build_policy_observation,
    initial_observation_state,
    update_position_integral,
)
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy  # noqa: E402
from structured_checkpoint import deployment_policy_hash  # noqa: E402
from structured_rollout import StructuredClosedLoopState, structured_observation  # noqa: E402


DEFAULT_Q2 = ROOT / "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt"
METRICS = ("position", "velocity", "omega")
METRIC_FLOORS = {"position": 0.05, "velocity": 0.10, "omega": 0.50}
SCENARIO_FIELDS = (
    "regime", "horizon", "scenario", "tail_position_q2", "tail_velocity_q2",
    "tail_omega_q2", "tail_position_structured", "tail_velocity_structured",
    "tail_omega_structured", "max_omega_q2", "max_omega_structured",
    "finite_q2", "finite_structured", "success_q2", "success_structured",
    "allocator_min_headroom", "allocator_headroom_violation",
    "allocator_wrench_residual_mean", "allocator_wrench_residual_max",
    "allocator_wrench_residual_p99",
    "identification_failed",
)
REQUIRED_PREREQUISITES = (
    "teacher_free_checkpoint", "calibration", "action_parity",
    "near_equilibrium_jvp", "equilibrium_gate", "allocator_gate",
)


def _device(name: str) -> torch.device:
    """Resolve ``auto`` before constructing torch.device."""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    return device


def _contains_teacher_weights(value: Any, *, teacher_branch: bool = False) -> bool:
    """Recursively reject teacher/model weights while allowing provenance.

    A ``teacher_checkpoint`` path or scalar teacher diagnostics are harmless;
    a tensor below a teacher/model/weights branch is not a student-only
    checkpoint and cannot be promoted.
    """

    weight_markers = (
        "state_dict", "model", "weight", "parameter", "network", "policy",
    )
    if torch.is_tensor(value):
        return teacher_branch
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            teacher_key = "teacher" in key_text
            marked_weight_key = any(marker in key_text for marker in weight_markers)
            branch = teacher_branch or (teacher_key and marked_weight_key)
            # A bare ``teacher`` mapping is conventionally a policy payload;
            # retain the strict check for nested tensors under it.
            if teacher_key and not marked_weight_key and key_text in {
                "teacher", "teacher_policy", "teacher_model",
            }:
                branch = True
            if _contains_teacher_weights(item, teacher_branch=branch):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_teacher_weights(item, teacher_branch=teacher_branch)
                   for item in value)
    return False


def _report_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    report = payload.get("report", {})
    return report if isinstance(report, Mapping) else {}


def _calibration_gate(policy: StructuredRecurrentPolicy,
                      payload: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    report = _report_mapping(payload)
    metadata = report.get("capability_calibration")
    if not isinstance(metadata, Mapping):
        metadata = payload.get("calibration_metadata", payload.get("capability_calibration"))
    if not isinstance(metadata, Mapping):
        return False, {"present": False, "promotion_gate_passed": False}
    current_hash = deployment_policy_hash(policy)
    promotion = bool(metadata.get("promotion_gate_passed", False))
    policy_hash = metadata.get("policy_config_hash")
    calibrated_hash = metadata.get("calibrated_parameter_hash")
    hash_match = policy_hash == current_hash and calibrated_hash == current_hash
    valid = bool(getattr(policy, "capability_calibration_valid").item())
    self_report = metadata.get("self_consistency")
    if isinstance(self_report, Mapping):
        self_consistency = bool(self_report.get(
            "passed", self_report.get("gate_passed", False)
        ))
    else:
        self_consistency = bool(
            metadata.get("self_calibration_converged", False)
            and metadata.get("q_recomputed_dominated_by_deployed_candidate", False)
        )
    return bool(promotion and hash_match and valid and self_consistency), {
        "present": True, "promotion_gate_passed": promotion,
        "policy_config_hash_match": policy_hash == current_hash,
        "calibrated_parameter_hash_match": calibrated_hash == current_hash,
        "calibration_buffer_valid": valid,
        "self_calibration_converged": bool(metadata.get("self_calibration_converged", False)),
        "q_recomputed_dominated_by_deployed_candidate": bool(
            metadata.get("q_recomputed_dominated_by_deployed_candidate", False)
        ),
        "self_consistency": self_consistency,
        "current_policy_config_hash": current_hash,
        "metadata_policy_config_hash": policy_hash,
        "metadata_calibrated_parameter_hash": calibrated_hash,
    }


def _layered_action_parity(payload: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Use source DAgger layered rows, never legacy replay-only parity."""

    report = _report_mapping(payload)
    postcheck = report.get("postcheck_action_parity")
    if report.get("evidence_stage") == "postcalibration":
        if not isinstance(postcheck, Mapping):
            return False, {"present": False, "reason": "missing postcheck parity evidence"}
        hash_a = postcheck.get("deployment_policy_hash")
        hash_b = report.get("deployment_policy_hash", report.get("evidence_hash"))
        same_hash = isinstance(hash_a, str) and hash_a == hash_b
        gate = bool(postcheck.get("gate_passed", False))
        return bool(gate and same_hash), {
            "present": True, "postcheck": True, "gate_passed": gate,
            "same_deployment_hash": same_hash,
            "deployment_policy_hash": hash_a,
            "away_samples": postcheck.get("away_samples"),
            "action_rms_threshold": postcheck.get("action_rms_threshold"),
        }
    rows = report.get("layered_gate_rows")
    markers = report.get("intervention_markers", {})
    if not isinstance(rows, list) or not rows:
        return False, {"present": False, "rows": 0}
    oracle = None
    for key in ("structured_residual_oracle", "heldout_action_parity",
                "heldout_report"):
        candidate = report.get(key)
        if isinstance(candidate, Mapping) and (
            "action_rms_threshold" in candidate
            or "whole_policy_action_rms_threshold" in candidate
        ):
            oracle = candidate
            break
    finite = all(bool(row.get("finite", False)) for row in rows if isinstance(row, Mapping))
    row_shape_ok = all(isinstance(row, Mapping) for row in rows)
    row_passed = all(bool(row.get("gate_passed", False)) for row in rows
                     if isinstance(row, Mapping))
    action_rms_max = max(
        (float(row.get("annulus_action_rms", row.get("action_rms", float("inf")))) for row in rows
         if isinstance(row, Mapping)), default=float("inf")
    )
    registered = bool(oracle and (
        oracle.get("pre_registered", False) or oracle.get("registered", False)
    ))
    threshold = float(oracle.get(
        "whole_policy_action_rms_threshold",
        oracle.get("action_rms_threshold", float("inf")),
    )) if oracle else float("inf")
    whole_policy_gate = bool(
        (oracle and oracle.get("whole_policy_gate_passed", False))
        or report.get("phase_c_whole_policy_gate_passed", False)
        or report.get("whole_policy_gate_passed", False)
    )
    no_final_teacher = (
        isinstance(markers, Mapping)
        and bool(markers.get("final_two_beta0", False))
        and int(report.get("final_teacher_execution_count", 1)) == 0
    )
    passed = bool(
        row_shape_ok and finite and row_passed and registered
        and math.isfinite(threshold) and threshold > 0.0
        and action_rms_max <= threshold and whole_policy_gate
        and report.get("active_phase") == "C" and no_final_teacher
    )
    return passed, {
        "present": True, "rows": len(rows), "all_rows_gate_passed": row_passed,
        "all_rows_finite": finite, "action_rms_max": action_rms_max,
        "pre_registered_oracle": registered, "action_rms_threshold": threshold,
        "whole_policy_gate_passed": whole_policy_gate,
        "active_phase": report.get("active_phase"),
        "final_two_beta0": bool(markers.get("final_two_beta0", False))
        if isinstance(markers, Mapping) else False,
        "final_teacher_execution_count": report.get("final_teacher_execution_count"),
        "layered_gate_passed": bool(report.get("layered_gate_passed", False)),
    }


def _allocator_gate(rows: Sequence[Mapping[str, Any]],
                   payload: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Require zero headroom violations and a registered residual release limit."""

    report = _report_mapping(payload)
    postcheck = report.get("postcheck_allocator")
    postcheck_gate = True
    postcheck_hash_ok = True
    if report.get("evidence_stage") == "postcalibration":
        postcheck_gate = bool(isinstance(postcheck, Mapping) and postcheck.get("gate_passed", False))
        postcheck_hash_ok = bool(
            isinstance(postcheck, Mapping)
            and postcheck.get("deployment_policy_hash") == report.get(
                "deployment_policy_hash", report.get("evidence_hash")
            )
        )
    oracle = None
    for key in ("structured_residual_oracle", "allocator_residual_oracle",
                "heldout_residual_report"):
        candidate = report.get(key)
        if isinstance(candidate, Mapping) and (
            "residual_p99_threshold" in candidate
            or "allocator_residual_p99_threshold" in candidate
        ):
            oracle = candidate
            break
    registered = bool(oracle and (
        oracle.get("pre_registered", False) or oracle.get("registered", False)
    ))
    threshold = float(oracle.get(
        "allocator_residual_p99_threshold",
        oracle.get("residual_p99_threshold", float("inf")),
    )) if oracle else float("inf")
    baseline = float(oracle.get(
        "baseline_residual_p99",
        oracle.get("residual_p99_baseline", float("nan")),
    )) if oracle else float("nan")
    multiplier = float(oracle.get(
        "relative_multiplier",
        oracle.get("residual_relative_multiplier", float("nan")),
    )) if oracle else float("nan")
    p99 = max((float(row.get("allocator_wrench_residual_p99", float("inf")))
               for row in rows), default=float("inf"))
    no_violation = all(float(row.get("allocator_headroom_violation", 1.0)) == 0.0
                       for row in rows)
    finite = all(math.isfinite(float(row.get("allocator_wrench_residual_p99", float("nan"))))
                 and math.isfinite(float(row.get("allocator_wrench_residual_max", float("nan"))))
                 for row in rows)
    threshold_ok = math.isfinite(threshold) and threshold > 0.0 and p99 <= threshold
    relative_ok = (
        math.isfinite(baseline) and baseline >= 0.0
        and math.isfinite(multiplier) and multiplier > 0.0
        and p99 <= baseline * multiplier
    )
    passed = bool(no_violation and finite and registered and threshold_ok and relative_ok
                  and postcheck_gate and postcheck_hash_ok)
    return passed, {
        "present": oracle is not None, "pre_registered_oracle": registered,
        "headroom_violation_zero": no_violation, "residual_finite": finite,
        "residual_p99_max": p99, "residual_p99_threshold": threshold,
        "baseline_residual_p99": baseline, "relative_multiplier": multiplier,
        "absolute_threshold_passed": threshold_ok,
        "relative_baseline_passed": relative_ok,
        "postcheck_gate_passed": postcheck_gate,
        "postcheck_same_deployment_hash": postcheck_hash_ok,
    }


def _explicit_report_gate(report: Mapping[str, Any], name: str) -> bool:
    for key in (name, f"{name}_passed", f"{name}_gate_passed", f"{name}_gate"):
        value = report.get(key)
        if isinstance(value, Mapping):
            value = value.get("gate_passed", value.get("passed", value.get("valid", False)))
        if isinstance(value, bool):
            return value
    gates = report.get("gates")
    if isinstance(gates, Mapping):
        for key in (name, f"{name}_passed", f"{name}_gate_passed", f"{name}_gate"):
            value = gates.get(key)
            if isinstance(value, bool):
                return value
    return False


def _augmented_stability_gate(payload: Mapping[str, Any], policy: StructuredRecurrentPolicy) -> Tuple[bool, Dict[str, Any]]:
    """Require fresh P1-6 evidence when consuming a post-calibration artifact."""

    report = _report_mapping(payload)
    if report.get("evidence_stage") != "postcalibration":
        return True, {"required": False, "reason": "non-postcalibration artifact"}
    evidence = report.get("augmented_closed_loop_stability")
    current_hash = deployment_policy_hash(policy)
    if not isinstance(evidence, Mapping):
        return False, {"required": True, "present": False,
                       "current_deployment_policy_hash": current_hash}
    rows = evidence.get("scenario_results")
    cells = evidence.get("by_authority_force_cell")
    row_hashes_ok = isinstance(rows, list) and all(
        isinstance(row, Mapping)
        and row.get("deployment_policy_hash") == current_hash
        and row.get("evidence_hash") == current_hash
        for row in rows
    )
    cell_hashes_ok = isinstance(cells, list) and len(cells) == 64 and all(
        isinstance(cell, Mapping)
        and cell.get("deployment_policy_hash") == current_hash
        and cell.get("evidence_hash") == current_hash
        for cell in cells
    )
    rows_gate_ok = isinstance(rows, list) and bool(rows) and all(
        bool(row.get("gate_passed", False)) and bool(row.get("finite", False))
        and bool(row.get("fixed_point_converged", False))
        and bool(row.get("fixed_point_anchor_passed", False))
        and isinstance(row.get("fixed_point_iterations"), int)
        and row.get("fixed_point_iterations", 0) >= 1
        and isinstance(row.get("fixed_point_anchor_hash"), str)
        and len(row.get("fixed_point_anchor_hash", "")) == 64
        for row in rows if isinstance(row, Mapping)
    ) and all(isinstance(row, Mapping) for row in rows)
    cells_gate_ok = isinstance(cells, list) and len(cells) == 64 and all(
        bool(cell.get("gate_passed", False)) and bool(cell.get("finite", False))
        and int(cell.get("samples", 0)) > 0
        for cell in cells if isinstance(cell, Mapping)
    ) and all(isinstance(cell, Mapping) for cell in cells)
    anchor_spec = evidence.get("fixed_point_anchor")
    anchor_spec_ok = isinstance(anchor_spec, Mapping) and (
        anchor_spec.get("method") == "projected_picard_on_25_step_poincare_map"
        and anchor_spec.get("residual_norm")
        == "normalized_non_yaw_non_discrete_l2"
        and isinstance(anchor_spec.get("max_iterations"), int)
        and anchor_spec.get("max_iterations", 0) >= 1
    )
    gate = bool(
        evidence.get("schema_version", 0) >= 2
        and evidence.get("poincare_map") == "complete_structured_boundary_codec_v2"
        and evidence.get("post_call") == 75
        and evidence.get("horizon_steps") == 25
        and evidence.get("evidence_hash") == current_hash
        and evidence.get("deployment_policy_hash") == current_hash
        and anchor_spec_ok
        and bool(evidence.get("gate_passed", False))
        and row_hashes_ok and cell_hashes_ok and rows_gate_ok and cells_gate_ok
    )
    return gate, {
        "required": True, "present": True, "gate_passed": gate,
        "schema_version": evidence.get("schema_version"),
        "post_call": evidence.get("post_call"),
        "horizon_steps": evidence.get("horizon_steps"),
        "scenario_count": len(rows) if isinstance(rows, list) else 0,
        "cell_count": len(cells) if isinstance(cells, list) else 0,
        "row_hashes_ok": row_hashes_ok, "cell_hashes_ok": cell_hashes_ok,
        "rows_gate_ok": rows_gate_ok, "cells_gate_ok": cells_gate_ok,
        "fixed_point_anchor_spec_ok": anchor_spec_ok,
        "current_deployment_policy_hash": current_hash,
        "evidence_deployment_policy_hash": evidence.get("deployment_policy_hash"),
        "evidence_hash": evidence.get("evidence_hash"),
    }


def _norm_history(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([torch.linalg.vector_norm(value, dim=-1) for value in values])


def cvar20(values: torch.Tensor) -> torch.Tensor:
    """Empirical mean of the largest 20 percent of per-scenario values."""
    flat = values.reshape(-1)
    if flat.numel() == 0:
        raise ValueError("values must be non-empty")
    count = max(1, int(math.ceil(0.20 * flat.numel())))
    return flat.topk(count, largest=True).values.mean()


def _success_from_tail(position: torch.Tensor, velocity: torch.Tensor,
                       omega: torch.Tensor, window: int = 100) -> torch.Tensor:
    tail = min(int(window), position.shape[0])
    return (
        (position[-tail:] < 0.05).all(dim=0)
        & (velocity[-tail:] < 0.10).all(dim=0)
        & (omega[-tail:] < 0.50).all(dim=0)
    )


@torch.no_grad()
def rollout_teacher(policy, initial: L2FState, horizon: int, simulator: L2FSimulator) -> Dict[str, torch.Tensor]:
    state = clone_state(initial)
    batch = state.position.shape[0]
    hidden = policy.initial_hidden(batch, device=state.position.device, dtype=state.position.dtype)
    observation_state = initial_observation_state(batch, device=state.position.device,
                                                  dtype=state.position.dtype)
    positions, velocities, omegas, actions = [], [], [], []
    for _ in range(horizon):
        observation, observed_position = build_policy_observation(
            state, observation_state, mode="integral25", integral_input_frame="body",
            noise_max=0.0,
        )
        action, hidden = policy(observation, hidden)
        observation_state = update_position_integral(
            observation_state, observed_position, dt=simulator.params.dt,
            integral_limit=0.5, integral_leak=0.0,
        )
        state = simulator.step(state, action, grad_decay=1.0)
        positions.append(state.position)
        velocities.append(state.velocity)
        omegas.append(state.omega)
        actions.append(action)
    return {
        "position": _norm_history(positions), "velocity": _norm_history(velocities),
        "omega": _norm_history(omegas), "action": torch.stack(actions),
    }


@torch.no_grad()
def rollout_structured(policy: StructuredRecurrentPolicy, initial: L2FState,
                       horizon: int, simulator: L2FSimulator) -> Dict[str, torch.Tensor]:
    physical = clone_state(initial)
    batch = physical.position.shape[0]
    observation = torch.cat(
        (
            physical.position, physical.velocity,
            physical.rotation.reshape(batch, 9), physical.omega,
            torch.zeros_like(physical.position), physical.previous_action,
        ), dim=-1
    )
    policy_state = policy.initial_state(observation)
    positions, velocities, omegas, actions = [], [], [], []
    headrooms, residuals, identification = [], [], []
    closed = StructuredClosedLoopState(physical, policy_state)
    for _ in range(horizon):
        observation = structured_observation(closed)
        output = policy.forward_with_aux(observation, closed.policy, simulator.params.dt)
        physical = simulator.step(closed.physical, output.action, grad_decay=1.0)
        closed = StructuredClosedLoopState(physical, output.next_state)
        positions.append(physical.position)
        velocities.append(physical.velocity)
        omegas.append(physical.omega)
        actions.append(output.action)
        allocator = output.auxiliary["allocator"]
        headrooms.append(allocator.minimum_headroom)
        residuals.append(allocator.wrench_residual)
        identification.append(output.auxiliary["identification_failed"])
    return {
        "position": _norm_history(positions), "velocity": _norm_history(velocities),
        "omega": _norm_history(omegas), "action": torch.stack(actions),
        "headroom": torch.stack(headrooms), "wrench_residual": torch.stack(residuals),
        "identification_failed": torch.stack(identification).bool(),
    }


def _scenario_rows(regime: str, horizon: int, teacher: Mapping[str, torch.Tensor],
                   student: Mapping[str, torch.Tensor]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    position_t, velocity_t, omega_t = (teacher[name] for name in METRICS)
    position_s, velocity_s, omega_s = (student[name] for name in METRICS)
    success_t = _success_from_tail(position_t, velocity_t, omega_t)
    success_s = _success_from_tail(position_s, velocity_s, omega_s)
    # Norm histories are [time, batch]; keep the batch axis so finite is
    # reported per scenario rather than accidentally reducing it away.
    finite_t = torch.stack(
        tuple(torch.isfinite(teacher[name]) for name in METRICS), dim=-1
    ).all(dim=(0, 2))
    finite_s = torch.stack(
        tuple(torch.isfinite(student[name]) for name in METRICS), dim=-1
    ).all(dim=(0, 2))
    finite_t &= torch.isfinite(teacher["action"]).all(dim=(0, 2))
    finite_s &= (
        torch.isfinite(student["action"]).all(dim=(0, 2))
        & torch.isfinite(student["headroom"]).all(dim=0)
        & torch.isfinite(student["wrench_residual"]).all(dim=0)
    )
    identification = student["identification_failed"].any(dim=0)
    for index in range(position_t.shape[1]):
        tail = min(100, horizon)
        row: Dict[str, Any] = {
            "regime": regime, "horizon": horizon, "scenario": index,
            "tail_position_q2": float(position_t[-tail:, index].mean()),
            "tail_velocity_q2": float(velocity_t[-tail:, index].mean()),
            "tail_omega_q2": float(omega_t[-tail:, index].mean()),
            "tail_position_structured": float(position_s[-tail:, index].mean()),
            "tail_velocity_structured": float(velocity_s[-tail:, index].mean()),
            "tail_omega_structured": float(omega_s[-tail:, index].mean()),
            "max_omega_q2": float(omega_t[:, index].max()),
            "max_omega_structured": float(omega_s[:, index].max()),
            "finite_q2": bool(finite_t[index]), "finite_structured": bool(finite_s[index]),
            "success_q2": bool(success_t[index]), "success_structured": bool(success_s[index]),
            "allocator_min_headroom": float(student["headroom"][:, index].min()),
            "allocator_headroom_violation": float((student["headroom"][:, index] < 0).any()),
            "allocator_wrench_residual_mean": float(student["wrench_residual"][:, index].mean()),
            "allocator_wrench_residual_max": float(student["wrench_residual"][:, index].max()),
            "allocator_wrench_residual_p99": float(torch.quantile(
                student["wrench_residual"][:, index], 0.99
            )),
            "identification_failed": bool(identification[index]),
        }
        rows.append(row)
    return rows


def _ratio(student: torch.Tensor, teacher: torch.Tensor, floor: float) -> float:
    return float(student.mean() / teacher.mean().clamp_min(float(floor)))


def migration_gates(rows: Sequence[Mapping[str, Any]], *, prerequisites: Mapping[str, bool]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("rows must be non-empty")
    gates: Dict[str, Any] = {}
    gates["finite_100pct"] = all(bool(row["finite_q2"] and row["finite_structured"]) for row in rows)
    gates["success_not_below_q2"] = (
        sum(bool(row["success_structured"]) for row in rows)
        >= sum(bool(row["success_q2"]) for row in rows)
    )
    for metric, floor in METRIC_FLOORS.items():
        teacher = torch.tensor([row[f"tail_{metric}_q2"] for row in rows])
        student = torch.tensor([row[f"tail_{metric}_structured"] for row in rows])
        gates[f"{metric}_mean_tail_ratio_le_1.10"] = _ratio(student, teacher, floor) <= 1.10
        gates[f"{metric}_cvar20_ratio_le_1.15"] = float(cvar20(student) / cvar20(teacher).clamp_min(floor)) <= 1.15
        gates[f"{metric}_mean_tail_ratio"] = _ratio(student, teacher, floor)
        gates[f"{metric}_cvar20_ratio"] = float(cvar20(student) / cvar20(teacher).clamp_min(floor))
    for regime in ("natural", "balanced"):
        for horizon in (250, 500):
            subset = [row for row in rows if row["regime"] == regime and row["horizon"] == horizon]
            if not subset:
                gates[f"{regime}_H{horizon}_present"] = False
                continue
            gates[f"{regime}_H{horizon}_success_not_below_q2"] = (
                sum(bool(row["success_structured"]) for row in subset)
                >= sum(bool(row["success_q2"]) for row in subset)
            )
            for metric, floor in METRIC_FLOORS.items():
                teacher = torch.tensor([row[f"tail_{metric}_q2"] for row in subset])
                student = torch.tensor([row[f"tail_{metric}_structured"] for row in subset])
                gates[f"{regime}_H{horizon}_{metric}_ratio_le_1.10"] = _ratio(student, teacher, floor) <= 1.10
                gates[f"{regime}_H{horizon}_{metric}_cvar20_le_1.15"] = float(
                    cvar20(student) / cvar20(teacher).clamp_min(floor)
                ) <= 1.15
    max_ok = all(
        float(row["max_omega_structured"]) <= max(5.0, 1.5 * float(row["max_omega_q2"]))
        for row in rows
    )
    gates["paired_max_omega"] = max_ok
    gates["identification_failures_zero"] = not any(bool(row["identification_failed"]) for row in rows)
    # Do not allow a caller to accidentally omit a release prerequisite.  A
    # missing calibration/runtime/parity record is a failed gate, not an
    # implicit smoke override.
    gates.update({
        f"prerequisite_{name}": bool(prerequisites.get(name, False))
        for name in REQUIRED_PREREQUISITES
    })
    gates["migration_gate_passed"] = all(
        bool(value) for name, value in gates.items()
        if name != "migration_gate_passed" and not name.endswith("_ratio")
    )
    return gates


def _load_structured(path: Path, device: torch.device) -> Tuple[StructuredRecurrentPolicy, Dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = StructuredPolicyConfig(**payload["config"])
    policy = StructuredRecurrentPolicy(config).to(device=device, dtype=torch.float32)
    policy.load_state_dict(payload["model"], strict=True)
    policy.eval()
    return policy, payload


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else list(SCENARIO_FIELDS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    device = _device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    teacher, teacher_args = load_q2_policy(args.teacher_checkpoint, device=device, dtype=torch.float32)
    student, student_payload = _load_structured(args.student_checkpoint, device)
    simulator = L2FSimulator(L2FParams(dt=float(teacher_args.get("dt", 0.01))))
    all_rows: List[Dict[str, Any]] = []
    rollout_completed = True
    try:
        for regime, balanced in (("natural", False), ("balanced", True)):
            initial = simulator.reset(
                256, device=device, dtype=torch.float32, sample_dynamics=True,
                sampled_dynamics_level="broad", broad_sampler="physical-fit",
                balanced_dynamics_sampling=balanced, sample_external_force=True,
            )
            for horizon in (250, 500):
                teacher_trace = rollout_teacher(teacher, initial, horizon, simulator)
                student_trace = rollout_structured(student, initial, horizon, simulator)
                all_rows.extend(_scenario_rows(regime, horizon, teacher_trace, student_trace))
    except (RuntimeError, ValueError, FloatingPointError):
        rollout_completed = False

    report_metadata = _report_mapping(student_payload)
    teacher_free = not _contains_teacher_weights(student_payload)
    calibration, calibration_details = _calibration_gate(student, student_payload)
    action_parity, action_parity_details = _layered_action_parity(student_payload)
    evidence_hash = report_metadata.get("deployment_policy_hash", report_metadata.get("evidence_hash"))
    if report_metadata.get("evidence_stage") == "postcalibration" and evidence_hash != deployment_policy_hash(student):
        action_parity = False
        action_parity_details = {
            **action_parity_details,
            "same_deployed_checkpoint_hash": False,
            "current_deployment_policy_hash": deployment_policy_hash(student),
            "report_deployment_policy_hash": evidence_hash,
        }
    allocator_passed, allocator_details = _allocator_gate(all_rows, student_payload)
    near_equilibrium_jvp = _explicit_report_gate(report_metadata, "near_equilibrium_jvp")
    equilibrium_gate = _explicit_report_gate(report_metadata, "equilibrium_gate")
    augmented_stability, augmented_stability_details = _augmented_stability_gate(
        student_payload, student
    )
    prerequisites = {
        "teacher_free_checkpoint": teacher_free,
        "calibration": calibration,
        "action_parity": action_parity,
        "near_equilibrium_jvp": near_equilibrium_jvp,
        "equilibrium_gate": equilibrium_gate,
        "allocator_gate": allocator_passed,
    }
    gates = migration_gates(all_rows, prerequisites=prerequisites) if all_rows else {
        "migration_gate_passed": False, "finite_100pct": False,
        **{f"prerequisite_{name}": bool(value)
           for name, value in prerequisites.items()},
    }
    gates["rollout_complete"] = rollout_completed
    # The post-calibration artifact is not consumable unless its complete-state
    # P1-6 evidence is present, fresh, and hash-bound to this exact policy.
    gates["prerequisite_augmented_closed_loop_stability"] = augmented_stability
    gates["migration_gate_passed"] = bool(
        gates.get("migration_gate_passed", False)
        and rollout_completed and augmented_stability
    )
    payload: Dict[str, Any] = {
        "phase": "structured-q2-torch-migration-validation",
        "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
        "student_checkpoint": str(Path(args.student_checkpoint).resolve()),
        "device": str(device), "seed": args.seed,
        "scenario_count_per_regime": 256, "regimes": ["natural", "balanced"],
        "horizons": [250, 500], "prerequisites": prerequisites, "gates": gates,
        "migration_gate_passed": bool(gates.get("migration_gate_passed", False)),
        "scenario_rows": len(all_rows),
        "rollout_completed": rollout_completed,
        "teacher_free_checkpoint": teacher_free,
        "calibration_details": calibration_details,
        "action_parity_details": action_parity_details,
        "allocator_details": allocator_details,
        "augmented_stability": augmented_stability,
        "augmented_stability_details": augmented_stability_details,
        "near_equilibrium_jvp": near_equilibrium_jvp,
        "equilibrium_gate": equilibrium_gate,
    }
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.csv_output, all_rows)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_checkpoint is not None and payload["migration_gate_passed"]:
        args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        release_report = dict(report_metadata)
        release_report.update({
            "migration_gate_passed": True,
            "migration_validator": "torch-structured-q2-v1",
            "migration_gates": gates,
            "migration_prerequisites": prerequisites,
        })
        torch.save({
            "architecture": "structured-recurrent-motor-policy",
            "model": student.state_dict(), "config": student_payload["config"],
            "report": release_report,
            "migration_gate_passed": True,
            "migration_validator": "torch-structured-q2-v1",
        }, args.output_checkpoint)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_Q2)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=ROOT / "runs/structured_q2_migration.csv")
    parser.add_argument("--json-output", type=Path, default=ROOT / "runs/structured_q2_migration.json")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    result = validate(parse_args())
    print(json.dumps({"migration_gate_passed": result["migration_gate_passed"],
                      "scenario_rows": result["scenario_rows"]}, sort_keys=True))

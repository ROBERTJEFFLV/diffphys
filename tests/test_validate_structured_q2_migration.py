from __future__ import annotations

import torch

from tools.validate_structured_q2_migration import (
    _contains_teacher_weights,
    _allocator_gate,
    _layered_action_parity,
    cvar20,
    migration_gates,
    _scenario_rows,
)
from structured_checkpoint import deployment_policy_hash
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy
from tools.calibrate_structured_capability import _policy_config_hash


def _rows(student_scale: float = 1.0):
    rows = []
    for scenario, (regime, horizon) in enumerate(
        ([(regime, horizon) for regime in ("natural", "balanced")
          for horizon in (250, 500)] * 4)
    ):
        row = {
            "regime": regime, "horizon": horizon, "scenario": scenario,
            "finite_q2": True, "finite_structured": True,
            "success_q2": True, "success_structured": True,
            "identification_failed": False,
            "max_omega_q2": 1.0, "max_omega_structured": 1.0,
        }
        for name, value in (("position", 0.05), ("velocity", 0.10), ("omega", 0.50)):
            row[f"tail_{name}_q2"] = value
            row[f"tail_{name}_structured"] = value * student_scale
        rows.append(row)
    return rows


def test_cvar20_selects_largest_twenty_percent() -> None:
    values = torch.arange(10.0)
    assert cvar20(values) == torch.tensor(8.5)


def test_scenario_rows_preserve_batch_finite_and_allocator_evidence() -> None:
    teacher = {
        "position": torch.ones(3, 2) * 0.02,
        "velocity": torch.ones(3, 2) * 0.03,
        "omega": torch.ones(3, 2) * 0.04,
        "action": torch.zeros(3, 2, 4),
    }
    student = {
        "position": torch.ones(3, 2) * 0.02,
        "velocity": torch.ones(3, 2) * 0.03,
        "omega": torch.ones(3, 2) * 0.04,
        "action": torch.zeros(3, 2, 4),
        "headroom": torch.ones(3, 2) * 0.2,
        "wrench_residual": torch.ones(3, 2) * 0.01,
        "identification_failed": torch.zeros(3, 2, dtype=torch.bool),
    }
    rows = _scenario_rows("natural", 250, teacher, student)
    assert len(rows) == 2
    assert all(row["finite_q2"] and row["finite_structured"] for row in rows)
    assert all(abs(row["allocator_min_headroom"] - 0.2) < 1.0e-6 for row in rows)


def test_migration_gates_require_prerequisites_and_thresholds() -> None:
    prerequisites = {
        "teacher_free_checkpoint": True, "calibration": True,
        "action_parity": True, "near_equilibrium_jvp": True,
        "equilibrium_gate": True, "allocator_gate": True,
    }
    good = migration_gates(
        _rows(), prerequisites=prerequisites
    )
    assert good["migration_gate_passed"]
    missing = migration_gates(
        _rows(), prerequisites={**prerequisites, "calibration": False}
    )
    assert not missing["migration_gate_passed"]
    absent = migration_gates(_rows(), prerequisites={})
    assert not absent["migration_gate_passed"]
    assert not absent["prerequisite_action_parity"]


def test_migration_gates_reject_cvar_and_identification_failure() -> None:
    rows = _rows(student_scale=1.2)
    rows[0]["identification_failed"] = True
    gates = migration_gates(
        rows, prerequisites={
            "teacher_free_checkpoint": True, "calibration": True,
            "action_parity": True, "near_equilibrium_jvp": True,
            "equilibrium_gate": True, "allocator_gate": True,
        }
    )
    assert not gates["omega_cvar20_ratio_le_1.15"]
    assert not gates["identification_failures_zero"]
    assert not gates["migration_gate_passed"]


def test_teacher_weights_are_rejected_but_provenance_is_allowed() -> None:
    assert not _contains_teacher_weights({
        "teacher_checkpoint": "/tmp/q2.pt", "report": {"teacher_actions": [0.1]},
    })
    assert _contains_teacher_weights({
        "teacher_state_dict": {"gru.weight": torch.zeros(2, 2)},
    })


def test_deployment_hash_includes_calibration_q_valid_and_sample_count() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig())
    base = deployment_policy_hash(policy)
    assert _policy_config_hash(policy) == base
    with torch.no_grad():
        policy.capability_conformal_q[0] += 0.1
    assert deployment_policy_hash(policy) != base
    with torch.no_grad():
        policy.capability_conformal_q[0] -= 0.1
        policy.capability_calibration_valid.fill_(True)
    assert deployment_policy_hash(policy) != base
    assert _policy_config_hash(policy) == deployment_policy_hash(policy)


def test_action_parity_requires_phase_c_whole_policy_oracle() -> None:
    rows = [{"finite": True, "gate_passed": True, "action_rms": 0.001}]
    payload = {"report": {
        "active_phase": "C", "layered_gate_passed": True,
        "layered_gate_rows": rows,
        "intervention_markers": {"final_two_beta0": True},
        "final_teacher_execution_count": 0,
        "structured_residual_oracle": {
            "pre_registered": True, "whole_policy_gate_passed": True,
            "action_rms_threshold": 0.002,
        },
    }}
    assert _layered_action_parity(payload)[0]
    payload["report"]["active_phase"] = "B"
    assert not _layered_action_parity(payload)[0]


def test_postcheck_action_parity_is_hash_bound_and_cannot_use_phase_c_rows() -> None:
    payload = {"report": {
        "evidence_stage": "postcalibration",
        "deployment_policy_hash": "h1",
        "postcheck_action_parity": {
            "gate_passed": True, "deployment_policy_hash": "h1",
            "away_samples": 128, "action_rms_threshold": 0.2,
        },
        # A stale/failed Phase-C row must not override the fresh postcheck.
        "layered_gate_rows": [{"finite": False, "gate_passed": False}],
    }}
    assert _layered_action_parity(payload)[0]
    payload["report"]["postcheck_action_parity"]["deployment_policy_hash"] = "stale"
    assert not _layered_action_parity(payload)[0]


def test_allocator_gate_requires_zero_headroom_and_registered_absolute_relative_limits() -> None:
    rows = [{
        "allocator_headroom_violation": 0.0,
        "allocator_wrench_residual_p99": 0.04,
        "allocator_wrench_residual_max": 0.05,
    }]
    payload = {"report": {"structured_residual_oracle": {
        "pre_registered": True, "residual_p99_threshold": 0.1,
        "baseline_residual_p99": 0.05, "relative_multiplier": 1.1,
    }}}
    assert _allocator_gate(rows, payload)[0]
    rows[0]["allocator_headroom_violation"] = 1.0
    assert not _allocator_gate(rows, payload)[0]

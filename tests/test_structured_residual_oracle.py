from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from structured_checkpoint import deployment_policy_hash
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy
from tools.distill_structured_dagger import _validate_external_h250_report
from tools.validate_structured_residual_oracle import _annulus_stats
from tools.postcalibration_local_evidence import _registered_oracle


def _episode() -> SimpleNamespace:
    return SimpleNamespace(
        student_actions=torch.tensor([
            [[0.1, 0.0, 0.0, 0.0]],
            [[0.2, 0.0, 0.0, 0.0]],
        ]),
        teacher_actions=torch.zeros(2, 1, 4),
    )


def test_oracle_separates_near_equilibrium_bias_from_annulus_rms() -> None:
    features = torch.zeros(2, 1, 15)
    features[0, 0, 0] = 0.1
    features[1, 0, 0] = 1.0
    stats = _annulus_stats(_episode(), features, 0.25, 3.0)
    assert stats["near_equilibrium_count"] == 1
    assert stats["annulus_count"] == 1
    assert abs(stats["oracle_rms"] - 0.1) < 1.0e-6
    assert abs(stats["near_equilibrium_q2_bias_rms"] - 0.05) < 1.0e-6


def test_phase_c_oracle_report_requires_matching_phase_b_hash(tmp_path: Path) -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig())
    report = {
        "phase": "structured-residual-oracle",
        "phase_b_deployment_hash": deployment_policy_hash(policy),
        "oracle_gate_passed": True,
        "annulus": {"r_min": 0.25, "r_max": 3.0},
        "preregistered_action_rms_threshold": 0.1,
    }
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    details = _validate_external_h250_report(path, policy)
    assert details["r_min"] == 0.25
    report["phase_b_deployment_hash"] = "mismatch"
    path.write_text(json.dumps(report), encoding="utf-8")
    try:
        _validate_external_h250_report(path, policy)
    except RuntimeError as exc:
        assert "hash" in str(exc)
    else:
        raise AssertionError("mismatched oracle hash unexpectedly accepted")


def test_postcheck_requires_complete_registered_oracle_contract() -> None:
    contract = {
        "schema_version": 1, "pre_registered": True,
        "phase_b_deployment_hash": "phase-b",
        "annulus": {"r_min": 0.25, "r_max": 3.0},
        "action_rms_threshold": 0.1,
        "allocator_residual_p99_threshold": 0.2,
        "baseline_residual_p99": 0.1, "relative_multiplier": 2.0,
    }
    result = _registered_oracle({"structured_residual_oracle": contract})
    assert result["r_min"] == 0.25
    assert result["allocator_threshold"] == 0.2

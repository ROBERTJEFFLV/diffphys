from __future__ import annotations

from types import SimpleNamespace

import torch

from probe_contract import (
    SHARED_TAU_MIN_WEIGHTED_FISHER_INFORMATION,
    WAVEFORM_SHA256,
    waveform_metadata,
    waveform_tensor,
)
from structured_policy import identification_probe_patterns
from tools.diagnose_probe_v4 import (
    _v4_gate,
    run,
    shared_tau_diagnostics,
    validate_shared_tau_configuration,
)


def _pooled_support_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    command = torch.zeros(50, 1, 4)
    motor = torch.zeros_like(command)
    command[:8, 0, 0] = 0.005
    command[8:12, 0, 1] = 0.005
    command[20:28, 0, 0] = -0.005
    command[28:32, 0, 1] = -0.005
    return command, motor


def test_shared_tau_pooled_gate_allows_per_motor_imbalance() -> None:
    command, motor = _pooled_support_fixture()
    result = shared_tau_diagnostics(command, motor)
    rise = result["per_scene"][0]["branches"]["rise"]
    assert rise["pooled_valid"] == 12
    assert rise["distinct_time_calls"] == 12
    assert rise["motors_contributing"] == 2
    assert rise["per_motor_counts"] == [8, 4, 0, 0]
    assert rise["gate_passed"]


def test_shared_tau_gate_rejects_zero_pooled_branch() -> None:
    command = torch.full((50, 1, 4), 0.005)
    motor = torch.zeros_like(command)
    result = shared_tau_diagnostics(command, motor)
    assert not result["gate_passed"]
    assert result["per_scene"][0]["branches"]["fall"]["pooled_valid"] == 0


def test_artificial_per_motor_tau_configuration_is_rejected() -> None:
    rising = torch.tensor([0.05, 0.05, 0.06, 0.05])
    falling = torch.full((4,), 0.10)
    result = validate_shared_tau_configuration(rising, falling)
    assert not result["gate_passed"]
    assert result["rising_spread"] > 0.0


def test_policy_probe_is_exactly_the_shared_v4_artifact() -> None:
    policy_probe = identification_probe_patterns(device=torch.device("cpu"), dtype=torch.float64)
    torch.testing.assert_close(policy_probe, waveform_tensor(dtype=torch.float64))
    assert len(WAVEFORM_SHA256) == 64
    assert (
        waveform_metadata()["shared_tau_min_weighted_fisher_information"]
        == SHARED_TAU_MIN_WEIGHTED_FISHER_INFORMATION
    )


def test_v4_gate_path_replaces_per_motor_check_with_shared_tau(monkeypatch) -> None:
    command, motor = _pooled_support_fixture()
    raw = {"command": command.tolist(), "motor_before": motor.tolist(),
           "motor_after": motor.tolist()}
    section = {
        metric: {"mean": 0.0, "p99": 0.0}
        for metric in ("position", "velocity", "omega")
    }
    probe = {
        "raw_transition": raw,
        "h75": section, "h125": section, "tail_h75_h125": section,
        "stats": {"position_max": 0.0, "velocity_max": 0.0, "omega_max": 0.0},
        "energy_retention_modal": [[1.0]],
        "standardized_X": {}, "coriolis": {},
    }
    zero = {
        "h75": section, "h125": section, "tail_h75_h125": section,
        "stats": {"position_max": 0.0, "velocity_max": 0.0, "omega_max": 0.0},
    }
    row = {"seed": 3707, "probe": probe, "zero": zero,
           "zero_arm_canonical_parity_max_abs": 0.0}
    checks = {"seed_results": [{"seed": 3707,
                                "checks": {"support_each_motor_rise_fall_ge_3": False,
                                            "physical_support_upper_bound_ge_3": False,
                                            "finite": True}}]}
    monkeypatch.setattr("tools.diagnose_probe_v4._formal_gate", lambda rows: (False, checks))
    passed, result = _v4_gate([row])
    assert passed
    assert result["seed_results"][0]["checks"]["shared_tau_pooled_time_motor_fisher"]
    assert "support_each_motor_rise_fall_ge_3" not in result["seed_results"][0]["checks"]


def test_formal_freeze_can_only_promote_registered_waveform(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "q2.pt"
    checkpoint.write_bytes(b"q2")

    def fake_rollout(_checkpoint, seed, scenarios, horizon, waveform):
        is_registered = torch.equal(waveform, waveform_tensor(dtype=waveform.dtype))
        return {
            "seed": seed,
            "probe": {
                "rise_support": [[3, 3, 3, 3]],
                "fall_support": [[3, 3, 3, 3]],
                "standardized_X": {"condition_max": 1.0 if is_registered else 2.0},
            },
        }

    monkeypatch.setattr("tools.diagnose_probe_v4._rollout_one", fake_rollout)
    monkeypatch.setattr(
        "tools.diagnose_probe_v4._v4_gate",
        lambda rows: (True, {"seed_results": [{"passed": True} for _ in rows]}),
    )
    report = run(SimpleNamespace(
        dry_run=False, checkpoint=checkpoint, scenarios=16, horizon=125,
        n_jobs=1, formal_freeze=True,
    ))
    assert report["status"] == "formal_frozen"
    assert report["gate_passed"] is True
    assert report["formal"]["eligible"] is True
    assert report["formal"]["frozen_sha256"] == WAVEFORM_SHA256
    assert report["seed_split"]["blind_consumed"] == []

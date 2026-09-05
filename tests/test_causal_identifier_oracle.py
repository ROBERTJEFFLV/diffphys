from __future__ import annotations

import hashlib
import math
import json
from pathlib import Path

import torch

from probe_contract import (
    WAVEFORM,
    WAVEFORM_SHA256,
    canonical_waveform_json,
    waveform_metadata,
)
from tools.diagnose_causal_identifier_oracle import (
    DEFAULT_Q2_CHECKPOINT,
    FORMAL_TRAIN_SEEDS,
    FORMAL_VALIDATION_SEED,
    PUBLICATION_STEPS,
    ZERO_PARITY_TOLERANCE,
    _collect_one,
    exact_physics_block_fit,
    initialize_observer_bank,
    load_probe_waveform,
    coverage_gate_failed_checks,
    physics_gate_failed_checks,
    residual_probe_step,
    probe_v4_eligibility,
    shared_tau_support,
    _hash_file,
)


def test_observer_bank_initializes_from_deployable_previous_action() -> None:
    previous = torch.tensor([[0.2, -0.3, 0.4, -0.5]])
    legacy, k15, k35 = initialize_observer_bank(previous, ((0.04, 0.08),) * 3)
    torch.testing.assert_close(legacy, previous)
    torch.testing.assert_close(k15, previous[:, None, :].expand_as(k15))
    torch.testing.assert_close(k35, previous[:, None, :].expand_as(k35))


def test_q2_checkpoint_smoke_zero_arm_parity_and_cadence() -> None:
    assert DEFAULT_Q2_CHECKPOINT.is_file()
    row = _collect_one(DEFAULT_Q2_CHECKPOINT, 3707, 0.0, 16, 126)
    assert row["q2_observation"] == {
        "mode": "integral25",
        "integral_input_frame": "body",
        "integral_input_multiplier": 1.0,
        "noise_max": 0.0,
        "integral_limit": 0.5,
        "integral_leak": 0.0,
        "integral_clamp_mode": "box",
        "dt": 0.01,
    }
    assert row["zero_parity"]["gate_passed"]
    assert max(row["zero_parity"][key] for key in (
        "max_action_error", "max_state_error", "max_integral_error"
    )) <= ZERO_PARITY_TOLERANCE
    assert row["features"]["S"].shape[:3] == (3, 16, 25)
    # Publication 25 contains transitions 1..25, and uses the post-step
    # observer bank with the response from that same transition.
    assert row["commands"].shape == (126, 16, 4)
    torch.testing.assert_close(row["previous_action_seen"][1], row["commands"][0])
    assert row["response_windows"].shape[:3] == (3, 16, 25)
    assert row["canonical_actions"].shape == row["q_actions"].shape
    assert tuple(PUBLICATION_STEPS) == (25, 100, 125)


def test_q2_residual_probe_is_bounded_and_actual_action_is_replayed() -> None:
    row = _collect_one(DEFAULT_Q2_CHECKPOINT, 3707, 0.005, 16, 126)
    residual = row["residual_probe"]
    prior = torch.cat((torch.zeros_like(residual[:1]), residual[:-1]), dim=0)
    assert float(residual.abs().max()) <= 0.005 + 1.0e-7
    assert float((residual - prior).abs().max()) <= 0.005 + 1.0e-7
    expected = row["q_actions"] + residual
    torch.testing.assert_close(row["commands"], expected)
    assert bool(((row["commands"] >= -1.0) & (row["commands"] <= 1.0)).all())
    torch.testing.assert_close(row["requested_probe"][75:], torch.zeros_like(row["requested_probe"][75:]))


def test_sol_residual_clip_uses_annulus_headroom_and_slews_back_after_call49() -> None:
    previous = torch.tensor([[0.004, -0.004, 0.005, -0.005]])
    q2 = torch.tensor([[1.0, -1.0, 0.0, 0.25]])
    waveform = torch.ones(50, 4)
    residual = residual_probe_step(previous, 0, 0.005, q_action=q2, waveform=waveform)
    assert float(residual[0, 0]) == 0.0
    assert bool(((q2 + residual) >= -1.0).all() and ((q2 + residual) <= 1.0).all())
    returned = residual_probe_step(torch.full((1, 4), 0.005), 50, 0.005,
                                   q_action=torch.zeros(1, 4), waveform=waveform)
    torch.testing.assert_close(returned, torch.zeros(1, 4))


def test_probe_waveform_file_must_match_v4_artifact(tmp_path) -> None:
    path = tmp_path / "waveform.json"
    path.write_text(str([list(row) for row in WAVEFORM]).replace("'", "\""), encoding="utf-8")
    waveform = load_probe_waveform(path)
    assert waveform.shape == (50, 4)
    torch.testing.assert_close(waveform, torch.tensor(WAVEFORM, dtype=torch.float32))
    bad = tmp_path / "bad_waveform.json"
    bad.write_text("[" + ",".join(["[0, 0, 0, 0]"] * 50) + "]", encoding="utf-8")
    try:
        load_probe_waveform(bad)
    except ValueError as error:
        assert "unique probe_contract.py v4 artifact" in str(error)
    else:  # pragma: no cover - defensive assertion for a fail-closed gate
        raise AssertionError("alternate waveform unexpectedly accepted")


def test_shared_tau_support_pools_time_and_motors_without_per_motor_gate() -> None:
    command = torch.zeros(50, 4)
    motor = torch.zeros_like(command)
    command[:8, 0] = 0.005
    command[8:12, 1] = 0.005
    command[20:28, 0] = -0.005
    command[28:32, 1] = -0.005
    result = shared_tau_support(command, motor)
    assert result["branches"]["rise"]["per_motor_counts"][:2].tolist() == [8, 4]
    assert result["branches"]["rise"]["gate_passed"]
    assert result["branches"]["fall"]["gate_passed"]


def test_shared_tau_support_has_explicit_information_floor() -> None:
    command = torch.zeros(50, 4)
    motor = torch.zeros_like(command)
    # Count/call/motor support passes, but every row is below the v4
    # amplitude/2 threshold and therefore contributes no information.
    command[:6, 0] = 0.001
    command[6:12, 1] = 0.001
    command[20:26, 0] = -0.001
    command[26:32, 1] = -0.001
    result = shared_tau_support(command, motor)
    assert not result["gate_passed"]
    assert result["branches"]["rise"]["weighted_fisher_information"] == 0.0


def test_probe_v4_eligibility_is_fail_closed_without_frozen_candidate(tmp_path) -> None:
    report = tmp_path / "probe.json"
    report.write_text(json.dumps({
        "contract": waveform_metadata(),
        "waveform_sha256": WAVEFORM_SHA256,
        "formal_frozen_sha256": None,
        "formal": {"eligible": False, "frozen_sha256": None},
    }), encoding="utf-8")
    result = probe_v4_eligibility(report)
    assert result["eligible"] is False
    assert result["reason"] == "no_frozen_probe_candidate"


def test_probe_v4_eligibility_requires_explicit_release_gate(tmp_path) -> None:
    report = tmp_path / "probe.json"
    report.write_text(json.dumps({
        "contract": waveform_metadata(),
        "waveform_sha256": WAVEFORM_SHA256,
        "formal_frozen_sha256": WAVEFORM_SHA256,
        "formal": {"eligible": True, "frozen_sha256": WAVEFORM_SHA256},
        "gate_passed": False,
    }), encoding="utf-8")
    result = probe_v4_eligibility(report)
    assert result["eligible"] is False
    assert result["reason"] == "frozen_probe_gate_not_passed"


def test_probe_v4_eligibility_binds_formal_seed_split_and_q2_hash(tmp_path) -> None:
    report = tmp_path / "probe.json"
    def seed_evidence(passed: bool) -> dict:
        paired = {}
        checks = {}
        for section in ("h75", "h125", "tail_h75_h125"):
            for metric in ("position", "velocity", "omega"):
                for statistic in ("mean", "p99"):
                    name = f"paired_{section}_{metric}_{statistic}"
                    value_passed = passed or bool(paired)
                    paired[name] = {
                        "actual": 0.5 if value_passed else 2.0,
                        "allowed": 1.0,
                        "passed": value_passed,
                    }
                    checks[name] = value_passed
        peaks = {
            name: {"actual": 0.5, "allowed": 1.0, "passed": True}
            for name in ("peak_position", "peak_velocity", "peak_omega")
        }
        checks.update({name: True for name in peaks})
        checks["paired_mean_p99_max_absolute_or_ratio"] = passed
        checks["shared_tau_pooled_time_motor_fisher"] = True
        return {
            "passed": passed,
            "checks": checks,
            "shared_tau": {"gate_passed": True},
            "safety_metrics": {"paired": paired, "peaks": peaks},
        }

    family = (
        WAVEFORM,
        tuple(tuple(-value for value in row) for row in WAVEFORM),
        tuple(reversed(WAVEFORM)),
    )
    candidate_scores = []
    for index, table in enumerate(family):
        passed = index == 0
        candidate_scores.append({
            "index": index,
            "sha256": hashlib.sha256(canonical_waveform_json(table)).hexdigest(),
            "train_gate_passed": passed,
            "worst_X_condition": float(index + 1),
            "checks": {"seed_results": [
                seed_evidence(passed) for _ in FORMAL_TRAIN_SEEDS
            ]},
        })
    payload = {
        "status": "formal_frozen",
        "contract": waveform_metadata(),
        "waveform_sha256": WAVEFORM_SHA256,
        "formal_frozen_sha256": WAVEFORM_SHA256,
        "formal": {
            "eligible": True,
            "frozen_sha256": WAVEFORM_SHA256,
            "gate_passed": True,
            "selection_train_gate_passed": True,
            "independent_validation_gate_passed": True,
            "blind_consumed": [],
        },
        "gate_passed": True,
        "seed_split": {
            "train": [3707, 4707, 5707, 6707],
            "validation": [7707],
            "blind_consumed": [],
        },
        "protocol": {
            "scenarios": 16,
            "horizon": 125,
            "formal_scenarios": 16,
            "formal_horizon": 125,
        },
        "producer_code_sha256": _hash_file(
            Path(__file__).resolve().parents[1] / "tools" / "diagnose_probe_v4.py"
        ),
        "candidate_scores": candidate_scores,
        "validation": {
            "index": 0,
            "gate_passed": True,
            "checks": {"seed_results": [seed_evidence(True)]},
        },
        "source_checkpoint_sha256": _hash_file(DEFAULT_Q2_CHECKPOINT),
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert probe_v4_eligibility(report)["eligible"] is True

    payload["source_checkpoint_sha256"] = "0" * 64
    report.write_text(json.dumps(payload), encoding="utf-8")
    rejected = probe_v4_eligibility(report)
    assert rejected["eligible"] is False
    assert rejected["reason"] == "probe_q2_checkpoint_sha256_mismatch"


def test_physics_gate_fault_injection_is_independent() -> None:
    passing = {
        "finite": True, "rank_min": 6, "condition_median": 1.0,
        "condition_p95": 2.0, "condition_max": 3.0,
        "normalized_z_rms_per_dim": [0.01] * 6,
        "absolute_error_p95_per_dim": [0.02] * 6,
        "shared_tau_support_passed": True,
    }
    faults = (
        ("rank_min", 5, "rank6_per_scene"),
        ("condition_median", 31.0, "condition_median<=30"),
        ("condition_p95", 101.0, "condition_p95<=100"),
        ("condition_max", 301.0, "condition_max<=300"),
        ("normalized_z_rms_per_dim", [0.021] + [0.01] * 5, "normalized_z_rms_per_dim<=0.02"),
        ("absolute_error_p95_per_dim", [0.051] + [0.02] * 5, "absolute_error_p95_per_dim<=0.05"),
        ("shared_tau_support_passed", False, "shared_tau_pooled_time_motor_fisher"),
    )
    for key, value, expected in faults:
        report = dict(passing)
        report[key] = value
        assert expected in physics_gate_failed_checks(report)
    per_motor_only = dict(passing, support_per_motor_passed=False)
    assert "shared_tau_pooled_time_motor_fisher" not in physics_gate_failed_checks(per_motor_only)
    missing_dimension_report = dict(passing)
    missing_dimension_report["normalized_z_rms_per_dim"] = []
    assert "normalized_z_rms_per_dim<=0.02" in physics_gate_failed_checks(
        missing_dimension_report
    )


def test_k35_coverage_gate_fault_injection_is_independent() -> None:
    passing = {
        "finite": True, "median_best_to_legacy": 0.2,
        "p95_best_to_legacy": 0.7, "motor_rms_p95": 0.0005,
        "motor_rms_max": 0.002, "authority_cells_finite": True,
    }
    faults = (
        ("finite", False, "finite_best_to_legacy_ratio"),
        ("median_best_to_legacy", 0.51, "median_best_to_legacy<=0.50"),
        ("p95_best_to_legacy", 0.81, "p95_best_to_legacy<=0.80"),
        ("motor_rms_p95", 0.0011, "motor_rms_p95<=0.001"),
        ("motor_rms_max", 0.0051, "motor_rms_max<=0.005"),
        ("authority_cells_finite", False, "authority_cells_finite"),
    )
    for key, value, expected in faults:
        report = dict(passing)
        report[key] = value
        assert expected in coverage_gate_failed_checks(report, strict_motor_rms=True)


def test_v3_cadence_report_excludes_call25_from_active_publications(tmp_path) -> None:
    from argparse import Namespace
    from tools.diagnose_causal_identifier_oracle import run

    result = run(Namespace(
        checkpoint=DEFAULT_Q2_CHECKPOINT, output=tmp_path / "report.json",
        stage="collect-ceiling", scenarios=16, horizon=126, n_jobs=1,
        max_updates=1, dry_run=True, waveform=None,
    ))
    cadence = result["cadence_semantics"]
    assert cadence["publication_calls"] == [100, 125]
    assert cadence["diagnostic_calls"] == [25]
    assert cadence["availability_t25"] == [0, 0, 0, 0, 0, 0]
    assert result["publications"] == [100, 125]


def test_exact_physics_oracle_recovers_multi_axis_noiseless_transition() -> None:
    """Exercise external-force projection, Coriolis terms, and pooled tau."""
    torch.manual_seed(73)
    n, dt, gravity = 160, 0.01, 9.80665
    tw, alpha_roll, eta_yaw, beta = 3.2, 87.0, 11.6 / 87.0, 0.63
    local_authority_scale = 2.0 * (tw - 1.0) / tw
    local_roll = local_authority_scale * alpha_roll
    local_yaw = local_authority_scale * eta_yaw * alpha_roll
    tau_up, tau_down = 0.075, 0.21
    command = torch.rand(n, 4) * 1.8 - 0.9
    motor_before = torch.zeros(n, 4)
    motor_next = torch.zeros_like(motor_before)
    for index in range(n):
        if index:
            motor_before[index] = motor_next[index - 1]
        tau = torch.where(command[index] >= motor_before[index], tau_up, tau_down)
        motor_next[index] = motor_before[index] + dt / tau * (command[index] - motor_before[index])

    angles = torch.linspace(-0.35, 0.35, n)
    rotation = torch.zeros(n, 3, 3)
    rotation[:, 0, 0] = torch.cos(angles)
    rotation[:, 0, 1] = -torch.sin(angles)
    rotation[:, 1, 0] = torch.sin(angles)
    rotation[:, 1, 1] = torch.cos(angles)
    rotation[:, 2, 2] = 1.0
    omega_before = torch.randn(n, 3) * 0.7
    x = torch.stack((
        motor_next.mean(-1),
        (motor_next[:, 1] - motor_next[:, 3]) / 2.0,
        (motor_next[:, 2] - motor_next[:, 0]) / 2.0,
        (motor_next[:, 0] - motor_next[:, 1] + motor_next[:, 2] - motor_next[:, 3]) / 4.0,
    ), dim=-1)
    force = (1 + (tw - 1) * motor_next).clamp_min(0)
    angular_modes = torch.stack(((force[:, 1] - force[:, 3]) / tw,
        (force[:, 2] - force[:, 0]) / tw,
        (force[:, 0] - force[:, 1] + force[:, 2] - force[:, 3]) / (2 * tw)), -1)
    omega_after = omega_before.clone()
    # Fixed-point solve makes the midpoint Coriolis response exact to float
    # precision while retaining nonzero omega in all three axes.
    for _ in range(12):
        mid = 0.5 * (omega_before + omega_after)
        acceleration = torch.stack((
            alpha_roll * angular_modes[:, 0] - beta * mid[:, 1] * mid[:, 2],
            alpha_roll * angular_modes[:, 1] + beta * mid[:, 2] * mid[:, 0],
            alpha_roll * eta_yaw * angular_modes[:, 2],
        ), dim=-1)
        omega_after = omega_before + dt * acceleration

    mass = torch.full((n,), 1.7)
    external_force = torch.randn(n, 3) * 0.8
    ext_body = torch.bmm(rotation.transpose(-1, -2),
                         (external_force / mass[:, None]).unsqueeze(-1)).squeeze(-1)
    corrected = torch.zeros(n, 3)
    corrected[:, 2] = gravity * force.mean(-1)
    specific_body = corrected + ext_body
    fit = exact_physics_block_fit(
        motor_before, motor_next, command, specific_body, rotation,
        omega_before, omega_after, external_force, mass, dt, gravity,
    )
    expected = torch.tensor((tw, alpha_roll, eta_yaw,
                             1.0 + beta, tau_up, tau_down))
    torch.testing.assert_close(fit["estimate"], expected, atol=2.0e-4, rtol=2.0e-4)
    assert fit["rank"] == 6
    assert math.isfinite(fit["condition"])
    assert fit["rise_support"] > 0
    assert fit["fall_support"] > 0
    assert fit["rise_support_per_motor"].shape == (4,)
    assert fit["fall_support_per_motor"].shape == (4,)
    assert bool((fit["rise_support_per_motor"] >= 1).all())
    assert bool((fit["fall_support_per_motor"] >= 1).all())
    assert fit["fit_rms"] < 2.0e-4

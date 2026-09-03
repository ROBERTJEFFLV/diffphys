from __future__ import annotations

import torch
import pytest
import json

from structured_distillation import (
    DAggerEpisode,
    capability_contract,
    capability_heteroscedastic_nll,
    build_dagger_scenario_bank,
    collect_dagger_episode,
    conformal_coverage_by_stratum,
    dagger_window_loss,
    distillation_phase_components,
    normalize_log_capability,
    fit_effectiveness_conformal_q,
    phase_a_equilibrium_gate,
    set_distillation_phase,
)
from env_l2f import L2FParams, L2FSimulator
from types import SimpleNamespace


def test_banked_distillation_requires_formal_causal_oracle(tmp_path) -> None:
    from tools.distill_structured_dagger import (
        _validate_identification_oracle_report,
    )
    from structured_checkpoint import CADENCE_SEMANTICS_VERSION
    from structured_policy import StructuredPolicyConfig

    source = tmp_path / "q2.pt"
    source.write_bytes(b"fixed-source-checkpoint")
    config = StructuredPolicyConfig(
        motor_observer_bank_size=35,
        motor_observer_mode="fixed_multi_tau_v2",
        motor_tau_grid_version=2,
    )
    with pytest.raises(RuntimeError, match="requires --identification-oracle-report"):
        _validate_identification_oracle_report(
            tmp_path / "missing.json", source_checkpoint=source, config=config
        )

    report = {
        "diagnostic": "causal-identifier-sequence-oracle",
        "formal_eligible": True,
        "gate_passed": True,
        "probe_amplitude": 0.005,
        "tau_grid_sizes": [35],
        "representation_grid_version": 2,
        "cadence_semantics_version": CADENCE_SEMANTICS_VERSION,
        "cadence_semantics": {
            "call_index_completed_transitions": True,
            "publication_calls": [50, 75],
            "availability_t25": [0, 0, 0, 0, 0, 0],
            "t50_call_index": 50,
        },
        "checkpoint_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
        "code_sha256": "0" * 64,
    }
    report_path = tmp_path / "oracle.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    contract = _validate_identification_oracle_report(
        report_path, source_checkpoint=source, config=config
    )
    assert contract["gate_passed"] is True
    report["probe_amplitude"] = 0.01
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="probe amplitude"):
        _validate_identification_oracle_report(
            report_path, source_checkpoint=source, config=config
        )


class _MockState:
    def __init__(self, hidden):
        self.hidden = hidden

    def detach(self):
        return _MockState(self.hidden.detach())


class _MockStudent:
    def initial_state(self, observation):
        return _MockState(torch.zeros(observation.shape[0], 3))

    def forward_with_aux(self, observation, state):
        action = torch.zeros(observation.shape[0], 4, dtype=observation.dtype)
        next_state = _MockState(state.hidden + 0.01)
        capability = observation.new_tensor((2.0, 100.0, 0.2, 1.7, 0.1, 0.2)).expand(observation.shape[0], 6)
        allocator = SimpleNamespace(
            minimum_headroom=torch.ones(observation.shape[0]),
            wrench_residual=torch.zeros(observation.shape[0]),
        )
        identifier = observation.new_full((observation.shape[0], 3), 100.0)
        identification_failed = torch.zeros(observation.shape[0], dtype=torch.bool)
        return SimpleNamespace(action=action, next_state=next_state,
                               auxiliary={
                                   "capability": capability, "allocator": allocator,
                                   "identifier": identifier,
                                   "identification_failed": identification_failed,
                                   "effectiveness_log_interval_width": torch.zeros(
                                       observation.shape[0], 3
                                   ),
                                   "disturbance_accel": torch.zeros(
                                       observation.shape[0], 3
                                   ),
                               })


class _MockTeacher:
    def initial_hidden(self, batch, *, device, dtype):
        return torch.zeros(batch, 3, device=device, dtype=dtype)

    def __call__(self, observation, hidden):
        return torch.ones(observation.shape[0], 4), hidden + 0.01


def test_dagger_bank_is_exact_4x4_with_four_scenarios_per_cell() -> None:
    bank = build_dagger_scenario_bank(seed=31)
    assert bank.count == 64
    counts = {(tw, alpha): int(((bank.tw_bin == tw) & (bank.log_alpha_bin == alpha)).sum())
              for tw in range(4) for alpha in range(4)}
    assert set(counts.values()) == {4}
    assert len(bank.stratum) == 64


def test_capability_contract_and_heteroscedastic_nll_are_finite() -> None:
    capability = torch.tensor([[2.0, 100.0, 0.2, 1.7, 0.1, 0.2]], requires_grad=True)
    contract = capability_contract({"capability": capability})
    target = normalize_log_capability(capability.detach())
    loss = capability_heteroscedastic_nll(
        contract.capability_z_mean, contract.capability_z_log_scale, target
    )
    loss.backward()
    assert torch.isfinite(contract.capability_ucb).all()
    assert torch.isfinite(loss)
    assert torch.isfinite(capability.grad).all()


def test_dagger_episode_uses_fixed_per_scenario_mask_and_real_window_loss() -> None:
    bank = build_dagger_scenario_bank(seed=37)
    simulator = L2FSimulator(L2FParams(dt=0.01))
    episode = collect_dagger_episode(_MockTeacher(), _MockStudent(), simulator, bank,
                                      beta=0.5, horizon=51, episode_seed=11)
    assert episode.intervention_mask.shape == (51, 64)
    assert torch.equal(episode.intervention_mask[0], episode.intervention_mask[-1])
    assert episode.teacher_same_latent_intercepts.shape == (51, 64, 4)
    assert episode.teacher_fast_delta_actions.shape == (51, 64, 4)
    assert episode.motor_trim_target.shape == (64, 4)
    assert episode.identification_norm_t50.shape == (64,)
    assert not bool(episode.identification_failure_t50.any())
    loss, components = dagger_window_loss(_MockStudent(), episode, prefix=25)
    assert torch.isfinite(loss)
    assert components["action"] >= 0.0
    assert components["analytic_trim"] >= 0.0
    assert components["same_latent_intercept_diagnostic"] >= 0.0
    assert components["body_z"] >= 0.0
    assert components["delta_action"] >= 0.0
    with pytest.raises(ValueError, match="local-JVP"):
        dagger_window_loss(_MockStudent(), episode, prefix=25, phase="B")
    pure = collect_dagger_episode(_MockTeacher(), _MockStudent(), simulator, bank,
                                  beta=0.0, horizon=51, episode_seed=11)
    assert int(pure.intervention_mask.sum()) == 0


def test_conformal_coverage_has_separate_calibration_and_validation_counts() -> None:
    bank = build_dagger_scenario_bank(seed=41)
    target = torch.zeros(1, 64, 6)
    episode = SimpleNamespace(
        tw_bin=bank.tw_bin, log_alpha_bin=bank.log_alpha_bin,
        capability_target_z=torch.zeros(64, 6),
        capability_z_mean=target,
        capability_z_log_scale=torch.full_like(target, -1.0),
    )
    rows = conformal_coverage_by_stratum(episode)
    assert len(rows) == 16
    assert all(row["calibration_effective_samples"] == 2 for row in rows)
    assert all(row["validation_effective_samples"] == 2 for row in rows)
    assert all(0.0 <= row["validation_coverage"] <= 1.0 for row in rows)


def test_conformal_helper_accepts_independent_validation_bank() -> None:
    calibration = build_dagger_scenario_bank(seed=51)
    validation = build_dagger_scenario_bank(seed=52)

    def summary(bank):
        values = torch.zeros(1, 64, 6)
        return SimpleNamespace(
            tw_bin=bank.tw_bin, log_alpha_bin=bank.log_alpha_bin,
            capability_target_z=torch.zeros(64, 6),
            capability_z_mean=values,
            capability_z_log_scale=torch.full_like(values, -1.0),
        )

    rows = conformal_coverage_by_stratum(summary(calibration), validation_episode=summary(validation))
    assert all(row["calibration_effective_samples"] == 4 for row in rows)
    assert all(row["validation_effective_samples"] == 4 for row in rows)


def test_staged_phase_freeze_is_explicit_and_residual_is_not_implicitly_enabled() -> None:
    from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy

    student = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=4, identifier_dim=4))
    active_a = set_distillation_phase(student, "A")
    assert set(distillation_phase_components("A")) == {
        "identifier", "capability", "explicit_disturbance_observer",
        "analytic_equilibrium", "observer",
    }
    assert any(name.startswith("identifier.") for name in active_a)
    assert any(name.startswith("capability_head.") for name in active_a)
    assert not any(name.startswith("residual_head.") for name in active_a)
    active_b = set_distillation_phase(student, "B")
    assert active_b and all(name.startswith("contextual_gain_head.") for name in active_b)
    active_c = set_distillation_phase(student, "C")
    assert any(name.startswith("encoder.") for name in active_c)
    assert any(name.startswith("gru.") for name in active_c)
    assert any(name.startswith("residual_head.") for name in active_c)


def test_phase_a1_mean_pretraining_freezes_only_uncertainty_scale() -> None:
    """Lock the A1 optimizer contract, including actual gradient flow.

    ``set_distillation_phase(..., "A")`` is the shared base selector.  The
    current CLI adds the scale freeze when NLL is disabled; keeping that small
    operation explicit here makes accidental scale training in A1 visible.
    """

    from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy

    student = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=4, identifier_dim=4))
    set_distillation_phase(student, "A1")
    trainable = {
        name for name, parameter in student.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert any(name.startswith("identifier.") for name in trainable)
    assert any(name.startswith("capability_head.") for name in trainable)
    assert not any(name.startswith("capability_log_scale_head.") for name in trainable)

    observation = torch.zeros(2, 25)
    observation[:, (6, 10, 14)] = 1.0
    state = student.initial_state(observation)
    for _ in range(50):
        state = student.forward_with_aux(observation, state).next_state
    # Capability heads publish a new value at the cadence boundary.
    output = student.forward_with_aux(observation, state)
    output.auxiliary["capability_z_mean"].square().mean().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in student.identifier.parameters()
    )
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in student.capability_head.parameters()
    )
    assert all(parameter.grad is None for parameter in student.capability_log_scale_head.parameters())


def test_phase_a2_scale_calibration_freezes_identifier_and_mean() -> None:
    """A2 must be a scale-only update after A1 has converged."""

    from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy

    student = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=4, identifier_dim=4))
    set_distillation_phase(student, "A2")
    trainable = {
        name for name, parameter in student.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("capability_log_scale_head.") for name in trainable)

    observation = torch.zeros(2, 25)
    observation[:, (6, 10, 14)] = 1.0
    state = student.initial_state(observation)
    for _ in range(50):
        state = student.forward_with_aux(observation, state).next_state
    output = student.forward_with_aux(observation, state)
    output.auxiliary["capability_z_log_scale"].square().mean().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in student.capability_log_scale_head.parameters()
    )
    assert all(parameter.grad is None for parameter in student.identifier.parameters())
    assert all(parameter.grad is None for parameter in student.capability_head.parameters())


def test_phase_a1_mean_gate_ignores_width_latch_but_a2_gate_does_not() -> None:
    """A1 evaluates the frozen mean; only A2 may promote calibrated width."""

    time, batch = 76, 2
    values = torch.zeros(time, batch, 6)
    episode = SimpleNamespace(
        motor_trim_target=torch.zeros(batch, 4),
        body_z_target=torch.tensor([[0.0, 0.0, 1.0]]).expand(batch, 3),
        disturbance_accel_target=torch.zeros(batch, 3),
        student_trim=torch.zeros(time, batch, 4),
        student_body_z=torch.tensor([0.0, 0.0, 1.0]).expand(time, batch, 3),
        student_disturbance_accel=torch.zeros(time, batch, 3),
        equilibrium_one_step_accel=torch.zeros(time, batch),
        equilibrium_one_step_omega=torch.zeros(time, batch),
        equilibrium_action_error=torch.zeros(time, batch),
        equilibrium_feasible=torch.ones(time, batch, dtype=torch.bool),
        identification_failure_t50=torch.tensor([True, True]),
        capability_z_mean=values,
        capability_z_log_scale=torch.zeros_like(values),
        capability_target_z=torch.zeros(batch, 6),
    )
    mean_report, mean_passed = phase_a_equilibrium_gate(
        episode, require_identification_width=False
    )
    width_report, width_passed = phase_a_equilibrium_gate(
        episode, require_identification_width=True
    )
    assert mean_passed and mean_report["identification_width_required"] is False
    assert not width_passed and width_report["identification_width_required"] is True


def test_a2_has_an_explicit_phase_selector_for_width_and_calibration() -> None:
    """Desired API test: only A2 may inspect width/install calibration."""

    from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy

    student = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=4, identifier_dim=4))
    active = set_distillation_phase(student, "A2")
    assert active
    assert all(name.startswith("capability_log_scale_head.") for name in active)


def test_four_stratum_effectiveness_conformal_uses_128_samples_each() -> None:
    bank = build_dagger_scenario_bank(512, seed=61, per_cell=32)
    values = torch.zeros(76, 512, 6)
    episode = SimpleNamespace(
        log_alpha_bin=bank.log_alpha_bin,
        capability_target_z=torch.zeros(512, 6),
        capability_z_mean=values,
        capability_z_log_scale=torch.zeros_like(values),
    )
    q, rows = fit_effectiveness_conformal_q(
        episode, validation_episode=episode, miscoverage=0.01
    )
    assert q.shape == (6,)
    assert len(rows) == 4
    assert all(row["calibration_samples"] == 128 for row in rows)
    assert all(row["validation_coverage"] == 1.0 for row in rows)
    assert all(row["phase_steps"] == [50, 75] for row in rows)


def test_phase_a_equilibrium_gate_rejects_latched_identification_failure() -> None:
    time, batch = 76, 2
    episode = SimpleNamespace(
        motor_trim_target=torch.zeros(batch, 4),
        body_z_target=torch.tensor([[0.0, 0.0, 1.0]]).expand(batch, 3),
        disturbance_accel_target=torch.zeros(batch, 3),
        student_trim=torch.zeros(time, batch, 4),
        student_body_z=torch.tensor([0.0, 0.0, 1.0]).expand(time, batch, 3),
        student_disturbance_accel=torch.zeros(time, batch, 3),
        equilibrium_one_step_accel=torch.zeros(time, batch),
        equilibrium_one_step_omega=torch.zeros(time, batch),
        equilibrium_action_error=torch.zeros(time, batch),
        equilibrium_feasible=torch.ones(time, batch, dtype=torch.bool),
        identification_failure_t50=torch.tensor([False, True]),
    )
    report, passed = phase_a_equilibrium_gate(episode)
    assert not passed
    assert report["identification_failure_t50_count"] == 1
    assert report["identification_failure_t50_required"] == 0

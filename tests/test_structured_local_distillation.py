from __future__ import annotations

import torch

from env_l2f import L2FParams, L2FSimulator
from structured_distillation import build_dagger_scenario_bank
from structured_local_distillation import (
    LOCAL_RADII,
    build_local_derivative_batch,
    collect_common_history,
    collect_equilibrium_history,
    fit_contextual_gain_local,
    h25_diagnostics,
    lifted_h250_diagnostics,
    local_derivative_diagnostics,
    local_mode_diagnostics,
    perturb_normalized_error,
    perturb_student_error_state,
    projection_diagnostics,
)
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy


class _Teacher:
    def initial_hidden(self, batch, *, device, dtype):
        return torch.zeros(batch, 8, device=device, dtype=dtype)

    def forward_with_aux(self, observation, hidden):
        action = torch.tanh(observation[:, :4] + hidden[:, :4])
        return action, hidden + 0.01, {}


def _fixture():
    simulator = L2FSimulator(L2FParams(dt=0.01))
    bank = build_dagger_scenario_bank(16, seed=13, dt=0.01, per_cell=1)
    policy = StructuredRecurrentPolicy(
        StructuredPolicyConfig(hidden_dim=8, identifier_dim=5, residual_scale=0.0)
    )
    snapshots = collect_common_history(
        _Teacher(), policy, simulator, bank, snapshot_steps=(50, 75)
    )
    return simulator, bank, policy, snapshots


def test_local_history_and_directional_batch_shapes():
    _, _, policy, snapshots = _fixture()
    batch = build_local_derivative_batch(
        _Teacher(), policy, snapshots, radii=LOCAL_RADII, seed=3
    )
    assert batch.directions.shape == (2, 3, 15, 15)
    assert batch.teacher_directional_derivative.shape == (2, 3, 15, 16, 4)
    # Every radius/snapshot gets a full orthonormal direction frame; this is
    # deliberately not one repeated random direction per radius.
    gram = batch.directions @ batch.directions.transpose(-1, -2)
    torch.testing.assert_close(
        gram,
        torch.eye(15).expand_as(gram),
        rtol=1e-4,
        atol=1e-4,
    )
    assert batch.direction_count >= 15
    heldout = build_local_derivative_batch(
        _Teacher(), policy, snapshots, radii=LOCAL_RADII, seed=4
    )
    assert not torch.equal(batch.directions, heldout.directions)
    assert torch.isfinite(batch.teacher_directional_derivative).all()


def test_local_jvp_uses_fixed_equilibrium_recurrent_history():
    simulator, bank, policy, _ = _fixture()
    snapshots = collect_equilibrium_history(
        _Teacher(), policy, simulator, bank, snapshot_steps=(50, 75)
    )
    assert [snapshot.step for snapshot in snapshots] == [50, 75]
    for snapshot in snapshots:
        torch.testing.assert_close(
            snapshot.observation, snapshot.equilibrium_observation
        )
        rotation = snapshot.equilibrium_observation[:, 6:15].reshape(-1, 3, 3)
        torch.testing.assert_close(
            rotation.transpose(-1, -2) @ rotation,
            torch.eye(3).expand_as(rotation),
            rtol=1e-4,
            atol=1e-4,
        )


def test_local_fit_keeps_reference_gain_fixed_and_reports_projection():
    simulator, bank, policy, snapshots = _fixture()
    before = policy.K_ref.detach().clone()
    batch, history = fit_contextual_gain_local(
        policy, _Teacher(), snapshots, iterations=1, radii=(0.1,), seed=4
    )
    assert len(history) == 1 and torch.isfinite(torch.tensor(history)).all()
    torch.testing.assert_close(policy.K_ref, before)
    projection = projection_diagnostics(policy, snapshots)
    assert projection["projection_bound_passed"]
    assert projection["projection_budget_min"] >= 0.0
    assert projection["post_induced_spectral_norm_max"] <= (
        projection["projection_budget_max"] + 1.0e-5
    )
    assert torch.isfinite(batch.student_directional_derivative).all()
    mode = local_mode_diagnostics(_Teacher(), policy, snapshots, batch)
    assert 0.0 <= mode["active_fraction"] <= 1.0
    assert len(mode["by_authority_cell"]) == 32


def test_local_mode_mask_is_per_scenario_not_global():
    simulator, bank, policy, snapshots = _fixture()
    # Make only half of each snapshot's scenarios confident.  The other half
    # must remain on the allocator-inverse surrogate branch.
    for snapshot in snapshots:
        snapshot.student_state.capability_log_scale[:8] = torch.log(
            torch.full_like(snapshot.student_state.capability_log_scale[:8], 0.05)
        )
    batch = build_local_derivative_batch(
        _Teacher(), policy, snapshots, radii=(0.1,), seed=12
    )
    mode = local_mode_diagnostics(_Teacher(), policy, snapshots, batch)
    assert 0.0 < mode["active_fraction"] < 1.0
    assert mode["active_deployment_jvp_error_mean"] is not None
    assert mode["inactive_surrogate_error_mean"] is not None


def test_local_diagnostics_and_lifted_h250_are_finite():
    simulator, bank, policy, snapshots = _fixture()
    batch = build_local_derivative_batch(
        _Teacher(), policy, snapshots, radii=LOCAL_RADII, seed=5
    )
    metrics = local_derivative_diagnostics(_Teacher(), policy, snapshots, batch)
    h25 = h25_diagnostics(policy, simulator, bank, horizon=25)
    lifted = lifted_h250_diagnostics(policy, simulator, bank, horizon=50, segment_length=25)
    assert metrics["equilibrium_action_trim_rms"] >= 0.0
    assert metrics["taylor_remainder_over_radius2_max"] >= 0.0
    assert h25["finite"]
    assert lifted["finite"]
    assert len(lifted["segments"]) == 2
    assert torch.isfinite(torch.tensor(metrics["jvp_normalized_error_mean"]))


def test_normalized_error_perturbation_preserves_width():
    observation = torch.zeros(2, 25)
    observation[:, (6, 10, 14)] = 1.0
    direction = torch.ones(2, 15)
    result = perturb_normalized_error(observation, direction, 0.1)
    assert result.shape == observation.shape
    assert torch.isfinite(result).all()
    rotation = result[:, 6:15].reshape(-1, 3, 3)
    identity = torch.eye(3).expand_as(rotation)
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation, identity)


def test_motor_error_perturbation_updates_student_state_and_keeps_hidden_fixed():
    _, _, policy, snapshots = _fixture()
    state = snapshots[0].student_state
    direction = torch.zeros(state.motor_estimate.shape[0], 15)
    direction[:, 11:15] = 1.0
    perturbed = perturb_student_error_state(state, direction, 0.1)
    torch.testing.assert_close(
        perturbed.motor_estimate - state.motor_estimate,
        torch.full_like(state.motor_estimate, 0.01),
    )
    torch.testing.assert_close(perturbed.hidden, state.hidden)
    torch.testing.assert_close(
        perturbed.previous_executed_action,
        state.previous_executed_action,
    )

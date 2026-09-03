from __future__ import annotations

import math

import torch

from diagnostics.physics import (
    branch_wrench_decomposition,
    steady_state_feasibility,
)
from diagnostics.provenance import canonical_config_hash, scenario_uid
from diagnostics.time_weighting import dense_tracking_time_weights
from env_l2f import L2FParams, L2FSimulator
from model import MotorGRUPolicy
from policy_observation import (
    PolicyObservationState,
    update_position_integral,
)


def _fixed_state(batch_size: int = 2):
    params = L2FParams()
    simulator = L2FSimulator(params)
    torch.manual_seed(11)
    state = simulator.reset(batch_size, device="cpu", sample_dynamics=False)
    state.position.zero_()
    state.velocity.zero_()
    state.omega.zero_()
    state.rotation.copy_(torch.eye(3).expand(batch_size, 3, 3))
    return params, simulator, state


def test_hover_feasibility_recovers_zero_command() -> None:
    params, _, state = _fixed_state()
    result = steady_state_feasibility(state, params)

    assert torch.all(result.feasible)
    assert not torch.any(result.numerical_failure)
    torch.testing.assert_close(result.required_thrust_ratio, torch.ones(2))
    torch.testing.assert_close(result.required_tilt_rad, torch.zeros(2))
    torch.testing.assert_close(result.required_motor_command, torch.zeros(2, 4), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(result.required_trim_ratio, torch.full((2,), 1.0 / 2.35))


def test_feasibility_distinguishes_actuator_upper_violation() -> None:
    params, _, state = _fixed_state(batch_size=1)
    state.external_force[:, 2] = -10.0 * state.mass * params.gravity
    result = steady_state_feasibility(state, params)

    assert not bool(result.feasible.item())
    assert bool(result.upper_thrust_violation.any().item())
    assert not bool(result.numerical_failure.item())


def test_branch_decomposition_uses_all_four_counterfactuals_after_motor_lag() -> None:
    params, _, state = _fixed_state(batch_size=1)
    main = torch.tensor([[0.1, -0.2, 0.3, -0.4]])
    integral = torch.tensor([[0.05, 0.05, 0.05, 0.05]])
    damping = torch.tensor([[0.1, -0.1, 0.1, -0.1]])
    result = branch_wrench_decomposition(
        state,
        {
            "main_logits": main,
            "integral_residual_logits": integral,
            "damping_residual_logits": damping,
        },
        dt=params.dt,
    )

    torch.testing.assert_close(result.integral_delta, result.main_integral - result.main_only)
    torch.testing.assert_close(result.damping_delta, result.main_damping - result.main_only)
    torch.testing.assert_close(
        result.damping_delta_after_integral,
        result.full - result.main_integral,
    )
    assert result.integral_delta[0, 0] > 0.0
    assert result.damping_delta[0, 3] > 0.0
    assert not torch.equal(result.damping_delta, result.damping_delta_after_integral)


def test_h250_tail_weight_is_six_times_an_ordinary_step() -> None:
    weights = dense_tracking_time_weights(
        episode_steps=500,
        segment_steps=250,
        tail_steps=50,
        lambda_tail=1.0,
    )
    ordinary = weights[0]
    assert weights.shape == (500,)
    torch.testing.assert_close(ordinary, torch.tensor(1.0 / 500.0, dtype=torch.float64))
    torch.testing.assert_close(weights[249], 6.0 * ordinary)
    torch.testing.assert_close(weights[499], 6.0 * ordinary)
    torch.testing.assert_close(weights.sum(), torch.tensor(2.0, dtype=torch.float64))


def test_world_axis_clamp_is_not_yaw_equivariant() -> None:
    state = PolicyObservationState(torch.tensor([[0.49, 0.49, 0.0]]))
    position = torch.tensor([[1.0, 0.0, 0.0]])
    updated = update_position_integral(
        state,
        position,
        dt=0.02,
        integral_limit=0.5,
    ).integral_position

    angle = math.pi / 4.0
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_state = PolicyObservationState(state.integral_position @ rotation.T)
    rotated_position = position @ rotation.T
    updated_rotated = update_position_integral(
        rotated_state,
        rotated_position,
        dt=0.02,
        integral_limit=0.5,
    ).integral_position
    rotated_updated = updated @ rotation.T

    assert torch.linalg.vector_norm(updated_rotated - rotated_updated) > 0.05


def test_provenance_is_stable_and_checkpoint_metadata_is_non_mutating() -> None:
    config_a = {"seed": 7, "horizons": (500, 1000), "path": "x"}
    config_b = {"path": "x", "horizons": [500, 1000], "seed": 7}
    assert canonical_config_hash(config_a) == canonical_config_hash(config_b)
    assert scenario_uid(7, 3) == "seed-7-sample-000003"

    torch.manual_seed(3)
    policy = MotorGRUPolicy(enable_integral_residual=True, enable_damping_residual=True)
    observation = torch.randn(2, 25)
    hidden = torch.randn(2, policy.hidden_dim)
    action_before, hidden_before = policy(observation, hidden)
    metadata = policy.architecture_metadata()
    action_after, hidden_after = policy(observation, hidden)
    torch.testing.assert_close(action_before, action_after)
    torch.testing.assert_close(hidden_before, hidden_after)
    assert metadata["architecture_version"] == 2
    assert metadata["deployment_privileged_inputs"] is False

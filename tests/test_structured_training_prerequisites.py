from __future__ import annotations

from dataclasses import replace

import torch

from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy


def observation() -> torch.Tensor:
    value = torch.zeros(1, 25)
    value[:, 6:15] = torch.eye(3).reshape(1, 9)
    return value


def test_inherited_action_recovery_preserves_executed_history_and_hard_slew() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(allocator_solver="box_qp"))
    obs = observation()
    obs[:, 21:25] = torch.tensor([0.4, -0.3, 0.2, -0.1])
    state = policy.initial_state(obs)
    executed = torch.tensor([[0.7, -0.6, 0.5, -0.4]])
    result = policy.forward_with_aux(obs, state, applied_action=executed)
    assert bool(result.auxiliary["inherited_command_recovery"].all())
    assert float((result.action - obs[:, 21:25]).abs().max()) <= 0.005001
    assert float(result.action.abs().max()) <= 0.400001
    assert result.next_state.slow_counter == 1
    torch.testing.assert_close(result.next_state.identification_actions[:, -1], executed)
    assert state.slow_counter == 0


def test_first_force_window_initializes_without_zero_prior_bias() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(allocator_solver="box_qp"))
    obs = observation()
    state = policy.initial_state(obs)
    history = torch.zeros(1, 25, 11)
    history[:, :, 2] = 1.0
    history[:, :, 3] = 1.0
    history[:, :, 5] = 9.80665
    history[:, :, 10] = 1.0
    state = replace(state, slow_counter=100, boot_progress=torch.full((1, 1), 100.0),
                    disturbance_history=history)
    obs[:, 3] = 0.01
    result = policy.forward_with_aux(obs, state)
    torch.testing.assert_close(result.auxiliary["disturbance_accel"],
                               torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6, rtol=1e-6)

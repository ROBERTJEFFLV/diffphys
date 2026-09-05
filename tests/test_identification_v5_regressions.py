from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

from diagnostics.formal_rollout import load_q2_policy
from env_l2f import L2FParams, L2FSimulator
from full_space_shooting import so3_exp, so3_local_residual
from policy_observation import build_policy_observation, initial_observation_state, update_position_integral
from structured_distillation import build_dagger_scenario_bank, collect_dagger_episode, _advance_student_with_executed_action
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy, replay_motor_history
from structured_stability import matrix_free_augmented_stability_report


def test_so3_log_has_identity_tangent_derivative_and_recovers_pi():
    identity = torch.eye(3, dtype=torch.float64)
    derivative = torch.func.jacfwd(lambda x: so3_local_residual(identity, so3_exp(x)))(torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(derivative, identity)
    axis = torch.tensor([1., -2., 3.], dtype=torch.float64)
    axis = axis / axis.norm()
    rotation = so3_exp(axis * torch.pi)
    recovered = so3_exp(so3_local_residual(rotation, identity))
    torch.testing.assert_close(recovered, rotation, atol=1e-9, rtol=1e-9)


def test_unstable_coordinate_cannot_be_removed_as_a_yaw_gauge():
    matrix = torch.diag(torch.tensor([1.4, .8, .7], dtype=torch.float64))
    with pytest.raises(ValueError, match="not a neutral symmetry"):
        matrix_free_augmented_stability_report(lambda x: matrix @ x,
            torch.zeros(3, dtype=torch.float64), yaw_basis=torch.tensor([1., 0., 0.], dtype=torch.float64))


def test_actual_action_replay_matches_every_recurrent_field_at_publication():
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=4, identifier_dim=4, allocator_solver="smooth_dls"))
    observation = torch.zeros(2, 25)
    observation[:, 6:15] = torch.eye(3).flatten()
    state = policy.initial_state(observation)
    state = replace(state, slow_counter=100, boot_progress=torch.full((2, 1), 100.),
                    identification_actions=torch.linspace(-.5, .5, 800).reshape(2, 100, 4),
                    motor_estimate=torch.full((2, 4), .8))
    action = torch.full((2, 4), -.15)
    candidate = policy.forward_with_aux(observation, state)
    actual = _advance_student_with_executed_action(policy, observation, state, candidate, action, dt=.01)
    reference = policy.forward_with_aux(observation, state, applied_action=action).next_state
    for field in fields(actual):
        left, right = getattr(actual, field.name), getattr(reference, field.name)
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right)
        else:
            assert left == right
    reconstructed = replay_motor_history(policy.motor_observer, state.identification_initial_motor,
        state.identification_actions, reference.capability[:, 4:5], reference.capability[:, 5:6], .01)
    torch.testing.assert_close(actual.previous_motor_estimate, reconstructed)
    assert actual.slow_counter == 101 and state.slow_counter == 100


def test_teacher_only_collection_preserves_q2_trajectory_and_integral():
    checkpoint = Path("reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt")
    teacher, args = load_q2_policy(checkpoint, device="cpu", dtype=torch.float32)
    student = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=4, identifier_dim=4, allocator_solver="smooth_dls"))
    simulator = L2FSimulator(L2FParams(dt=.01))
    bank = build_dagger_scenario_bank(16, seed=13707, per_cell=1)
    episode = collect_dagger_episode(teacher, student, simulator, bank, beta=1., horizon=101, teacher_probe=True)
    state, obs_state = bank.state, initial_observation_state(16, device="cpu", dtype=torch.float32)
    hidden = teacher.initial_hidden(16, device="cpu", dtype=torch.float32)
    student_integral = torch.zeros(16, 3)
    settings = teacher.q2_observation_settings
    with torch.no_grad():
        for call in range(101):
            obs, position = build_policy_observation(state, obs_state, **{k: settings[k] for k in ("mode", "noise_max", "integral_input_frame", "integral_input_multiplier")})
            action, hidden = teacher(obs, hidden)
            torch.testing.assert_close(episode.executed_actions[call], action, atol=0, rtol=0)
            torch.testing.assert_close(episode.observations[call, :, :18], obs[:, :18], atol=0, rtol=0)
            torch.testing.assert_close(episode.observations[call, :, 21:], obs[:, 21:], atol=0, rtol=0)
            integral_body = (state.rotation.transpose(1, 2) @ student_integral[:, :, None]).squeeze(-1)
            torch.testing.assert_close(episode.observations[call, :, 18:21], integral_body, atol=0, rtol=0)
            authority = (1 - action.abs().mean(-1)).clamp(0, 1)[:, None]
            student_integral = student.integrator(student_integral, state.position, .01, authority)
            obs_state = update_position_integral(obs_state, position, dt=.01, **{k: settings[k] for k in ("integral_limit", "integral_leak", "integral_clamp_mode")})
            state = simulator.step(state, action, grad_decay=1.)


def test_disturbance_residual_respects_zero_thrust_floor():
    simulator = L2FSimulator(L2FParams(dt=.01))
    bank = build_dagger_scenario_bank(16, seed=13707, per_cell=1)
    physical = replace(bank.state, motor=torch.full_like(bank.state.motor, -.95))
    end = simulator.step(physical, torch.full_like(physical.motor, -1.), grad_decay=1.)
    obs, _ = build_policy_observation(end, initial_observation_state(16, device="cpu", dtype=torch.float32), mode="integral25")
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=4, identifier_dim=4, allocator_solver="smooth_dls"))
    capability = torch.stack([physical.thrust_to_weight, physical.alpha_roll_max, physical.eta_yaw,
                              physical.jz_over_jxy, physical.motor_time_rising, physical.motor_time_falling], -1)
    state = replace(policy.initial_state(obs), capability=capability, motor_estimate=end.motor,
                    prev_velocity=physical.velocity, prev_omega=physical.omega, slow_counter=1)
    residual = policy.forward_with_aux(obs, state).auxiliary["disturbance_residual"]
    torch.testing.assert_close(residual, physical.external_force / physical.mass[:, None], atol=3e-5, rtol=3e-5)

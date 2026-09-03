from __future__ import annotations

from dataclasses import replace

import torch

from env_l2f import L2FParams, L2FSimulator
from equilibrium_control import (
    EquilibriumCenteredPolicy,
    PhaseSpaceConfig,
    analytic_equilibrium_target,
    contraction_objective,
    materialize_equilibrium_state,
    phase_space_energy,
)
from policy_observation import build_policy_observation, initial_observation_state


def _nominal_state(batch_size: int = 2) -> tuple[L2FSimulator, object]:
    simulator = L2FSimulator(
        L2FParams(
            dt=0.01,
            max_initial_position=0.0,
            max_initial_velocity=0.0,
            max_initial_angle=0.0,
            max_initial_omega=0.0,
        )
    )
    state = simulator.reset(batch_size, device="cpu", dtype=torch.float64)
    return simulator, state


def test_analytic_equilibrium_is_a_physical_fixed_point() -> None:
    simulator, state = _nominal_state()
    external_force = torch.tensor(
        ((0.025, -0.010, 0.0), (-0.015, 0.020, -0.005)),
        dtype=state.position.dtype,
    )
    state = replace(state, external_force=external_force)
    target = analytic_equilibrium_target(state)
    assert bool(target.feasible.all())
    equilibrium = materialize_equilibrium_state(state, target)
    next_state = simulator.step(equilibrium, target.motor_trim)

    assert torch.allclose(next_state.position, equilibrium.position, atol=2.0e-12)
    assert torch.allclose(next_state.velocity, equilibrium.velocity, atol=2.0e-12)
    assert torch.allclose(next_state.rotation, equilibrium.rotation, atol=2.0e-12)
    assert torch.allclose(next_state.omega, equilibrium.omega, atol=2.0e-12)
    assert torch.allclose(next_state.motor, equilibrium.motor, atol=2.0e-12)


def test_feedback_is_exactly_zero_at_predicted_equilibrium() -> None:
    simulator, state = _nominal_state(batch_size=1)
    target = analytic_equilibrium_target(state)
    equilibrium = materialize_equilibrium_state(state, target)
    policy = EquilibriumCenteredPolicy(encoder_dim=12, hidden_dim=12).double()
    with torch.no_grad():
        policy.trim_head.weight.zero_()
        policy.trim_head.bias.zero_()
        policy.direction_head.weight.zero_()
        policy.direction_head.bias.zero_()
        policy.feedback_gain_head.weight.zero_()
        policy.feedback_gain_head.bias.zero_()
        policy.base_feedback_gain.zero_()
    observation_state = initial_observation_state(1, device="cpu", dtype=torch.float64)
    observation, _ = build_policy_observation(
        equilibrium,
        observation_state,
        mode="integral25",
        integral_input_frame="body",
    )
    hidden = policy.initial_hidden(1, device="cpu", dtype=torch.float64)
    _, _, details = policy.forward_with_aux(observation, hidden)
    assert torch.equal(details["feedback_logits"], torch.zeros_like(details["feedback_logits"]))


def test_phase_space_energy_is_zero_only_at_target() -> None:
    _, state = _nominal_state(batch_size=1)
    target = analytic_equilibrium_target(state)
    equilibrium = materialize_equilibrium_state(state, target)
    config = PhaseSpaceConfig()
    assert torch.equal(
        phase_space_energy(equilibrium, target, config),
        torch.zeros(1, dtype=torch.float64),
    )
    displaced = replace(
        equilibrium,
        position=equilibrium.position + equilibrium.position.new_tensor(((0.1, 0.0, 0.0),)),
    )
    assert float(phase_space_energy(displaced, target, config).item()) > 0.0


def test_phase_space_sliding_term_uses_physical_units() -> None:
    _, state = _nominal_state(batch_size=1)
    target = analytic_equilibrium_target(state)
    equilibrium = materialize_equilibrium_state(state, target)
    config = PhaseSpaceConfig(
        position_scale=0.20,
        velocity_scale=0.05,
        lambda_position=2.0,
        weight_position=0.0,
        weight_tilt=0.0,
        weight_sliding_tilt=0.0,
        weight_yaw_rate=0.0,
        weight_motor=0.0,
        weight_previous_action=0.0,
    )
    position = equilibrium.position + equilibrium.position.new_tensor(((0.10, 0.0, 0.0),))
    velocity = equilibrium.velocity + equilibrium.velocity.new_tensor(((-0.20, 0.0, 0.0),))
    on_manifold = replace(equilibrium, position=position, velocity=velocity)
    assert torch.allclose(
        phase_space_energy(on_manifold, target, config),
        torch.zeros(1, dtype=torch.float64),
        atol=1.0e-14,
    )


def test_contraction_objective_penalizes_energy_growth() -> None:
    shrinking = torch.tensor(((10.0,), (5.0,), (2.0,)))
    growing = torch.tensor(((2.0,), (5.0,), (10.0,)))
    shrinking_result = contraction_objective(
        shrinking,
        interval_seconds=1.0,
        contraction_rate=0.0,
        terminal_weight=0.0,
    )
    growing_result = contraction_objective(
        growing,
        interval_seconds=1.0,
        contraction_rate=0.0,
        terminal_weight=0.0,
    )
    assert float(shrinking_result.loss) == 0.0
    assert float(growing_result.loss) > 0.0

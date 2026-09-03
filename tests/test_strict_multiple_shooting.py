from __future__ import annotations

import copy

import torch

from env_l2f import L2FParams, L2FSimulator
from equilibrium_control import EquilibriumCenteredPolicy, analytic_equilibrium_target
from policy_observation import initial_observation_state
from strict_multiple_shooting import (
    RecurrentSystemState,
    StrictShootingConfig,
    recurrent_tensors,
    strict_multiple_shooting_objective,
)


def _problem() -> tuple[L2FSimulator, RecurrentSystemState, object]:
    torch.manual_seed(71)
    simulator = L2FSimulator(L2FParams(dt=0.01))
    state = simulator.reset(2, device="cpu", dtype=torch.float64)
    integral = initial_observation_state(2, device="cpu", dtype=torch.float64)
    initial = RecurrentSystemState(
        state=state,
        hidden=torch.zeros(2, 10, dtype=torch.float64),
        integral=integral.integral_position,
    )
    return simulator, initial, analytic_equilibrium_target(state)


def _flat_gradient(loss: torch.Tensor, policy: torch.nn.Module) -> torch.Tensor:
    parameters = tuple(policy.parameters())
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat(
        tuple(
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        )
    )


def test_checkpoint_recompute_matches_uninterrupted_gradient() -> None:
    simulator, initial, target = _problem()
    direct_policy = EquilibriumCenteredPolicy(encoder_dim=10, hidden_dim=10).double()
    checkpoint_policy = copy.deepcopy(direct_policy)
    common = dict(
        segment_steps=4,
        segment_count=3,
        energy_interval_steps=2,
        state_step_decay=0.5**0.01,
        hidden_step_decay=0.7**0.01,
    )
    direct = strict_multiple_shooting_objective(
        direct_policy,
        simulator,
        initial,
        target,
        shooting_config=StrictShootingConfig(
            **common,
            use_checkpoint_recompute=False,
        ),
    )
    recomputed = strict_multiple_shooting_objective(
        checkpoint_policy,
        simulator,
        initial,
        target,
        shooting_config=StrictShootingConfig(
            **common,
            use_checkpoint_recompute=True,
        ),
    )
    direct_gradient = _flat_gradient(direct.loss, direct_policy)
    recomputed_gradient = _flat_gradient(recomputed.loss, checkpoint_policy)

    assert torch.allclose(direct.loss, recomputed.loss, rtol=1.0e-11, atol=1.0e-11)
    assert torch.allclose(
        direct_gradient,
        recomputed_gradient,
        rtol=2.0e-9,
        atol=2.0e-9,
    )


def test_final_objective_has_credit_through_first_boundary() -> None:
    simulator, initial, target = _problem()
    policy = EquilibriumCenteredPolicy(encoder_dim=10, hidden_dim=10).double()
    result = strict_multiple_shooting_objective(
        policy,
        simulator,
        initial,
        target,
        shooting_config=StrictShootingConfig(
            segment_steps=3,
            segment_count=2,
            energy_interval_steps=1,
            use_checkpoint_recompute=True,
        ),
    )
    first_boundary = recurrent_tensors(result.rollout.boundaries[0])
    gradients = torch.autograd.grad(
        result.rollout.energy[-1].sum(),
        first_boundary,
        allow_unused=True,
    )
    norm = torch.linalg.vector_norm(
        torch.cat(
            tuple(
                torch.zeros_like(value).reshape(-1)
                if gradient is None
                else gradient.reshape(-1)
                for value, gradient in zip(first_boundary, gradients)
            )
        )
    )
    assert torch.isfinite(norm)
    assert float(norm) > 0.0

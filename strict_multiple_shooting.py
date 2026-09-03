"""Deprecated implementation backing :mod:`checkpointed_exact_bptt`.

This historical filename is retained only for compatibility.  The code
recomputes segments to save memory but differentiates the exact single-
shooting trajectory; it is not a full-space multiple-shooting solver.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch.utils.checkpoint import checkpoint

from env_l2f import L2FSimulator, L2FState, apply_gradient_decay
from equilibrium_control import (
    ContractionResult,
    EquilibriumCenteredPolicy,
    EquilibriumTarget,
    PhaseSpaceConfig,
    contraction_objective,
    equilibrium_prediction_loss,
    phase_space_energy,
)
from l2f_cuda_backend import cuda_step
from policy_observation import (
    PolicyObservationState,
    build_policy_observation,
    update_position_integral,
)


DYNAMIC_STATE_FIELDS = (
    "position",
    "velocity",
    "rotation",
    "omega",
    "motor",
    "previous_action",
)


@dataclass(frozen=True)
class RecurrentSystemState:
    state: L2FState
    hidden: torch.Tensor
    integral: torch.Tensor


@dataclass(frozen=True)
class StrictShootingConfig:
    """Configuration for an exactly continuous, recomputed long rollout.

    Segment boundaries are not optimizer parameters.  They are the exact output
    of the preceding segment and therefore satisfy every continuity equality by
    construction.  Checkpointing only changes how backward is evaluated.
    """

    segment_steps: int = 250
    segment_count: int = 4
    energy_interval_steps: int = 25
    observation_noise_max: float = 0.0
    integral_input_frame: str = "body"
    integral_input_multiplier: float = 1.0
    integral_limit: float = 0.5
    integral_leak: float = 0.0
    integral_clamp_mode: str = "box"
    state_step_decay: float = 1.0
    hidden_step_decay: float = 1.0
    backend: str = "torch"
    use_checkpoint_recompute: bool = True

    @property
    def horizon(self) -> int:
        return self.segment_steps * self.segment_count

    def validate(self) -> None:
        if self.segment_steps < 1 or self.segment_count < 1:
            raise ValueError("segment_steps and segment_count must be positive")
        if self.energy_interval_steps < 1:
            raise ValueError("energy_interval_steps must be positive")
        if self.segment_steps % self.energy_interval_steps:
            raise ValueError("energy_interval_steps must divide segment_steps")
        if self.backend not in ("torch", "cuda"):
            raise ValueError("backend must be 'torch' or 'cuda'")
        if not 0.0 <= self.state_step_decay <= 1.0:
            raise ValueError("state_step_decay must be in [0,1]")
        if not 0.0 <= self.hidden_step_decay <= 1.0:
            raise ValueError("hidden_step_decay must be in [0,1]")


@dataclass(frozen=True)
class StrictShootingResult:
    end: RecurrentSystemState
    energy: torch.Tensor
    prediction_loss: torch.Tensor
    direction_loss: torch.Tensor
    trim_loss: torch.Tensor
    boundaries: tuple[RecurrentSystemState, ...]
    boundary_signatures: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class StrictObjectiveResult:
    loss: torch.Tensor
    rollout: StrictShootingResult
    contraction: ContractionResult


def recurrent_tensors(value: RecurrentSystemState) -> tuple[torch.Tensor, ...]:
    return (
        value.state.position,
        value.state.velocity,
        value.state.rotation,
        value.state.omega,
        value.state.motor,
        value.state.previous_action,
        value.hidden,
        value.integral,
    )


def _state_from_dynamic(
    template: L2FState,
    values: tuple[torch.Tensor, ...],
) -> L2FState:
    if len(values) != 8:
        raise ValueError("the recurrent state must contain exactly eight tensors")
    dynamic = dict(zip(DYNAMIC_STATE_FIELDS, values[:6]))
    return L2FState(
        **{
            field.name: dynamic.get(field.name, getattr(template, field.name))
            for field in fields(L2FState)
        }
    )


def _boundary_signature(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Small detached diagnostic; never participates in the objective."""

    return torch.stack(
        tuple(value.detach().float().square().mean().sqrt() for value in values)
    )


def _rollout_segment(
    policy: EquilibriumCenteredPolicy,
    simulator: L2FSimulator,
    template: L2FState,
    target: EquilibriumTarget,
    phase_config: PhaseSpaceConfig,
    config: StrictShootingConfig,
    values: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    state = _state_from_dynamic(template, values)
    hidden = values[6]
    observation_state = PolicyObservationState(values[7])
    energy_samples: list[torch.Tensor] = []
    prediction_sum = state.position.sum() * 0.0
    direction_sum = prediction_sum
    trim_sum = prediction_sum

    for local_step in range(config.segment_steps):
        observation, observed_position = build_policy_observation(
            state,
            observation_state,
            mode="integral25",
            noise_max=config.observation_noise_max,
            integral_input_frame=config.integral_input_frame,
            integral_input_multiplier=config.integral_input_multiplier,
        )
        action, hidden, details = policy.forward_with_aux(observation, hidden)
        hidden = apply_gradient_decay(hidden, config.hidden_step_decay)
        prediction, direction, trim = equilibrium_prediction_loss(details, target)
        prediction_sum = prediction_sum + prediction
        direction_sum = direction_sum + direction
        trim_sum = trim_sum + trim
        observation_state = update_position_integral(
            observation_state,
            observed_position,
            dt=simulator.params.dt,
            integral_limit=config.integral_limit,
            integral_leak=config.integral_leak,
            integral_clamp_mode=config.integral_clamp_mode,
        )
        if config.backend == "cuda":
            state = cuda_step(
                state,
                action,
                simulator.params,
                grad_decay=config.state_step_decay,
            )
        else:
            state = simulator.step(
                state,
                action,
                grad_decay=config.state_step_decay,
            )
        if (local_step + 1) % config.energy_interval_steps == 0:
            energy_samples.append(phase_space_energy(state, target, phase_config))

    if not energy_samples:
        raise RuntimeError("the segment produced no phase-space samples")
    end = (
        state.position,
        state.velocity,
        state.rotation,
        state.omega,
        state.motor,
        state.previous_action,
        hidden,
        observation_state.integral_position,
    )
    inverse_steps = 1.0 / float(config.segment_steps)
    return (
        *end,
        torch.stack(energy_samples, dim=0),
        prediction_sum * inverse_steps,
        direction_sum * inverse_steps,
        trim_sum * inverse_steps,
    )


def rollout_strict_multiple_shooting(
    policy: EquilibriumCenteredPolicy,
    simulator: L2FSimulator,
    initial: RecurrentSystemState,
    target: EquilibriumTarget,
    *,
    phase_config: PhaseSpaceConfig | None = None,
    config: StrictShootingConfig | None = None,
) -> StrictShootingResult:
    """Roll out checkpointed exact BPTT in H250-sized recompute blocks.

    ``z[k+1] = Phi(z[k], theta)`` is enforced by direct substitution.  The
    derivatives therefore equal uninterrupted full BPTT.  This function is a
    memory-saving gradient diagnostic, not a full-space multiple-shooting
    optimizer with independent boundary variables.
    """

    phase_config = phase_config or PhaseSpaceConfig()
    config = config or StrictShootingConfig()
    config.validate()
    if config.backend == "cuda" and not initial.state.position.is_cuda:
        raise ValueError("the CUDA backend requires CUDA state tensors")
    if config.observation_noise_max > 0.0 and config.use_checkpoint_recompute:
        # PyTorch checkpoint preserves RNG state, but forbidding noisy training
        # here makes exact-gradient validation and reproducibility unambiguous.
        raise ValueError("checkpoint-recomputed rollout requires zero observation noise")

    template = initial.state
    values = recurrent_tensors(initial)
    initial_energy = phase_space_energy(template, target, phase_config).unsqueeze(0)
    energy_parts = [initial_energy]
    prediction_parts: list[torch.Tensor] = []
    direction_parts: list[torch.Tensor] = []
    trim_parts: list[torch.Tensor] = []
    boundaries: list[RecurrentSystemState] = []
    signatures: list[torch.Tensor] = []

    for _ in range(config.segment_count):
        def segment_function(*inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return _rollout_segment(
                policy,
                simulator,
                template,
                target,
                phase_config,
                config,
                tuple(inputs),
            )

        outputs = (
            checkpoint(segment_function, *values, use_reentrant=False)
            if config.use_checkpoint_recompute
            else segment_function(*values)
        )
        values = tuple(outputs[:8])
        # Passing these exact tensors to the next segment is the continuity
        # constraint.  There is deliberately no detach, clone, or optimizer node.
        energy_parts.append(outputs[8])
        prediction_parts.append(outputs[9])
        direction_parts.append(outputs[10])
        trim_parts.append(outputs[11])
        boundaries.append(
            RecurrentSystemState(
                state=_state_from_dynamic(template, values),
                hidden=values[6],
                integral=values[7],
            )
        )
        signatures.append(_boundary_signature(values))

    end = RecurrentSystemState(
        state=_state_from_dynamic(template, values),
        hidden=values[6],
        integral=values[7],
    )
    return StrictShootingResult(
        end=end,
        energy=torch.cat(energy_parts, dim=0),
        prediction_loss=torch.stack(prediction_parts).mean(),
        direction_loss=torch.stack(direction_parts).mean(),
        trim_loss=torch.stack(trim_parts).mean(),
        boundaries=tuple(boundaries),
        boundary_signatures=tuple(signatures),
    )


def strict_multiple_shooting_objective(
    policy: EquilibriumCenteredPolicy,
    simulator: L2FSimulator,
    initial: RecurrentSystemState,
    target: EquilibriumTarget,
    *,
    phase_config: PhaseSpaceConfig | None = None,
    shooting_config: StrictShootingConfig | None = None,
    contraction_rate: float = 0.25,
    contraction_epsilon: float = 1.0e-6,
    cvar_fraction: float = 0.20,
    contraction_weight: float = 1.0,
    prediction_weight: float = 1.0,
    terminal_weight: float = 1.0,
    violation_cvar_weight: float = 1.0,
) -> StrictObjectiveResult:
    shooting_config = shooting_config or StrictShootingConfig()
    rollout = rollout_strict_multiple_shooting(
        policy,
        simulator,
        initial,
        target,
        phase_config=phase_config,
        config=shooting_config,
    )
    if not bool(target.feasible.any().item()):
        raise ValueError("the batch contains no feasible analytic equilibrium")
    contraction = contraction_objective(
        rollout.energy[:, target.feasible],
        interval_seconds=(
            shooting_config.energy_interval_steps * simulator.params.dt
        ),
        contraction_rate=contraction_rate,
        epsilon=contraction_epsilon,
        cvar_fraction=cvar_fraction,
        terminal_weight=terminal_weight,
        violation_cvar_weight=violation_cvar_weight,
    )
    loss = (
        float(contraction_weight) * contraction.loss
        + float(prediction_weight) * rollout.prediction_loss
    )
    return StrictObjectiveResult(loss=loss, rollout=rollout, contraction=contraction)

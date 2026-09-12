"""Deployable response-conditioned motor policy, without action teachers."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn

ARCHITECTURE = "response-conditioned-motor-policy-v2-actor-only"
OBSERVATION_DIM = 25


@dataclass(frozen=True)
class ResponsePolicyConfig:
    memory_dim: int = 64
    hidden_dim: int = 64
    dt: float = 0.01
    action_rate: float = 50.0
    integral_limit: float = 0.5
    integral_leak: float = 0.0

    def __post_init__(self) -> None:
        if min(self.memory_dim, self.hidden_dim) < 1:
            raise ValueError("network widths must be positive")
        if not all(math.isfinite(x) for x in (
            self.dt, self.action_rate, self.integral_limit, self.integral_leak
        )):
            raise ValueError("policy constants must be finite")
        if min(self.dt, self.action_rate, self.integral_limit) <= 0 or self.integral_leak < 0:
            raise ValueError("invalid policy integration or actuator constraints")


@dataclass(frozen=True)
class ResponsePolicyState:
    memory: torch.Tensor
    integral: torch.Tensor
    previous_velocity: torch.Tensor
    previous_omega: torch.Tensor
    previous_rotation: torch.Tensor
    last_action: torch.Tensor
    older_action: torch.Tensor
    calls: torch.Tensor


@dataclass(frozen=True)
class ResponsePolicyOutput:
    action: torch.Tensor
    next_state: ResponsePolicyState
    memory: torch.Tensor


def body_vector(rotation: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return (rotation.transpose(-1, -2) @ vector.unsqueeze(-1)).squeeze(-1)


class ResponseMotorPolicy(nn.Module):
    """Fixed deployment weights; recurrent memory adapts after real responses.

    Only the 25 deployable observation entries and executed actions enter this
    class. No motor truth, dynamics parameters, external-force truth, teacher,
    calibration flag, or privileged capability certificate is accepted.
    """

    def __init__(self, config: ResponsePolicyConfig = ResponsePolicyConfig()) -> None:
        super().__init__()
        self.config = config
        self.response_encoder = nn.Sequential(
            nn.Linear(26, config.hidden_dim), nn.SiLU()
        )
        self.response_memory = nn.GRUCell(config.hidden_dim, config.memory_dim)
        self.controller = nn.Sequential(
            nn.Linear(19 + config.memory_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 4),
        )
        # Small but nonzero weights preserve the cold-start task gradient into
        # memory. There is no zero authorization multiplier on learned control.
        nn.init.xavier_uniform_(self.controller[-1].weight, gain=0.1)
        nn.init.zeros_(self.controller[-1].bias)

    def initial_state(self, observation: torch.Tensor) -> ResponsePolicyState:
        self._check_observation(observation)
        rotation = observation[:, 6:15].reshape(-1, 3, 3)
        integral = (rotation @ observation[:, 18:21, None]).squeeze(-1)
        action = observation[:, 21:25]
        return ResponsePolicyState(
            observation.new_zeros(observation.shape[0], self.config.memory_dim),
            integral, observation[:, 3:6], observation[:, 15:18], rotation,
            action, action, observation.new_zeros(observation.shape[0], 1),
        )

    @staticmethod
    def _check_observation(observation: torch.Tensor) -> None:
        if observation.ndim != 2 or observation.shape[-1] != OBSERVATION_DIM:
            raise ValueError("expected [batch,25] deployable observation")

    def control_features(self, observation: torch.Tensor) -> torch.Tensor:
        rotation = observation[:, 6:15].reshape(-1, 3, 3)
        up = observation.new_tensor((0.0, 0.0, 1.0)).expand(observation.shape[0], 3)
        return torch.cat((
            body_vector(rotation, observation[:, :3]),
            body_vector(rotation, observation[:, 3:6]) / 3.0,
            body_vector(rotation, up),
            observation[:, 15:18] / 10.0,
            observation[:, 18:21] / self.config.integral_limit,
            observation[:, 21:25],
        ), -1)

    def forward(
        self,
        observation: torch.Tensor,
        state: Optional[ResponsePolicyState] = None,
        *,
        applied_action: Optional[torch.Tensor] = None,
        memory_enabled: bool = True,
    ) -> ResponsePolicyOutput:
        self._check_observation(observation)
        state = self.initial_state(observation) if state is None else state
        rotation = observation[:, 6:15].reshape(-1, 3, 3)
        # Observation previous_action is the command actually executed. This
        # is not a proposed action from a hypothetical policy evaluation.
        executed_previous = observation[:, 21:25]
        delta_velocity = body_vector(
            state.previous_rotation, observation[:, 3:6] - state.previous_velocity
        ) / (self.config.dt * 9.80665)
        delta_omega = (observation[:, 15:18] - state.previous_omega) / (
            self.config.dt * 1000.0
        )
        up = observation.new_tensor((0.0, 0.0, 1.0)).expand(observation.shape[0], 3)
        response_input = torch.cat((
            executed_previous, executed_previous - state.older_action,
            delta_velocity, delta_omega,
            (state.previous_omega + observation[:, 15:18]) / 20.0,
            body_vector(state.previous_rotation, up),
            body_vector(state.previous_rotation, state.previous_velocity) / 3.0,
            observation[:, 15:18] / 10.0,
        ), -1)
        proposed_memory = self.response_memory(
            self.response_encoder(torch.tanh(response_input)), state.memory
        )
        # At call0 no physical response exists yet; at call1 the first executed
        # action/response pair is available. No startup segment is detached.
        memory = torch.where(state.calls > 0, proposed_memory, state.memory)
        if not memory_enabled:
            memory = torch.zeros_like(memory)
        features = self.control_features(observation)
        proposed = torch.tanh(self.controller(torch.cat((features, memory), -1)))
        radius = self.config.action_rate * self.config.dt
        lower = (executed_previous - radius).clamp(-1.0, 1.0)
        upper = (executed_previous + radius).clamp(-1.0, 1.0)
        action = torch.maximum(lower, torch.minimum(upper, proposed))
        executed = action if applied_action is None else applied_action
        if executed.shape != action.shape:
            raise ValueError("executed action must have shape [batch,4]")
        if applied_action is not None and (
            not bool(torch.isfinite(executed).all()) or bool((executed.abs() > 1).any())
        ):
            raise ValueError("executed actions must be finite and in [-1,1]")
        integral = (
            max(0.0, 1.0 - self.config.integral_leak * self.config.dt) * state.integral
            + self.config.dt * observation[:, :3]
        ).clamp(-self.config.integral_limit, self.config.integral_limit)
        next_state = ResponsePolicyState(
            memory, integral, observation[:, 3:6], observation[:, 15:18], rotation,
            executed, executed_previous, state.calls + 1.0,
        )
        return ResponsePolicyOutput(action, next_state, memory)

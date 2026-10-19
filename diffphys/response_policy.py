"""Minimal recurrent motor policy with a hidden-only motor readout."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn

ARCHITECTURE = "gru16-hidden-only-readout-absolute-motor-policy-v1"
AGILE_ARCHITECTURE = "gru36-goal-conditioned-hidden-only-readout-absolute-motor-policy-v1"
OBSERVATION_DIM = 22
CONTROL_FEATURE_DIM = 16
GOAL_PACKET_DIM = 23
GOAL_FEATURE_DIM = 20


@dataclass(frozen=True)
class ResponsePolicyConfig:
    memory_dim: int = 64
    dt: float = 0.01
    goal_conditioning: bool = False

    def __post_init__(self) -> None:
        if (not isinstance(self.memory_dim, int) or isinstance(self.memory_dim, bool)
                or self.memory_dim < 1):
            raise ValueError("memory_dim must be a positive integer")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("invalid policy timestep or actuator constraints")
        if not isinstance(self.goal_conditioning, bool):
            raise ValueError("goal_conditioning must be boolean")


@dataclass(frozen=True)
class ResponsePolicyState:
    memory: torch.Tensor


@dataclass(frozen=True)
class ResponsePolicyOutput:
    action: torch.Tensor
    next_state: ResponsePolicyState


def body_vector(rotation: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return (rotation.transpose(-1, -2) @ vector.unsqueeze(-1)).squeeze(-1)


class ResponseMotorPolicy(nn.Module):
    """One native GRU and one hidden-only readout; no direct feature shortcut.

    Observation: p(3), v(3), measured R(9), body omega(3), previous command(4).
    Optional task goals are separate deployable commands, not simulator truth.
    No physical parameters, motor truth, learned encoder, explicit integral,
    response-difference cache, external controller or auxiliary network is used.
    Weights are fixed at deployment; memory updates from the FIRST observation.
    """

    def __init__(self, config: ResponsePolicyConfig = ResponsePolicyConfig()) -> None:
        super().__init__()
        self.config = config
        # Keep the native GRU and attribute name used by the existing VJP tools.
        feature_dim = CONTROL_FEATURE_DIM + (GOAL_FEATURE_DIM if config.goal_conditioning else 0)
        self.response_memory = nn.GRUCell(feature_dim, config.memory_dim)
        self.readout = nn.Linear(config.memory_dim, 4)
        # Only W_h remains. Current observations affect the action through GRU.
        with torch.no_grad():
            nn.init.xavier_uniform_(self.readout.weight, gain=0.1)
            self.readout.bias.zero_()
            if config.goal_conditioning:
                # Mission inputs can learn immediately, while an imported hover
                # controller initially keeps its original response to p-goal.
                self.response_memory.weight_ih[:, CONTROL_FEATURE_DIM:].zero_()

    @property
    def architecture(self) -> str:
        return AGILE_ARCHITECTURE if self.config.goal_conditioning else ARCHITECTURE

    def initial_state(self, observation: torch.Tensor) -> ResponsePolicyState:
        self._check_observation(observation)
        return ResponsePolicyState(
            observation.new_zeros(observation.shape[0], self.config.memory_dim)
        )

    @staticmethod
    def _check_observation(observation: torch.Tensor) -> None:
        if observation.ndim != 2 or observation.shape[-1] != OBSERVATION_DIM:
            raise ValueError("expected [batch,22] deployable observation")

    def control_features(self, observation: torch.Tensor, *,
                         goals: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Use measured state and one explicit world-frame mission packet.

        goals: current reference p(3), middle p(3), end p(3), middle R row-major(9),
        middle v(3), normalized motion time(1), remaining seconds(1).
        Task construction validates finite values once; this hot path checks
        shape/device/dtype without a device synchronization at every step.
        """
        self._check_observation(observation)
        if self.config.goal_conditioning:
            if (not isinstance(goals, torch.Tensor)
                    or goals.shape != (observation.shape[0], GOAL_PACKET_DIM)
                    or goals.device != observation.device or goals.dtype != observation.dtype):
                raise ValueError("goals must be [batch,23] and match observation device/dtype")
        elif goals is not None:
            raise ValueError("goals require a goal-conditioned policy")
        rotation = observation[:, 6:15].reshape(-1, 3, 3)
        position = observation[:, :3]
        error = position - goals[:, :3] if self.config.goal_conditioning else position
        base = torch.cat((
            body_vector(rotation, error),
            body_vector(rotation, observation[:, 3:6]) / 3.0,
            rotation[:, 2, :],  # R.T @ world-up, including the original sensor noise.
            observation[:, 15:18] / 10.0,
            observation[:, 18:22],  # Known command, not hidden actuator execution or motor truth.
        ), -1)
        if not self.config.goal_conditioning:
            return base
        relative_rotation = rotation.transpose(-1, -2) @ goals[:, 9:18].reshape(-1, 3, 3)
        identity = torch.eye(3, device=observation.device, dtype=observation.dtype)
        return torch.cat((
            base,
            body_vector(rotation, goals[:, 3:6] - position),
            body_vector(rotation, goals[:, 6:9] - position),
            (relative_rotation - identity).flatten(1),
            body_vector(rotation, goals[:, 18:21]) / 3.0,
            goals[:, 21:23],
        ), -1)

    def forward(
        self,
        observation: torch.Tensor,
        state: Optional[ResponsePolicyState] = None,
        *,
        goals: Optional[torch.Tensor] = None,
    ) -> ResponsePolicyOutput:
        self._check_observation(observation)
        state = self.initial_state(observation) if state is None else state
        if (state.memory.shape != (observation.shape[0], self.config.memory_dim)
                or state.memory.device != observation.device
                or state.memory.dtype != observation.dtype):
            raise ValueError("policy memory must match observation batch, device and dtype")
        features = self.control_features(observation, goals=goals)
        # Never skip call zero: current state must affect h_t and the first action.
        memory = self.response_memory(features, state.memory)
        proposed = torch.tanh(self.readout(memory))
        return ResponsePolicyOutput(proposed, ResponsePolicyState(memory))

"""Minimal recurrent motor policy with a direct state-to-action readout."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn

ARCHITECTURE = "gru16-direct-readout-absolute-motor-policy-v4"
OBSERVATION_DIM = 22
CONTROL_FEATURE_DIM = 16


@dataclass(frozen=True)
class ResponsePolicyConfig:
    memory_dim: int = 64
    dt: float = 0.01
    action_rate: float = 0.0  # Preserve the optional absolute-command slew projection.

    def __post_init__(self) -> None:
        if (not isinstance(self.memory_dim, int) or isinstance(self.memory_dim, bool)
                or self.memory_dim < 1):
            raise ValueError("memory_dim must be a positive integer")
        if not all(math.isfinite(x) for x in (self.dt, self.action_rate)):
            raise ValueError("policy constants must be finite")
        if self.dt <= 0 or self.action_rate < 0:
            raise ValueError("invalid policy timestep or actuator constraints")


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
    """One native GRU and one affine readout; hidden state is the only memory.

    Observation: p(3), v(3), measured R(9), body omega(3), executed action(4).
    No physical parameters, motor truth, learned encoder, explicit integral,
    response-difference cache, external controller or auxiliary network is used.
    Weights are fixed at deployment; memory updates from the FIRST observation.
    """

    def __init__(self, config: ResponsePolicyConfig = ResponsePolicyConfig()) -> None:
        super().__init__()
        self.config = config
        # Keep the native GRU and attribute name used by the existing VJP tools.
        self.response_memory = nn.GRUCell(CONTROL_FEATURE_DIM, config.memory_dim)
        self.readout = nn.Linear(CONTROL_FEATURE_DIM + config.memory_dim, 4)
        # One weight matrix [W_c | W_h], not two controllers or two optimizers.
        # Zero W_c is trainable immediately; nonzero W_h admits GRU gradients.
        with torch.no_grad():
            self.readout.weight[:, :CONTROL_FEATURE_DIM].zero_()
            nn.init.xavier_uniform_(self.readout.weight[:, CONTROL_FEATURE_DIM:], gain=0.1)
            self.readout.bias.zero_()

    def initial_state(self, observation: torch.Tensor) -> ResponsePolicyState:
        self._check_observation(observation)
        return ResponsePolicyState(
            observation.new_zeros(observation.shape[0], self.config.memory_dim)
        )

    @staticmethod
    def _check_observation(observation: torch.Tensor) -> None:
        if observation.ndim != 2 or observation.shape[-1] != OBSERVATION_DIM:
            raise ValueError("expected [batch,22] deployable observation")

    def control_features(self, observation: torch.Tensor) -> torch.Tensor:
        self._check_observation(observation)
        rotation = observation[:, 6:15].reshape(-1, 3, 3)
        return torch.cat((
            body_vector(rotation, observation[:, :3]),
            body_vector(rotation, observation[:, 3:6]) / 3.0,
            rotation[:, 2, :],  # R.T @ world-up, including the original sensor noise.
            observation[:, 15:18] / 10.0,
            observation[:, 18:22],  # Command actually executed, not motor truth.
        ), -1)

    def forward(
        self,
        observation: torch.Tensor,
        state: Optional[ResponsePolicyState] = None,
    ) -> ResponsePolicyOutput:
        self._check_observation(observation)
        state = self.initial_state(observation) if state is None else state
        if (state.memory.shape != (observation.shape[0], self.config.memory_dim)
                or state.memory.device != observation.device
                or state.memory.dtype != observation.dtype):
            raise ValueError("policy memory must match observation batch, device and dtype")
        features = self.control_features(observation)
        # Never skip call zero: current state must affect h_t and the first action.
        memory = self.response_memory(features, state.memory)
        proposed = torch.tanh(self.readout(torch.cat((features, memory), -1)))
        action = proposed
        if self.config.action_rate > 0:  # Same optional projection as the parent.
            executed_previous = observation[:, 18:22]
            radius = self.config.action_rate * self.config.dt
            lower = (executed_previous - radius).clamp(-1.0, 1.0)
            upper = (executed_previous + radius).clamp(-1.0, 1.0)
            action = torch.maximum(lower, torch.minimum(upper, proposed))
        return ResponsePolicyOutput(action, ResponsePolicyState(memory))

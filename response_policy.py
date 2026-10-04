"""Experimental geometric state feedback plus a bounded causal GRU residual.

Structural coefficient/input-gain bounds are NOT a closed-loop stability
certificate. The motor family, saturation and sensing delay still need audit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "geometric-gru-bounded-residual-absolute-motor-policy-v5"
OBSERVATION_DIM = 22
CONTROL_FEATURE_DIM = 16
CHANNEL_NAMES = ("collective", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class ResponsePolicyConfig:
    memory_dim: int = 64
    dt: float = 0.01
    residual_amplitude: tuple[float, ...] = (0.6, 0.12, 0.12, 0.06)
    residual_gain: tuple[float, ...] = (0.15, 0.06, 0.06, 0.03)
    horizontal_accel_limit: float = 6.0
    vertical_fraction: float = 0.8
    antipodal_epsilon: float = 1e-4

    def __post_init__(self) -> None:
        if (not isinstance(self.memory_dim, int) or isinstance(self.memory_dim, bool)
                or self.memory_dim < 1):
            raise ValueError("memory_dim must be a positive integer")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("invalid policy timestep or actuator constraints")
        for name in ("residual_amplitude", "residual_gain"):
            values = tuple(getattr(self, name))
            if (len(values) != 4 or any(not math.isfinite(v) for v in values)
                    or any(v < 0 if name == "residual_amplitude" else v <= 0 for v in values)):
                raise ValueError(name + " requires four finite channel limits (gain > 0, amplitude >= 0)")
            object.__setattr__(self, name, values)
        if not math.isfinite(self.horizontal_accel_limit) or self.horizontal_accel_limit <= 0:
            raise ValueError("horizontal_accel_limit must be finite and positive")
        if not math.isfinite(self.vertical_fraction) or not 0 < self.vertical_fraction < 1:
            raise ValueError("vertical_fraction must lie strictly between zero and one")
        if not math.isfinite(self.antipodal_epsilon) or not 0 < self.antipodal_epsilon < .01:
            raise ValueError("antipodal_epsilon must lie strictly between zero and .01")


@dataclass(frozen=True)
class ResponsePolicyState:
    memory: torch.Tensor


@dataclass(frozen=True)
class ResponsePolicyOutput:
    action: torch.Tensor
    next_state: ResponsePolicyState


@dataclass(frozen=True)
class ResponsePolicyComponents:
    """Optional observable decomposition; all commands are pre-tanh coordinates."""
    base_command: torch.Tensor
    residual_command: torch.Tensor
    motor_logits: torch.Tensor
    memory: torch.Tensor


def body_vector(rotation: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return (rotation.transpose(-1, -2) @ vector.unsqueeze(-1)).squeeze(-1)


def channels_to_motors(command: torch.Tensor) -> torch.Tensor:
    """Collective/roll/pitch/yaw -> FR/BR/BL/FL, FLU; not a wrench allocator."""
    t, r, p, y = command.unbind(-1)
    return torch.stack((t-r-p-y, t-r+p+y, t+r+p-y, t+r-p+y), -1)


def motors_to_channels(command: torch.Tensor) -> torch.Tensor:
    """Exact inverse of the fixed command-coordinate mixer (before saturation)."""
    a, b, c, d = command.unbind(-1)
    return torch.stack((a+b+c+d, -a-b+c+d, -a+b+c-d, -a+b-c+d), -1) / 4


class FeedbackCoefficients(nn.Module):
    """One small parameter vector, expanded per scene for exact group ownership.

    Order: kp_xy, kp_z, kv_xy, kv_z, k_tilt, k_rate_xy, k_rate_z,
    k_collective, common_hover_logit. Ranges are experimental, not certificates.
    Configuration constants are nonpersistent buffers: weights cannot overwrite
    the declared parameterization when loading a checkpoint.
    """
    def __init__(self) -> None:
        super().__init__()
        lower = torch.tensor((.2, .2, .2, .2, .01, .002, .002, .05, -1.))
        upper = torch.tensor((4., 6., 4., 5., .25, .15, .10, 1., 1.))
        initial = torch.tensor((1., 2., 1.5, 2., .06, .035, .025, .3, 0.))
        self.register_buffer("lower", lower, persistent=False)
        self.register_buffer("upper", upper, persistent=False)
        self.raw = nn.Parameter(torch.logit((initial-lower)/(upper-lower)))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # A tensor hook here sees per-scene partials before the shared sum.
        return self.raw.unsqueeze(0).expand(observation.shape[0], -1)

    def values(self, observation: torch.Tensor) -> torch.Tensor:
        return self.lower + (self.upper-self.lower) * torch.sigmoid(self(observation))


class GeometricFeedback(nn.Module):
    """Measured motion -> desired thrust direction -> reduced-attitude feedback.

    Uses no mass, inertia, inverse thrust curve or desired yaw. Velocity remains
    the existing delayed measurement. There is no new integral or action cache.
    """
    def __init__(self, config: ResponsePolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.coefficients = FeedbackCoefficients()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        kp, kpz, kv, kvz, kr, kw, kwz, kt, hover = self.coefficients.values(observation).unbind(-1)
        position, velocity = observation[:, :3], observation[:, 3:6]
        rotation = observation[:, 6:15].reshape(-1, 3, 3)
        omega = observation[:, 15:18]
        # Bounded horizontal acceleration preserves yaw equivariance (radial
        # rather than independent world-x/world-y clipping).
        horizontal = -kp[:, None]*position[:, :2] - kv[:, None]*velocity[:, :2]
        limit = self.config.horizontal_accel_limit
        horizontal = horizontal / torch.sqrt(1 + (horizontal/limit).square().sum(-1, keepdim=True))
        # Desired vertical specific force is strictly positive, so normalizing
        # the desired direction cannot divide by a vanishing desired force.
        g, span = 9.81, 9.81*self.config.vertical_fraction
        vertical = g - span*torch.tanh((kpz*position[:, 2] + kvz*velocity[:, 2])/span)
        force = torch.cat((horizontal, vertical[:, None]), -1)
        force_body = body_vector(rotation, force)
        direction = F.normalize(force_body, dim=-1, eps=self.config.antipodal_epsilon)
        cross = torch.stack((-direction[:, 1], direction[:, 0]), -1)
        sin2 = cross.square().sum(-1)
        eps = self.config.antipodal_epsilon
        sine = torch.sqrt(sin2 + eps*eps)
        cosine = direction[:, 2].clamp(-1, 1)
        angle = torch.atan2(sine, cosine)
        tilt = cross * (angle/sine)[:, None]
        # Exactly opposite directions have no unique shortest rotation axis.
        # Declare +body-x inside a tiny antipodal cap; this switching boundary
        # is not globally smooth and is excluded from smooth gain certificates.
        antipodal = (cosine < 0) & (sin2 < eps*eps)
        fallback = torch.stack((torch.full_like(angle, math.pi), torch.zeros_like(angle)), -1)
        tilt = torch.where(antipodal[:, None], fallback, tilt)
        differential = kr[:, None]*tilt - kw[:, None]*omega[:, :2]
        yaw = -kwz*omega[:, 2]  # Rate damping only: do not invent a heading target.
        collective = hover + kt*(force_body[:, 2]/g - 1)
        return torch.cat((collective[:, None], differential, yaw[:, None]), -1)


def gru_incremental_bounds(layer: nn.GRUCell) -> tuple[torch.Tensor, torch.Tensor]:
    """Conservative ||dh_new/dc||_2 and ||dh_new/dh_old||_2 upper bounds.

    Assumes h_old in [-1,1]^H, which zero initialization and native GRU updates
    preserve. F-norms upper-bound spectral norms. B bounds |W_hn h + b_hn|_inf;
    gate slopes are <=1/4, tanh slope <=1, and |h-n|<=2. This is a ONE-STEP
    controller bound, not contraction of the GRU or the physical closed loop.
    """
    ir, iz, inn = layer.weight_ih.chunk(3, 0)
    hr, hz, hn = layer.weight_hh.chunk(3, 0)
    bn = layer.bias_hh.chunk(3, 0)[2]
    b = (hn.abs().sum(-1) + bn.abs()).amax()
    lx = torch.linalg.vector_norm(inn) + .25*b*torch.linalg.vector_norm(ir) + .5*torch.linalg.vector_norm(iz)
    lh = 1 + torch.linalg.vector_norm(hn) + .25*b*torch.linalg.vector_norm(hr) + .5*torch.linalg.vector_norm(hz)
    return lx, lh


class IncrementalResidualReadout(nn.Linear):
    """Memory readout whose scale accounts for the WHOLE one-step GRU response.

    For channel i: delta_i = A_i tanh(w_eff_i h_new + b_i),
    w_eff_i = w_i / (1 + (A_i/G_i) ||w_i||_2 (L_c+L_h)).
    Thus ||d delta_i / d[c,h_old]||_2 <= G_i on the reachable hidden cube.
    Norm normalization is differentiable in the weights: never detach its scale.
    A bias can provide persistent adaptation without a large incremental gain.
    """
    def __init__(self, config: ResponsePolicyConfig) -> None:
        super().__init__(config.memory_dim, 4)
        self.register_buffer("amplitude", torch.tensor(config.residual_amplitude), persistent=False)
        self.register_buffer("gain_limit", torch.tensor(config.residual_gain), persistent=False)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.weight, gain=.1)
            self.bias.zero_()

    def effective_parameters(self, recurrent: nn.GRUCell) -> tuple[torch.Tensor, torch.Tensor]:
        lx, lh = gru_incremental_bounds(recurrent)
        norms = torch.linalg.vector_norm(self.weight, dim=-1)
        scale = 1 + (self.amplitude/self.gain_limit)*norms*(lx+lh)
        return self.weight/scale[:, None], self.bias

    def forward(self, memory: torch.Tensor, recurrent: nn.GRUCell) -> torch.Tensor:
        weight, bias = self.effective_parameters(recurrent)
        return F.linear(memory, weight, bias)


class ResponseMotorPolicy(nn.Module):
    """A structured instantaneous path and a directly acting recurrent residual.

    Observation: p(3), v(3), measured R(9), body omega(3), previous command(4).
    Only deployment observations are used. Weights are fixed during deployment;
    native GRU memory updates from the FIRST observation. The finite feedback
    ranges and residual derivative budgets require separate closed-loop testing.
    """
    def __init__(self, config: ResponsePolicyConfig = ResponsePolicyConfig()) -> None:
        super().__init__()
        self.config = config
        self.response_memory = nn.GRUCell(CONTROL_FEATURE_DIM, config.memory_dim)
        self.readout = IncrementalResidualReadout(config)
        self.base_feedback = GeometricFeedback(config)

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
            observation[:, 18:22],  # Known command, not hidden actuator execution or motor truth.
        ), -1)

    def components(
        self,
        observation: torch.Tensor,
        state: Optional[ResponsePolicyState] = None,
    ) -> ResponsePolicyComponents:
        self._check_observation(observation)
        state = self.initial_state(observation) if state is None else state
        if (state.memory.shape != (observation.shape[0], self.config.memory_dim)
                or state.memory.device != observation.device
                or state.memory.dtype != observation.dtype):
            raise ValueError("policy memory must match observation batch, device and dtype")
        features = self.control_features(observation)
        memory = self.response_memory(features, state.memory)
        base = self.base_feedback(observation)
        residual = self.readout.amplitude * torch.tanh(self.readout(memory, self.response_memory))
        logits = channels_to_motors(base + residual)
        return ResponsePolicyComponents(base, residual, logits, memory)

    def forward(
        self,
        observation: torch.Tensor,
        state: Optional[ResponsePolicyState] = None,
    ) -> ResponsePolicyOutput:
        parts = self.components(observation, state)
        return ResponsePolicyOutput(torch.tanh(parts.motor_logits), ResponsePolicyState(parts.memory))

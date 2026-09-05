"""Small physics-structured recurrent policy building blocks.

The module is deliberately independent of the legacy training entry points.  It
contains deployable, differentiable pieces that can be used for distillation or
for a later controlled experiment; it does not claim a stability certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from identification_features import (
    EXCITATION_HISTORY_LEN,
    EXCITATION_LAGS,
    EXCITATION_SCALE,
    bank_modal_features,
    normalize_response,
    production_legacy24,
    sol_response,
)
from probe_contract_v5 import (VERSION as PROBE_CONTRACT_VERSION, ACTIVE_STEPS as PROBE_PERIOD,
    PUBLISH_START, ProbeState, apply_probe, WAVEFORM)
from structured_allocator import ActiveSetBoxQPAllocator


ACTION_DIM = 4
OBSERVATION_DIM = 25
ERROR_DIM = 15
IDENTIFICATION_INPUT_DIM = 24
# Versioned fixed candidate motor time constants used by optional observer
# banks.  Version 1 is the original 5x3 bank; version 2 expands coverage to a
# 7x5 log grid.  The explicit version prevents a stale K=15 checkpoint from
# being silently interpreted as a different observer bank.
MOTOR_OBSERVER_TAU_RISE_GRID_V1 = (0.025, 0.04095, 0.06708, 0.10989, 0.18)
MOTOR_OBSERVER_TAU_RATIO_GRID_V1 = (1.0, math.sqrt(2.6), 2.6)


def motor_observer_tau_grid(version: int = 1) -> tuple[tuple[float, float], ...]:
    """Return the registered (rise, fall) observer modes for ``version``."""

    if version == 1:
        rises = MOTOR_OBSERVER_TAU_RISE_GRID_V1
        ratios = MOTOR_OBSERVER_TAU_RATIO_GRID_V1
    elif version == 2:
        rises = tuple(0.025 * (0.18 / 0.025) ** (index / 6.0)
                      for index in range(7))
        ratios = tuple(1.0 * (2.6 / 1.0) ** (index / 4.0)
                       for index in range(5))
    else:
        raise ValueError("motor observer tau grid version must be 1 or 2")
    return tuple(
        (float(rise), min(0.35, max(0.03, float(rise) * float(ratio))))
        for rise in rises for ratio in ratios
    )

from identification_information import INFORMATION_DIM, advance_information, capability_support

IDENTIFICATION_PROBE_PERIOD = PROBE_PERIOD


def identification_probe_patterns(*, device: torch.device,
                                  dtype: torch.dtype) -> torch.Tensor:
    """Return the v4 artifact exactly; no policy-local waveform is allowed."""
    return torch.tensor(WAVEFORM, device=device, dtype=dtype)[:, None].expand(-1, 4)


def _so3_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Small batched SO(3) exponential used to align one-step responses."""

    if rotation_vector.ndim != 2 or rotation_vector.shape[-1] != 3:
        raise ValueError("rotation_vector must have shape [batch,3]")
    x, y, z = rotation_vector.unbind(-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), -1).reshape(-1, 3, 3)
    theta2 = rotation_vector.square().sum(dim=-1, keepdim=True)
    theta = theta2.clamp_min(1.0e-12).sqrt()
    small = theta2 < 1.0e-8
    a = torch.where(
        small, 1.0 - theta2 / 6.0 + theta2.square() / 120.0,
        torch.sin(theta) / theta.clamp_min(1.0e-12),
    )
    b = torch.where(
        small, 0.5 - theta2 / 24.0 + theta2.square() / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1.0e-12),
    )
    identity = torch.eye(3, device=rotation_vector.device, dtype=rotation_vector.dtype)
    identity = identity.expand(rotation_vector.shape[0], 3, 3)
    return identity + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * torch.bmm(skew, skew)


def _lagged_cross_response_context(
    excitation_history: torch.Tensor,
    force_response: torch.Tensor,
    angular_response: torch.Tensor,
    motor_delta: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compatibility wrapper for the production shared feature function."""

    return production_legacy24(
        excitation_history=excitation_history,
        force_response=force_response,
        angular_response=angular_response,
        motor_delta=motor_delta,
        response_mask=response_mask,
    )

# Physical-fit sampler bounds.  Capability is represented in log space so a
# single additive uncertainty scale has the same meaning across the very wide
# alpha range.
CAPABILITY_LO = (1.45, 35.0, 0.02, 1.45, 0.025, 0.03)
CAPABILITY_HI = (5.50, 2200.0, 1.00, 1.95, 0.18, 0.35)
CAPABILITY_DEFAULT = (3.2, 100.0, 0.2, 1.7, 0.1, 0.15)


@dataclass(frozen=True)
class StructuredPolicyConfig:
    hidden_dim: int = 64
    identifier_dim: int = 64
    dt: float = 0.01
    observer_tau: float = 0.06
    # Optional fixed bank of motor observers.  K=0 is the legacy single
    # observer path and preserves old checkpoint parameter/state semantics.
    motor_observer_bank_size: int = 0
    # Explicit observer-bank version.  A non-legacy mode must agree with both
    # this version and the corresponding dynamic bank size.
    motor_observer_mode: str = "legacy"
    motor_tau_grid_version: int = 0
    # The identifier ingests the response sequence at every physical step;
    # only its capability output is published at ``slow_cadence``.  A small
    # per-step leak preserves motor-lag/order information while keeping the
    # latent slower than the fast feedback path.  The previous implementation
    # averaged 25 samples before one update with leak=.05 (about a five-second
    # time constant) and discarded the excitation/response ordering needed to
    # identify actuator lag and authority before the t50 gate.
    identifier_leak: float = 0.08
    identifier_limit: float = 1.0
    integral_limit: float = 0.5
    integral_leak: float = 0.0
    anti_windup_gain: float = 0.5
    residual_scale: float = 0.0
    residual_trainable: bool = False
    allocator_damping: float = 1.0e-3
    allocator_temperature: float = 0.25
    allocator_rate_limit: float = 0.0
    # ``smooth_dls`` preserves the experimental checkpoint semantics.  New
    # formal structured runs explicitly select ``box_qp``, which solves the
    # bounded/rate-constrained four-motor allocation problem on its active
    # face and exposes primal/KKT residuals.
    allocator_solver: str = "smooth_dls"
    capability_sigma_min: float = 0.05
    # The burn-in allocator uses an explicit full-prior maximum, so the learned
    # posterior need not start at sigma=2 (which weakens mean gradients by
    # 16x).  sigma=.5 with smoke beta=2 spans the normalized [-1,1] prior.
    capability_initial_sigma: float = 0.5
    # Uncertainty calibration must not reshape the shared identifier so that
    # a poor mean becomes easier to explain with a large sigma.  The mean path
    # owns the representation; the scale head fits conditional residuals on a
    # detached copy after/alongside mean training.
    capability_scale_detach_identifier: bool = True
    capability_ucb_beta: float = 2.0
    contextual_gain_rho: float = 0.25
    contextual_gain_scale: float = 1.0
    contextual_gain_jfloor: float = 1.0e-3
    contextual_blend_steps: int = 25
    burn_in_steps: int = 25
    burn_in_action_cap: float = 0.05
    burn_in_rate_limit: float = 0.5
    # Deterministic zero-mean identification excitation, expressed in motor
    # coordinates and mapped through the same conservative mixer/allocator.
    # A four-bank H50 screen selected .005; larger values violated the
    # pre-registered 1.5x paired angular-rate gate on at least one bank.
    burn_in_probe_amplitude: float = 0.0
    slow_cadence: int = 25
    # Capability/trim outputs are deliberately unavailable at call25.  The
    # recurrent identifier still consumes responses continuously, but the
    # conservative prior remains active until call50, when all 50 registered
    # probe transitions have been observed.
    identification_publish_start: int = PUBLISH_START
    # Constant external-force observer.  With cadence=25 and dt=.01, 0.2 s
    # leaves 8.2% startup bias at t50; the previous 0.5 s choice imposed a
    # 36.8% oracle bias and made the registered t50/t75 gate impossible.
    disturbance_observer_tau: float = 0.2


@dataclass
class StructuredPolicyState:
    hidden: torch.Tensor
    identifier: torch.Tensor
    motor_estimate: torch.Tensor
    integral: torch.Tensor
    motor_bank: Optional[torch.Tensor] = None
    slow_trim: Optional[torch.Tensor] = None
    slow_body_z: Optional[torch.Tensor] = None
    capability: Optional[torch.Tensor] = None
    slow_counter: int = 0
    prev_velocity: Optional[torch.Tensor] = None
    prev_omega: Optional[torch.Tensor] = None
    context_sum: Optional[torch.Tensor] = None
    capability_log_mean: Optional[torch.Tensor] = None
    capability_log_scale: Optional[torch.Tensor] = None
    capability_ucb: Optional[torch.Tensor] = None
    contextual_gain: Optional[torch.Tensor] = None
    contextual_gain_target: Optional[torch.Tensor] = None
    contextual_blend: Optional[torch.Tensor] = None
    boot_progress: Optional[torch.Tensor] = None
    capability_ucb_target: Optional[torch.Tensor] = None
    identification_failed: Optional[torch.Tensor] = None
    disturbance_accel: Optional[torch.Tensor] = None
    disturbance_residual_sum: Optional[torch.Tensor] = None
    disturbance_thrust_sum: Optional[torch.Tensor] = None
    previous_executed_action: Optional[torch.Tensor] = None
    disturbance_response_count: Optional[torch.Tensor] = None
    # Normalized command innovations from the most recent transitions.  Index
    # zero is the excitation which produced the response seen by this call.
    excitation_history: Optional[torch.Tensor] = None
    # Motor estimate immediately before the preceding executed command.  This
    # makes the observer's motor delta causal and action-aligned.
    previous_motor_estimate: Optional[torch.Tensor] = None
    probe_residual: Optional[torch.Tensor] = None
    probe_aborted: Optional[torch.Tensor] = None
    probe_omega_reference: Optional[torch.Tensor] = None
    identification_information: Optional[torch.Tensor] = None
    capability_authorized: Optional[torch.Tensor] = None
    identification_actions: Optional[torch.Tensor] = None
    identification_initial_motor: Optional[torch.Tensor] = None
    disturbance_history: Optional[torch.Tensor] = None

    @property
    def slow_hidden(self) -> torch.Tensor:
        return self.identifier

    @property
    def fast_hidden(self) -> torch.Tensor:
        return self.hidden

    def detach(self) -> "StructuredPolicyState":
        return StructuredPolicyState(**{
            field.name: (getattr(self, field.name).detach()
                         if torch.is_tensor(getattr(self, field.name)) else getattr(self, field.name))
            for field in fields(self)
        })


@dataclass(frozen=True)
class AllocatorDiagnostics:
    condition_number: torch.Tensor
    wrench_residual: torch.Tensor
    saturation: torch.Tensor
    rate_limited: torch.Tensor
    lower_headroom: torch.Tensor
    upper_headroom: torch.Tensor
    headroom_violation: torch.Tensor
    minimum_headroom: torch.Tensor
    trust_limited: torch.Tensor


@dataclass(frozen=True)
class StructuredPolicyOutput:
    action: torch.Tensor
    next_state: StructuredPolicyState
    auxiliary: dict


class MotorObserver(nn.Module):
    """Differentiable first-order motor observer.

    The observer intentionally models the deployable command-to-motor lag and
    has no access to privileged simulator motor state.
    """

    def __init__(self, tau: float = 0.06) -> None:
        super().__init__()
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        self.tau = float(tau)

    def forward(self, estimate: torch.Tensor, command: torch.Tensor,
                dt: float, tau_rise: Optional[torch.Tensor] = None,
                tau_fall: Optional[torch.Tensor] = None) -> torch.Tensor:
        if estimate.shape != command.shape or estimate.shape[-1] != ACTION_DIM:
            raise ValueError("estimate and command must have shape [batch,4]")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if tau_rise is None:
            tau_rise = estimate.new_full(estimate.shape, self.tau)
        if tau_fall is None:
            tau_fall = tau_rise
        if tau_rise.shape != estimate.shape or tau_fall.shape != estimate.shape:
            raise ValueError("tau tensors must match motor shape")
        tau = torch.where(command >= estimate, tau_rise, tau_fall).clamp_min(1.0e-4)
        alpha = (float(dt) / tau).clamp(0.0, 1.0)
        return estimate + alpha * (command.clamp(-1.0, 1.0) - estimate)


def replay_motor_history(observer: MotorObserver, initial: torch.Tensor,
                         actions: torch.Tensor, tau_rise: torch.Tensor,
                         tau_fall: torch.Tensor, dt: float, *, return_sequence: bool = False) -> torch.Tensor:
    """Reconstruct the current motor state with inferred continuous lag."""
    motor = initial
    sequence = []
    for command in actions.unbind(1):
        motor = observer(motor, command, dt, tau_rise.expand_as(motor), tau_fall.expand_as(motor))
        sequence.append(motor)
    return torch.stack(sequence, 1) if return_sequence else motor


class MultiTauMotorObserverBank(nn.Module):
    """A fixed bank of causal first-order motor observers.

    Each mode has its own rise/fall time constants.  The bank has no learned
    parameters and is advanced with the command actually applied to the
    simulator.  It is intentionally separate from :class:`MotorObserver` so
    that K=0 remains byte-for-byte compatible with legacy policy checkpoints.
    """

    def __init__(self, tau_pairs: Optional[tuple[tuple[float, float], ...]] = None,
                 *, version: int = 1) -> None:
        super().__init__()
        pairs = motor_observer_tau_grid(version) if tau_pairs is None else tuple(tau_pairs)
        if not pairs:
            raise ValueError("multi-tau observer bank must contain at least one mode")
        rise, fall = zip(*pairs)
        if any(value <= 0.0 for value in (*rise, *fall)):
            raise ValueError("multi-tau observer time constants must be positive")
        self.register_buffer("tau_rise", torch.tensor(rise, dtype=torch.float32))
        self.register_buffer("tau_fall", torch.tensor(fall, dtype=torch.float32))

    @property
    def modes(self) -> int:
        return int(self.tau_rise.numel())

    def forward(self, estimate: torch.Tensor, command: torch.Tensor, dt: float) -> torch.Tensor:
        if estimate.ndim != 3 or estimate.shape[-1] != ACTION_DIM:
            raise ValueError("bank estimate must have shape [batch,K,4]")
        if command.shape != (estimate.shape[0], ACTION_DIM):
            raise ValueError("command must have shape [batch,4]")
        if estimate.shape[1] != self.modes or dt <= 0.0:
            raise ValueError("bank mode count or dt is invalid")
        tau_rise = self.tau_rise.to(device=estimate.device, dtype=estimate.dtype).view(1, -1, 1)
        tau_fall = self.tau_fall.to(device=estimate.device, dtype=estimate.dtype).view(1, -1, 1)
        command_expanded = command[:, None, :]
        tau = torch.where(command_expanded >= estimate, tau_rise, tau_fall).clamp_min(1.0e-4)
        alpha = (float(dt) / tau).clamp(0.0, 1.0)
        return estimate + alpha * (command_expanded.clamp(-1.0, 1.0) - estimate)


class SlowIdentifier(nn.Module):
    """Causal response identifier with a bounded, rate-limited state.

    A feed-forward candidate followed by an EMA cannot distinguish two motor
    systems whose window means match but whose response order differs.  The
    GRU candidate retains that order (including rise/fall lag), while the
    explicit blend limits how quickly the deployment context itself moves.
    """

    def __init__(self, input_dim: int = OBSERVATION_DIM + ACTION_DIM,
                 identifier_dim: int = 8, leak: float = 0.05,
                 limit: float = 1.0) -> None:
        super().__init__()
        if identifier_dim < 1 or not 0.0 < leak <= 1.0 or limit <= 0.0:
            raise ValueError("invalid identifier dimensions, leak, or limit")
        self.cell = nn.GRUCell(input_dim, identifier_dim)
        self.leak = float(leak)
        self.limit = float(limit)

    def forward(self, context: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != self.cell.input_size:
            raise ValueError("context has incompatible shape")
        if previous.ndim != 2 or previous.shape[-1] != self.cell.hidden_size:
            raise ValueError("previous identifier state has incompatible shape")
        candidate = self.cell(context, previous)
        proposed = (1.0 - self.leak) * previous + self.leak * candidate
        return self.limit * torch.tanh(proposed / self.limit)


class AntiWindupIntegral(nn.Module):
    """Smooth bounded integral with differentiable back-calculation."""

    def __init__(self, limit: float = 0.5, leak: float = 0.0,
                 back_calculation: float = 0.5) -> None:
        super().__init__()
        if limit <= 0.0 or leak < 0.0 or back_calculation < 0.0:
            raise ValueError("invalid anti-windup parameters")
        self.limit = float(limit)
        self.leak = float(leak)
        self.back_calculation = float(back_calculation)

    def forward(self, integral: torch.Tensor, error: torch.Tensor, dt: float,
                authority: Optional[torch.Tensor] = None) -> torch.Tensor:
        if integral.shape != error.shape or integral.shape[-1] != 3:
            raise ValueError("integral and error must have shape [batch,3]")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if authority is None:
            authority = torch.ones_like(integral[..., :1])
        authority = authority.clamp(0.0, 1.0)
        retention = max(0.0, 1.0 - self.leak * float(dt))
        candidate = retention * integral + float(dt) * authority * error
        bounded = self.limit * torch.tanh(candidate / self.limit)
        corrected = candidate + self.back_calculation * (bounded - candidate)
        return self.limit * torch.tanh(corrected / self.limit)


class FastFeedbackInterface(nn.Module):
    """Verified/frozen fast feedback map.

    Gains are buffers by default.  A caller may explicitly request trainable
    gains, but the default is a zero, frozen interface suitable for a safe
    distillation baseline.
    """

    def __init__(self, action_dim: int = ACTION_DIM, error_dim: int = ERROR_DIM,
                 gain: Optional[torch.Tensor] = None, trainable: bool = False,
                 verified: bool = True, share_gain: bool = False) -> None:
        super().__init__()
        provided_gain = gain is not None
        if gain is None:
            gain = torch.zeros(action_dim, error_dim)
        if tuple(gain.shape) != (action_dim, error_dim):
            raise ValueError("gain must have shape [action_dim,error_dim]")
        if trainable:
            self.gain = nn.Parameter(gain.clone())
        else:
            self.register_buffer("gain", gain if share_gain else gain.clone())
        # A zero gain is a safe placeholder, not a verified controller.
        self._verified = bool(verified) and provided_gain and bool(
            gain.detach().norm().item() > 1.0e-8
        )
        self._trainable = bool(trainable)

    @property
    def verified(self) -> bool:
        return self._verified and bool(torch.isfinite(self.gain).all().item())

    @property
    def frozen(self) -> bool:
        return not self._trainable

    def freeze(self) -> None:
        self._trainable = False
        if isinstance(self.gain, nn.Parameter):
            self.gain.requires_grad_(False)

    def verify(self, maximum_gain: float = 100.0,
               minimum_norm: float = 1.0e-8) -> bool:
        self._verified = bool(torch.isfinite(self.gain).all().item()) and bool(
            self.gain.detach().abs().amax().item() <= maximum_gain
        ) and bool(self.gain.detach().norm().item() > minimum_norm)
        return self._verified

    def install_verified_gain(self, gain: torch.Tensor,
                              maximum_gain: float = 100.0,
                              minimum_norm: float = 1.0e-8) -> bool:
        if tuple(gain.shape) != tuple(self.gain.shape):
            raise ValueError("gain has incompatible shape")
        with torch.no_grad():
            self.gain.copy_(gain.to(device=self.gain.device, dtype=self.gain.dtype))
        self._verified = False
        return self.verify(maximum_gain, minimum_norm)

    def forward(self, error: torch.Tensor) -> torch.Tensor:
        if error.ndim != 2 or error.shape[-1] != self.gain.shape[-1]:
            raise ValueError("error has incompatible shape")
        return torch.einsum("ae,be->ba", self.gain, error)


def effective_wrench_mixer(capability: torch.Tensor, gravity: float = 9.80665) -> torch.Tensor:
    """Build the physical-fit specific-wrench mixer from capability six-vector.

    Capability columns are ``[thrust_to_weight, alpha_roll, eta_yaw,
    jz_over_jxy, tau_rise, tau_fall]``.  The last three are retained for the
    identifier/distillation interface but do not enter the instantaneous mixer.
    Signs match :func:`diagnostics.physics.thrust_to_wrench`.
    """
    if capability.ndim != 2 or capability.shape[-1] != 6:
        raise ValueError("capability must have shape [batch,6]")
    tw = capability[:, 0].clamp_min(1.01)
    alpha = capability[:, 1].clamp_min(1.0e-3)
    eta = capability[:, 2].clamp_min(1.0e-3)
    reserve = torch.maximum(2.0 - tw, torch.zeros_like(tw))
    q = (tw - 1.0) / (tw - reserve).clamp_min(1.0e-3)
    collective = (tw - 1.0) * float(gravity) / 4.0
    roll = q * alpha
    yaw = q * eta * alpha / 2.0
    zeros = torch.zeros_like(roll)
    return torch.stack((
        collective, collective, collective, collective,
        zeros, roll, zeros, -roll,
        -roll, zeros, roll, zeros,
        yaw, -yaw, yaw, -yaw,
    ), dim=-1).reshape(-1, 4, 4)


def equilibrium_from_capability_disturbance(
    capability: torch.Tensor,
    disturbance_accel_world: torch.Tensor,
    *,
    gravity: float = 9.80665,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the physical-fit hover equilibrium from deployable estimates.

    ``disturbance_accel_world`` is the estimated persistent external force per
    unit mass.  For the symmetric, linear-thrust physical-fit family, static
    hover requires ``b3*T/m = g*e3 - disturbance``.  The returned motor trim
    is therefore an analytic quantity, not another free neural action head.

    Returns ``(body_z_world, motor_trim, feasible)``.  The feasibility flag is
    diagnostic; the command is clamped to the deployable motor interval.
    """

    if capability.ndim != 2 or capability.shape[-1] != 6:
        raise ValueError("capability must have shape [batch,6]")
    if disturbance_accel_world.shape != (capability.shape[0], 3):
        raise ValueError("disturbance_accel_world must have shape [batch,3]")
    required = -disturbance_accel_world
    required = required.clone()
    required[:, 2] = required[:, 2] + float(gravity)
    required_norm = torch.linalg.vector_norm(required, dim=-1, keepdim=True)
    body_z = required / required_norm.clamp_min(1.0e-8)
    thrust_ratio = required_norm / float(gravity)
    thrust_to_weight = capability[:, 0:1]
    raw_trim = (thrust_ratio - 1.0) / (thrust_to_weight - 1.0).clamp_min(1.0e-4)
    minimum_ratio = torch.maximum(
        2.0 - thrust_to_weight, torch.zeros_like(thrust_to_weight)
    )
    feasible = (
        torch.isfinite(body_z).all(dim=-1)
        & torch.isfinite(raw_trim).all(dim=-1)
        & (body_z[:, 2] > 0.0)
        & (thrust_ratio[:, 0] >= minimum_ratio[:, 0])
        & (thrust_ratio[:, 0] <= thrust_to_weight[:, 0])
    )
    trim = raw_trim.clamp(-1.0, 1.0).expand(-1, ACTION_DIM)
    return body_z, trim, feasible


def reference_fast_gain(*, device=None, dtype=None) -> torch.Tensor:
    """Conservative nominal LQR gain in normalized 15D-error coordinates.

    This is an initialization candidate, not a certificate.  The lateral
    entries come from the hover linearization with motor lag omitted; every
    accepted deployment must still pass the augmented local and rollout gates.
    """

    gain = torch.zeros(4, ERROR_DIM, device=device, dtype=dtype)
    gain[0, 2] = -0.312
    gain[0, 5] = -1.430
    gain[1, 1] = 0.956
    gain[1, 4] = 0.971
    gain[1, 7] = -3.903
    gain[1, 8] = -4.443
    gain[2, 0] = -0.956
    gain[2, 3] = -0.971
    gain[2, 6] = 3.903
    gain[2, 9] = -4.443
    gain[3, 10] = -2.000
    return gain


class DampedConstrainedAllocator(nn.Module):
    """Damped least-squares wrench allocator with smooth box/rate limits."""

    def __init__(self, damping: float = 1.0e-3, temperature: float = 0.25,
                 rate_limit: float = 0.0) -> None:
        super().__init__()
        if damping <= 0.0 or temperature <= 0.0 or rate_limit < 0.0:
            raise ValueError("invalid allocator parameters")
        mixer = torch.tensor((
            (0.25, 0.25, 0.25, 0.25),
            (0.0, 1.0, 0.0, -1.0),
            (-1.0, 0.0, 1.0, 0.0),
            (1.0, -1.0, 1.0, -1.0),
        ))
        self.register_buffer("mixer", mixer)
        self.damping = float(damping)
        self.temperature = float(temperature)
        self.rate_limit = float(rate_limit)

    def forward(self, desired_wrench: torch.Tensor,
                previous_action: Optional[torch.Tensor] = None,
                dt: float = 0.01,
                mixer: Optional[torch.Tensor] = None,
                trim: Optional[torch.Tensor] = None,
                action_delta_cap: Optional[float] = None,
                rate_limit: Optional[float] = None) -> tuple[torch.Tensor, AllocatorDiagnostics]:
        if desired_wrench.ndim != 2 or desired_wrench.shape[-1] != 4:
            raise ValueError("desired_wrench must have shape [batch,4]")
        if previous_action is not None and previous_action.shape != desired_wrench.shape:
            raise ValueError("previous_action must match desired_wrench")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if mixer is None:
            mixer = self.mixer.expand(desired_wrench.shape[0], -1, -1)
        if mixer.shape != (desired_wrench.shape[0], 4, 4):
            raise ValueError("mixer must have shape [batch,4,4]")
        if trim is not None and trim.shape != desired_wrench.shape:
            raise ValueError("trim must have shape [batch,4]")
        gram = mixer.transpose(1, 2) @ mixer
        eye = torch.eye(4, device=mixer.device, dtype=mixer.dtype).expand_as(gram)
        inverse = torch.linalg.solve(gram + self.damping * eye, mixer.transpose(1, 2))
        unconstrained_delta = torch.zeros_like(desired_wrench)
        if trim is None:
            unconstrained = torch.bmm(inverse, desired_wrench.unsqueeze(-1)).squeeze(-1)
            boxed = torch.tanh(unconstrained / self.temperature)
            base = torch.zeros_like(boxed)
        else:
            delta_wrench = desired_wrench - torch.bmm(mixer, trim.unsqueeze(-1)).squeeze(-1)
            delta_action = torch.bmm(inverse, delta_wrench.unsqueeze(-1)).squeeze(-1)
            unconstrained_delta = delta_action
            trim_safe = trim.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
            base = trim_safe
            base_logit = torch.atanh(trim_safe)
            boxed = torch.tanh(base_logit + delta_action / (1.0 - trim_safe.square()).clamp_min(1.0e-4))
        lower_headroom = base + 1.0
        upper_headroom = 1.0 - base
        positive_delta = torch.relu(unconstrained_delta)
        negative_delta = torch.relu(-unconstrained_delta)
        headroom_violation = (
            torch.relu(positive_delta - upper_headroom)
            + torch.relu(negative_delta - lower_headroom)
        ).mean(dim=-1)
        directional_headroom = torch.where(
            unconstrained_delta >= 0.0, upper_headroom, lower_headroom
        )
        trust_limited = torch.zeros(desired_wrench.shape[0], device=desired_wrench.device,
                                    dtype=desired_wrench.dtype)
        if action_delta_cap is not None:
            cap = torch.as_tensor(action_delta_cap, device=boxed.device, dtype=boxed.dtype)
            if cap.ndim == 0:
                cap = cap.expand(boxed.shape[0], 1)
            elif cap.ndim == 1:
                cap = cap.reshape(-1, 1)
            if cap.shape != (boxed.shape[0], 1) or bool((cap <= 0.0).any().item()):
                raise ValueError("action_delta_cap must be positive scalar or [batch]")
            capped = base + cap * torch.tanh((boxed - base) / cap)
            trust_limited = (capped - boxed).abs().mean(dim=-1)
            boxed = capped
        effective_rate_limit = self.rate_limit if rate_limit is None else torch.as_tensor(
            rate_limit, device=boxed.device, dtype=boxed.dtype
        )
        if torch.is_tensor(effective_rate_limit):
            if effective_rate_limit.ndim == 0:
                effective_rate_limit = effective_rate_limit.expand(boxed.shape[0], 1)
            elif effective_rate_limit.ndim == 1:
                effective_rate_limit = effective_rate_limit.reshape(-1, 1)
            if effective_rate_limit.shape != (boxed.shape[0], 1) or bool(
                (effective_rate_limit < 0.0).any().item()
            ):
                raise ValueError("rate_limit must be non-negative scalar or [batch]")
        else:
            effective_rate_limit = boxed.new_full((boxed.shape[0], 1), float(effective_rate_limit))
        if previous_action is None:
            action = boxed
            rate_limited = torch.zeros(desired_wrench.shape[0], device=desired_wrench.device,
                                       dtype=desired_wrench.dtype)
        else:
            delta = effective_rate_limit.clamp_min(1.0e-8) * float(dt)
            previous_safe = previous_action.clamp(-1.0, 1.0)
            rate_action = previous_safe + delta * torch.tanh((boxed - previous_safe) / delta)
            has_rate = effective_rate_limit > 0.0
            action = torch.where(has_rate, rate_action, boxed)
            rate_limited = (action - boxed).abs().mean(dim=-1)
        # The smooth allocator is followed by a hard deployable-domain guard;
        # this also handles malformed replay previous_action values.
        action = action.clamp(-1.0, 1.0)
        residual = torch.bmm(mixer, action.unsqueeze(-1)).squeeze(-1) - desired_wrench
        cond = torch.linalg.cond(mixer).to(desired_wrench)
        diagnostics = AllocatorDiagnostics(
            condition_number=cond,
            wrench_residual=residual.norm(dim=-1),
            saturation=action.abs().mean(dim=-1),
            rate_limited=rate_limited,
            lower_headroom=lower_headroom.min(dim=-1).values,
            upper_headroom=upper_headroom.min(dim=-1).values,
            headroom_violation=headroom_violation,
            minimum_headroom=directional_headroom.min(dim=-1).values,
            trust_limited=trust_limited,
        )
        return action, diagnostics


class StructuredRecurrentPolicy(nn.Module):
    """Equilibrium-trimmed recurrent policy with an explicit slow state."""

    def __init__(self, config: StructuredPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or StructuredPolicyConfig()
        c = self.config
        if c.hidden_dim < 1 or c.identifier_dim < 1:
            raise ValueError("hidden_dim and identifier_dim must be positive")
        mode_sizes = {
            "legacy": (0, 0),
            "fixed_multi_tau_v1": (15, 1),
            "fixed_multi_tau_v2": (35, 2),
        }
        if c.motor_observer_mode not in mode_sizes:
            raise ValueError(
                "motor_observer_mode must be legacy, fixed_multi_tau_v1, or fixed_multi_tau_v2"
            )
        expected_size, expected_version = mode_sizes[c.motor_observer_mode]
        if (c.motor_observer_bank_size, c.motor_tau_grid_version) != (
            expected_size, expected_version
        ):
            raise ValueError(
                "motor_observer_bank_size and motor_tau_grid_version must match "
                f"mode {c.motor_observer_mode!r}: expected "
                f"({expected_size}, {expected_version})"
            )
        if c.disturbance_observer_tau <= 0.0:
            raise ValueError("disturbance_observer_tau must be positive")
        if c.allocator_solver not in ("smooth_dls", "box_qp"):
            raise ValueError("allocator_solver must be smooth_dls or box_qp")
        if c.slow_cadence < 1:
            raise ValueError("slow_cadence must be positive")
        if (c.identification_publish_start < 1
                or c.identification_publish_start % c.slow_cadence != 0):
            raise ValueError(
                "identification_publish_start must be a positive multiple of slow_cadence"
            )
        if c.identification_publish_start < c.burn_in_steps + c.contextual_blend_steps:
            raise ValueError(
                "identification_publish_start must cover burn-in and contextual transition"
            )
        if not 0.0 <= c.burn_in_probe_amplitude <= c.burn_in_action_cap:
            raise ValueError("burn-in probe must fit inside the burn-in action cap")
        self.encoder = nn.Sequential(nn.Linear(OBSERVATION_DIM + c.identifier_dim + ACTION_DIM, c.hidden_dim), nn.Tanh())
        self.gru = nn.GRUCell(c.hidden_dim, c.hidden_dim)
        self.identifier = SlowIdentifier(input_dim=IDENTIFICATION_INPUT_DIM,
                                         identifier_dim=c.identifier_dim, leak=c.identifier_leak,
                                         limit=c.identifier_limit)
        self.motor_observer = MotorObserver(c.observer_tau)
        self.motor_observer_bank = (
            MultiTauMotorObserverBank(version=c.motor_tau_grid_version)
            if c.motor_observer_bank_size else None
        )
        if c.motor_observer_bank_size:
            self.bank_adapter = nn.Linear(
                8 * c.motor_observer_bank_size, IDENTIFICATION_INPUT_DIM, bias=False
            )
            nn.init.zeros_(self.bank_adapter.weight)
        self.integrator = AntiWindupIntegral(c.integral_limit, c.integral_leak, c.anti_windup_gain)
        # Persistent external acceleration is updated by the explicit physical
        # residual observer below.  Trim and thrust direction are then solved
        # analytically; there is no learned slow motor-action/direction head in
        # which feedback can hide.
        # The mean is a tanh coordinate in normalized log(capability) space.  The
        # scale head is converted to a strictly positive log-space sigma.
        self.capability_head = nn.Linear(c.identifier_dim, 6)
        self.capability_log_scale_head = nn.Linear(c.identifier_dim, 6)
        # Contextual gain is capability-only by design.  Identifier and sigma
        # remain available to the confidence gate, but cannot directly change K.
        self.contextual_gain_head = nn.Linear(6, ACTION_DIM * ERROR_DIM)
        self.register_buffer(
            "K_ref", reference_fast_gain(device=torch.device("cpu"), dtype=torch.float32)
        )
        # Conformal calibration is intentionally explicit.  Until an external
        # calibration pass installs q, the fixed beta is only a smoke-test
        # uncertainty scale and is not a statistical coverage claim.
        self.register_buffer("capability_conformal_q", torch.zeros(6))
        self.register_buffer("capability_calibration_n", torch.zeros((), dtype=torch.long))
        self.register_buffer("capability_calibration_valid", torch.zeros((), dtype=torch.bool))
        # Retain the explicit interface for callers/checkpoint migration.  K_ref
        # itself is the deployable, permanently frozen base gain below.
        self.fast_feedback = FastFeedbackInterface(
            gain=self.K_ref, verified=True, share_gain=True
        )
        self.residual_head = nn.Linear(c.hidden_dim + ERROR_DIM, ACTION_DIM)
        self.allocator = (
            ActiveSetBoxQPAllocator(c.allocator_damping)
            if c.allocator_solver == "box_qp"
            else DampedConstrainedAllocator(
                c.allocator_damping, c.allocator_temperature,
                c.allocator_rate_limit,
            )
        )
        self.register_buffer("error_scales", torch.tensor((.1, .1, .1, .1, .1, .1,
                                                             .1, .1, .5, .5, .5,
                                                             .1, .1, .1, .1)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.capability_head.weight)
        log_lo = self._capability_log_lo(self.capability_head.bias)
        log_hi = self._capability_log_hi(self.capability_head.bias)
        center = (log_lo + log_hi) / 2.0
        half = (log_hi - log_lo) / 2.0
        default = self.capability_head.bias.new_tensor(CAPABILITY_DEFAULT).log()
        unit = ((default - center) / half).clamp(-0.999, 0.999)
        nn.init.constant_(self.capability_head.bias, 0.0)
        with torch.no_grad():
            self.capability_head.bias.copy_(torch.atanh(unit))
        nn.init.zeros_(self.capability_log_scale_head.weight)
        initial_raw_sigma = math.log(math.expm1(max(
            self.config.capability_initial_sigma - self.config.capability_sigma_min, 1.0e-4
        )))
        nn.init.constant_(self.capability_log_scale_head.bias, initial_raw_sigma)
        nn.init.zeros_(self.contextual_gain_head.weight)
        nn.init.zeros_(self.contextual_gain_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        if not self.config.residual_trainable:
            for parameter in self.residual_head.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _capability_log_lo(reference: torch.Tensor) -> torch.Tensor:
        return reference.new_tensor(CAPABILITY_LO).log()

    @staticmethod
    def _capability_log_hi(reference: torch.Tensor) -> torch.Tensor:
        return reference.new_tensor(CAPABILITY_HI).log()

    def _capability_statistics(
        self, identifier: torch.Tensor, *, initial: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return physical mean, log mean, log-sigma-z, and effectiveness UCB.

        The posterior scale lives in normalized log capability coordinates
        ``z=(log(c)-center)/half_range``.  A calibrated conformal score thus
        has one unambiguous meaning across thrust, angular authority, and time
        constants with very different physical units.
        """
        raw_mean = self.capability_head(identifier)
        log_lo = self._capability_log_lo(raw_mean)
        log_hi = self._capability_log_hi(raw_mean)
        center = (log_lo + log_hi) / 2.0
        half = (log_hi - log_lo) / 2.0
        mu_norm = torch.tanh(raw_mean)
        log_mean = center + half * mu_norm
        mean = log_mean.exp().clamp(log_mean.new_tensor(CAPABILITY_LO),
                                    log_mean.new_tensor(CAPABILITY_HI))
        scale_input = (
            identifier.detach()
            if self.config.capability_scale_detach_identifier else identifier
        )
        raw_scale = self.capability_log_scale_head(scale_input)
        sigma = (
            self.config.capability_sigma_min + F.softplus(raw_scale)
        ).clamp_max(2.0)
        log_sigma = sigma.clamp_min(self.config.capability_sigma_min).log()
        if bool(self.capability_calibration_valid.item()):
            multiplier = self.capability_conformal_q.to(log_mean)
        else:
            # Fixed beta is deliberately smoke-only.  A promotion checkpoint
            # must install held-out conformal calibration metadata.
            multiplier = log_mean.new_full(
                (6,), float(self.config.capability_ucb_beta)
            )
        upper_z = (mu_norm + multiplier * sigma).clamp(-1.0, 1.0)
        log_ucb = center + half * upper_z
        ucb = log_ucb.exp().clamp(log_ucb.new_tensor(CAPABILITY_LO),
                                  log_ucb.new_tensor(CAPABILITY_HI))
        if initial:
            maximum = ucb.new_tensor(CAPABILITY_HI).expand_as(ucb)
            ucb = torch.cat((maximum[:, :3], mean[:, 3:]), dim=-1)
        return mean, log_mean, log_sigma, ucb

    def _project_contextual_gain(
        self, delta_gain: torch.Tensor, mixer: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project a wrench gain using an allocator-induced spectral bound.

        The exact saturated allocator Jacobian is state dependent.  Its
        unsaturated local approximation is the damped inverse mixer, which is
        used here as an explicit differentiable projection surrogate.
        """
        gram = mixer.transpose(1, 2) @ mixer
        eye = torch.eye(ACTION_DIM, device=mixer.device, dtype=mixer.dtype).expand_as(gram)
        inverse = torch.linalg.solve(
            gram + self.config.allocator_damping * eye, mixer.transpose(1, 2)
        )
        induced = torch.bmm(inverse, delta_gain)
        pre_norm = torch.linalg.matrix_norm(induced, ord=2, dim=(-2, -1))
        reference = self.K_ref.to(device=mixer.device, dtype=mixer.dtype).expand(
            mixer.shape[0], -1, -1
        )
        reference_induced = torch.bmm(inverse, reference)
        reference_norm = torch.linalg.matrix_norm(reference_induced, ord=2, dim=(-2, -1))
        budget = self.config.contextual_gain_rho * torch.maximum(
            reference_norm,
            reference_norm.new_full(reference_norm.shape, self.config.contextual_gain_jfloor),
        )
        scale = (budget / pre_norm.clamp_min(1.0e-8)).clamp_max(1.0)
        projected = delta_gain * scale[:, None, None]
        post_norm = torch.linalg.matrix_norm(
            torch.bmm(inverse, projected), ord=2, dim=(-2, -1)
        )
        return projected, pre_norm, post_norm

    @torch.no_grad()
    def install_capability_conformal_q(self, q: torch.Tensor, sample_count: int) -> None:
        if tuple(q.shape) != (6,) or not bool(torch.isfinite(q).all().item()) or bool((q < 0).any().item()):
            raise ValueError("q must be a finite non-negative tensor with shape [6]")
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        self.capability_conformal_q.copy_(q.to(self.capability_conformal_q))
        self.capability_calibration_n.fill_(int(sample_count))
        self.capability_calibration_valid.fill_(True)

    @torch.no_grad()
    def invalidate_capability_calibration(self) -> None:
        """Invalidate closed-loop calibration after any policy-weight change."""

        self.capability_conformal_q.zero_()
        self.capability_calibration_n.zero_()
        self.capability_calibration_valid.fill_(False)

    def initial_state(self, observation: torch.Tensor) -> StructuredPolicyState:
        if observation.ndim != 2 or observation.shape[-1] != OBSERVATION_DIM:
            raise ValueError("observation must have shape [batch,25]")
        batch, device, dtype = observation.shape[0], observation.device, observation.dtype
        capability = observation.new_tensor(CAPABILITY_DEFAULT).expand(batch, 6).clone()
        log_mean = capability.log()
        log_sigma = observation.new_full((batch, 6), math.log(self.config.capability_initial_sigma))
        maximum = observation.new_tensor(CAPABILITY_HI).expand(batch, 6).clone()
        # During the initial observer/identifier burn-in, effectiveness is
        # intentionally treated as maximally uncertain.  Motor time constants
        # still use the posterior mean because they enter only the observer.
        contextual_gain = torch.zeros(batch, ACTION_DIM, ERROR_DIM, device=device, dtype=dtype)
        contextual_gain_target = contextual_gain.clone()
        contextual_blend = torch.zeros(batch, 1, device=device, dtype=dtype)
        boot_progress = torch.zeros(batch, 1, device=device, dtype=dtype)
        return StructuredPolicyState(
            hidden=torch.zeros(batch, self.config.hidden_dim, device=device, dtype=dtype),
            identifier=torch.zeros(batch, self.config.identifier_dim, device=device, dtype=dtype),
            motor_estimate=observation[:, 21:25].clone(), integral=observation[:, 18:21].clone(),
            motor_bank=(
                observation[:, 21:25].clone()[:, None, :].expand(
                    batch, self.config.motor_observer_bank_size, ACTION_DIM
                ).clone()
                if self.config.motor_observer_bank_size else None
            ),
            slow_trim=torch.zeros(batch, ACTION_DIM, device=device, dtype=dtype),
            slow_body_z=torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype).expand(batch, 3).clone(),
            capability=capability, slow_counter=0,
            prev_velocity=observation[:, 3:6].clone(), prev_omega=observation[:, 15:18].clone(),
            context_sum=torch.zeros(batch, 24, device=device, dtype=dtype),
            capability_log_mean=log_mean,
            capability_log_scale=log_sigma,
            capability_ucb=torch.cat((maximum[:, :3], capability[:, 3:]), dim=-1),
            contextual_gain=contextual_gain,
            contextual_gain_target=contextual_gain_target,
            contextual_blend=contextual_blend,
            boot_progress=boot_progress,
            capability_ucb_target=torch.cat((maximum[:, :3], capability[:, 3:]), dim=-1),
            identification_failed=torch.zeros(batch, dtype=torch.bool, device=device),
            disturbance_accel=torch.zeros(batch, 3, device=device, dtype=dtype),
            disturbance_residual_sum=torch.zeros(batch, 3, device=device, dtype=dtype),
            disturbance_thrust_sum=torch.zeros(batch, 3, device=device, dtype=dtype),
            previous_executed_action=observation[:, 21:25].clone(),
            disturbance_response_count=torch.zeros(batch, 1, device=device, dtype=dtype),
            excitation_history=torch.zeros(
                batch, EXCITATION_HISTORY_LEN, ACTION_DIM, device=device, dtype=dtype
            ),
            previous_motor_estimate=observation[:, 21:25].clone(),
            probe_residual=observation.new_zeros((batch, 1)),
            probe_omega_reference=observation.new_zeros((batch, 1)),
            identification_information=observation.new_zeros((batch, INFORMATION_DIM)),
            capability_authorized=torch.zeros(batch, 6, dtype=torch.bool, device=device),
            identification_actions=observation.new_zeros((batch, PUBLISH_START, 4)),
            identification_initial_motor=observation[:, 21:25].clone(),
            disturbance_history=observation.new_zeros((batch, self.config.slow_cadence, 11)),
            probe_aborted=torch.zeros(batch, dtype=torch.bool, device=device),
        )

    def forward_with_aux(self, observation: torch.Tensor,
                         state: Optional[StructuredPolicyState] = None,
                         dt: Optional[float] = None,
                         applied_action: Optional[torch.Tensor] = None,
                         applied_probe_residual: Optional[torch.Tensor] = None,
                         applied_probe_aborted: Optional[torch.Tensor] = None) -> StructuredPolicyOutput:
        if observation.ndim != 2 or observation.shape[-1] != OBSERVATION_DIM:
            raise ValueError("observation must have shape [batch,25]")
        if dt is None:
            dt = self.config.dt
        if state is None:
            state = self.initial_state(observation)
        if applied_action is not None:
            if applied_action.shape != (observation.shape[0], ACTION_DIM):
                raise ValueError("applied_action must have shape [batch,4]")
            applied_action = applied_action.to(device=observation.device, dtype=observation.dtype)
            if not bool(torch.isfinite(applied_action).all()) or bool((applied_action.abs() > 1).any()):
                raise ValueError("applied_action must be finite and inside [-1,1]")
        identification_failed = state.identification_failed
        if identification_failed is None:
            identification_failed = torch.zeros(
                observation.shape[0], dtype=torch.bool, device=observation.device
            )
        if self.config.slow_cadence < 1:
            raise ValueError("slow_cadence must be positive")
        previous_action = observation[:, 21:25]
        boot_progress = state.boot_progress
        if boot_progress is None:
            boot_progress = observation.new_zeros((observation.shape[0], 1))
        burn_in_mask = (boot_progress < float(self.config.burn_in_steps)).to(observation.dtype)
        transition_mask = (
            boot_progress < float(self.config.identification_publish_start)
        ).to(observation.dtype)
        # The state already contains the previous motor estimate.  Updating it
        # toward ``previous_action`` here and toward the current action again
        # would double-advance the observer in one policy call.
        motor_estimate = state.motor_estimate
        capability_mean = state.capability
        if capability_mean is None:
            capability_mean = observation.new_tensor(CAPABILITY_DEFAULT).expand(
                observation.shape[0], 6
            )
        tau_rise = capability_mean[:, 4:5].expand_as(motor_estimate)
        tau_fall = capability_mean[:, 5:6].expand_as(motor_estimate)
        prev_velocity = state.prev_velocity if state.prev_velocity is not None else observation[:, 3:6]
        prev_omega = state.prev_omega if state.prev_omega is not None else observation[:, 15:18]
        old_context = state.context_sum if state.context_sum is not None else torch.zeros(
            observation.shape[0], 24,
            device=observation.device, dtype=observation.dtype)
        rotation = observation[:, 6:15].reshape(-1, 3, 3)
        body_z_world = rotation[:, :, 2]
        omega_current = observation[:, 15:18]
        # env_l2f advances R with the midpoint angular velocity, so reconstruct
        # the exact orientation that produced the measured velocity increment.
        rotation_previous = torch.bmm(
            rotation,
            _so3_exp(-0.5 * float(dt) * (prev_omega + omega_current)),
        )
        previous_body_z_world = rotation_previous[:, :, 2]
        linear_accel_world = (observation[:, 3:6] - prev_velocity) / float(dt)
        gravity_world = observation.new_tensor((0.0, 0.0, 9.80665)).expand_as(
            linear_accel_world
        )
        specific_thrust_world = linear_accel_world + gravity_world
        specific_thrust_body = torch.bmm(
            rotation_previous.transpose(1, 2), specific_thrust_world.unsqueeze(-1)
        ).squeeze(-1)
        angular_accel_raw = (observation[:, 15:18] - prev_omega) / float(dt)
        motor_mean = motor_estimate.mean(dim=-1, keepdim=True)
        motor_rms = (
            (motor_estimate - motor_mean).square().mean(dim=-1, keepdim=True)
            + 1.0e-12
        ).sqrt()
        response_mask = (boot_progress > 0.0).to(observation.dtype)
        force_feature, angular_feature, collective_feature = normalize_response(
            specific_thrust_body, angular_accel_raw, response_mask=response_mask
        )
        excitation_history = state.excitation_history
        if excitation_history is None:
            excitation_history = observation.new_zeros(
                observation.shape[0], EXCITATION_HISTORY_LEN, ACTION_DIM
            )
        previous_motor_estimate = state.previous_motor_estimate
        if previous_motor_estimate is None:
            previous_motor_estimate = motor_estimate
        motor_delta = motor_estimate - previous_motor_estimate
        # Exactly 24 causal features: three lagged force products (9), three
        # lagged roll/pitch/yaw response products (9), rise/fall products (2),
        # and four aligned excitation energies.  No absolute fast state enters
        # the identifier.
        context = _lagged_cross_response_context(
            excitation_history, force_feature, angular_feature, motor_delta,
            response_mask,
        )
        ledger = state.identification_information
        if ledger is None:
            ledger = observation.new_zeros((observation.shape[0], INFORMATION_DIM))
        ledger = advance_information(ledger, excitation_history[:, 0], force_feature,
                                     angular_feature, prev_omega,
                                     response_mask * float(state.slow_counter <= self.config.identification_publish_start))
        supported_axes = capability_support(ledger.detach())
        authorized_axes = state.capability_authorized
        if authorized_axes is None:
            authorized_axes = torch.zeros_like(supported_axes)
        legacy_identification_context = context
        bank_identification_features = None
        motor_bank = state.motor_bank
        if self.config.motor_observer_bank_size:
            if motor_bank is None:
                motor_bank = motor_estimate[:, None, :].expand(
                    observation.shape[0], self.config.motor_observer_bank_size, ACTION_DIM
                ).clone()
            # Convert each candidate motor state into the four physical mixer
            # modes (collective, roll, pitch, yaw).  Pair those modal values
            # with the measured force/angular response, and include x^2 so the
            # bank can identify signed and quadratic authority while the
            # identifier input remains the fixed 24D interface.
            observed_response = sol_response(collective_feature, angular_feature)
            # A candidate motor bank exists at call0, but no physical response
            # exists yet.  Do not let the unpaired x^2 terms move the
            # identifier before the first measured transition response.
            modal_features = bank_modal_features(
                motor_bank, observed_response, response_mask=response_mask
            )
            bank_identification_features = modal_features.reshape(
                observation.shape[0], -1
            )
            modal_context = torch.tanh(
                self.bank_adapter(bank_identification_features)
            )
            context = context + modal_context
        # Keep the measured cadence window: clipping negative rotor thrust
        # makes force piecewise affine in TW, so two affine sums are not
        # sufficient statistics when a new estimate crosses a clipping knee.
        disturbance_history = state.disturbance_history
        if disturbance_history is None:
            disturbance_history = observation.new_zeros((observation.shape[0], self.config.slow_cadence, 11))
        response_row = torch.cat((previous_body_z_world, specific_thrust_world,
                                  motor_estimate, response_mask), -1)
        disturbance_history = torch.cat((disturbance_history[:, 1:], response_row[:, None]), 1)
        force_now = (1.0 + (capability_mean[:, 0:1] - 1.0) * motor_estimate).clamp_min(0).mean(-1, keepdim=True)
        disturbance_residual = specific_thrust_world - previous_body_z_world * (9.80665 * force_now)
        # Retain zero-valued fields only for explicit legacy codec readers.
        disturbance_residual_sum = torch.zeros_like(disturbance_residual)
        disturbance_thrust_sum = torch.zeros_like(disturbance_residual)
        disturbance_response_count = disturbance_history[:, :, 10].sum(1, keepdim=True)
        # Preserve the ordered excitation/response history.  The recurrent
        # identifier advances each fast step, while capability/trim remain
        # sample-and-held at the slow cadence.  ``context_sum`` is retained as
        # an all-zero compatibility field in the boundary codec.
        candidate_identifier = self.identifier(context, state.identifier)
        # Keep the recurrent identifier exactly at its initial value on call0;
        # a zero-context GRUCell can otherwise move through its learned bias.
        ident = torch.where(
            response_mask.bool(), candidate_identifier, state.identifier
        )
        context_sum = torch.zeros_like(old_context)
        # ``boot_progress`` is the call index: at call ``t`` it records that
        # exactly ``t`` physics transitions have completed.  The response
        # available to this call is therefore the one from transition
        # ``t-1``.  Publish only after at least one response and on the
        # cadence boundary itself.  In particular, call 0 has no response and
        # must not publish the initial fallback capability.
        response_available = bool((boot_progress > 0.0).all().item())
        publication_available = (
            boot_progress >= float(self.config.identification_publish_start)
        )
        update_slow = (
            response_available
            and bool(publication_available.all().item())
            and state.slow_counter % self.config.slow_cadence == 0
        )
        if update_slow:
            capability, log_mean, log_sigma, capability_ucb = self._capability_statistics(
                ident
            )
            authorized_axes = supported_axes
            # Retain candidate log statistics for supervised fitting.  Only
            # authorized physical means/authority reach the control branches.
            prior = capability.new_tensor(CAPABILITY_DEFAULT).expand_as(capability)
            capability = torch.where(authorized_axes, capability, prior)
            capability_ucb = torch.where(authorized_axes & bool(self.capability_calibration_valid.item()), capability_ucb,
                capability_ucb.new_tensor(CAPABILITY_HI).expand_as(capability_ucb))
            previous_disturbance = state.disturbance_accel
            if previous_disturbance is None:
                previous_disturbance = observation.new_zeros((observation.shape[0], 3))
            observer_gain = 1.0 - math.exp(
                -float(self.config.slow_cadence) * float(dt)
                / float(self.config.disturbance_observer_tau)
            )
            # Reconstruct the startup motor trajectory once under the newly
            # inferred lag, before using its last cadence window for force.
            if state.slow_counter == self.config.identification_publish_start:
                if state.identification_actions is not None and state.identification_initial_motor is not None:
                    motors = replay_motor_history(self.motor_observer,
                        state.identification_initial_motor,
                        state.identification_actions[:, -min(state.slow_counter, PUBLISH_START):],
                        capability[:, 4:5], capability[:, 5:6], float(dt), return_sequence=True)
                    motor_estimate = motors[:, -1]
                    count = min(motors.shape[1], self.config.slow_cadence)
                    corrected = torch.cat((disturbance_history[:, -count:, :6],
                        motors[:, -count:], disturbance_history[:, -count:, 10:]), -1)
                    disturbance_history = torch.cat((disturbance_history[:, :-count], corrected), 1)
            rotor_force = (1 + (capability[:, None, 0:1] - 1) * disturbance_history[:, :, 6:10]).clamp_min(0)
            force_world = disturbance_history[:, :, :3] * (9.80665 * rotor_force.mean(-1, keepdim=True))
            valid = disturbance_history[:, :, 10:]
            mean_residual = ((disturbance_history[:, :, 3:6] - force_world) * valid).sum(1) / valid.sum(1).clamp_min(1)
            disturbance_accel = (
                (1.0 - observer_gain) * previous_disturbance
                + observer_gain * mean_residual
            )
            disturbance_limit = 9.80665 * torch.minimum(
                disturbance_accel.new_full((observation.shape[0],), 0.75),
                0.8 * (capability[:, 0] - 1.0).clamp_min(0.0),
            )
            disturbance_norm = torch.linalg.vector_norm(
                disturbance_accel, dim=-1
            ).clamp_min(1.0e-8)
            disturbance_scale = (
                disturbance_limit / disturbance_norm
            ).clamp_max(1.0)
            disturbance_accel = disturbance_accel * disturbance_scale[:, None]
            disturbance_residual_sum = torch.zeros_like(disturbance_residual_sum)
            disturbance_thrust_sum = torch.zeros_like(disturbance_thrust_sum)
            disturbance_response_count = torch.zeros_like(disturbance_response_count)
            body_z, trim, equilibrium_feasible = equilibrium_from_capability_disturbance(
                capability, disturbance_accel
            )
            gain_context = ((log_mean - log_mean.new_tensor(CAPABILITY_LO).log())
                            / (log_mean.new_tensor(CAPABILITY_HI).log()
                               - log_mean.new_tensor(CAPABILITY_LO).log())) * 2.0 - 1.0
            contextual_candidate = torch.tanh(self.contextual_gain_head(gain_context)).reshape(
                -1, ACTION_DIM, ERROR_DIM
            ) * self.config.contextual_gain_scale
            capability_ucb_target = capability_ucb
            contextual_target = None
        else:
            trim, body_z, capability = state.slow_trim, state.slow_body_z, capability_mean
            disturbance_accel = state.disturbance_accel
            if disturbance_accel is None:
                disturbance_accel = observation.new_zeros((observation.shape[0], 3))
            analytic_body_z, analytic_trim, equilibrium_feasible = (
                equilibrium_from_capability_disturbance(
                    capability, disturbance_accel
                )
            )
            # Held values must equal the analytic solve.  Recomputing also
            # keeps feasibility truthful for all 24 inter-cadence steps.
            trim, body_z = analytic_trim, analytic_body_z
            log_mean = state.capability_log_mean
            log_sigma = state.capability_log_scale
            capability_ucb = state.capability_ucb
            capability_ucb_target = state.capability_ucb_target
            contextual_candidate = state.contextual_gain
            contextual_target = state.contextual_gain_target
            if log_mean is None:
                log_mean = capability.clamp_min(1.0e-6).log()
            if log_sigma is None:
                log_sigma = observation.new_full((observation.shape[0], 6),
                                                 math.log(self.config.capability_initial_sigma))
            if capability_ucb is None:
                capability_ucb = observation.new_tensor(CAPABILITY_HI).expand_as(capability).clone()
            if capability_ucb_target is None:
                capability_ucb_target = capability_ucb
            if contextual_candidate is None:
                contextual_candidate = torch.zeros(observation.shape[0], ACTION_DIM, ERROR_DIM,
                    device=observation.device, dtype=observation.dtype)
            if contextual_target is None:
                contextual_target = contextual_candidate

        tau_rise = capability[:, 4:5].expand_as(motor_estimate)
        tau_fall = capability[:, 5:6].expand_as(motor_estimate)
        recurrent_input = torch.cat((observation, ident, motor_estimate), -1)
        latent = self.gru(self.encoder(recurrent_input), state.hidden)

        if contextual_candidate is None:
            contextual_candidate = torch.zeros(
                observation.shape[0], ACTION_DIM, ERROR_DIM,
                device=observation.device, dtype=observation.dtype
            )
        if contextual_target is None:
            contextual_target = state.contextual_gain_target
        if contextual_target is None:
            contextual_target = contextual_candidate
        contextual_gain = state.contextual_gain
        if contextual_gain is None:
            contextual_gain = torch.zeros_like(contextual_candidate)

        # Both capability and contextual gain transitions are first-order
        # interpolations.  This keeps the t25->t50 boundary rate limited even
        # when a new slow estimate changes sharply.
        active_ucb = state.capability_ucb
        if active_ucb is None:
            active_ucb = capability_ucb
        ucb_rate = 1.0 / max(self.config.contextual_blend_steps, 1)
        active_ucb = torch.exp(
            active_ucb.clamp_min(1.0e-6).log()
            + ucb_rate * (
                capability_ucb_target.clamp_min(1.0e-6).log()
                - active_ucb.clamp_min(1.0e-6).log()
            )
        )
        active_ucb = active_ucb.clamp(
            active_ucb.new_tensor(CAPABILITY_LO), active_ucb.new_tensor(CAPABILITY_HI)
        )
        # Effective authority is the posterior mean for observer channels and
        # the one-sided UCB only for instantaneous effectiveness channels.  The
        # first 25 actions deliberately retain the maximal-UCB safety envelope,
        # including the cadence boundary where the first slow estimate arrives.
        initial_effectiveness = observation.new_tensor(CAPABILITY_HI).expand(
            observation.shape[0], 6
        )[:, :3]
        # Do not consume a capability estimate before the first registered
        # publication.  In particular call25 still uses the maximum-authority
        # conservative prior even though the recurrent state has evolved.
        effectiveness = torch.where(
            publication_available.bool(), active_ucb[:, :3], initial_effectiveness
        )
        allocation_capability = torch.cat((effectiveness, capability[:, 3:]), dim=-1)
        mixer = effective_wrench_mixer(allocation_capability)
        if update_slow:
            contextual_target, _, _ = self._project_contextual_gain(
                contextual_candidate, mixer
            )
        contextual_blend = state.contextual_blend
        if contextual_blend is None:
            contextual_blend = torch.zeros(
                observation.shape[0], 1, device=observation.device, dtype=observation.dtype
            )
        contextual_blend = (
            contextual_blend
            + publication_available.to(contextual_blend.dtype)
            * (1.0 / max(self.config.contextual_blend_steps, 1))
        ).clamp_max(1.0)
        gain_rate = 1.0 / max(self.config.contextual_blend_steps, 1)
        contextual_gain = contextual_gain + gain_rate * (contextual_target - contextual_gain)
        projected_gain, pre_induced_norm, induced_norm = self._project_contextual_gain(
            contextual_gain, mixer
        )
        equilibrium_trim = trim
        equilibrium_wrench = torch.bmm(mixer, equilibrium_trim.unsqueeze(-1)).squeeze(-1)
        rotation = observation[:, 6:15].reshape(-1, 3, 3)
        desired_body_z = torch.bmm(rotation.transpose(1, 2), body_z.unsqueeze(-1)).squeeze(-1)
        tilt_error = desired_body_z[:, :2]
        error = torch.cat((observation[:, 0:6], tilt_error, observation[:, 15:18],
                           motor_estimate - equilibrium_trim), -1)
        features = torch.tanh(error / self.error_scales.to(error))
        reference_gain = self.K_ref.to(device=features.device, dtype=features.dtype)
        log_lo = log_mean.new_tensor(CAPABILITY_LO).log()
        log_hi = log_mean.new_tensor(CAPABILITY_HI).log()
        center = 0.5 * (log_lo + log_hi)
        half = 0.5 * (log_hi - log_lo)
        mu_z = (log_mean - center) / half
        sigma_z = log_sigma.exp()
        uncertainty_multiplier = (
            self.capability_conformal_q.to(log_mean)
            if bool(self.capability_calibration_valid.item())
            else log_mean.new_full((6,), float(self.config.capability_ucb_beta))
        )
        lower_z = (mu_z - uncertainty_multiplier * sigma_z).clamp(-1.0, 1.0)
        upper_z = (mu_z + uncertainty_multiplier * sigma_z).clamp(-1.0, 1.0)
        effectiveness_log_width = half[:3] * (upper_z[:, :3] - lower_z[:, :3])
        allowed_log_width = effectiveness_log_width.new_tensor(
            (math.log(1.5), math.log(2.0), math.log(2.0))
        )
        capability_confidence = (
            1.0
            - torch.relu(effectiveness_log_width - allowed_log_width)
            / allowed_log_width
        ).clamp(0.0, 1.0).amin(dim=-1, keepdim=True)
        # The t50 diagnostic is evaluated at call index 50, after the first
        # 50 transitions/responses (u0..u49) have been consumed.
        t50_reached = boot_progress >= float(
            self.config.identification_publish_start
        )
        width_ok = (effectiveness_log_width <= allowed_log_width).all(dim=-1)
        identification_failed = identification_failed | (t50_reached.squeeze(-1) & ~width_ok)
        safety_ok = ~identification_failed
        if state.probe_aborted is not None:
            safety_ok = safety_ok & ~state.probe_aborted
        authorized_axes = authorized_axes & safety_ok[:, None]
        wrench_authorized = torch.stack((authorized_axes[:, (0, 4, 5)].all(-1),
            authorized_axes[:, 1], authorized_axes[:, 1], authorized_axes[:, (2, 3)].all(-1)), -1)
        wrench_authorized = wrench_authorized & bool(self.capability_calibration_valid.item())
        gain_weight = contextual_blend * capability_confidence
        effective_gain = reference_gain.unsqueeze(0) + gain_weight[:, :, None] * projected_gain * wrench_authorized[:, :, None]
        fast = torch.einsum("bae,be->ba", effective_gain, features)
        # Multiplication by ||e||² is deliberate: residual is O(||e||²).
        residual = torch.tanh(self.residual_head(torch.cat((latent, features), -1))) * (
            features.square().sum(-1, keepdim=True) / (1.0 + features.square().sum(-1, keepdim=True))
        ) * self.config.residual_scale
        residual = residual * wrench_authorized.to(residual)
        correction_wrench = fast + residual
        desired_wrench = equilibrium_wrench + correction_wrench
        action_cap = torch.where(
            transition_mask > 0.0,
            observation.new_full((observation.shape[0], 1), self.config.burn_in_action_cap),
            observation.new_full((observation.shape[0], 1), 1.0e6),
        )
        effective_rate = torch.where(
            transition_mask > 0.0,
            observation.new_full((observation.shape[0], 1), self.config.burn_in_rate_limit),
            observation.new_full((observation.shape[0], 1), self.config.allocator_rate_limit),
        )
        action, allocation = self.allocator(
            desired_wrench, previous_action=previous_action, dt=dt,
            mixer=mixer, trim=equilibrium_trim,
            action_delta_cap=action_cap, rate_limit=effective_rate,
        )
        # Preserve the exact collective direction AFTER allocation, intersecting
        # every actuator/rate/trust bound before applying a scalar residual.
        probe_state = ProbeState(
            state.probe_residual if state.probe_residual is not None else observation.new_zeros((observation.shape[0], 1)),
            state.probe_aborted if state.probe_aborted is not None else torch.zeros(observation.shape[0], dtype=torch.bool, device=observation.device), state.probe_omega_reference)
        probe_lower = torch.maximum(torch.full_like(action, -1.0), equilibrium_trim - action_cap)
        probe_upper = torch.minimum(torch.full_like(action, 1.0), equilibrium_trim + action_cap)
        probe_lower = torch.where(effective_rate > 0, torch.maximum(probe_lower, previous_action - effective_rate * float(dt)), probe_lower)
        probe_upper = torch.where(effective_rate > 0, torch.minimum(probe_upper, previous_action + effective_rate * float(dt)), probe_upper)
        base_action = action
        action, next_probe, requested = apply_probe(action, probe_state, boot_progress,
            position=observation[:, :3], velocity=observation[:, 3:6], omega=observation[:, 15:18],
            body_z=body_z_world, amplitude=self.config.burn_in_probe_amplitude,
            lower=probe_lower, upper=probe_upper)
        probe_action = action - base_action
        # An externally selected action cannot inherit the unexecuted probe.
        if applied_action is not None:
            executed_scalar = (torch.zeros_like(next_probe.residual) if applied_probe_residual is None
                        else applied_probe_residual.to(observation))
            aborted = (probe_state.aborted if applied_probe_aborted is None
                       else applied_probe_aborted.to(device=observation.device, dtype=torch.bool))
            if executed_scalar.shape != next_probe.residual.shape or not bool(torch.isfinite(executed_scalar).all()) or bool((executed_scalar.abs() > .005001).any()):
                raise ValueError("executed probe residual must be finite [batch,1] within the registered amplitude")
            if aborted.shape != next_probe.aborted.shape:
                raise ValueError("executed probe abort mask must be [batch]")
            next_probe = ProbeState(executed_scalar, aborted | probe_state.aborted, next_probe.omega_reference)
        executed_action = action if applied_action is None else applied_action
        # Advance the observer with the command the simulator actually
        # receives.  The resulting innovation is stored at history index zero
        # for the next call, where it is paired with that transition's
        # measured response.
        next_motor_estimate = self.motor_observer(
            motor_estimate, executed_action, dt, tau_rise, tau_fall
        )
        next_motor_bank = None
        if self.config.motor_observer_bank_size:
            next_motor_bank = self.motor_observer_bank(
                motor_bank, executed_action, dt
            )
        next_excitation = (
            (executed_action - motor_estimate) / EXCITATION_SCALE
        ).clamp(-5.0, 5.0)
        next_excitation_history = torch.cat(
            (next_excitation.unsqueeze(1), excitation_history[:, :-1]), dim=1
        )
        # Anti-windup follows the command that the simulator actually receives.
        # This matters during DAgger teacher execution or an external shield.
        authority = (1.0 - executed_action.abs().mean(dim=-1)).clamp(
            0.0, 1.0
        ).unsqueeze(-1)
        next_integral = self.integrator(state.integral, observation[:, 0:3], dt, authority)
        action_history = state.identification_actions
        if action_history is None:
            action_history = observation.new_zeros((observation.shape[0], PUBLISH_START, 4))
        if state.slow_counter < self.config.identification_publish_start:
            action_history = torch.cat((action_history[:, 1:], executed_action[:, None]), 1)
        next_state = StructuredPolicyState(
            disturbance_history=disturbance_history,
            identification_actions=action_history,
            identification_initial_motor=state.identification_initial_motor,
            hidden=latent, identifier=ident,
            probe_residual=next_probe.residual, probe_aborted=next_probe.aborted,
            probe_omega_reference=next_probe.omega_reference,
            identification_information=ledger, capability_authorized=authorized_axes,
            motor_estimate=next_motor_estimate, integral=next_integral,
            motor_bank=next_motor_bank,
            slow_trim=trim, slow_body_z=body_z, capability=capability,
            slow_counter=state.slow_counter + 1,
            prev_velocity=observation[:, 3:6], prev_omega=observation[:, 15:18],
            context_sum=context_sum,
            capability_log_mean=log_mean,
            capability_log_scale=log_sigma,
            capability_ucb=active_ucb,
            contextual_gain=contextual_gain,
            contextual_gain_target=contextual_target,
            contextual_blend=contextual_blend,
            boot_progress=(boot_progress + 1.0).clamp_max(float(
                self.config.identification_publish_start
            )),
            capability_ucb_target=capability_ucb_target,
            identification_failed=identification_failed,
            disturbance_accel=disturbance_accel,
            disturbance_residual_sum=disturbance_residual_sum,
            disturbance_thrust_sum=disturbance_thrust_sum,
            # On the next call this is u_(t-1) while the observation carries
            # u_t, so their difference is the latest executed action rate.
            previous_executed_action=previous_action,
            disturbance_response_count=disturbance_response_count,
            excitation_history=next_excitation_history,
            previous_motor_estimate=motor_estimate,
        )
        return StructuredPolicyOutput(action, next_state, {
            "latent": latent, "identifier": ident, "motor_estimate": motor_estimate,
            "motor_bank": motor_bank,
            "trim_action": trim, "equilibrium_wrench": equilibrium_wrench,
            "body_z": body_z, "capability": capability,
            "disturbance_accel": disturbance_accel,
            "disturbance_residual": disturbance_residual,
            "disturbance_limit": disturbance_limit if update_slow else torch.full(
                (observation.shape[0],), float("nan"), device=observation.device,
                dtype=observation.dtype
            ),
            "equilibrium_feasible": equilibrium_feasible,
            "capability_log_mean": log_mean, "capability_log_scale": log_sigma,
            "capability_sigma": log_sigma.exp(), "capability_ucb": active_ucb,
            "capability_z_mean": (
                (log_mean - 0.5 * (
                    log_mean.new_tensor(CAPABILITY_LO).log()
                    + log_mean.new_tensor(CAPABILITY_HI).log()
                ))
                / (0.5 * (
                    log_mean.new_tensor(CAPABILITY_HI).log()
                    - log_mean.new_tensor(CAPABILITY_LO).log()
                ))
            ),
            "capability_z_log_scale": log_sigma,
            "capability_ucb_target": capability_ucb_target,
            "capability_conformal_q": self.capability_conformal_q.to(log_mean),
            "capability_calibration_valid": log_mean.new_full(
                (observation.shape[0],), float(self.capability_calibration_valid.item())
            ),
            "capability_calibration_n": log_mean.new_full(
                (observation.shape[0],), float(self.capability_calibration_n.item())
            ),
            "allocation_capability": allocation_capability,
            "error": error, "feedback_features": features, "fast_feedback": fast,
            "residual": residual, "desired_wrench": desired_wrench,
            "applied_action": executed_action,
            "K_ref": reference_gain, "contextual_gain": contextual_gain,
            "contextual_blend": contextual_blend, "contextual_induced_norm": induced_norm,
            "contextual_gain_weight": gain_weight,
            "capability_confidence": capability_confidence,
            # Read-only parity hooks.  The offline identification oracle uses
            # these tensors to prove that it is evaluating the exact deployed
            # causal feature contract, rather than a separately reimplemented
            # approximation.
            "identification_legacy_context": legacy_identification_context,
            "identification_bank_features": bank_identification_features,
            "identification_context": context,
            "identification_publication_available": publication_available.squeeze(-1),
            "identification_published": torch.full(
                (observation.shape[0],), bool(update_slow),
                device=observation.device, dtype=torch.bool,
            ),
            "identification_failed": identification_failed,
            "t50_identification_failed": identification_failed,
            "effectiveness_log_interval_width": effectiveness_log_width,
            "contextual_induced_norm_pre": pre_induced_norm,
            "burn_in": burn_in_mask.squeeze(-1),
            "identification_prior_active": (~publication_available.bool()).squeeze(-1),
            "identification_probe_action": probe_action,
            "probe_aborted": next_probe.aborted,
            "capability_supported_axes": supported_axes,
            "capability_authorized_axes": authorized_axes,
            "capability_ucb_active": publication_available.to(
                dtype=observation.dtype
            ).squeeze(-1),
            "action_delta_cap": action_cap.squeeze(-1),
            "rate_limit": effective_rate.squeeze(-1),
            "boot_progress": next_state.boot_progress,
            "allocator": allocation,
        })

    def forward(self, observation: torch.Tensor,
                state: Optional[StructuredPolicyState] = None,
                dt: Optional[float] = None,
                applied_action: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, StructuredPolicyState]:
        output = self.forward_with_aux(observation, state, dt, applied_action)
        return output.action, output.next_state


# Short aliases make the intended execution API easy to discover.
StructuredPolicy = StructuredRecurrentPolicy

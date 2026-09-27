"""RAPTOR constant Gaussian force and independent, causal noisy measurements.

One distribution is used by TRAIN and fixed EVAL. Gaussian draws are unbounded;
these settings are not actuator-reserve bounds or learned-stability guarantees.
All randomness is sampled once on CPU. The Actor only receives measured state
and its last known command, never disturbance metadata or physical parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from env_raptor import RaptorState

NOISE_VERSION = "raptor-force-gaussian-observation-delay-v2"
RAPTOR_FORCE_COEFFICIENT = 0.3
CONTROL_DT = 0.01
VELOCITY_HISTORY_STEPS = 3
MEASUREMENT_DIM = 12


@dataclass(frozen=True)
class DisturbanceConfig:
    enabled: bool = True
    position_std: float = 0.001       # metres, per axis per acquired sample
    velocity_std: float = 0.002       # metres/second
    attitude_std: float = 0.001       # rotation-vector radians, NOT matrix entries
    omega_std: float = 0.002          # radians/second
    velocity_delay_min: float = 0.010 # seconds; a constant latency per episode
    velocity_delay_max: float = 0.030

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("disturbance enabled must be boolean")
        for name in ("position_std", "velocity_std", "attitude_std", "omega_std",
                     "velocity_delay_min", "velocity_delay_max"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if not self.velocity_delay_min <= self.velocity_delay_max <= VELOCITY_HISTORY_STEPS * CONTROL_DT:
            raise ValueError("velocity delay must satisfy 0 <= min <= max <= 0.030 seconds")

    @classmethod
    def clean(cls):
        """Zero-noise fixture for deterministic regressions, not another EVAL."""
        return cls(enabled=False)

    @classmethod
    def from_args(cls, args):
        return cls(enabled=not args.disable_disturbances,
                   position_std=args.position_noise_std, velocity_std=args.velocity_noise_std,
                   attitude_std=args.attitude_noise_std, omega_std=args.omega_noise_std,
                   velocity_delay_min=args.velocity_delay_min,
                   velocity_delay_max=args.velocity_delay_max)


def _generator(seed: int, stream: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed((int(seed) ^ stream) & ((1 << 64) - 1))


def raptor_force_std(mass: torch.Tensor, thrust_to_weight: torch.Tensor,
                     unit_uniform: torch.Tensor) -> torch.Tensor:
    """Pinned RAPTOR formula, not a dimensional reinterpretation (no extra g).

    r ~ U[0, .3 * max(TWR-1, 0)]; sigma_F = r*TWR*mass/3.
    The original /3 is a scale convention, not clipping at three sigma.
    """
    r = unit_uniform * RAPTOR_FORCE_COEFFICIENT * (thrust_to_weight - 1).clamp_min(0)
    return r * thrust_to_weight * mass / 3


@torch.no_grad()
def attach_disturbances(state: RaptorState, config: DisturbanceConfig, *, seed: int,
                        horizon: int) -> RaptorState:
    """Initialize a pooled CPU bank, retaining independent RNG streams.

    Force, latency and measurement draws never depend on the Actor or on an
    episode's success. Sensor standard deviations do not encode airframe ID.
    History before reset is held at the first acquired noisy measurement.
    """
    if state.position.device.type != "cpu" or horizon < 1:
        raise ValueError("attach disturbances to a CPU bank with a positive horizon")
    n, dtype = len(state.mass), state.mass.dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("disturbances require float32 or float64")
    if bool((state.step_index != 0).any()):
        raise ValueError("disturbances may only be attached at reset")
    rows = torch.arange(n, dtype=torch.long)
    zero = torch.zeros(n, dtype=dtype)
    force = torch.zeros(n, 3, dtype=dtype)
    std = torch.zeros(n, MEASUREMENT_DIM, dtype=dtype)
    tape = torch.zeros(n, 1, MEASUREMENT_DIM, dtype=dtype)
    sigma_force, delay = zero, zero
    if config.enabled:
        force_rng = _generator(seed, 0xF04CE)
        noise_rng = _generator(seed, 0x5E4502)
        delay_rng = _generator(seed, 0xDE1A7)
        # FP64 at reset avoids avoidable loss in the physical scale calculation.
        u = torch.rand(n, generator=force_rng, dtype=torch.float64)
        sigma_force = raptor_force_std(state.mass.double(), state.thrust_to_weight.double(), u).to(dtype)
        force = torch.randn(n, 3, generator=force_rng, dtype=dtype) * sigma_force[:, None]
        std = torch.tensor((config.position_std, config.velocity_std,
                            config.attitude_std, config.omega_std), dtype=dtype)
        std = std.repeat_interleave(3)[None].expand(n, MEASUREMENT_DIM).clone()
        tape = torch.randn(n, horizon + 1, MEASUREMENT_DIM, generator=noise_rng, dtype=dtype) * std[:, None]
        delay = (config.velocity_delay_min +
                 (config.velocity_delay_max - config.velocity_delay_min) *
                 torch.rand(n, generator=delay_rng, dtype=torch.float64)).to(dtype)
    rotation_tape = rotation_error(tape[:, :, 6:9].reshape(-1, 3)).reshape(n, tape.shape[1], 3, 3)
    if not all(bool(torch.isfinite(x).all()) for x in (force, sigma_force, tape, delay, rotation_tape)):
        raise FloatingPointError("nonfinite sampled disturbance")
    return replace(state, external_force=force, external_torque=torch.zeros_like(force),
                   force_std=sigma_force, noise_std=std, noise_tape=tape,
                   rotation_tape=rotation_tape, noise_row=rows, velocity_delay=delay,
                   previous_velocity=state.velocity[:, None, :].expand(n, VELOCITY_HISTORY_STEPS, 3).clone())


def noise_at(state: RaptorState, *, previous: bool = False) -> torch.Tensor:
    index = (state.step_index - int(previous)).clamp_min(0)
    if state.noise_tape.shape[1] == 1:
        index = torch.zeros_like(index)
    return state.noise_tape[state.noise_row, index]


def rotation_error(vector: torch.Tensor) -> torch.Tensor:
    """SO(3) exponential with stable zero-angle derivatives."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    angle = vector.norm(dim=-1)
    a = torch.sinc(angle/math.pi)[:, None, None]
    b = (.5*torch.sinc(angle/(2*math.pi)).square())[:, None, None]
    return torch.eye(3, device=vector.device, dtype=vector.dtype) + a*skew + b*(skew@skew)


def measured_observation(state: RaptorState, dt: float = CONTROL_DT) -> torch.Tensor:
    """Whitelist: measured p, delayed measured world v, measured R/omega, command.

    Four acquisition times support any latency in [0,30] ms at 100 Hz.
    Previous velocities stay differentiable; reuse their original noise samples.
    At exactly 30 ms both interpolation indices are 3: never read a fifth frame.
    The sampled latency and all noise/force/airframe metadata remain hidden.
    """
    if not math.isfinite(dt) or abs(dt - CONTROL_DT) > 1e-12:
        raise ValueError("measurement history requires dt=0.01 seconds")
    if state.noise_tape.shape[1] == 1:
        return torch.cat((state.position, state.velocity, state.rotation.flatten(1),
                          state.omega, state.previous_action), -1)
    eps = noise_at(state)
    offsets = torch.arange(VELOCITY_HISTORY_STEPS + 1, device=state.step_index.device)
    times = (state.step_index[:, None] - offsets[None]).clamp_min(0)
    acquired_noise = state.noise_tape[state.noise_row[:, None], times, 3:6]
    values = torch.cat((state.velocity[:, None, :], state.previous_velocity), 1) + acquired_noise
    q = (state.velocity_delay / dt).clamp(0, VELOCITY_HISTORY_STEPS)
    lower = q.floor().long()
    upper = (lower + 1).clamp_max(VELOCITY_HISTORY_STEPS)
    weight = (q - lower.to(q.dtype))[:, None]
    rows = torch.arange(len(q), device=q.device)
    velocity = (1 - weight) * values[rows, lower] + weight * values[rows, upper]
    rotation = state.rotation @ state.rotation_tape[state.noise_row, state.step_index]
    return torch.cat((state.position + eps[:, :3], velocity, rotation.flatten(1),
                      state.omega + eps[:, 9:12], state.previous_action), -1)


def executed_command(state: RaptorState, command: torch.Tensor) -> torch.Tensor:
    """No additive execution noise; retain the physical motor-response model."""
    return command.clamp(-1, 1)


@torch.no_grad()
def disturbance_report(state: RaptorState) -> dict:
    def span(value):
        return [float(value.min()), float(value.max())]
    return {"version": NOISE_VERSION, "deployment_authorized": False,
            "force_model": "raptor-source-gaussian-episode-constant-no-extra-g",
            "force_std_N": span(state.force_std),
            "force_norm_N": span(state.external_force.norm(dim=-1)),
            "measurement_std": {name: span(state.noise_std[:, part]) for name, part in
                                (("position_m", slice(0, 3)), ("velocity_m_s", slice(3, 6)),
                                 ("attitude_rotvec_rad", slice(6, 9)), ("omega_rad_s", slice(9, 12)))},
            "velocity_delay_s": span(state.velocity_delay),
            "external_torque_noise": False, "motor_command_noise": False,
            "transient_force_schedule": False}

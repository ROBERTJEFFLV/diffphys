"""Bounded uncertainty in one joint, dimensionless actuator-reserve budget.

The certificate is static hover allocation + an explicit reference error model,
NOT a region of attraction of the learned Actor. See docs/disturbance_budget.md.
All randomness is sampled on CPU once per episode; step/observe never draw RNG.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from env_raptor import RaptorState

NOISE_VERSION = "joint-reserve-bounded-v1"
COMPONENTS = ("force", "torque", "action", "position", "velocity", "attitude", "omega", "delay")
# Fixed by the retained RAPTOR parameter support, not per-vehicle sensor quality.
REFERENCE_TIME = 0.30
FASTEST_MOTOR_TIME = 0.03


@dataclass(frozen=True)
class DisturbanceConfig:
    budget: float = 0.10
    pool: tuple[float, ...] = (1., 1., 1., 1., 1.)

    def __post_init__(self):
        if not math.isfinite(self.budget) or not 0 <= self.budget <= 0.10:
            raise ValueError("disturbance-budget must be finite and in [0,0.10]")
        pool = tuple(float(x) for x in self.pool)
        if (len(pool) != 5 or any(not math.isfinite(x) or x < 0 for x in pool)
                or not math.isfinite(sum(pool)) or sum(pool) <= 0):
            raise ValueError("disturbance-pool needs five nonnegative weights with positive sum")
        object.__setattr__(self, "pool", pool)

    @classmethod
    def from_args(cls, args):
        return cls(args.disturbance_budget, tuple(args.disturbance_pool))


def sample_bounded(bounds: torch.Tensor, generator: torch.Generator, *, steps: int | None = None):
    """One sampler for both classes: supplied physical/SI bounds times U[-1,1]."""
    shape = bounds.shape if steps is None else (bounds.shape[0], steps, bounds.shape[1])
    scale = bounds if steps is None else bounds[:, None, :]
    return (2 * torch.rand(shape, generator=generator, dtype=bounds.dtype) - 1) * scale


@torch.no_grad()
def capacities(state: RaptorState, budget: float = .10) -> dict[str, torch.Tensor]:
    """Exact per-airframe box margins; common sensor envelope over the WHOLE pool.

    B maps rotor thrusts to [collective, roll, pitch, yaw]. No TTI shortcut is
    used for yaw. The absolute-value allocation bound accounts for all axes
    acting together. Double precision is used only at reset, not in the rollout.
    """
    m, j = state.mass.double(), state.inertia.double()
    r, k, c = (x.double() for x in (state.rotor_positions, state.rotor_torque_constant,
                                  state.thrust_coefficients))
    lo, hi = state.motor_min.double()[:, None], state.motor_max.double()[:, None]
    fmin = c[..., 0] + c[..., 1] * lo + c[..., 2] * lo.square()
    fmax = c[..., 0] + c[..., 1] * hi + c[..., 2] * hi.square()
    b = torch.stack((torch.ones_like(k), r[..., 1], -r[..., 0],
                     k * k.new_tensor((-1, 1, -1, 1))), 1)
    inverse = torch.linalg.inv(b)
    hover = inverse[..., 0] * (m * 9.81)[:, None]
    reserve = torch.minimum(hover - fmin, fmax - hover)
    if not bool((torch.isfinite(reserve) & (reserve > 0)).all()):
        raise ValueError("sampled airframe has no positive two-sided hover thrust reserve")
    inv = inverse.abs()
    def ratio(numerator, denominator):
        return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-300), torch.inf)
    force = ratio(reserve, inv[..., 0]).amin(-1)  # vector norm in N, before /sqrt(3)
    torque = ratio(reserve[..., None], 3 * inv[..., 1:]).amin(1)
    slope = .5 * (hi-lo) * (c[..., 1] + 2 * c[..., 2] * hi)
    if not bool((slope > 0).all() & (c[..., 2] >= 0).all()):
        raise ValueError("bounded action mapping requires the monotone RAPTOR thrust curve")
    action = reserve / slope

    kp, kv = 1 / REFERENCE_TIME**2, 2 / REFERENCE_TIME
    p = (force / (m * kp * math.sqrt(3))).amin()
    v = (force / (m * kv * math.sqrt(3))).amin()
    gyro_load = (inv[..., 1:] * (j*kv)[:, None, :]).sum(-1) / reserve
    omega = gyro_load.amax().reciprocal()
    rotation_load = (inv[..., 1:] * (j*kp)[:, None, :]).sum(-1) / reserve
    # The same attitude error can affect angular correction AND thrust direction.
    rotation_load = rotation_load.amax(-1) + math.sqrt(3)*fmax.sum(-1)/force
    angle = rotation_load.amax().reciprocal()
    acceleration = 9.81 + fmax.sum(-1)/m + budget*force/m
    delay = (force/(m*kv*acceleration)).amin().clamp_max(FASTEST_MOTOR_TIME)
    sensor = torch.stack((p, v, angle, omega)).repeat_interleave(3)
    values = dict(allocation=b, inverse=inverse, hover=hover, reserve=reserve,
                  force=force[:, None]/math.sqrt(3), torque=torque, action=action,
                  sensor=sensor, delay=delay, acceleration=acceleration)
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise ValueError("nonfinite disturbance capacity")
    return values


@torch.no_grad()
def attach_disturbances(state: RaptorState, config: DisturbanceConfig, *, seed: int, horizon: int) -> RaptorState:
    """Sample on a pooled CPU bank, then transfer the complete bank once to GPU.

    Pool weights select 0/25/50/75/100% of the total budget. A random simplex
    splits that budget across ALL eight components, including the existing
    world force. No eight-times-10% accumulation and no unbounded Gaussian tail.
    """
    if state.position.device.type != "cpu" or horizon < 1:
        raise ValueError("attach disturbances to a CPU bank with a positive horizon")
    n, dtype = len(state.mass), state.mass.dtype
    rows = torch.arange(n, dtype=torch.long)
    empty = state.position.new_zeros(n, 16)
    if config.budget == 0 or not any(config.pool[1:]):
        return replace(state, external_force=torch.zeros_like(state.external_force),
                       external_torque=torch.zeros_like(state.external_torque),
                       noise_tape=empty[:, None],
                       rotation_tape=torch.eye(3,dtype=dtype).expand(n,1,3,3).clone(), noise_bounds=empty,
                       noise_fraction=state.position.new_zeros(n, len(COMPONENTS)),
                       velocity_delay=state.mass.new_zeros(n), noise_row=rows)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    cap = capacities(state, config.budget)
    # Retain numerical headroom when casting double-precision capacities to FP32.
    margin = 1 - 32*torch.finfo(dtype).eps
    severity = torch.multinomial(torch.tensor(config.pool, dtype=torch.float64), n, True, generator=g)
    total = severity.double() * (config.budget / 4) * margin
    simplex = -torch.rand(n, len(COMPONENTS), generator=g, dtype=torch.float64).clamp_min(1e-300).log()
    fractions = simplex / simplex.sum(-1, keepdim=True) * total[:, None]
    fraction = fractions.to(dtype)
    def bounds(name, column):
        return (cap[name] * fractions[:, column, None] * margin).to(dtype)
    force = sample_bounded(bounds("force", 0).expand(n, 3), g)
    torque = sample_bounded(bounds("torque", 1), g)
    sensors = cap["sensor"][None] * fractions[:, 3:7].repeat_interleave(3, -1)
    noise_bounds = torch.cat(((sensors*margin).to(dtype), bounds("action", 2)), -1)
    tape = sample_bounded(noise_bounds, g, steps=horizon+1)
    delay = (cap["delay"] * fractions[:, 7] * margin).to(dtype)
    return replace(state, external_force=force, external_torque=torque,
                   noise_tape=tape,
                   rotation_tape=rotation_error(tape[:,:,6:9].reshape(-1,3)).reshape(n,horizon+1,3,3),
                   noise_bounds=noise_bounds, noise_fraction=fraction,
                   velocity_delay=delay, noise_row=rows)


def noise_at(state: RaptorState, *, previous: bool = False) -> torch.Tensor:
    index = (state.step_index-1).clamp_min(0) if previous else state.step_index
    if state.noise_tape.shape[1] == 1:
        index = torch.zeros_like(index)
    return state.noise_tape[state.noise_row, index]


def rotation_error(vector: torch.Tensor) -> torch.Tensor:
    """SO(3) exponential; stable at zero, with ||angle|| bounded by the tape."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    angle = vector.norm(dim=-1)
    a = torch.sinc(angle/math.pi)[:, None, None]
    b = (.5*torch.sinc(angle/(2*math.pi)).square())[:, None, None]
    return torch.eye(3, device=vector.device, dtype=vector.dtype) + a*skew + b*(skew@skew)


def measured_observation(state: RaptorState, dt: float = .01) -> torch.Tensor:
    """Truth and loss stay clean; only the 22D deployable observation is changed.

    Fractional delay interpolates successive acquired VELOCITY measurements,
    including their original noise. R is current: never rotate an old body-frame
    velocity using a different timestamp's attitude. The stored velocities are
    in the world frame. At reset the history is held at the first measurement.
    """
    if state.noise_tape.shape[1] == 1:
        return torch.cat((state.position, state.velocity, state.rotation.flatten(1),
                          state.omega, state.previous_action), -1)
    eps = noise_at(state)
    old_eps = noise_at(state, previous=True)
    lag = (state.velocity_delay/dt)[:, None]
    current = state.velocity + eps[:, 3:6]
    old = state.previous_velocity + old_eps[:, 3:6]
    velocity = current + lag*(old-current)
    rotation = state.rotation @ state.rotation_tape[state.noise_row, state.step_index]
    return torch.cat((state.position+eps[:, :3], velocity, rotation.flatten(1),
                      state.omega+eps[:, 9:12], state.previous_action), -1)


def executed_command(state: RaptorState, command: torch.Tensor) -> torch.Tensor:
    if state.noise_tape.shape[1] == 1:
        return command.clamp(-1, 1)
    return (command + noise_at(state)[:, 12:16]).clamp(-1, 1)


@torch.no_grad()
def disturbance_report(state: RaptorState) -> dict:
    def span(value):
        return [float(value.min()), float(value.max())]
    return {"version": NOISE_VERSION, "certificate": "hover-allocation-and-reference-error-model-only",
            "deployment_authorized": False, "maximum_total_fraction": float(state.noise_fraction.sum(-1).max()),
            "fraction_ranges": {name: span(state.noise_fraction[:, i]) for i, name in enumerate(COMPONENTS)},
            "bound_ranges": {name: span(state.noise_bounds[:, part]) for name, part in
                             (("position_m", slice(0,3)), ("velocity_m_s", slice(3,6)),
                              ("attitude_rad_per_axis", slice(6,9)), ("omega_rad_s", slice(9,12)),
                              ("normalized_action", slice(12,16)))},
            "velocity_delay_s": span(state.velocity_delay),
            "force_norm_N": span(state.external_force.norm(dim=-1)),
            "torque_Nm_by_axis": [span(state.external_torque[:, j]) for j in range(3)]}

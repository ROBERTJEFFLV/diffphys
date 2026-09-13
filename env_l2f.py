"""Differentiable L2F/RAPTOR reference dynamics and paper-first initialization.

Actions are absolute normalized motor commands in [-1, 1], NOT hover residuals.
FLU X-frame order: front-right, back-right, back-left, front-left.
Reference constants/algorithms: see docs/raptor_reference.md and THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math

import torch
import torch.nn.functional as F

ENVIRONMENT_VERSION = "l2f-raptor-reference-v1"
ACTION_CONVENTION = "absolute-normalized-motor-FR-BR-BL-FL-FLU-v1"
RAPTOR_SOURCE = "e43ae4bcda4556321a63f4eb5dcc826cd637aa39"
L2F_SOURCE = "d07592d5c5dea3c90954d2be6f04cfa68581ebe8"
PROFILES = ("l2f", "raptor")


@dataclass(frozen=True)
class L2FParams:
    """One explicit protocol shared by sampling, simulation, EVAL and checkpoints."""
    dt: float = 0.01
    protocol: str = "raptor"

    def __post_init__(self):
        if self.protocol not in PROFILES:
            raise ValueError("protocol must be l2f or raptor")
        if not math.isfinite(self.dt) or abs(self.dt - 0.01) > 1e-12:
            raise ValueError("reference protocols require dt=0.01 s (100 Hz)")


def environment_contract(params: L2FParams) -> dict:
    return {
        "version": ENVIRONMENT_VERSION,
        "protocol": params.protocol,
        "dt": params.dt,
        "action_convention": ACTION_CONVENTION,
        "integrator": "joint-rk4-quaternion-normalization",
        "initialization": "raptor-paper-90deg-size-scaled" if params.protocol == "raptor" else "l2f-2024-default",
        "source": RAPTOR_SOURCE if params.protocol == "raptor" else L2F_SOURCE,
        "noise": "source-default-frozen-tape-v1",
    }


def quaternion_rotation(q: torch.Tensor) -> torch.Tensor:
    """Hamilton wxyz, body to world; also used at intermediate RK4 stages."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ), -1).reshape(*q.shape[:-1], 3, 3)


@dataclass(frozen=True)
class L2FState:
    position: torch.Tensor
    velocity: torch.Tensor
    orientation: torch.Tensor  # Hamilton wxyz; rotation is a derived observation
    omega: torch.Tensor       # body angular velocity
    motor: torch.Tensor       # physical rotor state: RPM (L2F), [0,1] (RAPTOR)
    previous_action: torch.Tensor
    external_force: torch.Tensor  # constant world force for this episode
    external_torque: torch.Tensor # constant body torque for this episode
    mass: torch.Tensor
    inertia: torch.Tensor
    rotor_positions: torch.Tensor
    thrust_coefficients: torch.Tensor
    rotor_torque_constant: torch.Tensor
    motor_time_rising: torch.Tensor
    motor_time_falling: torch.Tensor
    motor_min: torch.Tensor
    motor_max: torch.Tensor
    arm_length: torch.Tensor      # center-to-rotor distance, not an x/y coordinate
    thrust_to_weight: torch.Tensor
    torque_to_inertia: torch.Tensor
    force_std: torch.Tensor
    initial_position_limit: torch.Tensor
    position_limit: torch.Tensor
    velocity_limit: torch.Tensor
    omega_limit: torch.Tensor
    guidance: torch.Tensor
    profile_code: torch.Tensor
    noise_std: torch.Tensor
    noise_tape: torch.Tensor      # immutable [B,H+1,18]; zero-noise uses [B,1,18]
    step_index: torch.Tensor      # int64; repeatable observations at window boundaries

    @property
    def rotation(self):
        return quaternion_rotation(self.orientation)

    def to(self, device, dtype):
        return type(self)(**{
            f.name: getattr(self, f.name).to(
                device=device, dtype=dtype if getattr(self, f.name).is_floating_point() else None
            ) for f in fields(self)
        })


def _orientation(n, generator, dtype, *, haar):
    if haar:
        # L2F-2024 samples a uniform quaternion and rejects angles >90 degrees.
        q = torch.empty((n, 4), dtype=dtype)
        todo = torch.arange(n)
        while todo.numel():
            proposal = F.normalize(torch.randn((todo.numel(), 4), generator=generator, dtype=dtype), dim=-1)
            valid = proposal[:, 0] >= math.cos(math.pi / 4)
            q[todo[valid]] = proposal[valid]
            todo = todo[~valid]
        return q
    # Paper-first: random axis, uniform rotation angle up to pi/2. Do not copy
    # the pinned RAPTOR bug which ignores `limit` and samples only [0,1] rad.
    axis = F.normalize(torch.randn((n, 3), generator=generator, dtype=dtype), dim=-1)
    half = torch.rand((n, 1), generator=generator, dtype=dtype) * (math.pi / 4)
    return torch.cat((half.cos(), half.sin() * axis), -1)


class L2FSimulator:
    def __init__(self, params: L2FParams = L2FParams()):
        self.params = params

    @torch.no_grad()
    def reset(self, batch_size: int, *, device="cpu", dtype=torch.float32,
              seed: int = 0, horizon: int = 500) -> L2FState:
        if batch_size < 1 or horizon < 1:
            raise ValueError("batch_size and horizon must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("reference dynamics require float32 or float64")
        # Local CPU RNG makes the fixed bank independent of optimizer RNG and
        # accelerator placement. No global RNG mutation and no random draws in step.
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        n = batch_size
        def uniform(low, high, shape=None):
            return low + (high-low) * torch.rand((n,) if shape is None else shape, generator=g, dtype=dtype)
        def normal(shape):
            return torch.randn(shape, generator=g, dtype=dtype)
        def constant(value):
            return torch.full((n,), float(value), dtype=dtype)
        def rotor(value):
            return value[:, None].expand(n, 4).clone()

        raptor = self.params.protocol == "raptor"
        if raptor:
            # Match the pinned generator's joint distribution, not the old
            # log-mass/4x4 physical-fit sampler. No clipping of sampled authority.
            mass = uniform(0.02**(1/3), 5.0**(1/3)).pow(3)
            tw = uniform(1.5, 5.0)
            base_mass = 0.0306
            base_c = torch.tensor((0.00352526, 0.01437313, 0.09223048), dtype=dtype)
            coefficient_scale = tw * mass * 9.81 / (4 * base_c.sum())
            coefficients = coefficient_scale[:, None, None] * base_c[None, None, :].expand(n, 4, 3)
            tti = uniform(40.0, 1200.0)
            # Upstream uses Normal(mean=-0.1,std=0.1), then a reciprocal map.
            deviation = -0.1 + 0.1 * normal((n,))
            size_factor = torch.where(deviation < 0, 1/(1-deviation), 1+deviation)
            distance_factor = (mass/base_mass).pow(1/3) * size_factor
            xy = 0.028 * distance_factor
            base_j = torch.tensor((9.416556729130406e-6, 9.644051701582312e-6, 1.745951732253285e-5), dtype=dtype)
            tti_nominal = 0.028 * math.sqrt(2) * (tw*mass*9.81/4) / base_j[0]
            inertia = base_j[None, :] * (distance_factor * tti_nominal / tti)[:, None]
            km = rotor(uniform(0.005, 0.05))
            rising = rotor(uniform(0.03, 0.10))
            falling = rotor(uniform(0.03, 0.30))
            motor_min, motor_max = constant(0), constant(1)
            # Preserve the published code's units/formula (there is no g factor).
            force_std = uniform(0.0, 0.3*(tw-1)) * tw * mass / 3
            torque_std = constant(0)
            initial_limit = 10 * math.sqrt(2) * xy
            position_limit = 2 * initial_limit
            velocity_limit, omega_limit = constant(2), constant(35)
            noise_std = torch.zeros((n, 18), dtype=dtype)
        else:
            mass = constant(0.027)
            xy = constant(0.028)
            inertia = torch.tensor((3.85e-6, 3.85e-6, 5.9675e-6), dtype=dtype).expand(n, 3).clone()
            coefficients = torch.tensor((0.0, 0.0, 3.16e-10), dtype=dtype).expand(n, 4, 3).clone()
            km = rotor(constant(0.005964552))
            rising = falling = rotor(constant(0.15))
            motor_min, motor_max = constant(0), constant(21702)
            tw = 4 * 3.16e-10 * motor_max.square() / (mass * 9.81)
            tti = 2 * xy * (3.16e-10 * motor_max.square()) / inertia[:, 0]
            force_std = constant(0.027 * 9.81 / 20)
            torque_std = constant(0.027 * 9.81 / 10000)
            initial_limit, position_limit = constant(0.2), constant(0.6)
            velocity_limit = omega_limit = constant(1000)
            # Order used by the actor: p, v, row-major R, body omega.
            noise_std = torch.tensor([0.001]*3 + [0.002]*3 + [0.001]*9 + [0.002]*3, dtype=dtype).expand(n, 18).clone()

        xy_signs = torch.tensor(((1,-1), (-1,-1), (-1,1), (1,1)), dtype=dtype)
        positions_xy = xy[:, None, None] * xy_signs[None, :, :]
        rotor_positions = torch.cat((positions_xy, torch.zeros((n, 4, 1), dtype=dtype)), -1)
        guidance = uniform(0., 1.) < 0.1
        position = uniform(-1., 1., (n, 3)) * initial_limit[:, None]
        velocity = uniform(-1., 1., (n, 3))
        omega = uniform(-1., 1., (n, 3))
        q = _orientation(n, g, dtype, haar=not raptor)
        position[guidance] = 0
        q[guidance] = q.new_tensor((1, 0, 0, 0))
        if raptor:
            velocity[guidance] = 0
            omega[guidance] = 0
        # Rotor initialization is distinct from the kinematic guidance state.
        # RAPTOR's relative [-1,0] maps to [0,0.5], not a shared hover throttle.
        motor = uniform(0., 0.5, (n, 4)) if raptor else rotor(motor_max*0.5)
        previous_action = 2*(motor-motor_min[:, None])/(motor_max-motor_min)[:, None]-1
        force = normal((n, 3)) * force_std[:, None]
        torque = normal((n, 3)) * torque_std[:, None]
        noise_tape = (torch.zeros((n, 1, 18), dtype=dtype) if raptor else
                      normal((n, horizon+1, 18)) * noise_std[:, None, :])
        state = L2FState(
            position, velocity, q, omega, motor, previous_action,
            force, torque, mass, inertia, rotor_positions, coefficients, km,
            rising, falling, motor_min, motor_max, math.sqrt(2)*xy, tw, tti, force_std,
            initial_limit, position_limit, velocity_limit, omega_limit,
            guidance.to(dtype), torch.full((n,), int(raptor), dtype=torch.long),
            noise_std, noise_tape, torch.zeros(n, dtype=torch.long),
        )
        return state.to(device, dtype)

    @staticmethod
    def motor_command(state: L2FState, action: torch.Tensor) -> torch.Tensor:
        return state.motor_min[:, None] + (action.clamp(-1, 1)+1)*0.5 * (state.motor_max-state.motor_min)[:, None]

    @staticmethod
    def thrust(state: L2FState, motor: torch.Tensor) -> torch.Tensor:
        c = state.thrust_coefficients
        return c[..., 0] + c[..., 1]*motor + c[..., 2]*motor.square()

    @staticmethod
    def body_torque(state: L2FState, thrust: torch.Tensor) -> torch.Tensor:
        x, y = state.rotor_positions[..., 0], state.rotor_positions[..., 1]
        signs = thrust.new_tensor((-1, 1, -1, 1))
        return torch.stack(((y*thrust).sum(-1), -(x*thrust).sum(-1),
                            (signs*state.rotor_torque_constant*thrust).sum(-1)), -1)

    def step(self, state: L2FState, action: torch.Tensor) -> L2FState:
        if action.shape != state.motor.shape or action.device != state.motor.device or action.dtype != state.motor.dtype:
            raise ValueError("action must match rotor state [batch,4], dtype and device")
        command = action.clamp(-1, 1)
        setpoint = self.motor_command(state, command)
        dt = self.params.dt
        # Joint RK4, including the motor response at each stage. No Euler motor
        # shortcut, hover offsets, authority normalization or hidden truth inputs.
        def dynamics(values):
            p, v, q, w, m = values
            thrust = self.thrust(state, m)
            body_z = quaternion_rotation(q)[..., 2]
            acceleration = body_z * (thrust.sum(-1)/state.mass)[:, None]
            acceleration = acceleration + v.new_tensor((0., 0., -9.81)) + state.external_force/state.mass[:, None]
            torque = self.body_torque(state, thrust) + state.external_torque
            w_dot = (torque - torch.linalg.cross(w, state.inertia*w, dim=-1))/state.inertia
            qw, qv = q[:, :1], q[:, 1:]
            q_dot = 0.5 * torch.cat((-(qv*w).sum(-1, keepdim=True), qw*w + torch.linalg.cross(qv, w, dim=-1)), -1)
            tau = torch.where(setpoint >= m, state.motor_time_rising, state.motor_time_falling)
            return v, acceleration, q_dot, w_dot, (setpoint-m)/tau
        values = (state.position, state.velocity, state.orientation, state.omega, state.motor)
        def shifted(k, fraction):
            return tuple(x + (dt*fraction)*d for x, d in zip(values, k))
        k1 = dynamics(values)
        k2 = dynamics(shifted(k1, 0.5))
        k3 = dynamics(shifted(k2, 0.5))
        k4 = dynamics(shifted(k3, 1.0))
        p, v, q, w, m = tuple(x+(dt/6)*(a+2*b+2*c+d) for x,a,b,c,d in zip(values,k1,k2,k3,k4))
        q = F.normalize(q, dim=-1)
        if self.params.protocol == "raptor":
            # Upstream numerical state limits; these are NOT episode boundaries.
            p, v, w = (value.clamp(-100000, 100000) for value in (p, v, w))
        m = torch.maximum(state.motor_min[:, None], torch.minimum(state.motor_max[:, None], m))
        return replace(state, position=p, velocity=v, orientation=q, omega=w,
                       motor=m, previous_action=command, step_index=state.step_index+1)

    @staticmethod
    def terminated(state: L2FState) -> torch.Tensor:
        return ((state.position.abs() > state.position_limit[:, None]).any(-1)
                | (state.velocity.abs() > state.velocity_limit[:, None]).any(-1)
                | (state.omega.abs() > state.omega_limit[:, None]).any(-1))

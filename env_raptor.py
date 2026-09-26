"""Multi-airframe RAPTOR dynamics: absolute motor commands, joint RK4, FLU X frame.

Only this physical family is supported. Observation/actuation uncertainty is
isolated in response_noise; true-state costs and rigid-body equations stay clean.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
import torch
import torch.nn.functional as F

from response_noise import (DisturbanceConfig, attach_disturbances, executed_command,
                            NOISE_VERSION)

ENVIRONMENT_VERSION = "raptor-multi-airframe-bounded-v3"
ACTION_CONVENTION = "absolute-normalized-motor-FR-BR-BL-FL-FLU-v1"
IMMUTABLE_TAPES = ("noise_tape", "rotation_tape")
RAPTOR_SOURCE = "e43ae4bcda4556321a63f4eb5dcc826cd637aa39"


@dataclass(frozen=True)
class RaptorParams:
    dt: float = .01

    def __post_init__(self):
        if not math.isfinite(self.dt) or abs(self.dt-.01) > 1e-12:
            raise ValueError("RAPTOR requires dt=0.01 s (100 Hz)")


def environment_contract(params: RaptorParams) -> dict:
    return {"version": ENVIRONMENT_VERSION, "dt": params.dt,
            "action_convention": ACTION_CONVENTION,
            "integrator": "joint-rk4-quaternion-normalization",
            "initialization": "raptor-paper-90deg-size-scaled", "source": RAPTOR_SOURCE,
            "noise": NOISE_VERSION, "action_history": "command-not-hidden-execution",
            "termination": "position-only-per-axis-strict-exceedance-v1"}


def quaternion_rotation(q: torch.Tensor) -> torch.Tensor:
    """Hamilton wxyz, body to world; also used at intermediate RK4 stages."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ), -1).reshape(*q.shape[:-1], 3, 3)


@dataclass(frozen=True)
class RaptorState:
    position: torch.Tensor
    velocity: torch.Tensor
    orientation: torch.Tensor
    omega: torch.Tensor
    motor: torch.Tensor
    previous_action: torch.Tensor  # Last known command, NOT hidden execution noise.
    external_force: torch.Tensor  # Episode-constant world force.
    external_torque: torch.Tensor # Episode-constant body torque.
    mass: torch.Tensor
    inertia: torch.Tensor
    rotor_positions: torch.Tensor
    thrust_coefficients: torch.Tensor
    rotor_torque_constant: torch.Tensor
    motor_time_rising: torch.Tensor
    motor_time_falling: torch.Tensor
    motor_min: torch.Tensor
    motor_max: torch.Tensor
    arm_length: torch.Tensor
    thrust_to_weight: torch.Tensor
    torque_to_inertia: torch.Tensor
    initial_position_limit: torch.Tensor
    position_limit: torch.Tensor
    guidance: torch.Tensor
    noise_tape: torch.Tensor      # Immutable ORIGINAL pool [N,H+1,16], or [N,1,16].
    rotation_tape: torch.Tensor   # Precomputed SO(3) errors: no trigonometric kernels in step.
    noise_row: torch.Tensor       # Stable row ID: compacting live states never copies the tape.
    noise_bounds: torch.Tensor
    noise_fraction: torch.Tensor
    velocity_delay: torch.Tensor
    previous_velocity: torch.Tensor  # Differentiable world-velocity history, one control step.
    step_index: torch.Tensor

    @property
    def rotation(self):
        return quaternion_rotation(self.orientation)

    def to(self, device, dtype):
        return type(self)(**{
            f.name: getattr(self, f.name).to(
                device=device, dtype=dtype if getattr(self, f.name).is_floating_point() else None
            ) for f in fields(self)
        })


class RaptorSimulator:
    def __init__(self, params: RaptorParams = RaptorParams()):
        self.params = params

    @torch.no_grad()
    def reset(self, batch_size: int, *, device="cpu", dtype=torch.float32,
              seed: int = 0, horizon: int = 500,
              disturbances: DisturbanceConfig = DisturbanceConfig()) -> RaptorState:
        if batch_size < 1 or horizon < 1:
            raise ValueError("batch_size and horizon must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("reference dynamics require float32 or float64")
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

        mass = uniform(0.02**(1/3), 5.0**(1/3)).pow(3)
        tw = uniform(1.5, 5.0)
        base_mass = .0306
        base_c = torch.tensor((.00352526, .01437313, .09223048), dtype=dtype)
        coefficient_scale = tw * mass * 9.81 / (4 * base_c.sum())
        coefficients = coefficient_scale[:, None, None] * base_c[None, None, :].expand(n, 4, 3)
        tti = uniform(40., 1200.)
        deviation = -.1 + .1 * normal((n,))
        size_factor = torch.where(deviation < 0, 1/(1-deviation), 1+deviation)
        distance_factor = (mass/base_mass).pow(1/3) * size_factor
        xy = .028 * distance_factor
        base_j = torch.tensor((9.416556729130406e-6, 9.644051701582312e-6,
                               1.745951732253285e-5), dtype=dtype)
        tti_nominal = .028 * math.sqrt(2) * (tw*mass*9.81/4) / base_j[0]
        inertia = base_j[None, :] * (distance_factor * tti_nominal / tti)[:, None]
        km = rotor(uniform(.005, .05))
        rising = rotor(uniform(.03, .10))
        falling = rotor(uniform(.03, .30))
        signs = torch.tensor(((1,-1), (-1,-1), (-1,1), (1,1)), dtype=dtype)
        positions_xy = xy[:, None, None] * signs[None]
        rotor_positions = torch.cat((positions_xy, torch.zeros((n,4,1), dtype=dtype)), -1)
        initial_limit = 10 * math.sqrt(2) * xy
        guidance = uniform(0., 1.) < .1
        position = uniform(-1., 1., (n,3)) * initial_limit[:, None]
        velocity = uniform(-1., 1., (n,3))
        omega = uniform(-1., 1., (n,3))
        axis = F.normalize(normal((n,3)), dim=-1)
        half = uniform(0., math.pi/4, (n,1))
        q = torch.cat((half.cos(), half.sin()*axis), -1)
        position[guidance] = 0
        velocity[guidance] = 0
        omega[guidance] = 0
        q[guidance] = q.new_tensor((1,0,0,0))
        motor = uniform(0., .5, (n,4))
        zero = torch.zeros((n,3), dtype=dtype)
        state = RaptorState(
            position=position, velocity=velocity, orientation=q, omega=omega, motor=motor,
            previous_action=2*motor-1, external_force=zero, external_torque=zero,
            mass=mass, inertia=inertia, rotor_positions=rotor_positions,
            thrust_coefficients=coefficients, rotor_torque_constant=km,
            motor_time_rising=rising, motor_time_falling=falling,
            motor_min=constant(0), motor_max=constant(1), arm_length=math.sqrt(2)*xy,
            thrust_to_weight=tw, torque_to_inertia=tti,
            initial_position_limit=initial_limit, position_limit=2*initial_limit,
            guidance=guidance.to(dtype), noise_tape=torch.zeros(n,1,16,dtype=dtype),
            rotation_tape=torch.eye(3,dtype=dtype).expand(n,1,3,3).clone(),
            noise_row=torch.arange(n), noise_bounds=torch.zeros(n,16,dtype=dtype),
            noise_fraction=torch.zeros(n,8,dtype=dtype), velocity_delay=constant(0),
            previous_velocity=velocity, step_index=torch.zeros(n,dtype=torch.long))
        state = attach_disturbances(state, disturbances, seed=seed ^ 0x5A17C9E3, horizon=horizon)
        return state.to(device, dtype)

    @staticmethod
    def motor_command(state: RaptorState, action: torch.Tensor) -> torch.Tensor:
        return state.motor_min[:, None] + (action.clamp(-1, 1)+1)*.5 * (state.motor_max-state.motor_min)[:, None]

    @staticmethod
    def thrust(state: RaptorState, motor: torch.Tensor) -> torch.Tensor:
        c = state.thrust_coefficients
        return c[..., 0] + c[..., 1]*motor + c[..., 2]*motor.square()

    @staticmethod
    def body_torque(state: RaptorState, thrust: torch.Tensor) -> torch.Tensor:
        x, y = state.rotor_positions[..., 0], state.rotor_positions[..., 1]
        signs = torch.tensor((-1,1,-1,1), device=thrust.device, dtype=thrust.dtype)
        return torch.stack(((y*thrust).sum(-1), -(x*thrust).sum(-1),
                            (signs*state.rotor_torque_constant*thrust).sum(-1)), -1)

    def step(self, state: RaptorState, action: torch.Tensor) -> RaptorState:
        if action.shape != state.motor.shape or action.device != state.motor.device or action.dtype != state.motor.dtype:
            raise ValueError("action must match rotor state [batch,4], dtype and device")
        command = action.clamp(-1, 1)
        setpoint = self.motor_command(state, executed_command(state, command))
        dt = self.params.dt
        def dynamics(values):
            p, v, q, w, m = values
            thrust = self.thrust(state, m)
            body_z = quaternion_rotation(q)[..., 2]
            acceleration = body_z * (thrust.sum(-1)/state.mass)[:, None]
            gravity = torch.tensor((0.,0.,-9.81), device=v.device, dtype=v.dtype)
            acceleration = acceleration + gravity + state.external_force/state.mass[:, None]
            torque = self.body_torque(state, thrust) + state.external_torque
            w_dot = (torque - torch.linalg.cross(w, state.inertia*w, dim=-1))/state.inertia
            qw, qv = q[:, :1], q[:, 1:]
            q_dot = .5 * torch.cat((-(qv*w).sum(-1, keepdim=True),
                                    qw*w + torch.linalg.cross(qv,w,dim=-1)), -1)
            tau = torch.where(setpoint >= m, state.motor_time_rising, state.motor_time_falling)
            return v, acceleration, q_dot, w_dot, (setpoint-m)/tau
        values = (state.position, state.velocity, state.orientation, state.omega, state.motor)
        def shifted(k, fraction):
            return tuple(x + (dt*fraction)*d for x,d in zip(values,k))
        k1 = dynamics(values)
        k2 = dynamics(shifted(k1,.5))
        k3 = dynamics(shifted(k2,.5))
        k4 = dynamics(shifted(k3,1.))
        p,v,q,w,m = tuple(x+(dt/6)*(a+2*b+2*c+d) for x,a,b,c,d in zip(values,k1,k2,k3,k4))
        q = F.normalize(q, dim=-1)
        p,v,w = (x.clamp(-100000,100000) for x in (p,v,w))
        m = torch.maximum(state.motor_min[:, None], torch.minimum(state.motor_max[:, None],m))
        return replace(state, position=p, velocity=v, orientation=q, omega=w, motor=m,
                       previous_action=command, previous_velocity=state.velocity,
                       step_index=state.step_index+1)

    @staticmethod
    def terminated(state: RaptorState) -> torch.Tensor:
        return RaptorSimulator.position_terminated(state.position,state.position_limit)

    @staticmethod
    def position_terminated(position: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
        return (position.abs() > limit[..., None]).any(-1)

"""Differentiable kinematic readouts of the EXISTING motor-level RK4 rollout.

No second integrator, virtual acceleration actuator, or surrogate backward is
introduced here. Velocity losses backpropagate through the native simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from env_raptor import RaptorSimulator, quaternion_rotation


@dataclass(frozen=True)
class KinematicReadout:
    acceleration: torch.Tensor
    angular_acceleration: torch.Tensor
    jerk: torch.Tensor
    jerk_valid: torch.Tensor


def interval_kinematics(trace, dt: float = .01) -> KinematicReadout:
    """RK4 interval-mean acceleration, NOT an independent approximation to step.

    Without numerical safety clamps, (v[k+1]-v[k])/dt is exactly the RK4-weighted
    acceleration. Keep BOTH velocity endpoints attached; no measured/noisy
    finite difference or second physical rollout is used.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    v = torch.cat((trace.initial.velocity[None], trace.velocities), 0)
    w = torch.cat((trace.initial.omega[None], trace.omegas), 0)
    active = trace.valid[..., None]
    acceleration = torch.where(active, (v[1:]-v[:-1])/dt, 0.)
    angular = torch.where(active, (w[1:]-w[:-1])/dt, 0.)
    valid_jerk = trace.valid[1:] & trace.valid[:-1]
    jerk = torch.where(valid_jerk[..., None], acceleration.diff(dim=0)/dt, 0.)
    return KinematicReadout(acceleration, angular, jerk, valid_jerk)


def instantaneous_acceleration(state) -> tuple[torch.Tensor, torch.Tensor]:
    """Unforced AFT terminal force/moment balance; original force law, once.

    This task has no pulses/constant forces. Hidden physical truth is used only
    for task scoring, NEVER as an Actor observation.
    """
    thrust = RaptorSimulator.thrust(state, state.motor)
    R = quaternion_rotation(state.orientation)
    gravity = state.velocity.new_tensor((0., 0., -9.81))
    acceleration = R[..., 2]*(thrust.sum(-1)/state.mass)[..., None] + gravity
    torque = RaptorSimulator.body_torque(state, thrust)
    angular = (torque-torch.linalg.cross(state.omega, state.inertia*state.omega, dim=-1))/state.inertia
    return acceleration, angular


def segment_ball_geometry(previous, current, centre, radius):
    """Chord/sphere intersections on the current measured control interval.

    Return hit, closest fraction, entry fraction, exit fraction. This detects a
    straight chord that crosses the ball with both endpoints outside. It is NOT
    a certificate for the curved continuous RK4 path between samples. Tangency
    is not traversal. Fractions are differentiable on a fixed nondegenerate
    branch; the hit decision itself is discrete.
    """
    d = current-previous
    r = previous-centre
    a = d.square().sum(-1)
    moving = a > 0
    safe_a = torch.where(moving, a, torch.ones_like(a))
    b = (r*d).sum(-1)
    closest = (-b/safe_a).clamp(0., 1.)
    distance2 = (r+closest[..., None]*d).square().sum(-1)
    hit = (distance2 < radius.square()) & moving
    inside0 = r.square().sum(-1) <= radius.square()
    inside1 = (current-centre).square().sum(-1) <= radius.square()
    hit = hit | inside0 | inside1
    discriminant = b.square()-a*(r.square().sum(-1)-radius.square())
    # Prevent a 0 * inf backward on the unused miss/tangent branch.
    transverse = hit & moving & (discriminant > 0)
    safe_disc = torch.where(transverse, discriminant, torch.ones_like(a))
    root = torch.where(transverse, safe_disc.sqrt(), torch.zeros_like(a))
    entry = ((-b-root)/safe_a).clamp(0., 1.)
    exit_ = ((-b+root)/safe_a).clamp(0., 1.)
    entry = torch.where(moving, entry, torch.zeros_like(entry))
    exit_ = torch.where(moving, exit_, torch.ones_like(exit_))
    return hit, closest, entry, exit_


def advance_phase(previous, current, centre, radius, phase):
    """0 approach -> 1 entered -> 2 STOP, locked forever after first exit.

    No speed/attitude/dwell gate. Full outside-to-outside chord passage switches
    directly from 0 to 2 at the next control call.
    """
    hit, _, _, _ = segment_ball_geometry(previous, current, centre, radius)
    inside = (current-centre).square().sum(-1) <= radius.square()
    entered = (phase >= 1) | hit
    stop = (phase == 2) | (entered & ~inside)
    return torch.where(stop, 2, torch.where(entered, 1, 0)).to(phase.dtype)


def interpolate_quaternion(before, after, fraction):
    """Shortest-sign normalized interpolation used ONLY for substep scoring."""
    aligned = torch.where(((before*after).sum(-1) < 0)[..., None], -after, after)
    return torch.nn.functional.normalize(before+fraction[..., None]*(aligned-before), dim=-1)

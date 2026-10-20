"""Fixed point commands and deployable hard-switch state, without a time reference."""
from __future__ import annotations

from dataclasses import dataclass, replace, fields
import math
import torch

from .env_raptor import RaptorState, quaternion_rotation
from .waypoint_config import WaypointConfig


@dataclass(frozen=True)
class WaypointMission:
    points: torch.Tensor             # [N,3,3], fixed world positions
    rotation: torch.Tensor           # [N,3,3], requested midpoint attitude (never downgraded)
    level_rotation: torch.Tensor
    direction: torch.Tensor          # fixed unit crossing-plane normal
    band_half_width: torch.Tensor    # fixed per mission, not based on policy performance

    def to(self, device, dtype):
        return type(self)(**{f.name: getattr(self, f.name).to(device=device, dtype=dtype) for f in fields(self)})

    def select(self, rows):
        return type(self)(**{f.name: getattr(self, f.name).index_select(0, rows) for f in fields(self)})

    def validate(self, position_limit, config):
        n = len(position_limit)
        expected = {'points':(n,3,3), 'rotation':(n,3,3), 'level_rotation':(n,3,3),
                    'direction':(n,3), 'band_half_width':(n,)}
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or not bool(torch.isfinite(value).all()):
                raise ValueError("invalid mission " + name)
        if not bool((self.points.abs() < position_limit[:,None,None]).all()):
            raise ValueError("fixed point is outside the original world boundary")
        if not torch.allclose(self.direction.norm(dim=-1), torch.ones_like(position_limit), atol=1e-6, rtol=1e-6):
            raise ValueError("crossing direction must be unit length")
        identity = torch.eye(3, dtype=self.rotation.dtype, device=self.rotation.device)
        for r in (self.rotation, self.level_rotation):
            if (not torch.allclose(r.transpose(-1,-2)@r, identity.expand_as(r), atol=1e-5, rtol=1e-5)
                    or not bool((torch.linalg.det(r) > .999).all())):
                raise ValueError("target attitude must belong to SO(3)")
        along = ((self.points-self.points[:,1,None])*self.direction[:,None]).sum(-1)
        if not bool(((along[:,0] < -self.band_half_width-config.arrival_radius_m)
                     & (along[:,2] > self.band_half_width+config.arrival_radius_m)
                     & (self.band_half_width > config.pose_half_width_m)).all()):
            raise ValueError("fixed geometry lacks separate approach, passage and braking regions")


@torch.no_grad()
def build_mission(initial: RaptorState, config: WaypointConfig, *, seed: int):
    """Keep each sampled aircraft, its boundary, noise and nonzero velocity.

    Birth is point1, level attitude, zero angular velocity and equilibrium motors,
    as in the previous agile reset. Only originally zero initial speeds are filled.
    No reference curve, duration optimization or actuator-based tilt reduction.
    """
    if initial.position.device.type != 'cpu':
        raise ValueError("sample missions on CPU before upload")
    n, dtype = len(initial.mass), initial.mass.dtype
    generator = torch.Generator().manual_seed(int(seed))
    yaw = 2*math.pi*torch.rand(n, generator=generator, dtype=dtype)
    roll = yaw.new_full((n,), math.radians(config.tilt_degrees))
    q_yaw = torch.stack((torch.cos(yaw/2), yaw*0, yaw*0, torch.sin(yaw/2)), -1)
    q_roll = torch.stack((torch.cos(roll/2), torch.sin(roll/2), roll*0, roll*0), -1)
    level, rolled = quaternion_rotation(q_yaw), quaternion_rotation(q_roll)
    target = level@rolled
    direction = level[:,:,0]
    # Geometry only: bend opposite the acceleration of the requested attitude.
    up = yaw.new_tensor((0.,0.,1.))
    bend_dir = up - 1.2*target[:,:,2]
    bend_dir = bend_dir/bend_dir.norm(dim=-1,keepdim=True)
    B = initial.position_limit
    half_span = config.half_span_fraction*B
    points = torch.stack((-half_span[:,None]*direction,
                          config.bend_fraction*B[:,None]*bend_dir,
                          half_span[:,None]*direction), 1)
    width = torch.minimum(B.new_full(B.shape, config.no_slowdown_half_width_m),
                          config.band_max_leg_fraction*half_span)
    mission = WaypointMission(points, target, level, direction, width)
    mission.validate(B, config)
    velocity_rng = torch.Generator().manual_seed((int(seed)^0x9A47F1) & ((1<<63)-1))
    drawn = 5*torch.rand(n,3,generator=velocity_rng,dtype=dtype)-2.5
    velocity = torch.where((initial.velocity == 0).all(-1,keepdim=True), drawn, initial.velocity)
    c = initial.thrust_coefficients
    desired = initial.mass[:,None]*9.81/4
    motor = (-c[...,1] + (c[...,1].square()+4*c[...,2]*(desired-c[...,0])).sqrt())/(2*c[...,2])
    if not bool(((motor >= initial.motor_min[:,None]) & (motor <= initial.motor_max[:,None])).all()):
        raise ValueError("sampled airframe has no unforced hover equilibrium")
    state = replace(initial, position=points[:,0], velocity=velocity, orientation=q_yaw,
                    omega=torch.zeros_like(initial.omega), motor=motor,
                    previous_action=torch.zeros_like(initial.previous_action),
                    previous_velocity=velocity[:,None].expand_as(initial.previous_velocity).clone())
    return state, mission


@torch.no_grad()
def advance_target(measured_position, mission, index, radius):
    """One position-only switch per acquisition; index2 (point3) is permanent.

    No threshold on attitude, speed or dwell. Use measurements, not hidden truth.
    """
    target = mission.points[torch.arange(len(index),device=index.device), index]
    hit = (measured_position-target).square().sum(-1) <= radius*radius
    return torch.where((index < 2) & hit, index+1, index)


def goal_packet(observation22, mission, index, passage_speed):
    """Same 23-value carrier/36-feature native GRU, NEW versioned semantics.

    current p, next p, final p, pending-midpoint-or-level R, fixed crossing v,
    is-final flag, passage-pending flag. NO moving p_ref, clock or countdown.
    Point3 can be commanded while the midpoint attitude obligation is still
    pending; this does not delay the position-only command switch.
    """
    n = len(index)
    rows = torch.arange(n, device=index.device)
    current = mission.points[rows, index]
    next_point = mission.points[rows, (index+1).clamp_max(2)]
    along = ((observation22[:,:3]-mission.points[:,1])*mission.direction).sum(-1)
    pending = (along <= mission.band_half_width) | (index < 2)
    rotation = torch.where(pending[:,None,None], mission.rotation, mission.level_rotation)
    velocity = torch.where(pending[:,None], passage_speed*mission.direction, 0.)
    return torch.cat((current, next_point, mission.points[:,2], rotation.flatten(1), velocity,
                      (index == 2).to(observation22)[:,None], pending.to(observation22)[:,None]), -1)

"""Public plane-task geometry and differentiable direction errors."""
from __future__ import annotations

import hashlib
import math
import random
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

if TYPE_CHECKING:
    from env_raptor import RaptorState
    from pass_stop import GoalConfig


def pass_region_radius(arm_length: torch.Tensor, radius_arms: float) -> torch.Tensor:
    """Keep the native arm scaling, with a0.1m minimum physical ball radius."""
    return (radius_arms*arm_length).clamp_min(.1)


def perpendicular_body_normal(heading: torch.Tensor) -> torch.Tensor:
    """Choose the most upward body z perpendicular to a unit travel direction."""
    horizontal = torch.hypot(heading[..., 0], heading[..., 1])
    safe_horizontal = horizontal.clamp_min(torch.finfo(heading.dtype).tiny)
    normal = torch.stack((-heading[..., 2]*(heading[..., 0]/safe_horizontal),
                          -heading[..., 2]*(heading[..., 1]/safe_horizontal), horizontal), -1)
    # Vertical travel has no preferred upward perpendicular; use world x.
    normal = torch.where((horizontal > 0)[..., None], normal,
                         heading.new_tensor((1., 0., 0.)))
    return F.normalize(normal, dim=-1)


@torch.no_grad()
def sample_plane_goals(initial: RaptorState, mission: GoalConfig
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = initial.noise_row.detach().cpu().tolist()
    spawn = initial.position.detach().cpu().double().tolist()
    arms = initial.arm_length.detach().cpu().double().tolist()
    radii = pass_region_radius(initial.arm_length, mission.region_radius_arms).detach().cpu().double().tolist()
    triples = []
    for row, entropy, arm, radius in zip(rows, spawn, arms, radii):
        key = repr(('plane-task-v1', mission.task_seed, row, entropy)).encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))

        def surface():
            point = [rng.uniform(-40*arm, 40*arm) for _ in range(3)]
            face = rng.randrange(6)
            point[face//2] = (-1 if face % 2 == 0 else 1)*40*arm
            return point

        first_gap = max(radius, 2*mission.radius_m)+1e-5
        last_gap = max(radius+mission.stop_position_m,
                       mission.radius_m+mission.stop_position_m)+1e-5
        for _ in range(256):
            start, stop = surface(), surface()
            via = [rng.uniform(-30*arm, 30*arm) for _ in range(3)]
            if (math.dist(start, via) > first_gap and math.dist(stop, via) > last_gap
                    and math.dist(start, stop) > 2*mission.stop_position_m+1e-5):
                break
        else:
            raise ValueError(f'plane-task geometry cannot fit scene {row}; no aircraft removed')
        triples.append((start, via, stop))
    points = initial.position.new_tensor(triples)
    heading = F.normalize(points[:, 2]-points[:, 1], dim=-1)
    # Body z is perpendicular to travel; its sign and body spin remain free.
    normal = perpendicular_body_normal(heading)
    q = F.normalize(torch.cat((1+normal[:, 2:3], -normal[:, 1:2], normal[:, :1],
                               torch.zeros_like(normal[:, :1])), -1), dim=-1)
    velocities = torch.zeros_like(points)
    velocities[:, 1] = math.hypot(*mission.velocities[1])*heading
    return points, q, velocities


def body_normal(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), -1)


def _normal_ratio(actual: torch.Tensor, target: torch.Tensor, tolerance_deg: float,
                  *, unoriented: bool) -> torch.Tensor:
    if unoriented:
        sign = torch.where((actual*target).sum(-1, keepdim=True) >= 0, 1., -1.)
        target = sign*target
    return (actual-target).norm(dim=-1)/(2*math.sin(math.radians(tolerance_deg)/2))


def plane_attitude_ratio(q: torch.Tensor, normal: torch.Tensor,
                         tolerance_deg: float) -> torch.Tensor:
    return _normal_ratio(body_normal(q), normal, tolerance_deg, unoriented=True)


def directed_velocity_ratio(velocity: torch.Tensor, direction: torch.Tensor,
                            minimum_speed: torch.Tensor, tolerance_deg: float) -> torch.Tensor:
    speed = velocity.norm(dim=-1)
    denominator = torch.maximum(speed, minimum_speed)
    unit = velocity/denominator[..., None]
    fraction = speed/denominator
    # At rest the direction is missing, not halfway between good and opposite.
    # The squared error is 4-2*r*(1+cos(angle)), monotone as speed decreases.
    missing = (1-fraction).clamp_min(0)
    squared = (unit-direction).square().sum(-1)+missing*(3+fraction)
    scale = 2*math.sin(math.radians(tolerance_deg)/2)
    epsilon = torch.finfo(velocity.dtype).eps
    return ((squared/(scale*scale)+epsilon)/(1+epsilon)).sqrt()


def velocity_z_orthogonality_ratio(q: torch.Tensor, velocity: torch.Tensor,
                                   minimum_speed: torch.Tensor, tolerance_deg: float) -> torch.Tensor:
    """One at the angular tolerance from velocity perpendicular to body z."""
    speed = velocity.norm(dim=-1)
    denominator = torch.maximum(speed, minimum_speed)
    unit = velocity/denominator[..., None]
    normal = F.normalize(body_normal(q), dim=-1)
    missing = (1-speed/denominator).clamp_min(0)
    error = (unit*normal).sum(-1).abs()+missing
    return error/math.sin(math.radians(min(tolerance_deg, 90.)))


def segment_plane_geometry(start: torch.Tensor, end: torch.Tensor, centre: torch.Tensor,
                            normal: torch.Tensor, radius: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    before = ((start-centre)*normal).sum(-1)
    after = ((end-centre)*normal).sum(-1)
    delta = after-before
    moving = delta != 0
    safe_delta = torch.where(moving, delta, torch.ones_like(delta))
    fraction = -before/safe_delta
    point = start+fraction[..., None]*(end-start)
    hit = (moving & (fraction >= 0) & (fraction <= 1)
           & ((point-centre).square().sum(-1) <= radius.square()))
    return hit, fraction.clamp(0., 1.), delta > 0

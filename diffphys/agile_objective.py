"""Continuous three-point losses through unchanged differentiable physics.

Targets and time weights are fixed before an episode. No crossing event,
success threshold or state-based phase switch controls the loss or curriculum.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch

from .env_raptor import RaptorSimulator, RaptorState

AGILE_OBJECTIVE_VERSION = "continuous-three-point-flatness-tracking-final-rest-v1"
GOAL_PACKET_DIM = 23
AGILE_STEP_COMPONENTS = (
    "tracking_position", "tracking_velocity", "midpoint_position",
    "midpoint_velocity", "midpoint_attitude", "action_delta", "boundary",
    "rest_position", "rest_velocity", "rest_attitude", "rest_omega",
)
AGILE_TERMINAL_COMPONENTS = (
    "final_position", "final_velocity", "final_attitude", "final_omega",
)
AGILE_COMPONENTS = AGILE_STEP_COMPONENTS + AGILE_TERMINAL_COMPONENTS


@dataclass(frozen=True)
class AgileLossConfig:
    """Weights and normalization scales, never success/failure tolerances."""
    tracking_position_weight: float = 1.0
    tracking_velocity_weight: float = .25
    midpoint_position_weight: float = 4.0
    midpoint_velocity_weight: float = 1.0
    midpoint_attitude_weight: float = 2.0
    terminal_position_weight: float = 1.0
    terminal_velocity_weight: float = .5
    terminal_attitude_weight: float = .25
    terminal_omega_weight: float = .05
    action_delta_weight: float = .02
    boundary_weight: float = 5.0
    final_weight: float = 2.0
    hold_weight: float = 1.0
    boundary_margin_fraction: float = .15
    midpoint_window_fraction: float = .06
    midpoint_position_scale_fraction: float = .15
    rest_position_scale_fraction: float = .04
    rest_velocity_scale_m_s: float = .3
    rest_attitude_scale_rad: float = .2
    rest_omega_scale_rad_s: float = .5
    velocity_scale_floor_m_s: float = .5
    smooth_epsilon: float = .01

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(field.name + " must be a finite number")
            if field.name.endswith("_weight"):
                if value < 0:
                    raise ValueError(field.name + " must be nonnegative")
            elif value <= 0:
                raise ValueError(field.name + " must be positive")
        for name in ("boundary_margin_fraction", "midpoint_window_fraction",
                     "midpoint_position_scale_fraction", "rest_position_scale_fraction"):
            if getattr(self, name) >= 1:
                raise ValueError(name + " must be less than one")
        if self.rest_attitude_scale_rad >= math.pi:
            raise ValueError("rest_attitude_scale_rad must be less than pi")


@dataclass(frozen=True)
class AgileSchedule:
    """Detached, batched targets [H+1,N,...] and normalized time weights."""
    position: torch.Tensor
    velocity: torch.Tensor
    rotation: torch.Tensor
    phase: torch.Tensor
    midpoint_weight: torch.Tensor
    rest_weight: torch.Tensor
    horizon: int
    dt: float


def _smooth_norm(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    squared = value.square().sum(-1)
    radius = torch.hypot(value.norm(dim=-1), value.new_full((), epsilon))
    return squared / (radius + epsilon)


def _attitude_cost(rotation: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth SO(3) error 1-cos(theta), stable at identical rotations."""
    return .25*(rotation-target).square().sum((-1, -2))


def _attitude_angle(rotation, target):
    # Only descriptive metrics use acos; it never participates in gradients.
    trace = (rotation*target).sum((-1, -2))
    return torch.acos(((trace-1.)*.5).clamp(-1., 1.))


@torch.no_grad()
def prepare_agile_schedule(mission, config: AgileLossConfig, *, horizon: int, dt: float) -> AgileSchedule:
    if type(horizon) is not int or horizon < 1 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("horizon and dt must be positive")
    if not 0 < mission.hold_seconds < horizon*dt:
        raise ValueError("horizon must include motion and the final rest interval")
    if (not bool(torch.isfinite(mission.duration).all())
            or not bool((mission.duration > 0).all())
            or bool((mission.duration+mission.hold_seconds > horizon*dt+1e-6).any())):
        raise ValueError("mission motion plus rest must fit the planned horizon")
    time = torch.arange(horizon+1, device=mission.duration.device,
                        dtype=mission.duration.dtype)[:, None]*dt
    reference = mission.nominal_reference(time)
    phase = (time/mission.duration).clamp(0., 1.)
    # Follow the reference near the middle: distance to a fixed waypoint
    # throughout a time window would instead encourage the aircraft to slow down.
    middle = torch.exp(-.5*((time[1:]/mission.duration-.5)
                           /config.midpoint_window_fraction).square())
    middle = middle/middle.sum(0, keepdim=True).clamp_min(torch.finfo(time.dtype).tiny)
    # This final-time window does not depend on whether the aircraft has arrived.
    u = ((time[1:, 0]-(horizon*dt-mission.hold_seconds))/mission.hold_seconds).clamp(0., 1.)
    rest = u.pow(3)*(10.+u*(-15.+6.*u))
    rest = rest/rest.sum().clamp_min(torch.finfo(time.dtype).tiny)
    return AgileSchedule(reference.position, reference.velocity, reference.rotation,
                         phase, middle, rest, horizon, dt)


def make_goal_packet(observation22: torch.Tensor, mission, time_seconds,
                     remaining_seconds, *, schedule: AgileSchedule | None = None,
                     step: int | None = None) -> torch.Tensor:
    """23 public world-coordinate task inputs; phase is a clock, not a decision.

    Current reference p(3), middle p(3), end p(3), middle R(9), middle v(3),
    normalized motion time(1), remaining seconds(1). The policy transforms these
    using measured p/R. Hidden force, motor truth and hover equilibrium attitude
    are absent from the packet.
    """
    n = len(mission.start)
    if observation22.shape != (n, 22):
        raise ValueError("expected [scene,22] observation")
    if schedule is None:
        reference = mission.nominal_reference(time_seconds)
        position = reference.position
        phase = (torch.as_tensor(time_seconds, device=mission.duration.device,
                                 dtype=mission.duration.dtype)/mission.duration).clamp(0., 1.)
    else:
        if step is None or not 0 <= step <= schedule.horizon:
            raise ValueError("schedule requires a valid step")
        position, phase = schedule.position[step], schedule.phase[step]
    remaining = torch.as_tensor(remaining_seconds, device=observation22.device,
                                dtype=observation22.dtype)
    if remaining.numel() == 1:
        remaining = remaining.reshape(1).expand(n)
    if remaining.shape != (n,):
        raise ValueError("remaining_seconds must be scalar or [scene]")
    return torch.cat((position, mission.waypoint, mission.goal, mission.rotation.flatten(1),
                      mission.velocity, phase[:, None], remaining.clamp_min(0.)[:, None]), -1)


def _rest_terms(physical, mission, config):
    position = _smooth_norm((physical.position-mission.goal)
                           /(mission.position_scale*config.rest_position_scale_fraction)[:, None],
                           config.smooth_epsilon)
    velocity = _smooth_norm(physical.velocity/config.rest_velocity_scale_m_s, config.smooth_epsilon)
    angle_scale = 2.*math.sin(config.rest_attitude_scale_rad/2.)**2
    attitude = _attitude_cost(physical.rotation, mission.hover_rotation)/angle_scale
    omega = _smooth_norm(physical.omega/config.rest_omega_scale_rad_s, config.smooth_epsilon)
    return position, velocity, attitude, omega


def agile_step(before: RaptorState, after: RaptorState, active: torch.Tensor, mission,
               config: AgileLossConfig, *, step: int, horizon: int, dt: float,
               schedule: AgileSchedule | None = None):
    """Per-row smooth costs at a predetermined time, with no arrival decisions.

    Tracking/rest still penalize frozen failure states at each planned time.
    Early failure cannot eliminate their remaining loss. The original simulator
    stops evaluating the Actor and RK4 for failed rows as before.
    """
    if not 1 <= step <= horizon:
        raise ValueError("step must lie inside the planned horizon")
    if active.shape != before.step_index.shape or active.dtype != torch.bool:
        raise ValueError("active must match the original scene rows")
    if schedule is None:
        schedule = prepare_agile_schedule(mission, config, horizon=horizon, dt=dt)
    if schedule.horizon != horizon or schedule.dt != dt:
        raise ValueError("agile schedule horizon/dt mismatch")
    p_ref, v_ref, r_ref = (value[step] for value in
                          (schedule.position, schedule.velocity, schedule.rotation))
    speed_scale = mission.velocity.norm(dim=-1).clamp_min(config.velocity_scale_floor_m_s)
    p_error, v_error = after.position-p_ref, after.velocity-v_ref
    p = _smooth_norm(p_error/mission.position_scale[:, None], config.smooth_epsilon)
    local_p = _smooth_norm(p_error/(mission.position_scale
                                   *config.midpoint_position_scale_fraction)[:, None], config.smooth_epsilon)
    v = _smooth_norm(v_error/speed_scale[:, None], config.smooth_epsilon)
    rotation = after.rotation
    middle = schedule.midpoint_weight[step-1]
    rest = config.hold_weight*schedule.rest_weight[step-1]
    rp, rv, rr, rw = _rest_terms(after, mission, config)
    delta = torch.where(active[:, None], after.previous_action-before.previous_action, 0.)
    warning = ((after.position.abs()/after.position_limit[:, None]
                -(1.-config.boundary_margin_fraction))/config.boundary_margin_fraction
               ).clamp_min(0.).square().sum(-1)
    components = {
        "tracking_position": config.tracking_position_weight*p/horizon,
        "tracking_velocity": config.tracking_velocity_weight*v/horizon,
        "midpoint_position": config.midpoint_position_weight*middle*local_p,
        "midpoint_velocity": config.midpoint_velocity_weight*middle*v,
        "midpoint_attitude": config.midpoint_attitude_weight*middle*_attitude_cost(rotation, r_ref),
        "action_delta": config.action_delta_weight*_smooth_norm(delta, config.smooth_epsilon)/horizon,
        "boundary": config.boundary_weight*warning/horizon,
        "rest_position": rest*config.terminal_position_weight*rp,
        "rest_velocity": rest*config.terminal_velocity_weight*rv,
        "rest_attitude": rest*config.terminal_attitude_weight*rr,
        "rest_omega": rest*config.terminal_omega_weight*rw,
    }
    with torch.no_grad():
        # Timestamp selection is independent of flight state, with no threshold
        # turning the measured residual into a success/failure signal.
        at_middle = (torch.round(mission.duration/(2.*dt)).long() == step)
        selected = at_middle.to(after.position.dtype)
        metrics = {
            "midpoint_position_sq": selected*(after.position-mission.waypoint).square().sum(-1),
            "midpoint_velocity_sq": selected*(after.velocity-mission.velocity).square().sum(-1),
            "midpoint_attitude_sq": selected*_attitude_angle(rotation, mission.rotation).square(),
            "midpoint_speed": selected*after.velocity.norm(dim=-1),
            "reference_position_sq": p_error.square().sum(-1)/horizon,
            "reference_velocity_sq": v_error.square().sum(-1)/horizon,
            "rest_speed_sq": schedule.rest_weight[step-1]*after.velocity.square().sum(-1),
            "rest_omega_sq": schedule.rest_weight[step-1]*after.omega.square().sum(-1),
        }
    return components, metrics


def agile_terminal(physical: RaptorState, mission, config: AgileLossConfig) -> dict[str, torch.Tensor]:
    """Final residual once at H, including every original failed/frozen row."""
    p, v, rotation, omega = _rest_terms(physical, mission, config)
    return {
        "final_position": config.final_weight*config.terminal_position_weight*p,
        "final_velocity": config.final_weight*config.terminal_velocity_weight*v,
        "final_attitude": config.final_weight*config.terminal_attitude_weight*rotation,
        "final_omega": config.final_weight*config.terminal_omega_weight*omega,
    }


@torch.no_grad()
def agile_metrics(physical: RaptorState, mission, config: AgileLossConfig, *,
                  dt: float = .01, horizon: int | None = None, measurements=None) -> dict:
    """Continuous error reports, with no success labels or training-stage gates."""
    p = physical.position-mission.goal
    finite = all(bool(torch.isfinite(value).all()) for value in
                 (p, physical.velocity, physical.orientation, physical.omega))
    result = {
        "objective_version": AGILE_OBJECTIVE_VERSION,
        "scenario_count": len(p), "finite": finite,
        "final_position_rms_m": float(p.square().sum(-1).mean().sqrt()),
        "final_velocity_rms_m_s": float(physical.velocity.square().sum(-1).mean().sqrt()),
        "final_attitude_rms_rad": float(_attitude_angle(physical.rotation, mission.hover_rotation).square().mean().sqrt()),
        "final_omega_rms_rad_s": float(physical.omega.square().sum(-1).mean().sqrt()),
        "boundary_failure_rate": float(RaptorSimulator.terminated(physical).double().mean()),
        "reference_feasible_fraction": float(mission.feasible.double().mean()),
        "requested_difficulty": float(mission.requested_difficulty),
        "achieved_difficulty_mean": float(mission.difficulty.double().mean()),
    }
    if measurements is not None:
        mapping = {
            "midpoint_position_sq": "midpoint_position_rms_m",
            "midpoint_velocity_sq": "midpoint_velocity_rms_m_s",
            "midpoint_attitude_sq": "midpoint_attitude_rms_rad",
            "reference_position_sq": "reference_position_rms_m",
            "reference_velocity_sq": "reference_velocity_rms_m_s",
            "rest_speed_sq": "rest_velocity_rms_m_s",
            "rest_omega_sq": "rest_omega_rms_rad_s",
        }
        result.update({name: float(measurements[key].double().mean().sqrt()) for key, name in mapping.items()})
        result["midpoint_speed_mean_m_s"] = float(measurements["midpoint_speed"].double().mean())
    return result

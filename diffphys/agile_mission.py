"""Three-point missions without changing the sampled airframes or their boundary.

The reference is a quadratic spatial Bezier with a ninth-order time law.  The
motor audit is a *nominal sampled reference* audit, not a closed-loop guarantee:
the motion reference does not encode the hidden force/pulse realization.  The
reset and final hover attitude compensate the episode-constant force; pulses
remain an unchanged disturbance.  No aircraft is resampled or dropped when a
requested maneuver is too hard.  Its amplitude is reduced deterministically
and both the requested and the actual task are recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Optional

import torch

from .env_raptor import RaptorState

MISSION_VERSION = "agile-three-point-s9-actuator-audit-v1"
GRAVITY = 9.81
CONTROL_DT = .01
MID_SPEED_COEFFICIENT = 315. / 64.
MID_ACCELERATION_COEFFICIENT = 99225. / 2048.
GEOMETRY_LIMIT = 1
FRAME_LIMIT = 2
ROTOR_THRUST_LIMIT = 4
MOTOR_SETPOINT_LIMIT = 8
HOVER_EQUILIBRIUM_LIMIT = 16
NUMERICAL_LIMIT = 32
REASON_NAMES = {
    GEOMETRY_LIMIT: "reference_boundary", FRAME_LIMIT: "reference_attitude",
    ROTOR_THRUST_LIMIT: "individual_rotor_thrust",
    MOTOR_SETPOINT_LIMIT: "rk4_motor_setpoint",
    HOVER_EQUILIBRIUM_LIMIT: "constant_force_hover", NUMERICAL_LIMIT: "nonfinite",
}


@dataclass(frozen=True)
class AgileMissionConfig:
    tilt_degrees: float = 60.
    thrust_ratio: float = 1.2
    boundary_margin_fraction: float = .2
    half_span_fraction: float = .45
    bend_fraction: float = .2
    hold_seconds: float = 1.
    hover_fraction: float = .1
    yaw_degrees: Optional[float] = None
    center_fraction: tuple[float, float, float] = (0., 0., 0.)
    actuator_margin_fraction: float = .01
    shrink_factors: tuple[float, ...] = (1., .8, .6, .4, .25, .125, .0625, .03125, 0.)
    screen_batch_size: int = 128

    def __post_init__(self):
        for name in ("tilt_degrees", "thrust_ratio", "boundary_margin_fraction",
                     "half_span_fraction", "bend_fraction", "hold_seconds",
                     "hover_fraction", "actuator_margin_fraction"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(name + " must be finite")
        if not 0 <= self.tilt_degrees < 89:
            raise ValueError("tilt_degrees must be in [0,89); inverted maneuvers are not this family")
        if self.thrust_ratio <= 0 or self.hold_seconds <= 0:
            raise ValueError("thrust_ratio and hold_seconds must be positive")
        if not 0 <= self.boundary_margin_fraction < 1:
            raise ValueError("boundary_margin_fraction must be in [0,1)")
        if not 0 < self.half_span_fraction < 1 or not 0 < self.bend_fraction < 1:
            raise ValueError("half_span_fraction and bend_fraction must be in (0,1)")
        if not 0 <= self.hover_fraction <= 1 or not 0 <= self.actuator_margin_fraction < .5:
            raise ValueError("invalid hover fraction or actuator margin")
        if self.yaw_degrees is not None and not math.isfinite(self.yaw_degrees):
            raise ValueError("yaw_degrees must be finite or None")
        if len(self.center_fraction) != 3 or any(
                not math.isfinite(x) or abs(x) >= 1-self.boundary_margin_fraction
                for x in self.center_fraction):
            raise ValueError("center_fraction must be a finite three-vector strictly inside the margin")
        if (not self.shrink_factors or self.shrink_factors[0] != 1.
                or self.shrink_factors[-1] != 0.
                or any(not math.isfinite(x) or not 0 <= x <= 1 for x in self.shrink_factors)
                or any(a <= b for a, b in zip(self.shrink_factors, self.shrink_factors[1:]))):
            raise ValueError("shrink_factors must strictly descend from 1 to 0")
        if (not isinstance(self.screen_batch_size, int) or isinstance(self.screen_batch_size, bool)
                or self.screen_batch_size < 1):
            raise ValueError("screen_batch_size must be a positive integer")


@dataclass(frozen=True)
class AgileReference:
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    jerk: torch.Tensor
    snap: torch.Tensor
    rotation: torch.Tensor
    omega: torch.Tensor
    omega_dot: torch.Tensor


@dataclass(frozen=True)
class AgileMission:
    start: torch.Tensor
    waypoint: torch.Tensor
    goal: torch.Tensor
    rotation: torch.Tensor
    velocity: torch.Tensor
    direction: torch.Tensor
    duration: torch.Tensor
    hold_seconds: float
    position_scale: torch.Tensor
    difficulty: torch.Tensor
    requested_difficulty: float
    feasible: torch.Tensor
    is_hover: torch.Tensor  # Intentional hover replay only; failed maneuvers stay agile rows.
    report: dict
    half_span: torch.Tensor
    hover_rotation: torch.Tensor
    center: torch.Tensor
    bend: torch.Tensor
    limit_reason: torch.Tensor
    hover_feasible: torch.Tensor

    def to(self, device, dtype=None) -> AgileMission:
        return replace(self, **{
            f.name: value.to(device=device, dtype=(dtype if value.is_floating_point() else None))
            for f in fields(self) if isinstance(value := getattr(self, f.name), torch.Tensor)
        })

    def select(self, indices: torch.Tensor) -> AgileMission:
        """Select rows; the report remains provenance of the original complete bank."""
        return replace(self, **{
            f.name: value.index_select(0, indices.to(value.device))
            for f in fields(self) if isinstance(value := getattr(self, f.name), torch.Tensor)
        })

    def nominal_reference(self, time: float | torch.Tensor) -> AgileReference:
        """Unforced geometric reference, not a privileged disturbed reference.

        A scalar, [N] per-aircraft times, or [K,1]/[K,N] grid is accepted.  The
        final *task* attitude is hover_rotation; under constant force this can
        differ from the unforced reference attitude returned here.
        """
        time = torch.as_tensor(time, device=self.duration.device, dtype=self.duration.dtype)
        p, v, a, j, snap = reference_kinematics(
            self.center, self.half_span, self.bend, self.direction, self.duration, time)
        rotation, omega, omega_dot, _ = flatness_attitude(a, j, snap, self.direction)
        return AgileReference(p, v, a, j, snap, rotation, omega, omega_dot)


def smootherstep9(u: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """S9 and its first four derivatives with respect to normalized time.

    Symmetry avoids cancellation at the right endpoint.  S9 has zero velocity,
    acceleration, jerk and snap at both endpoints.
    """
    x = torch.minimum(u, 1-u)
    x2, x3, x4 = x.square(), x.pow(3), x.pow(4)
    s = x.pow(5) * (126 + x*(-420 + x*(540 + x*(-315 + 70*x))))
    q = torch.where(u <= .5, s, 1-s)
    q1 = 630*x4*(1-x).pow(4)
    q2_left = 2520*x3*(1-x).pow(3)*(1-2*x)
    q3 = 2520*x2*(1-x).square()*(3-14*x+14*x2)
    q4_left = 15120*x*(1-x)*(1-2*x)*(1-7*x+7*x2)
    q2 = torch.where(u <= .5, q2_left, -q2_left)
    q4 = torch.where(u <= .5, q4_left, -q4_left)
    return q, q1, q2, q3, q4


def reference_kinematics(center, half_span, bend, direction, duration, time):
    """Position through snap, with an exact rest extension outside [0,T]."""
    u = (time / duration).clamp(0., 1.)
    q, q1, q2, q3, q4 = smootherstep9(u)
    moving = (time >= 0) & (time <= duration)
    q1, q2, q3, q4 = (
        torch.where(moving, value / duration.pow(order), 0.)
        for order, value in enumerate((q1, q2, q3, q4), 1))
    forward = half_span[..., None] * direction
    tangent = 2*forward + 4*(1-2*q)[..., None]*bend
    p = center + (2*q-1)[..., None]*forward + (4*q*(1-q))[..., None]*bend
    v = tangent*q1[..., None]
    a = tangent*q2[..., None] - 8*bend*q1.square()[..., None]
    j = tangent*q3[..., None] - 24*bend*(q1*q2)[..., None]
    snap = tangent*q4[..., None] - bend*(32*q1*q3 + 24*q2.square())[..., None]
    return p, v, a, j, snap


def _normalize_with_derivatives(value, first, second):
    norm = value.norm(dim=-1).clamp_min(1e-12)
    unit = value/norm[..., None]
    dnorm = (unit*first).sum(-1)
    ddnorm = (first.square().sum(-1)+(value*second).sum(-1)-dnorm.square())/norm
    dunit = (first-unit*dnorm[..., None])/norm[..., None]
    ddunit = (second-unit*ddnorm[..., None]-2*dunit*dnorm[..., None])/norm[..., None]
    return unit, dunit, ddunit


def _vee(matrix):
    return .5*torch.stack((matrix[..., 2, 1]-matrix[..., 1, 2],
                           matrix[..., 0, 2]-matrix[..., 2, 0],
                           matrix[..., 1, 0]-matrix[..., 0, 1]), -1)


def flatness_attitude(acceleration, jerk, snap, direction):
    """Continuous fixed-yaw frame and exact angular rates of the unforced path."""
    up = acceleration.new_tensor((0., 0., 1.))
    force = acceleration + GRAVITY*up
    b3, db3, ddb3 = _normalize_with_derivatives(force, jerk, snap)
    # This x axis has horizontal projection along direction, hence fixed Euler
    # yaw when force_z>0.  Do not confuse body tilt with velocity direction.
    x, dx, ddx = (value[..., 2:3]*direction - (value*direction).sum(-1, keepdim=True)*up
                  for value in (force, jerk, snap))
    b1, db1, ddb1 = _normalize_with_derivatives(x, dx, ddx)
    b2 = torch.linalg.cross(b3, b1, dim=-1)
    db2 = torch.linalg.cross(db3, b1, dim=-1)+torch.linalg.cross(b3, db1, dim=-1)
    ddb2 = (torch.linalg.cross(ddb3, b1, dim=-1)
            +2*torch.linalg.cross(db3, db1, dim=-1)+torch.linalg.cross(b3, ddb1, dim=-1))
    rotation = torch.stack((b1, b2, b3), -1)
    first = torch.stack((db1, db2, db3), -1)
    second = torch.stack((ddb1, ddb2, ddb3), -1)
    omega = _vee(rotation.transpose(-1, -2) @ first)
    omega_dot = _vee(rotation.transpose(-1, -2) @ second)
    valid = ((force[..., 2] > .02*GRAVITY) & (force.norm(dim=-1) > .05*GRAVITY)
             & (x.norm(dim=-1) > 1e-8))
    return rotation, omega, omega_dot, valid


def rk4_motor_setpoint(now, following, rising, falling, *, dt=CONTROL_DT):
    """Exact constant setpoint for one step of the source motor RK4 equation.

    All source time constants satisfy dt/tau <= 1/3, so the chosen rising or
    falling branch remains unchanged at the four intermediate RK4 stages.
    """
    tau = torch.where(following >= now, rising, falling)
    x = dt/tau
    one_minus_r = x*(1-x/2+x.square()/6-x.pow(3)/24)
    return now+(following-now)/one_minus_r


def _motor_from_thrust(thrust, coefficients):
    delta = thrust-coefficients[..., 0]
    c1, c2 = coefficients[..., 1], coefficients[..., 2]
    discriminant = (c1.square()+4*c2*delta).clamp_min(0.)
    # The stable quadratic root also handles constant/linear thrust curves.
    return 2*delta/(c1+discriminant.sqrt()).clamp_min(1e-15)


def _actuators(initial):
    names = ("mass", "inertia", "thrust_coefficients", "motor_time_rising",
             "motor_time_falling", "motor_min", "motor_max", "external_force", "external_torque")
    values = {name: getattr(initial, name).detach().to("cpu", torch.float64) for name in names}
    positions = initial.rotor_positions.detach().to("cpu", torch.float64)
    km = initial.rotor_torque_constant.detach().to("cpu", torch.float64)
    mixer = torch.stack((torch.ones_like(km), positions[..., 1], -positions[..., 0],
                         km*km.new_tensor((-1., 1., -1., 1.))), 1)
    values["inverse_mixer"] = torch.linalg.inv(mixer)
    c = values["thrust_coefficients"]
    low, high = values["motor_min"][:, None], values["motor_max"][:, None]
    values["minimum_thrust"] = c[..., 0]+c[..., 1]*low+c[..., 2]*low.square()
    values["maximum_thrust"] = c[..., 0]+c[..., 1]*high+c[..., 2]*high.square()
    return values


def _hover_equilibrium(actuators, direction):
    mass = actuators["mass"]
    force = mass[:, None]*mass.new_tensor((0., 0., GRAVITY))-actuators["external_force"]
    magnitude = force.norm(dim=-1)
    up = torch.where((magnitude > 1e-12)[:, None], force/magnitude.clamp_min(1e-12)[:, None],
                     force.new_tensor((0., 0., 1.)))
    # A hover under a very unusual force can have undefined Euler yaw.  Choose
    # a finite orthogonal heading there; actuator bounds still decide validity.
    alternate = torch.stack((-direction[:, 1], direction[:, 0], direction[:, 2]), -1)
    heading = torch.where(((up*direction).sum(-1).abs() > .99)[:, None], alternate, direction)
    y = torch.linalg.cross(up, heading, dim=-1)
    y = y/y.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x = torch.linalg.cross(y, up, dim=-1)
    rotation = torch.stack((x, y, up), -1)
    wrench = torch.cat((magnitude[:, None], -actuators["external_torque"]), -1)
    thrust = (actuators["inverse_mixer"] @ wrench[..., None]).squeeze(-1)
    valid = (torch.isfinite(thrust).all(-1) & (magnitude > 1e-12)
             & (thrust >= actuators["minimum_thrust"]-1e-12).all(-1)
             & (thrust <= actuators["maximum_thrust"]+1e-12).all(-1))
    motor = _motor_from_thrust(thrust, actuators["thrust_coefficients"])
    # An impossible disturbance is retained and flagged; clipping only keeps
    # the initial actuator state physically representable and does not pass it.
    motor = torch.nan_to_num(motor).clamp(actuators["motor_min"][:, None],
                                        actuators["motor_max"][:, None])
    return rotation, motor, valid


def _rotation_quaternion(rotation):
    r = rotation
    terms = torch.stack((1+r[:, 0, 0]+r[:, 1, 1]+r[:, 2, 2],
                         1+r[:, 0, 0]-r[:, 1, 1]-r[:, 2, 2],
                         1-r[:, 0, 0]+r[:, 1, 1]-r[:, 2, 2],
                         1-r[:, 0, 0]-r[:, 1, 1]+r[:, 2, 2]), -1).clamp_min(0.)
    candidates = torch.stack((
        torch.stack((terms[:, 0], r[:, 2, 1]-r[:, 1, 2], r[:, 0, 2]-r[:, 2, 0], r[:, 1, 0]-r[:, 0, 1]), -1),
        torch.stack((r[:, 2, 1]-r[:, 1, 2], terms[:, 1], r[:, 1, 0]+r[:, 0, 1], r[:, 0, 2]+r[:, 2, 0]), -1),
        torch.stack((r[:, 0, 2]-r[:, 2, 0], r[:, 1, 0]+r[:, 0, 1], terms[:, 2], r[:, 1, 2]+r[:, 2, 1]), -1),
        torch.stack((r[:, 1, 0]-r[:, 0, 1], r[:, 2, 0]+r[:, 0, 2], r[:, 2, 1]+r[:, 1, 2], terms[:, 3]), -1)), 1)
    candidates = candidates/(2*terms.sqrt().clamp_min(.1)[..., None])
    q = candidates[torch.arange(len(r)), terms.argmax(-1)]
    q = q/q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where((q[:, :1] < 0), -q, q)


def _screen_reference(actuators, center, half_span, bend, direction, duration, bound,
                      config):
    """Check every 100 Hz reference sample and the required rotor setpoints."""
    count = len(duration)
    controls = torch.stack((center-half_span[:, None]*direction, center+2*bend,
                            center+half_span[:, None]*direction), 1)
    geometry_ok = (controls.abs() <= ((1-config.boundary_margin_fraction)*bound)[:, None, None]).all((1, 2))
    time = torch.arange(int(round(float(duration.max())/CONTROL_DT))+1,
                        dtype=duration.dtype)[:, None]*CONTROL_DT
    _, _, a, j, snap = reference_kinematics(center, half_span, bend, direction, duration, time)
    rotation, omega, omega_dot, frame_ok = flatness_attitude(a, j, snap, direction)
    del rotation
    force = a+a.new_tensor((0., 0., GRAVITY))
    total_thrust = force.norm(dim=-1)*actuators["mass"]
    inertia = actuators["inertia"]
    torque = inertia*omega_dot+torch.linalg.cross(omega, inertia*omega, dim=-1)
    wrench = torch.cat((total_thrust[..., None], torque), -1)
    thrust = torch.einsum("nij,tnj->tni", actuators["inverse_mixer"], wrench)
    reserve = config.actuator_margin_fraction*(actuators["maximum_thrust"]-actuators["minimum_thrust"])
    thrust_ok = ((thrust >= actuators["minimum_thrust"]+reserve-1e-12)
                 & (thrust <= actuators["maximum_thrust"]-reserve+1e-12)).all((0, 2))
    motor = _motor_from_thrust(thrust, actuators["thrust_coefficients"])
    setpoint = rk4_motor_setpoint(motor[:-1], motor[1:], actuators["motor_time_rising"],
                                 actuators["motor_time_falling"])
    motor_reserve = config.actuator_margin_fraction*(actuators["motor_max"]-actuators["motor_min"])[:, None]
    setpoint_ok = ((setpoint >= actuators["motor_min"][:, None]+motor_reserve-1e-12)
                   & (setpoint <= actuators["motor_max"][:, None]-motor_reserve+1e-12)).all((0, 2))
    finite = (torch.isfinite(wrench).all((0, 2)) & torch.isfinite(thrust).all((0, 2))
              & torch.isfinite(setpoint).all((0, 2)))
    reason = torch.zeros(count, dtype=torch.long)
    for flag, ok in ((GEOMETRY_LIMIT, geometry_ok), (FRAME_LIMIT, frame_ok.all(0)),
                     (ROTOR_THRUST_LIMIT, thrust_ok), (MOTOR_SETPOINT_LIMIT, setpoint_ok),
                     (NUMERICAL_LIMIT, finite)):
        reason |= (~ok).long()*flag
    return reason == 0, reason


def _summary(value, mask=None):
    value = value if mask is None else value[mask]
    if not value.numel():
        return {"min": None, "mean": None, "max": None}
    return {"min": float(value.min()), "mean": float(value.mean()), "max": float(value.max())}


@torch.no_grad()
def sample_agile_mission(initial: RaptorState, config: AgileMissionConfig, *, seed: int,
                         difficulty: float, horizon: int) -> tuple[RaptorState, AgileMission]:
    """Retask an already sampled bank; no aircraft or disturbance redraws occur."""
    if not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError("difficulty must be in [0,1]")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    hold_steps = math.ceil(config.hold_seconds/CONTROL_DT-1e-10)
    maximum_steps = 2*((horizon-hold_steps)//2)
    if maximum_steps < 4:
        raise ValueError("horizon must allow at least four motion steps plus the requested hold")
    if bool((initial.step_index != 0).any()):
        raise ValueError("agile missions must be attached at reset")
    n = len(initial.mass)
    if n < 1:
        raise ValueError("mission bank must be nonempty")
    device, dtype = initial.position.device, initial.position.dtype
    # One independent stream, one fixed-size draw.  Backoff consumes no RNG.
    generator = torch.Generator(device="cpu").manual_seed((int(seed)^0xA611EF17)&((1<<63)-1))
    draw = torch.rand(n, 3, generator=generator, dtype=torch.float64)
    yaw = ((draw[:, 0]*2-1)*math.pi if config.yaw_degrees is None
           else torch.full((n,), math.radians(config.yaw_degrees), dtype=torch.float64))
    direction = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), -1)
    side = torch.where(draw[:, 1] < .5, -1., 1.)
    lateral = side[:, None]*torch.stack((-yaw.sin(), yaw.cos(), torch.zeros_like(yaw)), -1)
    planned_hover = (draw[:, 2] < config.hover_fraction) | (difficulty == 0)
    bound = initial.position_limit.detach().to("cpu", torch.float64)
    if not bool((torch.isfinite(bound) & (bound > 0)).all()):
        raise ValueError("original position limits must be finite and positive")
    center = bound[:, None]*bound.new_tensor(config.center_fraction)
    base_span = config.half_span_fraction*bound
    theta = math.radians(config.tilt_degrees)
    d_direction = (config.thrust_ratio*math.sin(theta)*lateral
                   + (1-config.thrust_ratio*math.cos(theta))*bound.new_tensor((0., 0., 1.)))
    d_norm = d_direction.norm(dim=-1)
    # Exact upright mg midpoint has zero curvature; still permit a straight
    # three-point translation with the full available motion time.
    base_bend = config.bend_fraction*bound[:, None]*d_direction/d_norm.clamp_min(1e-12)[:, None]
    desired_duration = (MID_ACCELERATION_COEFFICIENT*config.bend_fraction*bound
                        / (GRAVITY*d_norm.clamp_min(1e-12))).sqrt()
    desired_duration = torch.where(d_norm > 1e-12, desired_duration,
                                    torch.full_like(desired_duration, maximum_steps*CONTROL_DT))
    motion_steps = (2*torch.ceil(desired_duration/(2*CONTROL_DT))).long().clamp(4, maximum_steps)
    duration = motion_steps.to(torch.float64)*CONTROL_DT
    actuators = _actuators(initial)
    hover_rotation, hover_motor, hover_ok = _hover_equilibrium(actuators, direction)
    used = torch.zeros(n, dtype=torch.float64)
    accepted = planned_hover.clone()
    first_reason = torch.zeros(n, dtype=torch.long)
    first_checked = torch.zeros(n, dtype=torch.bool)
    for factor in config.shrink_factors:
        pending = (~accepted & ~planned_hover).nonzero(as_tuple=True)[0]
        if not pending.numel():
            break
        amplitude = difficulty*factor
        for rows in pending.split(config.screen_batch_size):
            subset = {name: value.index_select(0, rows) for name, value in actuators.items()}
            ok, reason = _screen_reference(subset, center[rows], amplitude*base_span[rows],
                                           amplitude*base_bend[rows], direction[rows], duration[rows],
                                           bound[rows], config)
            first = ~first_checked[rows]
            first_reason[rows[first]] = reason[first]
            first_checked[rows] = True
            passing = rows[ok]
            used[passing] = amplitude
            accepted[passing] = True
    # A requested maneuver that only passes at zero amplitude is a failed
    # maneuver, not an extra successful hover example.
    feasible = hover_ok & accepted & (planned_hover | (used > 0))
    stationary = planned_hover | (used == 0)
    first_reason |= (~hover_ok).long()*HOVER_EQUILIBRIUM_LIMIT
    span, bend = used*base_span, used[:, None]*base_bend
    start, waypoint, goal = center-span[:, None]*direction, center+bend, center+span[:, None]*direction
    velocity = (MID_SPEED_COEFFICIENT*span/duration)[:, None]*direction
    _, _, mid_a, mid_j, mid_snap = reference_kinematics(center, span, bend, direction, duration, duration/2)
    rotation, _, _, _ = flatness_attitude(mid_a, mid_j, mid_snap, direction)
    # Even an intentional hover/fallback keeps a public unforced reference.
    # Sending the force-balanced hover_rotation as the public middle target
    # would reveal the hidden constant-force direction through the goal packet.
    # hover_rotation belongs only to physical reset and truth-based end losses.
    moving = ~stationary
    tilt = torch.rad2deg(torch.acos(rotation[:, 2, 2].clamp(-1., 1.)))
    actual_thrust_ratio = (mid_a+mid_a.new_tensor((0., 0., GRAVITY))).norm(dim=-1)/GRAVITY
    report = {
        "version": MISSION_VERSION, "aircraft_rows": n, "aircraft_rows_retained": n,
        "aircraft_resampled": 0, "horizon": horizon, "hold_seconds": config.hold_seconds,
        "requested_difficulty": float(difficulty), "requested_tilt_degrees": config.tilt_degrees,
        "requested_mid_thrust_ratio": config.thrust_ratio,
        "planned_hover_count": int(planned_hover.sum()), "moving_count": int(moving.sum()),
        "fallback_hover_count": int((stationary & ~planned_hover).sum()),
        "feasible_count": int(feasible.sum()), "infeasible_count": int((~feasible).sum()),
        "difficulty_capped_count": int(((used < difficulty) & ~planned_hover).sum()),
        "time_capped_count": int((desired_duration > maximum_steps*CONTROL_DT).sum()),
        "actual_difficulty": _summary(used, ~planned_hover),
        "actual_mid_tilt_degrees": _summary(tilt, moving),
        "actual_mid_speed_m_s": _summary(velocity.norm(dim=-1), moving),
        "actual_mid_thrust_ratio": _summary(actual_thrust_ratio, moving),
        "motion_duration_seconds": _summary(duration),
        "initial_rejection_counts": {name: int(((first_reason & flag) != 0).sum())
                                     for flag, name in REASON_NAMES.items()},
        "boundary": "unchanged world-fixed position_limit; reference Bezier control hull inside margin",
        "actuator_screen": "unforced reference at every 100 Hz grid sample; individual rotor thrust and exact motor RK4 setpoints",
        "difficulty_selection": "first passing descending amplitude; not a global maximum certificate",
        "constant_force": "unchanged; compensated only in initial/final hover, not in maneuver targets",
        "pulses": "unchanged; no future pulse data used in goals, resets or feasibility screen",
        "guarantee": "no continuous-time, disturbed-tracking or learned-control guarantee",
    }
    def output(value):
        return value.to(device=device, dtype=(dtype if value.is_floating_point() else None))
    mission = AgileMission(
        start=output(start), waypoint=output(waypoint), goal=output(goal), rotation=output(rotation),
        velocity=output(velocity), direction=output(direction), duration=output(duration),
        hold_seconds=config.hold_seconds, position_scale=output(bound), difficulty=output(used),
        requested_difficulty=float(difficulty), feasible=output(feasible), is_hover=output(planned_hover),
        report=report, half_span=output(span), hover_rotation=output(hover_rotation), center=output(center),
        bend=output(bend), limit_reason=output(first_reason), hover_feasible=output(hover_ok))
    zero = torch.zeros_like(initial.velocity)
    hover_initial = replace(initial, position=mission.start, velocity=zero, omega=zero.clone(),
                            orientation=output(_rotation_quaternion(hover_rotation)), motor=output(hover_motor),
                            previous_action=torch.zeros_like(initial.previous_action),
                            previous_velocity=torch.zeros_like(initial.previous_velocity))
    return hover_initial, mission

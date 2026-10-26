"""Typed pose/velocity goals with an entry-traverse-exit region and temporal loss.

This task is an opt-in addition to master's native GRU/RK4/Adam pipeline.
The controller switches by measured position. Losses use true states and a PASS
speed floor. Hard switches are not assigned fictitious derivatives.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from env_raptor import RaptorSimulator, quaternion_rotation
from response_noise import measured_observation, disturbance_report
from response_policy import (ResponseMotorPolicy, ResponsePolicyConfig,
                             ResponsePolicyState, ResponsePolicyOutput, body_vector)
from pass_stop_physics import (advance_phase, segment_ball_geometry,
                               interpolate_quaternion, interval_kinematics,
                               instantaneous_acceleration)
from response_task import (TaskLossConfig, ResponseClosedLoopState, rollout,
                           smooth_l2, tensors_finite)
from compiled_execution import tensor_kernel

VERSION = "pass-region-min-speed-v3"
LEGACY_ARCHITECTURE = "gru26-hidden-only-typed-goal-motor-policy-v1"
GOAL_FEATURE_DIM = 69
ARCHITECTURE = "gru69-hidden-only-pose-velocity-region-v2"


@dataclass(frozen=True)
class GoalConfig:
    # Coordinates are fractions of the ORIGINAL world-centred per-axis limit.
    points: tuple = ((-.7, 0., 0.), (0., .38, 0.), (.7, 0., 0.))
    relative_to_boundary: bool = True
    via_quaternion: tuple = (0.6087614290087207, 0.7933533402912352, 0., 0.)
    stop_quaternion: tuple = (1., 0., 0., 0.)
    start_quaternion: tuple = (1., 0., 0., 0.)
    # Explicit task choices, NOT a claimed dynamically feasible optimum.
    velocities: tuple = ((0., 0., 0.), (2., 0., 0.), (0., 0., 0.))
    region_radius_arms: float = 2.
    pass_speed_scale_m_s: float = .5  # Loss normalization, never slack below the speed floor.
    velocity_weight: float = 10.
    direction_weight: float = .01
    acceleration_weight: float = 0.
    jerk_weight: float = 0.
    stop_acceleration_m_s2: float = .5
    stop_angular_acceleration_rad_s2: float = 2.
    stop_motor_rate_per_s: float = .1
    radius_m: float = .1
    pose_tolerance_deg: float = 1.
    stop_position_m: float = .1
    stop_velocity_m_s: float = .1
    stop_pose_deg: float = 5.
    stop_omega_rad_s: float = .5
    point_weight: float = 10.
    time_weight: float = 1.
    smooth_weight: float = .01
    version: str = VERSION

    def __post_init__(self):
        object.__setattr__(self, 'points', tuple(tuple(float(v) for v in p) for p in self.points))
        for name in ('start_quaternion', 'via_quaternion', 'stop_quaternion'):
            q = tuple(float(v) for v in getattr(self, name))
            if len(q) != 4 or not all(math.isfinite(v) for v in q) or abs(sum(v*v for v in q)-1) > 1e-6:
                raise ValueError(name + ' must be a unit wxyz quaternion')
            object.__setattr__(self, name, q)
        object.__setattr__(self, 'velocities', tuple(tuple(float(v) for v in a) for a in self.velocities))
        if (len(self.velocities) != 3 or any(len(v) != 3 or not all(math.isfinite(a) for a in v)
                                            for v in self.velocities)):
            raise ValueError('three finite world-frame target velocity vectors are required')
        if any(a != 0 for i in (0, 2) for a in self.velocities[i]):
            raise ValueError('START and STOP velocities must be zero; their direction is undefined')
        if sum(a*a for a in self.velocities[1]) <= 0:
            raise ValueError('PASS velocity must be nonzero')
        if self.version != VERSION or type(self.relative_to_boundary) is not bool:
            raise ValueError('unsupported task version/coordinate convention')
        if len(self.points) != 3 or any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in self.points):
            raise ValueError('exactly three finite positions are required')
        for name in ('radius_m', 'pose_tolerance_deg', 'stop_position_m', 'stop_velocity_m_s',
                     'stop_pose_deg', 'stop_omega_rad_s', 'point_weight', 'time_weight',
                     'region_radius_arms', 'pass_speed_scale_m_s', 'velocity_weight',
                     'stop_acceleration_m_s2', 'stop_angular_acceleration_rad_s2',
                     'stop_motor_rate_per_s'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + ' must be positive and finite')
        if max(self.pose_tolerance_deg, self.stop_pose_deg) >= 180:
            raise ValueError('angular tolerance must be less than 180 degrees')
        for name in ('smooth_weight', 'direction_weight',
                     'acceleration_weight', 'jerk_weight'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        # A stationary ordinary quadrotor cannot support a persistently tilted thrust axis.
        if any(abs(q[1])+abs(q[2]) > 1e-6 for q in (self.start_quaternion, self.stop_quaternion)):
            raise ValueError('START/STOP must be level (yaw allowed) for an ordinary quadrotor')


@dataclass(frozen=True)
class PassStopLossConfig(TaskLossConfig):
    mission: dict | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.mission is None:
            raise ValueError('typed task requires a mission')
        GoalConfig(**self.mission)


def loss_config(value):
    return (PassStopLossConfig if 'mission' in value else TaskLossConfig)(**value)


def policy_class(config):
    return GoalMotorPolicy if isinstance(config, PassStopLossConfig) else ResponseMotorPolicy


def objective_version(config):
    from response_task import TASK_OBJECTIVE_VERSION
    return VERSION if isinstance(config, PassStopLossConfig) else TASK_OBJECTIVE_VERSION


def architecture(config):
    from response_policy import ARCHITECTURE as HOVER_ARCHITECTURE
    return ARCHITECTURE if isinstance(config, PassStopLossConfig) else HOVER_ARCHITECTURE


@dataclass(frozen=True)
class GoalState(ResponsePolicyState):
    via: torch.Tensor
    stop: torch.Tensor
    via_rotation: torch.Tensor
    stop_rotation: torch.Tensor
    radius: torch.Tensor
    phase: torch.Tensor
    previous_position: torch.Tensor
    start: torch.Tensor
    start_rotation: torch.Tensor
    target_velocities: torch.Tensor

    @property
    def is_stop(self):
        return self.phase == 2


class GoalMotorPolicy(ResponseMotorPolicy):
    """Native GRU64 + Linear4 only; all three fixed pose/velocity requirements.

    The first 26 columns keep the previous AFT feature layout for weights-only
    migration. Added columns: start/via positions(6), all relative attitudes(18),
    all target velocities(9), measured orientation(6), phase onehot(3), radius(1).
    No motor truth, inertial parameters, reference clock or planner output.
    """
    def __init__(self, config=ResponsePolicyConfig()):
        super().__init__(config)
        self.response_memory = nn.GRUCell(GOAL_FEATURE_DIM, config.memory_dim)

    def initial_state(self, observation):
        raise ValueError('use task_state for typed pose/velocity goals')

    def task_state(self, observation, via, stop, via_rotation, stop_rotation, radius,
                   *, start=None, start_rotation=None, velocities=None, is_stop=False):
        self._check_observation(observation)
        n = len(observation)
        if via.shape != (n,3) or stop.shape != (n,3) or via_rotation.shape != (n,3,3) or stop_rotation.shape != (n,3,3):
            raise ValueError('goal tensor shapes must match the observed batch')
        start = observation[:,:3].clone() if start is None else start
        start_rotation = stop_rotation if start_rotation is None else start_rotation
        velocities = (observation.new_tensor(GoalConfig().velocities)[None].expand(n,-1,-1)
                      if velocities is None else velocities)
        radius = torch.as_tensor(radius,device=observation.device,dtype=observation.dtype).expand(n)
        tensors = (via,stop,via_rotation,stop_rotation,start,start_rotation,velocities,radius)
        if (start.shape != (n,3) or start_rotation.shape != (n,3,3) or velocities.shape != (n,3,3)
                or any(a.device != observation.device or a.dtype != observation.dtype for a in tensors)):
            raise ValueError('task shapes/dtypes/devices must match observations')
        if (not all(bool(torch.isfinite(a).all()) for a in tensors) or not bool((radius > 0).all())):
            raise ValueError('task data must be finite and radius positive')
        return GoalState(observation.new_zeros(n,self.config.memory_dim),via,stop,via_rotation,
                         stop_rotation,radius,torch.full((n,),2 if is_stop else 0,
                         device=observation.device,dtype=torch.long),observation[:,:3].clone(),
                         start,start_rotation,velocities)

    def goal_features(self, observation, state, phase):
        stop = phase == 2
        goal = torch.where(stop[:,None],state.stop,state.via)
        target_R = torch.where(stop[:,None,None],state.stop_rotation,state.via_rotation)
        R = observation[:,6:15].reshape(-1,3,3)
        eye = torch.eye(3,device=R.device,dtype=R.dtype)
        relative = R.transpose(-1,-2) @ target_R
        original = torch.cat((observation[:,:3]-goal,observation[:,3:]),-1)
        base = super().control_features(original)
        legacy = torch.cat((body_vector(R,state.stop-observation[:,:3]),
                            (relative[...,:2]-eye[:,:2]).flatten(1),
                            stop.to(observation)[:,None]),-1)
        rotations = torch.stack((state.start_rotation,state.via_rotation,state.stop_rotation),1)
        relative_all = R.transpose(-1,-2)[:,None] @ rotations
        velocity_body = (R.transpose(-1,-2)[:,None] @ state.target_velocities.unsqueeze(-1)).squeeze(-1)
        added = torch.cat((body_vector(R,state.start-observation[:,:3]),
                           body_vector(R,state.via-observation[:,:3]),
                           (relative_all[...,:2]-eye[:,:2]).flatten(1),
                           velocity_body.flatten(1)/3.,R[...,:2].flatten(1),
                           F.one_hot(phase,num_classes=3).to(observation),state.radius[:,None]),-1)
        return torch.cat((base,legacy,added),-1)

    def forward(self, observation, state=None):
        self._check_observation(observation)
        if not isinstance(state,GoalState) or state.memory.shape != (len(observation),self.config.memory_dim):
            raise ValueError('typed Actor requires its matching GoalState')
        phase = tensor_kernel(advance_phase, state.previous_position,observation[:,:3],state.via,state.radius,state.phase,
                              compiled=getattr(self, '_compiled_execution', False))
        memory = self.response_memory(self.goal_features(observation,state,phase),state.memory)
        next_state = replace(state,memory=memory,phase=phase,previous_position=observation[:,:3])
        return ResponsePolicyOutput(torch.tanh(self.readout(memory)),next_state)


def import_motor_weights(policy, weights):
    """Weights-only migration, not optimizer resume. Keep one self-contained Actor."""
    expected = policy.state_dict()
    if set(weights) != set(expected):
        raise ValueError('Actor tensor names mismatch')
    value = dict(weights)
    key = 'response_memory.weight_ih'
    if isinstance(policy, GoalMotorPolicy) and value[key].shape[0] == expected[key].shape[0] and value[key].shape[1] in (16,26):
        expanded = torch.zeros_like(expected[key])
        expanded[:,:value[key].shape[1]] = value[key].to(expanded)
        value[key] = expanded
    if any(value[n].shape != x.shape or not bool(torch.isfinite(value[n]).all()) for n,x in expected.items()):
        raise ValueError('incompatible or nonfinite Actor weights')
    policy.load_state_dict(value,strict=True)


def prepare(policy, initial, config):
    """Retask a sampled master scene; leave sampler, dynamics and sensor code intact."""
    if not isinstance(config, PassStopLossConfig):
        return initial
    c = GoalConfig(**config.mission)
    if not isinstance(policy, GoalMotorPolicy):
        raise ValueError('typed loss requires typed Actor')
    n = len(initial.mass)
    points = initial.position.new_tensor(c.points)[None].expand(n,-1,-1)
    if c.relative_to_boundary:
        points = points*initial.position_limit[:,None,None]
    if bool((points.abs() >= initial.position_limit[:,None,None]).any()):
        raise ValueError('task points must lie strictly inside the original world boundary')
    if bool(((points[:,1]-points[:,0]).norm(dim=-1) <= 2*c.radius_m).any()
            or ((points[:,2]-points[:,1]).norm(dim=-1) <= c.radius_m+c.stop_position_m).any()):
        raise ValueError('task goal regions must be disjoint; no silent geometric rescaling')
    radius = c.region_radius_arms*initial.arm_length
    if bool((points[:,1].abs()+radius[:,None] >= initial.position_limit[:,None]).any()):
        raise ValueError('the entire PASS ball must fit inside the original boundary')
    if bool(((points[:,0]-points[:,1]).norm(dim=-1) <= radius).any()
            or ((points[:,2]-points[:,1]).norm(dim=-1) <= radius+c.stop_position_m).any()):
        raise ValueError('START/STOP must lie outside the PASS ball')
    q_start = initial.orientation.new_tensor(c.start_quaternion).expand(n,-1)
    q_end = initial.orientation.new_tensor(c.stop_quaternion).expand(n,-1)
    q_via = q_end.new_tensor(c.via_quaternion).expand(n,-1)
    # Exact positive root of the original thrust polynomial at force balance.
    coef = initial.thrust_coefficients
    d = coef[...,0]-initial.mass[:,None]*9.81/4
    motor = -2*d/(coef[...,1]+torch.sqrt(coef[...,1].square()-4*coef[...,2]*d))
    if bool(((motor < initial.motor_min[:,None]) | (motor > initial.motor_max[:,None])).any()):
        raise ValueError('sampled actuator cannot initialize hover equilibrium')
    # The new task starts at rest. START/STOP velocity vectors are exactly zero.
    # No external forces or pulses in this task. Measurement noise/delay survive.
    state = replace(initial, position=points[:,0], orientation=q_start, velocity=torch.zeros_like(initial.velocity), omega=torch.zeros_like(initial.omega),
                    motor=motor, previous_action=torch.zeros_like(initial.previous_action),
                    external_force=torch.zeros_like(initial.external_force),
                    external_torque=torch.zeros_like(initial.external_torque),
                    force_std=torch.zeros_like(initial.force_std),
                    pulse_tape=torch.zeros_like(initial.pulse_tape),
                    pulse_active_tape=torch.zeros_like(initial.pulse_active_tape),
                    previous_velocity=torch.zeros_like(initial.previous_velocity))
    obs = measured_observation(state)
    ps = policy.task_state(obs,points[:,1],points[:,2],quaternion_rotation(q_via),quaternion_rotation(q_end),radius,
                           start=points[:,0],start_rotation=quaternion_rotation(q_start),
                           velocities=obs.new_tensor(c.velocities)[None].expand(n,-1,-1))
    return ResponseClosedLoopState(state,ps)


def task_rollout(policy, simulator, initial, horizon, config, **kwargs):
    return rollout(policy,simulator,prepare(policy,initial,config),horizon,**kwargs)


def attitude_ratio(q, target_R, tolerance_deg):
    # Rotation-matrix chord distance is sign-independent; no acos in the graph.
    scale = 2*math.sqrt(2)*math.sin(math.radians(tolerance_deg)/2)
    return (quaternion_rotation(q)-target_R).flatten(-2).norm(dim=-1)/scale


def temporal_proxy(pass_ratio, stop_ratio, *, dt=.01, first_hit=None, alive=None):
    """Best PAST simultaneous PASS quality, worst FUTURE STOP quality.

    Ratios <= 1 mean within all relevant tolerances. Cummax/cummin are continuous,
    piecewise differentiable; ties use PyTorch's subgradient convention. They do
    not accumulate repeated weak evidence. No softmax, hazard product, or detach.
    The optional hard-event cap prevents a bad first visit being erased by later
    revisits; derivatives are conditional on that discrete event selection.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be positive and finite')
    if pass_ratio.ndim != 2 or pass_ratio.shape != stop_ratio.shape or len(pass_ratio) < 2:
        raise ValueError('temporal ratios must share [H+1,N]')
    qp = 1/(1+(pass_ratio-1).clamp_min(0))
    qs = 1/(1+(stop_ratio-1).clamp_min(0))
    if alive is not None:
        qp = torch.where(alive,qp,0.)
        qs = torch.where(alive,qs,0.)
    past = qp.cummax(0).values
    if first_hit is not None:
        row = torch.arange(qp.shape[1],device=qp.device)
        cap = qp[first_hit.clamp_min(0),row]
        cap = torch.where(first_hit >= 0,cap,torch.ones_like(cap))
        past = torch.minimum(past,cap[None])
    future = qs.flip(0).cummin(0).values.flip(0)
    # Logical AND via minimum: avoids multiplying two already small gradients.
    completion = torch.minimum(past,future)
    return dt*(1-completion[:-1]).sum(0), completion



def pose_ratio_quaternion(q, target, tolerance_deg):
    """Sign-invariant chordal SO(3) error, without storing rotation matrices."""
    a, b = target[..., :1], target[..., 1:]
    c, d = q[..., :1], q[..., 1:]
    relative_vector = a*d-c*b-torch.linalg.cross(b.expand_as(d),d,dim=-1)
    denominator = math.sin(math.radians(tolerance_deg)/2)
    return relative_vector.norm(dim=-1)/denominator


def _region_schedule(observations, state, valid):
    measured = observations[...,:3]
    hit, _, enter, exit_ = segment_ball_geometry(measured[:-1],measured[1:],state.via,state.radius)
    hit = hit & valid
    inside = ((measured[1:]-state.via).square().sum(-1) <= state.radius.square()) & valid
    has_entered = hit.long().cumsum(0) > 0
    exited = (has_entered & ~inside & valid).long().cumsum(0) > 0
    before_exit = torch.cat((torch.zeros_like(exited[:1]),exited[:-1]),0)
    region = hit & ~before_exit & valid
    phases = torch.where(exited,2,torch.where(has_entered,1,0))
    phases = torch.cat((torch.zeros_like(phases[:1]),phases),0)
    exit_exists = exited.any(0)
    exit_index = torch.where(exit_exists,exited.long().argmax(0)+1,-1)
    return region, enter, exit_, phases, exit_index


def _tensor_costs(trace, c, config):
    state = trace.end.policy
    if not isinstance(state,GoalState) or trace.observations is None:
        raise ValueError('typed costs require native observations and goal state')
    p = torch.cat((trace.initial.position[None],trace.positions),0)
    v = torch.cat((trace.initial.velocity[None],trace.velocities),0)
    q = torch.cat((trace.initial.orientation[None],trace.post_orientations),0)
    w = torch.cat((trace.initial.omega[None],trace.omegas),0)
    H,N = trace.actions.shape[:2]
    dt = .01
    alive = ~RaptorSimulator.position_terminated(p,trace.initial.position_limit)
    row = torch.arange(N,device=p.device)
    tq = q.new_tensor(c.via_quaternion)
    eq = q.new_tensor(c.stop_quaternion)
    target_v = state.target_velocities[:,1]
    minimum_speed = target_v.norm(dim=-1)
    pos = (p-state.via).norm(dim=-1)/c.radius_m
    rot = pose_ratio_quaternion(q,tq,c.pose_tolerance_deg)
    vel = 1+(minimum_speed-v.norm(dim=-1))/c.pass_speed_scale_m_s
    pair = torch.maximum(torch.maximum(pos,rot),vel)
    kinematic = interval_kinematics(trace,dt)
    stop_parts = torch.stack(((p-state.stop).norm(dim=-1)/c.stop_position_m,
                             (v-state.target_velocities[:,2]).norm(dim=-1)/c.stop_velocity_m_s,
                             pose_ratio_quaternion(q,eq,c.stop_pose_deg),
                             w.norm(dim=-1)/c.stop_omega_rad_s),-1)
    # End-force balance prevents the last-frame zero-velocity turning-point cheat.
    end_acceleration,end_angular = instantaneous_acceleration(trace.end.physical)
    final_state = trace.end.physical
    motor_command = RaptorSimulator.motor_command(final_state,final_state.previous_action)
    tau = torch.where(motor_command >= final_state.motor,final_state.motor_time_rising,final_state.motor_time_falling)
    motor_rate = (motor_command-final_state.motor)/tau
    balance_parts = torch.stack((end_acceleration.norm(dim=-1)/c.stop_acceleration_m_s2,
                                 end_angular.norm(dim=-1)/c.stop_angular_acceleration_rad_s2,
                                 motor_rate.abs().amax(-1)/c.stop_motor_rate_per_s),-1)
    end_balance = balance_parts.amax(-1)
    stop = stop_parts.amax(-1)
    stop = torch.cat((stop[:-1],torch.maximum(stop[-1],end_balance)[None]),0)
    region,entry_fraction,exit_fraction,phases,exit_index = _region_schedule(trace.observations,state,trace.valid)
    exists = region.any(0)
    finished_region = exit_index >= 0
    # Score entry/middle/exit on every first-traversal chord. This has fixed O(HN)
    # cost, no extra RK4 step, and catches straight between-sample passages.
    fractions = torch.stack((entry_fraction,(entry_fraction+exit_fraction)*.5,exit_fraction),0)
    ps = p[:-1][None]+fractions[...,None]*(p[1:]-p[:-1])[None]
    vs = v[:-1][None]+fractions[...,None]*(v[1:]-v[:-1])[None]
    qs = interpolate_quaternion(q[:-1][None].expand(3,-1,-1,-1),
                                 q[1:][None].expand(3,-1,-1,-1),fractions)
    rp = (ps-state.via).norm(dim=-1)/c.radius_m
    rr = pose_ratio_quaternion(qs,tq,c.pose_tolerance_deg)
    rv = 1+(minimum_speed-vs.norm(dim=-1))/c.pass_speed_scale_m_s
    # Missing a region still gets a joint same-time recovery gradient.
    joint = (pos-1).clamp_min(0).square()+(rot-1).clamp_min(0).square()+(vel-1).clamp_min(0).square()
    eligible = alive.clone()
    eligible[0] = ~alive[1:].any(0)
    witness = joint.masked_fill(~eligible,torch.inf).argmin(0)
    fallback_p, fallback_r, fallback_v = pos[witness,row],rot[witness,row],vel[witness,row]
    # The whole interval's pose/velocity is scored; making the visit shorter does
    # not remove its peak error, nor can a later good revisit erase the first.
    region_p = rp.masked_fill(~region[None],torch.inf).amin((0,1))
    region_r = torch.where(region[None],rr,0.).amax((0,1))
    region_v = rv.masked_fill(~region[None],-torch.inf).amax((0,1))
    selected_p = torch.where(exists,region_p,fallback_p)
    selected_r = torch.where(exists,region_r,fallback_r)
    selected_v = torch.where(exists,region_v,fallback_v)
    pass_position = c.point_weight*(selected_p-1).clamp_min(0).square()
    pass_attitude = c.point_weight*(selected_r-1).clamp_min(0).square()
    pass_speed = c.velocity_weight*(selected_v-1).clamp_min(0).square()
    missing = c.point_weight*(~finished_region).to(p)
    end_cost = c.point_weight*((stop_parts[-1]-1).clamp_min(0).square().sum(-1)
                               +(balance_parts-1).clamp_min(0).square().sum(-1))
    # Use interpolated joint states to inform the retained time proxy as well.
    interval_quality = torch.maximum(torch.maximum(rp,rr),rv).amin(0)
    interval_quality = torch.where(region,interval_quality,pair[1:])
    time_pair = torch.cat((pair[:1],interval_quality),0)
    # Worst first-traversal quality caps all future completion evidence.
    cap_ratio = torch.maximum(torch.maximum(selected_p,selected_r),selected_v)
    time_pair = torch.where(exists[None],torch.maximum(time_pair,cap_ratio[None]),time_pair)
    time_s,completion = temporal_proxy(time_pair,stop,dt=.01,alive=alive)
    # Weak fixed approach direction: no moving reference, no target speed floor.
    heading = F.normalize(state.via-state.start,dim=-1)
    progress_speed = (v[:-1]*heading).sum(-1)
    lateral = v[:-1]-progress_speed[...,None]*heading
    scale = target_v.norm(dim=-1).clamp_min(.1)
    heading_cost = F.softplus(-progress_speed/scale)+.1*(lateral/scale[...,None]).square().sum(-1)
    approach = (phases[:-1] == 0) & ~region & trace.valid
    direction = c.direction_weight*torch.where(approach,heading_cost,0.).sum(0)/H
    du = torch.where(trace.valid[...,None],trace.action_deltas,0.)
    action_cost = c.smooth_weight*smooth_l2(du,config.epsilon_a).sum(0)/H
    acceleration_cost = c.acceleration_weight*kinematic.acceleration.square().sum(-1).sum(0)/H
    jerk_cost = c.jerk_weight*kinematic.jerk.square().sum(-1).sum(0)/max(1,H-1)
    failed = ~alive[-1]
    lengths = trace.valid.sum(0)
    failure = failed.to(p)*(config.terminal_cost+config.dead_cost*(H-lengths))/H
    boundary = (p.abs()/trace.initial.position_limit[None,:,None]-1).clamp_min(0).square().amax((0,2))
    components = {'pass_position':pass_position,'pass_attitude':pass_attitude,
                  'pass_speed':pass_speed,'pass_missing':missing,'stop':end_cost,
                  'time':c.time_weight*time_s/(H*dt),'direction':direction,
                  'action_delta':action_cost,
                  'acceleration':acceleration_cost,'jerk':jerk_cost,
                  'failure':failure,'boundary':c.point_weight*boundary}
    costs = sum(components.values())
    with torch.no_grad():
        passed = finished_region & (selected_p <= 1) & (selected_r <= 1) & (selected_v <= 1)
        after_exit = torch.arange(H+1,device=p.device)[:,None] >= torch.where(finished_region,exit_index,H+1)[None]
        stopped = (stop <= 1) & alive & passed[None] & after_exit
        outside = torch.where(~stopped,torch.arange(H+1,device=p.device)[:,None],-1)
        completed = stopped[-1] & ~failed
        hard_time = (outside.amax(0)+1).clamp_max(H).to(p)*dt
        hard_time = torch.where(completed,hard_time,torch.full_like(hard_time,H*dt))
        shortfall = (minimum_speed-vs.norm(dim=-1)).clamp_min(0)
        metrics = {'task_objective':costs.mean(),
                   'task_components':{k:a.mean() for k,a in components.items()},
                   'time_proxy_seconds':time_s.mean(),'completion_time_seconds':hard_time.mean(),
                   'pass_fraction':passed.float().mean(),'success_fraction':completed.float().mean(),
                   'region_enter_fraction':exists.float().mean(),'region_exit_fraction':finished_region.float().mean(),
                   'pass_position_m':(selected_p*c.radius_m).mean(),
                   'pass_chordal_ratio':selected_r.mean(),
                   'pass_minimum_speed_m_s':minimum_speed.mean(),
                   'pass_speed_shortfall_m_s':((selected_v-1)*c.pass_speed_scale_m_s).clamp_min(0).mean(),
                   'region_max_speed_shortfall_m_s':torch.where(region[None],shortfall,0.).amax((0,1)).mean(),
                   'region_samples':region.sum(),'region_radius_m_min':state.radius.min(),
                   'region_radius_m_max':state.radius.max(),
                   'terminal_acceleration_m_s2':end_acceleration.norm(dim=-1).mean(),
                   'terminal_angular_acceleration_rad_s2':end_angular.norm(dim=-1).mean(),
                   'terminal_motor_rate_per_s':motor_rate.abs().amax(-1).mean(),
                   'acceleration_rms_m_s2':kinematic.acceleration.square().sum(-1).mean().sqrt(),
                   'position_rms':(p[1:]-state.stop).square().sum(-1).mean().sqrt(),
                   'velocity_rms':v[1:].square().sum(-1).mean().sqrt(),
                   'omega_rms':w[1:].square().sum(-1).mean().sqrt(),
                   'physical_transitions':trace.valid.sum(),'scenario_count':N,'finite':True,
                   'motor_saturation_fraction':(trace.actions.abs() >= 1-1e-6).float().mean()}
    return costs,metrics,components


def typed_costs(trace, config, *, compiled=False):
    c = GoalConfig(**config.mission)
    costs, values, components = tensor_kernel(_tensor_costs, trace, c, config, compiled=compiled)
    if not tensors_finite([costs]):
        raise FloatingPointError('nonfinite typed task objective')
    integer_metrics = {'physical_transitions', 'scenario_count', 'region_samples'}
    metrics = {name: (int(value) if name in integer_metrics else float(value))
               if isinstance(value, torch.Tensor) else value for name, value in values.items()}
    metrics['task_components'] = {name: float(value) for name, value in values['task_components'].items()}
    metrics['disturbances'] = disturbance_report(trace.initial)
    return costs, metrics, components

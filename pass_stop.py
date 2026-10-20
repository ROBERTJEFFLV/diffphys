"""Typed fixed goals and a temporal loss; no planner, critic, or solver gradient.

This task is an opt-in addition to master's native GRU/RK4/Adam pipeline.
The controller switches by measured position. Losses use true states and never
request a PASS velocity. Hard switches are not assigned fictitious derivatives.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import nn

from env_raptor import RaptorSimulator, quaternion_rotation
from response_noise import measured_observation, disturbance_report
from response_policy import (ResponseMotorPolicy, ResponsePolicyConfig,
                             ResponsePolicyState, ResponsePolicyOutput, body_vector)
from response_task import (TaskLossConfig, ResponseClosedLoopState, rollout,
                           smooth_l2, tensors_finite)

VERSION = "pass-stop-prefix-suffix-v1"
ARCHITECTURE = "gru26-hidden-only-typed-goal-motor-policy-v1"


@dataclass(frozen=True)
class GoalConfig:
    # Coordinates are fractions of the ORIGINAL world-centred per-axis limit.
    points: tuple = ((-.7, 0., 0.), (0., .38, 0.), (.7, 0., 0.))
    relative_to_boundary: bool = True
    via_quaternion: tuple = (0.6087614290087207, 0.7933533402912352, 0., 0.)
    stop_quaternion: tuple = (1., 0., 0., 0.)
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
        for name in ('via_quaternion', 'stop_quaternion'):
            q = tuple(float(v) for v in getattr(self, name))
            if len(q) != 4 or not all(math.isfinite(v) for v in q) or abs(sum(v*v for v in q)-1) > 1e-6:
                raise ValueError(name + ' must be a unit wxyz quaternion')
            object.__setattr__(self, name, q)
        if self.version != VERSION or type(self.relative_to_boundary) is not bool:
            raise ValueError('unsupported task version/coordinate convention')
        if len(self.points) != 3 or any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in self.points):
            raise ValueError('exactly three finite positions are required')
        for name in ('radius_m', 'pose_tolerance_deg', 'stop_position_m', 'stop_velocity_m_s',
                     'stop_pose_deg', 'stop_omega_rad_s', 'point_weight', 'time_weight'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + ' must be positive and finite')
        if max(self.pose_tolerance_deg, self.stop_pose_deg) >= 180:
            raise ValueError('angular tolerance must be less than 180 degrees')
        if not math.isfinite(self.smooth_weight) or self.smooth_weight < 0:
            raise ValueError('smooth_weight must be nonnegative')
        # A stationary ordinary quadrotor cannot support a persistently tilted thrust axis.
        if abs(self.stop_quaternion[1])+abs(self.stop_quaternion[2]) > 1e-6:
            raise ValueError('STOP must be level (yaw is allowed) for the ordinary quadrotor')


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
    is_stop: torch.Tensor


class GoalMotorPolicy(ResponseMotorPolicy):
    """Master's single native GRU + hidden-only readout, with ten goal features.

    The first sixteen feature meanings are preserved, except position is relative
    to the current FIXED goal. Extra: end position(3), relative rotation(6), type(1).
    Goal state contains only public task data; no mass, motor truth or clock.
    """
    def __init__(self, config=ResponsePolicyConfig()):
        super().__init__(config)
        self.response_memory = nn.GRUCell(26, config.memory_dim)

    def initial_state(self, observation):
        raise ValueError('use task_state(observation, via, stop, rotations, radius) for typed goals')

    def task_state(self, observation, via, stop, via_rotation, stop_rotation, radius, *, is_stop=False):
        self._check_observation(observation)
        n = len(observation)
        if via.shape != (n,3) or stop.shape != (n,3) or via_rotation.shape != (n,3,3) or stop_rotation.shape != (n,3,3):
            raise ValueError('goal tensor shapes must match the observed batch')
        return GoalState(observation.new_zeros(n, self.config.memory_dim), via, stop,
                         via_rotation, stop_rotation, observation.new_full((n,), radius),
                         torch.full((n,), is_stop, dtype=torch.bool, device=observation.device))

    def forward(self, observation, state=None):
        self._check_observation(observation)
        if not isinstance(state, GoalState) or state.memory.shape != (len(observation),self.config.memory_dim):
            raise ValueError('typed Actor requires its matching GoalState')
        # Position-only command transition; pose never delays switching.
        reached = (observation[:,:3]-state.via).square().sum(-1) <= state.radius.square()
        stop = state.is_stop | reached
        goal = torch.where(stop[:,None], state.stop, state.via)
        target_R = torch.where(stop[:,None,None], state.stop_rotation, state.via_rotation)
        R = observation[:,6:15].reshape(-1,3,3)
        relative = R.transpose(-1,-2) @ target_R
        eye = torch.eye(3, device=R.device, dtype=R.dtype)
        original = torch.cat((observation[:,:3]-goal, observation[:,3:]),-1)
        base = super().control_features(original)
        extra = torch.cat((body_vector(R, state.stop-observation[:,:3]),
                           (relative[...,:2]-eye[:,:2]).flatten(1), stop.to(observation)[:,None]),-1)
        memory = self.response_memory(torch.cat((base,extra),-1), state.memory)
        return ResponsePolicyOutput(torch.tanh(self.readout(memory)), replace(state,memory=memory,is_stop=stop))


def import_motor_weights(policy, weights):
    """Weights-only migration, not optimizer resume. Keep one self-contained Actor."""
    expected = policy.state_dict()
    if set(weights) != set(expected):
        raise ValueError('Actor tensor names mismatch')
    value = dict(weights)
    key = 'response_memory.weight_ih'
    if isinstance(policy, GoalMotorPolicy) and value[key].shape == (expected[key].shape[0],16):
        expanded = torch.zeros_like(expected[key])
        expanded[:,:16] = value[key].to(expanded)
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
    q_end = initial.orientation.new_tensor(c.stop_quaternion).expand(n,-1)
    q_via = q_end.new_tensor(c.via_quaternion).expand(n,-1)
    # Exact positive root of the original thrust polynomial at force balance.
    coef = initial.thrust_coefficients
    d = coef[...,0]-initial.mass[:,None]*9.81/4
    motor = -2*d/(coef[...,1]+torch.sqrt(coef[...,1].square()-4*coef[...,2]*d))
    if bool(((motor < initial.motor_min[:,None]) | (motor > initial.motor_max[:,None])).any()):
        raise ValueError('sampled actuator cannot initialize hover equilibrium')
    # Preserve sampled initial velocity, including master's 10% zero-speed rows.
    # No external forces or pulses in this task. Measurement noise/delay survive.
    state = replace(initial, position=points[:,0], orientation=q_end, omega=torch.zeros_like(initial.omega),
                    motor=motor, previous_action=torch.zeros_like(initial.previous_action),
                    external_force=torch.zeros_like(initial.external_force),
                    external_torque=torch.zeros_like(initial.external_torque),
                    force_std=torch.zeros_like(initial.force_std),
                    pulse_tape=torch.zeros_like(initial.pulse_tape),
                    pulse_active_tape=torch.zeros_like(initial.pulse_active_tape),
                    previous_velocity=initial.velocity[:,None].expand_as(initial.previous_velocity).clone())
    obs = measured_observation(state)
    ps = policy.task_state(obs,points[:,1],points[:,2],quaternion_rotation(q_via),quaternion_rotation(q_end),c.radius_m)
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


def typed_costs(trace, config):
    c = GoalConfig(**config.mission)
    state = trace.end.policy
    if not isinstance(state,GoalState) or trace.observations is None:
        raise ValueError('typed costs require native observation history and goal state')
    p = torch.cat((trace.initial.position[None],trace.positions),0)
    v = torch.cat((trace.initial.velocity[None],trace.velocities),0)
    q = torch.cat((trace.initial.orientation[None],trace.post_orientations),0)
    w = torch.cat((trace.initial.omega[None],trace.omegas),0)
    H,N = trace.actions.shape[:2]
    alive = ~RaptorSimulator.position_terminated(p,trace.initial.position_limit)
    pos = (p-state.via).norm(dim=-1)/c.radius_m
    rot = attitude_ratio(q,state.via_rotation,c.pose_tolerance_deg)
    pair = torch.maximum(pos,rot)
    stop_parts = torch.stack(((p-state.stop).norm(dim=-1)/c.stop_position_m,
                             v.norm(dim=-1)/c.stop_velocity_m_s,
                             attitude_ratio(q,state.stop_rotation,c.stop_pose_deg),
                             w.norm(dim=-1)/c.stop_omega_rad_s),-1)
    stop = stop_parts.amax(-1)
    # Command event uses the same acquired positions as GoalMotorPolicy.
    hit = ((trace.observations[:,:,:3]-state.via).square().sum(-1) <= c.radius_m**2) & alive
    exists = hit.any(0)
    first = torch.where(exists,hit.long().argmax(0),-1)
    # Fallback jointly selects position AND attitude at one real, live state.
    joint = (pos-1).clamp_min(0).square()+(rot-1).clamp_min(0).square()
    eligible = alive.clone()
    # Spawn is fixed, not an action-controlled witness. Avoid a zero-gradient
    # fallback at x0 when all future poses are initially equally poor.
    eligible[0] = ~alive[1:].any(0)
    available = joint.masked_fill(~eligible,torch.inf)
    candidate = available.argmin(0)
    witness = torch.where(exists,first,candidate)
    row = torch.arange(N,device=p.device)
    pr = (pos[witness,row]-1).clamp_min(0)
    rr = (rot[witness,row]-1).clamp_min(0)
    pass_cost = c.point_weight*(pr.square()+rr.square()+(~exists).to(p))
    end_cost = c.point_weight*(stop_parts[-1]-1).clamp_min(0).square().sum(-1)
    time_s,completion = temporal_proxy(pair,stop,first_hit=first,alive=alive)
    # Weak original command-change regularizer, but no 25x attitude-motion cost.
    du = torch.where(trace.valid[...,None],trace.action_deltas,0.)
    action_cost = c.smooth_weight*smooth_l2(du,config.epsilon_a).sum(0)/H
    # Original failure accounting, plus differentiable residual to the ORIGINAL boundary.
    failed = ~alive[-1]
    lengths = trace.valid.sum(0)
    failure = failed.to(p)*(config.terminal_cost+config.dead_cost*(H-lengths))/H
    boundary = (p.abs()/trace.initial.position_limit[None,:,None]-1).clamp_min(0).square().amax((0,2))
    components = {'pass_pose':pass_cost, 'stop':end_cost, 'time':c.time_weight*time_s/(H*.01),
                  'action_delta':action_cost, 'failure':failure, 'boundary':c.point_weight*boundary}
    costs = sum(components.values())
    if not tensors_finite([costs]):
        raise FloatingPointError('nonfinite typed task objective')
    with torch.no_grad():
        passed = exists & (pos[first.clamp_min(0),row] <= 1) & (rot[first.clamp_min(0),row] <= 1)
        before = torch.arange(H+1,device=p.device)[:,None] < first[None]
        stopped = (stop <= 1) & alive & passed[None] & ~before
        outside = torch.where(~stopped,torch.arange(H+1,device=p.device)[:,None],-1)
        last_outside = outside.amax(0)
        completed = stopped[-1] & ~failed
        hard_time = (last_outside+1).clamp_max(H).to(p)*.01
        hard_time = torch.where(completed,hard_time,torch.full_like(hard_time,H*.01))
        metrics = {'task_objective':float(costs.mean()), 'task_components':{k:float(a.mean()) for k,a in components.items()},
                   'time_proxy_seconds':float(time_s.mean()), 'completion_time_seconds':float(hard_time.mean()),
                   'pass_fraction':float(passed.float().mean()), 'success_fraction':float(completed.float().mean()),
                   'pass_position_m':float((pos[witness,row]*c.radius_m).mean()),
                   'pass_chordal_ratio':float(rot[witness,row].mean()),
                   'position_rms':float((p[1:]-state.stop).square().sum(-1).mean().sqrt()),
                   'velocity_rms':float(v[1:].square().sum(-1).mean().sqrt()),
                   'omega_rms':float(w[1:].square().sum(-1).mean().sqrt()),
                   'physical_transitions':int(trace.valid.sum()), 'scenario_count':N,'finite':True,
                   'motor_saturation_fraction':float((trace.actions.abs() >= 1-1e-6).float().mean()),
                   'disturbances':disturbance_report(trace.initial)}
    return costs,metrics,components

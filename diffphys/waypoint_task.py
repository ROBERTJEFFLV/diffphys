"""Actual 100 Hz hard commands, continuous physics, and event/terminal constraints.

Full BPTT is retained. Discrete index/region choices are NOT differentiated with a
straight-through estimator. Their effects are included by paired HARD rollouts.
Fractional RK4 is used for supervision only and never advances the Actor clock.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from types import SimpleNamespace
import math
import torch

from .env_raptor import RaptorSimulator, RaptorState, IMMUTABLE_TAPES, quaternion_rotation
from .response_policy import ResponsePolicyState
from .response_noise import measured_observation
from .waypoint_mission import advance_target, goal_packet
from .waypoint_gradients import phr

DYNAMIC = ('position', 'velocity', 'orientation', 'omega', 'motor',
           'previous_action', 'previous_velocity', 'step_index')
CONSTRAINTS = (
    'midpoint_command_entry', 'midpoint_plane_crossing', 'midpoint_position',
    'midpoint_attitude', 'midpoint_forward_speed', 'passage_band_attitude',
    'passage_no_deceleration', 'passage_forward_speed', 'final_position',
    'final_speed', 'final_attitude', 'final_omega', 'final_acceleration',
    'final_angular_acceleration', 'final_motor_rate', 'world_boundary', 'body_rate',
)


def select_state(state, rows):
    return replace(state, **{f.name:getattr(state,f.name).index_select(0,rows)
                            for f in fields(state) if f.name not in IMMUTABLE_TAPES})


def merge_state(full, before, after, rows):
    if len(rows) == len(full.mass):
        return after
    return replace(full, **{name:getattr(full,name).index_copy(0,rows,getattr(after,name))
                            for name in DYNAMIC})


@dataclass
class WaypointTrace:
    initial: RaptorState
    final: RaptorState
    state: dict[str, torch.Tensor]        # [H+1,N,...]
    actions: torch.Tensor                # [H,N,4]
    target_index: torch.Tensor           # [H+1,N], post-measurement index
    valid: torch.Tensor                  # [H,N], last boundary-crossing step included
    final_memory: torch.Tensor


@dataclass
class WaypointResult:
    costs: torch.Tensor
    constraints: torch.Tensor
    time_cost: torch.Tensor
    report: dict
    trace: WaypointTrace | None = None
    event_data: dict | None = None


def rollout(policy, simulator, initial, mission, config, *, horizon=500):
    """No curriculum, no trajectory reference, no reset/detach at a waypoint."""
    if horizon < 1 or abs(simulator.params.dt-.01) > 1e-12 or not policy.config.goal_conditioning:
        raise ValueError("waypoints require goal-conditioned 100 Hz physics")
    if initial.noise_tape.shape[1] > 1 and initial.noise_tape.shape[1] < horizon+1:
        raise ValueError("measurement tape shorter than rollout")
    if any(bool(value.count_nonzero()) for value in
           (initial.external_force, initial.external_torque, initial.pulse_tape)):
        raise ValueError("waypoint equilibrium supervision requires force-free initial tapes")
    physical = initial
    obs = measured_observation(physical)
    memory = policy.initial_state(obs).memory
    target = torch.zeros_like(initial.step_index)
    target = advance_target(obs[:,:3], mission, target, config.arrival_radius_m)
    states = {name:[getattr(initial,name)] for name in DYNAMIC}
    actions, indices, valids = [], [target], []
    live_rows = (~simulator.terminated(initial)).nonzero(as_tuple=True)[0]
    for _ in range(horizon):
        before = physical
        active = torch.zeros_like(initial.step_index, dtype=torch.bool)
        action = physical.previous_action
        if live_rows.numel():
            active[live_rows] = True
            goals = goal_packet(obs, mission, target, config.passage_speed_m_s)
            out = policy(obs.index_select(0,live_rows), ResponsePolicyState(memory.index_select(0,live_rows)),
                         goals=goals.index_select(0,live_rows))
            small = select_state(physical,live_rows)
            moved = simulator.step(small,out.action)
            physical = merge_state(physical,small,moved,live_rows)
            memory = memory.index_copy(0,live_rows,out.next_state.memory)
            action = action.index_copy(0,live_rows,out.action)
            keep = (~simulator.terminated(moved)).nonzero(as_tuple=True)[0]
            live_rows = live_rows.index_select(0,keep)
        obs = measured_observation(physical)
        proposed = advance_target(obs[:,:3],mission,target,config.arrival_radius_m)
        target = torch.where(active,proposed,target)
        for name in states:
            states[name].append(getattr(physical,name))
        actions.append(action)
        indices.append(target)
        valids.append(active)
    return WaypointTrace(initial,physical,{k:torch.stack(v) for k,v in states.items()},
                         torch.stack(actions),torch.stack(indices),torch.stack(valids),memory)


def fractional_rk4(state, action, seconds):
    """Reuse the ORIGINAL RK4 equations at a fractional dt for an audit sample.

    The real simulator still uses RaptorParams(.01). Returned history/step_index
    are discarded: no substep is fed back into physics, measurements, or GRU.
    This adapter deliberately calls the native source kernel, not a new RHS.
    """
    clock = torch.as_tensor(seconds,device=state.position.device,dtype=state.position.dtype)
    if clock.numel() == 1:
        clock = clock.reshape(1).expand(len(state.mass))
    audit = object.__new__(RaptorSimulator)
    audit.params = SimpleNamespace(dt=clock[:,None])
    audit._rotation = quaternion_rotation
    return RaptorSimulator.step(audit,state,action)


def _state_at(trace, times, rows):
    base = select_state(trace.initial,rows)
    return replace(base,**{name:value[times,rows] for name,value in trace.state.items()})


def rates(state_values, initial):
    """True physical acceleration and actuator equilibrium, no measured inputs."""
    thrust = RaptorSimulator.thrust(initial,state_values['motor'])
    rotation = quaternion_rotation(state_values['orientation'])
    acceleration = rotation[...,2]*(thrust.sum(-1)/initial.mass)[...,None]
    acceleration = acceleration + acceleration.new_tensor((0.,0.,-9.81))
    acceleration = acceleration + initial.external_force/initial.mass[:,None]
    w = state_values['omega']
    torque = RaptorSimulator.body_torque(initial,thrust)+initial.external_torque
    angular = (torque-torch.linalg.cross(w,initial.inertia*w,dim=-1))/initial.inertia
    setpoint = RaptorSimulator.motor_command(initial,state_values['previous_action'])
    motor = state_values['motor']
    tau = torch.where(setpoint >= motor,initial.motor_time_rising,initial.motor_time_falling)
    return acceleration, angular, (setpoint-motor)/tau


def rotation_residual(rotation, target, degrees):
    """Chordal norm / chordal tolerance - 1; sign-independent, no acos in loss.

    The tangent gradient can vanish at exactly 180 deg relative error; do not
    mislabel this representation as globally nondegenerate.
    """
    scale = 2*math.sqrt(2)*math.sin(math.radians(degrees)/2)
    return torch.linalg.vector_norm((rotation-target).flatten(-2),dim=-1)/scale-1


def audit_samples(trace, substeps):
    """Sample intervals at j/substeps with held commands; no feedback/replanning.

    One block avoids materializing all repeated static airframe data at once.
    Supervision is sampled, not a certificate for every continuous instant.
    """
    H,N = trace.actions.shape[:2]
    names = ('position','velocity','orientation','omega','motor','previous_action')
    blocks = {name:[] for name in names}
    for first in range(0,H,16):
        last = min(H,first+16)
        times = torch.arange(first,last,device=trace.actions.device)[:,None].expand(-1,N).reshape(-1)
        rows = torch.arange(N,device=times.device)[None].expand(last-first,-1).reshape(-1)
        base = _state_at(trace,times,rows)
        action = trace.actions[times,rows]
        live = trace.valid[times,rows]
        local = {name:[] for name in names}
        for j in range(substeps):
            sampled = base if j == 0 else fractional_rk4(base,action,.01*j/substeps)
            for name in names:
                v = getattr(sampled,name)
                if j and name != 'previous_action':
                    v = torch.where(live.reshape(-1,*([1]*(v.ndim-1))),v,getattr(base,name))
                # This interval's known command is used for instantaneous motor derivative.
                if name == 'previous_action':
                    v = action
                local[name].append(v.reshape(last-first,N,*v.shape[1:]))
        for name in names:
            val = torch.stack(local[name],1).flatten(0,1)
            blocks[name].append(val)
    return {name:torch.cat(values+[trace.state[name][-1:]],0) for name,values in blocks.items()}


def midpoint_event(trace, mission, config):
    """First transverse forward plane crossing, with implicit root gradient.

    HARD choice of crossing interval is part of the real scalar program. Paired
    rollouts account for changed intervals. Within an interval the root state
    is actual fractional RK4, not a linear position/quaternion interpolation.
    """
    p = trace.state['position']
    s = ((p-mission.points[:,1])*mission.direction).sum(-1)
    crossing = (s[:-1] < 0) & (s[1:] >= 0) & trace.valid
    exists = crossing.any(0)
    where = crossing.to(torch.long).argmax(0)
    distance = (p-mission.points[:,1]).square().sum(-1)
    closest = distance.detach().argmin(0)
    rows_all = torch.arange(len(exists),device=p.device)
    event = {name:trace.state[name][closest,rows_all] for name in
             ('position','velocity','orientation','omega','motor','previous_action')}
    event_time = closest.to(p)*.01
    good = exists.clone()
    rows = exists.nonzero(as_tuple=True)[0]
    if not rows.numel():
        return event, good, event_time, closest
    base = _state_at(trace,where[rows],rows)
    action = trace.actions[where[rows],rows]
    direction, waypoint = mission.direction[rows],mission.points[rows,1]
    with torch.no_grad():
        lo = p.new_zeros(len(rows)); hi = lo+.01
        for _ in range(config.root_iterations):
            mid = (lo+hi)/2
            state = fractional_rk4(base,action,mid)
            g = ((state.position-waypoint)*direction).sum(-1)
            lo,hi = torch.where(g < 0,mid,lo),torch.where(g >= 0,mid,hi)
        t0 = (lo+hi)/2
    if torch.is_grad_enabled():
        time = t0.detach().requires_grad_(True)
        at_root = fractional_rk4(base,action,time)
        g = ((at_root.position-waypoint)*direction).sum(-1)
        dgdt = torch.autograd.grad(g.sum(),time,retain_graph=True)[0].detach()
        transverse = dgdt > 1e-6
        # No fake unbounded derivative for grazing crossings. Mark them invalid.
        # Zero forward correction: both grad/no_grad modes use the identical
        # bisection root, while the VJP follows the implicit event equation.
        refined = time-(g-g.detach())/dgdt.clamp_min(1e-6)
        refined = torch.where(transverse,refined,time)
        at_root = fractional_rk4(base,action,refined)
    else:
        refined = t0
        at_root = fractional_rk4(base,action,refined)
        transverse = (at_root.velocity*direction).sum(-1) > 1e-6
    good = good.index_copy(0,rows,transverse)
    for name in event:
        event[name] = event[name].index_copy(0,rows,getattr(at_root,name))
    event_time = event_time.index_copy(0,rows,where[rows].to(p)*.01+refined)
    return event,good,event_time,closest


def _masked_max(value, mask, fallback):
    chosen = torch.where(mask,value,value.new_full((),-torch.inf)).amax(0)
    return torch.where(mask.any(0),chosen,fallback)


@torch.no_grad()
def completion_time(native_stop, target_index, crossed, event_time, dt):
    """Last outside time, not first entry and not a fixed dwell interval.

    The sampled clock is a discontinuous task objective; paired hard rollouts,
    rather than a fabricated ordinary derivative, learn its parameter dependence.
    """
    horizon = len(native_stop)-1
    if horizon < 1 or target_index.shape != native_stop.shape:
        raise ValueError("completion clock shape mismatch")
    steps = torch.arange(horizon+1,device=target_index.device)[:,None]
    times = steps.to(event_time)*dt
    route_ok = (target_index >= 2) & crossed[None] & (times >= event_time.detach()[None])
    stop_ok = native_stop & route_ok
    outside = torch.where(~stop_ok,steps,-1)
    completed = (outside.amax(0)+1).clamp_max(horizon)
    return completed.to(event_time)/horizon, stop_ok


def constraints_for_trace(trace, mission, config):
    """No missed-event exemption, no Gaussian pose weight, no signed averaging."""
    initial = trace.initial
    H,N = trace.actions.shape[:2]
    samples = audit_samples(trace,config.audit_substeps)
    p,v,w = (samples[k] for k in ('position','velocity','omega'))
    R = quaternion_rotation(samples['orientation'])
    a,_,_ = rates(samples,initial)
    event,exists,event_time,closest = midpoint_event(trace,mission,config)
    revent = quaternion_rotation(event['orientation'])
    center_pose = rotation_residual(revent,mission.rotation,config.pose_tolerance_degrees)
    center_speed = 1-(event['velocity']*mission.direction).sum(-1)/config.passage_speed_m_s
    position_error = (event['position']-mission.points[:,1]).norm(dim=-1)/config.arrival_radius_m-1
    distance = (trace.state['position']-mission.points[:,1]).norm(dim=-1)
    near_distance = distance[closest,torch.arange(N,device=p.device)]
    missing = 1+near_distance/config.arrival_radius_m
    entered = (trace.target_index >= 2).any(0)
    along = ((p-mission.points[:,1])*mission.direction).sum(-1)
    band = along.abs() <= mission.band_half_width
    pose_band = along.abs() <= config.pose_half_width_m
    pose_samples = rotation_residual(R,mission.rotation,config.pose_tolerance_degrees)
    # Direct physical condition d|v|^2/dt=2v.a >=0, NOT just a speed floor.
    no_deceleration = -(v*a).sum(-1)/(config.passage_speed_m_s*9.81)
    speed_samples = 1-(v*mission.direction).sum(-1)/config.passage_speed_m_s
    native_R = quaternion_rotation(trace.state['orientation'])
    native_a,native_wdot,native_mdot = rates(trace.state,initial)
    endpoint = {
        'final_position': (trace.state['position']-mission.points[:,2]).norm(dim=-1)/config.terminal_position_m-1,
        'final_speed': trace.state['velocity'].norm(dim=-1)/config.terminal_speed_m_s-1,
        'final_attitude': rotation_residual(native_R,mission.level_rotation,config.terminal_attitude_degrees),
        'final_omega': trace.state['omega'].norm(dim=-1)/config.terminal_omega_rad_s-1,
        'final_acceleration': native_a.norm(dim=-1)/config.terminal_acceleration_m_s2-1,
        'final_angular_acceleration': native_wdot.norm(dim=-1)/config.terminal_angular_acceleration_rad_s2-1,
        'final_motor_rate': native_mdot.abs().amax(-1)/config.terminal_motor_rate_s-1,
    }
    values = {
        'midpoint_command_entry':torch.where(entered,-torch.ones_like(missing),missing),
        'midpoint_plane_crossing':torch.where(exists,-torch.ones_like(missing),missing),
        'midpoint_position':position_error,
        'midpoint_attitude':center_pose,
        'midpoint_forward_speed':center_speed,
        'passage_band_attitude':torch.maximum(center_pose,_masked_max(pose_samples,pose_band,center_pose)),
        'passage_no_deceleration':_masked_max(no_deceleration,band,torch.zeros_like(missing)),
        'passage_forward_speed':torch.maximum(center_speed,_masked_max(speed_samples,band,center_speed)),
        **{name:value[-1] for name,value in endpoint.items()},
        'world_boundary':(p.abs()/initial.position_limit[None,:,None]-1).amax((0,2)),
        'body_rate':w.norm(dim=-1).amax(0)/config.maximum_omega_rad_s-1,
    }
    # Stopping obligations are unconditional; all later departures count.
    with torch.no_grad():
        native_stop = torch.stack([x <= 0 for x in endpoint.values()]).all(0)
        time_cost, stop_ok = completion_time(native_stop, trace.target_index, exists, event_time, .01)
    g = torch.stack([values[name] for name in CONSTRAINTS],-1)
    with torch.no_grad():
        command_entry = trace.target_index >= 2
        entry_index = command_entry.long().argmax(0)
        native_along = ((trace.state['position']-mission.points[:,1])*mission.direction).sum(-1)
        exited = native_along > mission.band_half_width
        exit_index = exited.long().argmax(0)
        entry_time = torch.where(entered,entry_index.to(p)*.01,p.new_full((N,),-1.))
        exit_time = torch.where(exited.any(0),exit_index.to(p)*.01,p.new_full((N,),-1.))
    diagnostics = {
        'command_entry_time_s':entry_time, 'first_band_exit_time_s':exit_time,
        'entered':entered, 'crossed':exists, 'event_time_s':event_time,
        'event_position':event['position'], 'event_velocity':event['velocity'],
        'event_rotation':revent, 'settled_at_H':stop_ok[-1],
        'completion_time_s':time_cost*(H*.01),
        'band_half_width_m':mission.band_half_width,
    }
    return time_cost,g,diagnostics


def evaluate_rollout(policy, simulator, initial, mission, config, multipliers=None, *, horizon=500,
                     record_trace=False, make_report=True):
    trace = rollout(policy,simulator,initial,mission,config,horizon=horizon)
    time_cost,g,details = constraints_for_trace(trace,mission,config)
    mu = torch.zeros_like(g) if multipliers is None else multipliers
    costs = phr(time_cost,g,mu,config.rho)
    report = {}
    if make_report:
        with torch.no_grad():
            violations = g.clamp_min(0)
            all_ok = (g <= 1e-6).all(-1) & details['settled_at_H']
            report = {
                'objective_mean':float(costs.mean()),
                'completion_time_mean_s':float(details['completion_time_s'].mean()),
                'all_constraints_fraction':float(all_ok.to(g).mean()),
                'midpoint_command_entry_fraction':float(details['entered'].to(g).mean()),
                'midpoint_plane_crossing_fraction':float(details['crossed'].to(g).mean()),
                'settled_at_H_fraction':float(details['settled_at_H'].to(g).mean()),
                'constraint_max':{name:float(violations[:,i].max()) for i,name in enumerate(CONSTRAINTS)},
                'constraint_violation_fraction':{name:float((g[:,i]>1e-6).to(g).mean()) for i,name in enumerate(CONSTRAINTS)},
                'physical_transitions':int(trace.valid.sum()),
                'band_half_width_range_m':[float(mission.band_half_width.min()),float(mission.band_half_width.max())],
                'constraint_sample_interval_s':.01/config.audit_substeps,
                'deployment_authorized':False,
            }
    return WaypointResult(costs,g,time_cost,report,trace if record_trace else None,
                          {k:v.detach() for k,v in details.items()} if record_trace else None)

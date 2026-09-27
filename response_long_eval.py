"""Independent, inference-only moving-target / surface-force evaluation.

Training modules do not import this environment. The force is world-fixed during
each pulse, while its application point is body-fixed. The EVAL-only RK4 kernel
therefore updates r x R(q).T F at every stage, not just every controller call.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Callable

import torch
import torch.nn.functional as F

from env_raptor import RaptorParams, RaptorSimulator, RaptorState, quaternion_rotation
from response_noise import executed_command, measured_observation
from response_policy import ResponseMotorPolicy
from response_task import _merge_rows, _select_rows

LONG_EVAL_VERSION = 'moving-hover-60s-surface-pulses-v2-gaussian-history'


@dataclass(frozen=True)
class LongEvalConfig:
    duration_seconds: float = 60.
    target_period_seconds: float = 10.
    force_period_seconds: float = 1.
    pulse_seconds: float = .1
    force_fraction: float = .2
    arena_side: float = 4.
    target_margin: float = .5
    seed: int = 20260927
    surface_padding_ratio: float = .12
    surface_half_height_ratio: float = .10
    hover_radius: float = .20
    hover_speed: float = .20
    hover_omega: float = .50
    hover_tail_seconds: float = 2.
    dt: float = .01

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{f.name} must be finite')
        if self.dt != .01 or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('EVAL requires 100 Hz and a nonnegative integer seed')
        for name in ('duration_seconds', 'target_period_seconds', 'force_period_seconds',
                     'pulse_seconds', 'hover_tail_seconds'):
            value = getattr(self, name)/self.dt
            if value < 1 or abs(value-round(value)) > 1e-8:
                raise ValueError(f'{name} must be a positive whole number of control steps')
        if self.steps % self.target_steps or self.steps % self.force_steps:
            raise ValueError('duration must contain complete target and force periods')
        if self.pulse_seconds > self.force_period_seconds or self.hover_tail_seconds > self.target_period_seconds:
            raise ValueError('pulse / hover tail must fit their respective periods')
        if self.arena_side <= 0 or not 0 < self.target_margin < self.arena_side/2:
            raise ValueError('target margin must leave a nonempty interior')
        if self.force_fraction < 0 or self.surface_padding_ratio < 0 or self.surface_half_height_ratio <= 0:
            raise ValueError('invalid force magnitude or surface dimensions')
        if min(self.hover_radius, self.hover_speed, self.hover_omega) <= 0:
            raise ValueError('hover reporting thresholds must be positive')

    @property
    def steps(self): return round(self.duration_seconds/self.dt)
    @property
    def target_steps(self): return round(self.target_period_seconds/self.dt)
    @property
    def force_steps(self): return round(self.force_period_seconds/self.dt)
    def target_index(self, step): return step//self.target_steps
    def pulse_active(self, step): return step % self.force_steps < round(self.pulse_seconds/self.dt)


@dataclass(frozen=True)
class EvalSchedule:
    targets: torch.Tensor  # [target,3], same route for all airframes
    forces_world: torch.Tensor  # [pulse,scene,3], force DURING the pulse
    points_body: torch.Tensor  # [pulse,scene,3], held at the sampled material point
    face_ids: torch.Tensor
    surface_half_extents: torch.Tensor


def arena_exited(position: torch.Tensor, config: LongEvalConfig) -> torch.Tensor:
    half = config.arena_side/2
    low = position.new_tensor((-half,-half,0.))
    high = position.new_tensor((half,half,config.arena_side))
    return ((position < low) | (position > high)).any(-1)


def sample_box_surface(half_extents: torch.Tensor, samples: int, generator: torch.Generator):
    """Uniform AREA measure, not uniform face selection or center-of-mass kicks."""
    if half_extents.ndim != 2 or half_extents.shape[1] != 3 or samples < 1:
        raise ValueError('expected positive samples and [scene,3] box half extents')
    h = half_extents.detach().cpu()
    if not bool((torch.isfinite(h) & (h > 0)).all()):
        raise ValueError('surface extents must be positive and finite')
    x,y,z = h.unbind(-1)
    areas = (4*torch.stack((y*z,x*z,x*y),-1)).repeat_interleave(2,-1)
    faces = torch.multinomial(areas,samples,replacement=True,generator=generator).T
    points = (2*torch.rand((samples,len(h),3),generator=generator,dtype=h.dtype)-1)*h
    axis = faces//2
    sign = 2*(faces % 2)-1
    surface_coordinate = h[None].expand(samples,-1,-1).gather(2,axis[...,None])[...,0]*sign
    points.scatter_(2,axis[...,None],surface_coordinate[...,None])
    return points.to(half_extents), faces.to(half_extents.device)


def make_schedule(initial: RaptorState, config: LongEvalConfig) -> EvalSchedule:
    dtype = initial.position.dtype
    n = len(initial.mass)
    target_count = config.steps//config.target_steps
    events = config.steps//config.force_steps
    targets = torch.rand((target_count,3),generator=torch.Generator().manual_seed(config.seed),dtype=dtype)
    width = config.arena_side-2*config.target_margin
    targets = (targets-.5)*width
    targets[:,2] += config.arena_side/2
    gen = torch.Generator().manual_seed(config.seed ^ 0x4F726365)
    direction = torch.randn((events,n,3),generator=gen,dtype=dtype)
    direction = F.normalize(direction,dim=-1)
    forces = direction*(config.force_fraction*9.81*initial.mass.cpu()[None,:,None])
    half = initial.rotor_positions.abs().amax(1)
    half = half + config.surface_padding_ratio*initial.arm_length[:,None]
    half[:,2] = config.surface_half_height_ratio*initial.arm_length
    points,faces = sample_box_surface(half,events,torch.Generator().manual_seed(config.seed ^ 0x706F696E))
    return EvalSchedule(targets.to(initial.position), forces.to(initial.position),points,faces,half)


def point_torque(orientation, point_body, force_world):
    body_force = (quaternion_rotation(orientation).transpose(-1,-2) @ force_world[...,None])[...,0]
    return torch.linalg.cross(point_body,body_force,dim=-1)


class PointForceSimulator(RaptorSimulator):
    """The retained joint RK4 equations plus an attitude-dependent point wrench.

    The isolated kernel is intentionally not imported into production training.
    Zero-lever force is checked bitwise against RaptorSimulator.step; nonzero
    lever force is checked against an independent NumPy RK4 reference.
    """
    def step(self, state, action, force_world, point_body):
        if action.shape != state.motor.shape or action.device != state.motor.device or action.dtype != state.motor.dtype:
            raise ValueError('action must match motor tensor')
        if any(x.shape != state.position.shape or x.device != state.position.device or x.dtype != state.position.dtype
               for x in (force_world,point_body)):
            raise ValueError('point and force must match [scene,3] state tensors')
        command = action.clamp(-1,1)
        setpoint = self.motor_command(state,executed_command(state,command))
        dt = self.params.dt
        def dynamics(values):
            p,v,q,w,m = values
            thrust = self.thrust(state,m)
            body_z = quaternion_rotation(q)[...,2]
            acceleration = body_z*(thrust.sum(-1)/state.mass)[:,None]
            gravity = torch.tensor((0.,0.,-9.81),device=v.device,dtype=v.dtype)
            acceleration = acceleration+gravity+force_world/state.mass[:,None]
            torque = self.body_torque(state,thrust)+point_torque(q,point_body,force_world)
            w_dot = (torque-torch.linalg.cross(w,state.inertia*w,dim=-1))/state.inertia
            qw,qv = q[:,:1],q[:,1:]
            q_dot = .5*torch.cat((-(qv*w).sum(-1,keepdim=True),qw*w+torch.linalg.cross(qv,w,dim=-1)),-1)
            tau = torch.where(setpoint>=m,state.motor_time_rising,state.motor_time_falling)
            return v,acceleration,q_dot,w_dot,(setpoint-m)/tau
        values = (state.position,state.velocity,state.orientation,state.omega,state.motor)
        def shifted(k,fraction): return tuple(x+(dt*fraction)*d for x,d in zip(values,k))
        k1=dynamics(values);k2=dynamics(shifted(k1,.5));k3=dynamics(shifted(k2,.5));k4=dynamics(shifted(k3,1.))
        p,v,q,w,m = tuple(x+(dt/6)*(a+2*b+2*c+d) for x,a,b,c,d in zip(values,k1,k2,k3,k4))
        q=F.normalize(q,dim=-1)
        p,v,w=(x.clamp(-100000,100000) for x in (p,v,w))
        m=torch.maximum(state.motor_min[:,None],torch.minimum(state.motor_max[:,None],m))
        return replace(state,position=p,velocity=v,orientation=q,omega=w,motor=m,
                       previous_action=command,
                       previous_velocity=torch.cat((state.velocity[:,None,:],state.previous_velocity[:,:-1,:]),1),
                       step_index=state.step_index+1,
                       external_force=force_world,external_torque=point_torque(q,point_body,force_world))


def hover_initial_state(initial: RaptorState, config: LongEvalConfig):
    """Level, stationary start in the arena center; no Actor warm-up/reset midflight."""
    p=torch.zeros_like(initial.position);p[:,2]=config.arena_side/2
    zero=torch.zeros_like(p)
    q=torch.zeros_like(initial.orientation);q[:,0]=1
    c=initial.thrust_coefficients
    required=initial.mass[:,None]*9.81/4
    motor=(-c[...,1]+(c[...,1].square()+4*c[...,2]*(required-c[...,0])).sqrt())/(2*c[...,2])
    if not bool((torch.isfinite(motor)&(motor>=initial.motor_min[:,None])&(motor<=initial.motor_max[:,None])).all()):
        raise ValueError('airframe has no feasible static hover motor state')
    # Replace TRAIN pulses as well as its constant wrench. Immutable tape rows
    # retain ORIGINAL pool IDs even when this environment evaluates a subset.
    return replace(initial,position=p,velocity=zero,
        previous_velocity=torch.zeros_like(initial.previous_velocity),orientation=q,omega=zero,
        motor=motor,previous_action=2*motor-1,external_force=zero,external_torque=zero,
        pulse_tape=torch.zeros_like(initial.pulse_tape[:,:1]),
        pulse_active_tape=torch.zeros_like(initial.pulse_active_tape[:,:1]),
        step_index=torch.zeros_like(initial.step_index))


@dataclass(frozen=True)
class LongEvalResult:
    trajectory: dict[str, torch.Tensor]
    summary: dict


class LongHoverEval:
    """A separate EVAL environment: reset once, retain all memory across goals."""
    @torch.no_grad()
    def __init__(self, policy: ResponseMotorPolicy, initial: RaptorState,
                 config: LongEvalConfig = LongEvalConfig(), schedule: EvalSchedule | None = None):
        if policy.config.dt != config.dt:
            raise ValueError('policy and EVAL control frequency must match')
        if initial.noise_tape.shape[1] != 1 and initial.noise_tape.shape[1] < config.steps+1:
            raise ValueError('sample the noise tape for the entire EVAL horizon')
        self.policy=policy
        self.config=config
        self.state=hover_initial_state(initial,config)
        self.initial=self.state
        self.schedule=make_schedule(initial,config) if schedule is None else schedule
        self.simulator=PointForceSimulator(RaptorParams(config.dt))
        self.step_index=0
        self.alive=torch.ones_like(initial.mass,dtype=torch.bool)
        self.failure_step=torch.zeros_like(initial.mass,dtype=torch.long)
        self.failure_reason=torch.zeros_like(self.failure_step)
        self.memory=policy.initial_state(self.observation())
        self.records={name:[getattr(self.state,name).detach().cpu().clone()]
                      for name in ('position','velocity','omega','orientation')}
        self.records.update({name:[] for name in ('action','valid','force_world','torque_body','memory_norm')})

    def observation(self):
        if self.step_index >= self.config.steps:
            raise RuntimeError('episode complete')
        obs=measured_observation(self.state)
        target=self.schedule.targets[self.config.target_index(self.step_index)]
        return torch.cat((obs[:,:3]-target,obs[:,3:]),-1)

    @torch.no_grad()
    def step(self):
        if self.step_index >= self.config.steps:
            raise RuntimeError('episode complete')
        before=self.state
        alive_before=self.alive.clone()
        pulse=self.step_index//self.config.force_steps
        point=self.schedule.points_body[pulse]
        force=self.schedule.forces_world[pulse]
        force=force*(alive_before & self.config.pulse_active(self.step_index))[:,None]
        torque=point_torque(before.orientation,point,force)
        action=before.previous_action.clone()
        valid=torch.zeros_like(self.alive)
        indices=self.alive.nonzero(as_tuple=True)[0]
        if indices.numel():
            live=_select_rows(before,indices)
            memory=_select_rows(self.memory,indices)
            output=self.policy(self.observation().index_select(0,indices),memory)
            after=self.simulator.step(live,output.action,force[indices],point[indices])
            tensors=[getattr(after,name) for name in ('position','velocity','orientation','omega','motor')]
            tensors += [output.action,output.next_state.memory]
            finite=torch.stack([torch.isfinite(x).flatten(1).all(1) for x in tensors]).all(0)
            bad_global=indices[~finite]
            self.failure_step[bad_global]=self.step_index+1
            self.failure_reason[bad_global]=2
            self.alive[bad_global]=False
            good_local=finite.nonzero(as_tuple=True)[0]
            good_global=indices[finite]
            if good_global.numel():
                good_after=_select_rows(after,good_local)
                good_memory=_select_rows(output.next_state,good_local)
                self.state=_merge_rows(before,_select_rows(before,good_global),good_after,good_global)
                self.memory=_merge_rows(self.memory,_select_rows(self.memory,good_global),good_memory,good_global)
                action[good_global]=output.action[finite]
                valid[good_global]=True
                exited=arena_exited(good_after.position,self.config)
                outside=good_global[exited]
                self.failure_step[outside]=self.step_index+1
                self.failure_reason[outside]=1
                self.alive[outside]=False
        self.step_index+=1
        for name in ('position','velocity','omega','orientation'):
            self.records[name].append(getattr(self.state,name).detach().cpu().clone())
        event=dict(action=action,valid=valid,force_world=force,torque_body=torque,
                   memory_norm=self.memory.memory.norm(dim=-1))
        for name,value in event.items():
            self.records[name].append(value.detach().cpu().clone())
        return dict(event,alive_before=alive_before)

    def run(self, progress: Callable[[dict], None] | None = None) -> LongEvalResult:
        while self.step_index < self.config.steps:
            self.step()
            if progress and self.step_index % self.config.target_steps == 0:
                progress(dict(simulated_seconds=self.step_index*self.config.dt,
                              alive=int(self.alive.sum()),count=len(self.alive)))
        trace={name:torch.stack(values) for name,values in self.records.items()}
        summary=summarize_long_eval(trace,self.schedule,self.config,
                                   self.failure_step.cpu(),self.failure_reason.cpu())
        return LongEvalResult(trace,summary)


def summarize_long_eval(trace, schedule, config, failure_step, failure_reason):
    """Report all attempts; frozen padding is never flight or settled hovering."""
    targets=schedule.targets.detach().cpu()
    steps=config.steps
    target_by_step=targets[torch.arange(steps)//config.target_steps]
    error=trace['position'][1:]-target_by_step[:,None]
    distance=error.norm(dim=-1)
    velocity=trace['velocity'][1:].norm(dim=-1)
    omega=trace['omega'][1:].norm(dim=-1)
    valid=trace['valid']
    n=valid.shape[1]
    reason_names={0:None,1:'arena_exit',2:'nonfinite'}
    def rms(value,mask):
        x=value[mask].double()
        return float(x.square().mean().sqrt()) if x.numel() else None
    tail=round(config.hover_tail_seconds/config.dt)
    scene_reports=[]
    for i in range(n):
        reports=[]
        failure=int(failure_step[i])
        for j in range(len(targets)):
            start=j*config.target_steps;end=start+config.target_steps
            mask=valid[start:end,i]
            completed=bool(mask.all()) and (failure==0 or failure>end)
            settled=((distance[end-tail:end,i]<=config.hover_radius)
                     & (velocity[end-tail:end,i]<=config.hover_speed)
                     & (omega[end-tail:end,i]<=config.hover_omega))
            reports.append(dict(target_index=j,target=targets[j].tolist(),
                attempted=bool(mask.any()),completed=completed,hovered=completed and bool(settled.all()),
                valid_seconds=float(mask.sum())*config.dt,
                position_error_rms=rms(distance[start:end,i],mask),
                velocity_rms=rms(velocity[start:end,i],mask),omega_rms=rms(omega[start:end,i],mask),
                final_position_error=float(distance[end-1,i]) if completed else None,
                tail_position_error_rms=rms(distance[end-tail:end,i],valid[end-tail:end,i]) if completed else None,
                tail_velocity_rms=rms(velocity[end-tail:end,i],valid[end-tail:end,i]) if completed else None,
                tail_omega_rms=rms(omega[end-tail:end,i],valid[end-tail:end,i]) if completed else None))
        scene_reports.append(dict(scene_index=i,completed=failure==0,
            failure_step=failure or None,failure_seconds=failure*config.dt if failure else None,
            failure_reason=reason_names[int(failure_reason[i])],
            valid_seconds=float(valid[:,i].sum())*config.dt,
            position_error_rms=rms(distance[:,i],valid[:,i]),velocity_rms=rms(velocity[:,i],valid[:,i]),
            omega_rms=rms(omega[:,i],valid[:,i]),targets=reports))
    complete=int((failure_step==0).sum())
    goal_count=n*len(targets)
    hovered=sum(t['hovered'] for row in scene_reports for t in row['targets'])
    return dict(protocol=LONG_EVAL_VERSION,requested_seconds=config.duration_seconds,
        scene_count=n,completed_count=complete,completion_fraction=complete/n,
        arena_exit_count=int((failure_reason==1).sum()),nonfinite_count=int((failure_reason==2).sum()),
        position_error_rms=rms(distance,valid),velocity_rms=rms(velocity,valid),omega_rms=rms(omega,valid),
        scheduled_target_count=goal_count,hovered_target_count=hovered,hovered_target_fraction=hovered/goal_count,
        hover_definition=dict(last_seconds=config.hover_tail_seconds,position_norm_m=config.hover_radius,
                              speed_norm_m_s=config.hover_speed,omega_norm_rad_s=config.hover_omega,
                              requires_complete_segment=True),
        metrics_note='RMS includes valid flight through the first arena crossing; early failures remain in completion denominators.',
        scenes=scene_reports)

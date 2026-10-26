"""Physical gradient and region semantics, not learned-flight qualification."""
from pathlib import Path
import sys
from dataclasses import asdict, replace
import math
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from env_raptor import RaptorSimulator, quaternion_rotation
from response_noise import DisturbanceConfig, measured_observation
from response_policy import ResponseMotorPolicy
from response_task import rollout, ResponseClosedLoopState
from pass_stop import (GoalMotorPolicy,GoalConfig,PassStopLossConfig,prepare,typed_costs,
                       pose_ratio_quaternion,GOAL_FEATURE_DIM,import_motor_weights,
                       _region_schedule)
from pass_stop_physics import (advance_phase,segment_ball_geometry,interval_kinematics,
                               instantaneous_acceleration)
from test_pass_stop import cfg, system, synthetic

torch.set_num_threads(1)
DTYPES=(torch.float32,torch.float64)

@pytest.mark.parametrize('dtype',DTYPES)
def test_swept_region_entry_exit_and_miss(dtype):
    a=torch.tensor([[-2.,0,0],[-2.,2,0],[-2.,0,0]],dtype=dtype)
    b=torch.tensor([[2.,0,0],[2.,2,0],[0.,0,0]],dtype=dtype)
    c=torch.zeros_like(a);r=torch.ones(3,dtype=dtype)
    phase=advance_phase(a,b,c,r,torch.zeros(3,dtype=torch.long))
    assert phase.tolist()==[2,0,1]
    hit,_,en,ex=segment_ball_geometry(a,b,c,r)
    torch.testing.assert_close(en[0],en.new_tensor(.25))
    torch.testing.assert_close(ex[0],ex.new_tensor(.75))
    assert hit.tolist()==[True,False,True]
    assert advance_phase(b,c,c,r,phase).tolist()==[2,1,1]

@pytest.mark.parametrize('dtype',DTYPES)
def test_phase_tangent_and_zero_length_are_finite(dtype):
    a=torch.tensor([[-2.,1.,0.],[2.,0.,0.],[0.,0.,0.]],dtype=dtype,requires_grad=True)
    b=torch.tensor([[2.,1.,0.],[2.,0.,0.],[0.,0.,0.]],dtype=dtype,requires_grad=True)
    hit,f,en,ex=segment_ball_geometry(a,b,torch.zeros_like(a),torch.ones(3,dtype=dtype))
    assert hit.tolist()==[False,False,True]
    gs=torch.autograd.grad((f+en+ex).sum(),(a,b))
    assert all(torch.isfinite(g).all() for g in gs)

@pytest.mark.parametrize('dtype',DTYPES)
def test_all_target_information_is_visible_and_zero_velocity_direction_finite(dtype):
    sim,s,a=system(1,dtype,10)
    closed=prepare(a,s,cfg());state=closed.policy;obs=measured_observation(closed.physical)
    feature=a.goal_features(obs,state,state.phase)
    assert feature.shape==(1,GOAL_FEATURE_DIM) and torch.isfinite(feature).all()
    assert feature[:,50:53].count_nonzero()==0 and feature[:,56:59].count_nonzero()==0
    for field in ('start','via','stop','start_rotation','via_rotation','stop_rotation','target_velocities'):
        value=getattr(state,field).clone()
        if 'rotation' in field:
            angle=.1
            R=value.new_tensor(((math.cos(angle),-math.sin(angle),0),
                                (math.sin(angle),math.cos(angle),0),(0,0,1)))
            value=value@R
        else: value=value+.1
        changed=a.goal_features(obs,replace(state,**{field:value}),state.phase)
        assert not torch.equal(feature,changed),field
    rotated=obs.clone();rotated[:,6:15]=quaternion_rotation(obs.new_tensor([[.92387953,0,0,.38268343]])).flatten(1)
    assert not torch.equal(feature[:,59:65],a.goal_features(rotated,state,state.phase)[:,59:65])

@pytest.mark.parametrize('dtype',DTYPES)
def test_minimum_speed_loss_to_pose_and_motor_uses_native_rk4(dtype):
    sim,ini,a=system(1,dtype,12)
    closed=prepare(a,ini,cfg())
    angle=torch.tensor(.17,dtype=dtype,requires_grad=True)
    q=torch.stack((torch.cos(angle/2),torch.sin(angle/2),angle*0,angle*0))[None]
    cmd=(2*closed.physical.motor.detach()-1+.03).requires_grad_()
    state=replace(closed.physical,orientation=q)
    for _ in range(8):state=sim.step(state,cmd)
    loss=10*((2-state.velocity.norm(dim=-1))/.5).clamp_min(0).square().sum()
    ga,gu=torch.autograd.grad(loss,(angle,cmd))
    assert torch.isfinite(ga) and torch.isfinite(gu).all()
    assert ga.abs()>0 and gu.abs().max()>0
    if dtype==torch.float64:
        def value(ang,action):
            q_=torch.stack((torch.cos(ang/2),torch.sin(ang/2),ang*0,ang*0))[None]
            st=replace(closed.physical,orientation=q_)
            for _ in range(8):st=sim.step(st,action)
            return 10*((2-st.velocity.norm(dim=-1))/.5).clamp_min(0).square().sum()
        eps=1e-6
        fd=(value(angle.detach()+eps,cmd.detach())-value(angle.detach()-eps,cmd.detach()))/(2*eps)
        torch.testing.assert_close(ga,fd,rtol=1e-6,atol=1e-7)
        z=torch.tensor([[.4,-.2,.1,.3]],dtype=dtype)
        fd=(value(angle.detach(),cmd.detach()+eps*z)-value(angle.detach(),cmd.detach()-eps*z))/(2*eps)
        torch.testing.assert_close((gu*z).sum(),fd,rtol=1e-6,atol=1e-7)

@pytest.mark.parametrize('degrees',[0.,45.,90.,105.,170.])
def test_rk4_matches_constant_acceleration_and_analytic_pose_gradient(degrees):
    dtype=torch.float64
    sim,ini,a=system(1,dtype,2);closed=prepare(a,ini,cfg())
    angle=torch.tensor(math.radians(degrees),dtype=dtype,requires_grad=True)
    q=torch.stack((torch.cos(angle/2),torch.sin(angle/2),angle*0,angle*0))[None]
    velocity=torch.tensor([[.3,-.1,.2]],dtype=dtype,requires_grad=True)
    st=replace(closed.physical,orientation=q,velocity=velocity)
    command=2*st.motor-1
    result=sim.step(st,command)
    acceleration,_=instantaneous_acceleration(st)
    torch.testing.assert_close(result.velocity,velocity+.01*acceleration,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(result.position,st.position+.01*velocity+.5*.01**2*acceleration,rtol=1e-12,atol=1e-12)
    derivative=torch.autograd.grad(result.velocity[0,1],angle,retain_graph=True)[0]
    expected=-.01*9.81*torch.cos(angle)
    torch.testing.assert_close(derivative,expected,rtol=1e-10,atol=1e-10)
    gp=torch.autograd.grad(result.position.sum(),velocity)[0]
    torch.testing.assert_close(gp,torch.full_like(gp,.01),rtol=1e-12,atol=1e-12)

@pytest.mark.parametrize('dtype',DTYPES)
def test_acceleration_and_jerk_reuse_original_differentiable_endpoints(dtype):
    sim,ini,a=system(1,dtype,6);closed=prepare(a,ini,cfg())
    tr=rollout(a,sim,closed,6)
    k=interval_kinematics(tr)
    v=torch.cat((tr.initial.velocity[None],tr.velocities))
    torch.testing.assert_close(k.acceleration,(v[1:]-v[:-1])/.01)
    g=torch.autograd.grad(k.acceleration.square().sum()+1e-3*k.jerk.square().sum(),a.readout.weight)[0]
    assert torch.isfinite(g).all() and g.abs().max()>0

@pytest.mark.parametrize('dtype',DTYPES)
def test_fast_average_does_not_hide_an_instantaneous_speed_dip(dtype):
    p=torch.tensor([[-1.,0,0],[0.,0,0],[1.,0,0]],dtype=dtype)
    v=p.new_tensor([[3.,0,0],[1.,0,0],[3.,0,0]])[:,None]
    tr,c=synthetic(p,v=v)
    assert typed_costs(tr,c)[2]['pass_speed'].item()==pytest.approx(40.)

@pytest.mark.parametrize('dtype',DTYPES)
def test_region_loss_does_not_disappear_with_fewer_samples_or_good_revisit(dtype):
    p=torch.tensor([[-1.,0,0],[0.,0,0],[1.,0,0]],dtype=dtype)
    q=p.new_tensor(GoalConfig().via_quaternion).expand(3,1,4)
    v=p.new_tensor((1.,0.,0.)).expand(3,1,3)
    tr,c=synthetic(p,q=q,v=v)
    base=typed_costs(tr,c)[2]['pass_speed']
    # A single segment through the ball is also supervised.
    quick,_=synthetic(p[[0,2]],q=q[[0,2]],v=v[[0,2]])
    torch.testing.assert_close(typed_costs(quick,c)[2]['pass_speed'],base)
    again=torch.cat((p,p.flip(0),p[1:]),0)
    qs=q[:1].expand(len(again),1,4)
    vs=v[:1].expand(len(again),1,3).clone();vs[3:]=p.new_tensor((2.,0,0))
    twice,_=synthetic(again,q=qs,v=vs)
    torch.testing.assert_close(typed_costs(twice,c)[2]['pass_speed'],base)

@pytest.mark.parametrize('dtype',DTYPES)
def test_pose_ratio_matches_matrix_formula_and_is_sign_invariant(dtype):
    from pass_stop import attitude_ratio
    target=torch.tensor(GoalConfig().via_quaternion,dtype=dtype)
    q=torch.nn.functional.normalize(torch.tensor([[.3,.5,.1,-.2]],dtype=dtype),dim=-1).requires_grad_()
    ratio=pose_ratio_quaternion(q,target,1.)
    torch.testing.assert_close(ratio,pose_ratio_quaternion(-q,target,1.))
    torch.testing.assert_close(ratio,attitude_ratio(q,quaternion_rotation(target),1.),rtol=1e-5,atol=1e-5)
    assert torch.isfinite(torch.autograd.grad(ratio.sum(),q)[0]).all()

@pytest.mark.parametrize('key,value',[('region_radius_arms',0),('velocity_weight',-1),
                                    ('pass_speed_scale_m_s',0),('acceleration_weight',float('nan'))])
def test_invalid_config(key,value):
    with pytest.raises(ValueError):GoalConfig(**{key:value})

def test_zero_start_stop_velocity_contract():
    for v in (((1,0,0),(2,0,0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0)),((0,0,0),(2,0,0),(0,1,0))):
        with pytest.raises(ValueError):GoalConfig(velocities=v)

@pytest.mark.parametrize('dtype',DTYPES)
def test_optional_nmi_decay_preserves_forward_but_changes_adjoint(dtype):
    from response_adjoints import collect_rollout,backward_actor
    from response_groups import GroupBalanceConfig
    torch.manual_seed(15)
    sim,ini,a=system(32,dtype,12)
    group=GroupBalanceConfig(max_groups=1,min_scenarios=32,clip_norm=1e30)
    values=[];grads=[]
    for alpha in (0.,1.):
        r=collect_rollout(a,sim,ini,cfg(),horizon=12,time_decay=alpha,group_config=group)
        values.append(r.costs.detach())
        backward_actor(a,sim,r,cfg())
        grads.append(torch.cat([p.grad.flatten() for p in a.parameters()]))
    assert torch.equal(values[0],values[1])
    assert torch.isfinite(grads[1]).all() and not torch.equal(grads[0],grads[1])


def test_late_zero_velocity_with_unbalanced_motors_is_not_stopped():
    p=torch.tensor([[-1.,0,0],[0.,0,0],[1.,0,0]],dtype=torch.float64)
    tr,c=synthetic(p)
    # Position/velocity alone would say stationary; change the real final motors.
    balanced=replace(tr.end.physical,previous_action=2*tr.end.physical.motor-1)
    tr.end=replace(tr.end,physical=balanced)
    cost=typed_costs(tr,c)[2]['stop']
    tr.end=replace(tr.end,physical=replace(balanced,motor=balanced.motor+.05))
    assert typed_costs(tr,c)[2]['stop']>cost


def test_dead_padding_does_not_create_exit():
    sim,ini,a=system(1,torch.float64,4);closed=prepare(a,ini,cfg())
    state=closed.policy
    centre=state.via[0]
    measured=torch.zeros(5,1,22,dtype=torch.float64)
    measured[:,:,:3]=torch.stack((centre-3*state.radius[0],centre,centre,centre,centre))[:,None]
    valid=torch.tensor([[True],[False],[False],[False]])
    region,_,_,_,exit_index=_region_schedule(measured,state,valid)
    assert region.sum()==1 and exit_index.item()==-1

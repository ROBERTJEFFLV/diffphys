"""Focused regressions for the two new task concepts; not learned-flight tests."""
from dataclasses import asdict, replace
from pathlib import Path
import math
import sys
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_raptor import RaptorParams, RaptorSimulator, quaternion_rotation
from response_noise import DisturbanceConfig, measured_observation
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig, rollout
from pass_stop import (GoalConfig, GoalMotorPolicy, PassStopLossConfig, temporal_proxy,
                       typed_costs, prepare, task_rollout, attitude_ratio, import_motor_weights)

torch.set_num_threads(1)
DTYPES=(torch.float32,torch.float64)

def cfg(**kwargs):
    return PassStopLossConfig(.01,.01,25,mission=asdict(GoalConfig(**kwargs)))

def system(n=2,dtype=torch.float64,steps=20,noise=False):
    sim=RaptorSimulator()
    ini=sim.reset(n,seed=77,dtype=dtype,horizon=steps,
                  disturbances=DisturbanceConfig() if noise else DisturbanceConfig.clean())
    actor=GoalMotorPolicy().to(dtype)
    return sim,ini,actor

def synthetic(p,q=None,v=None,w=None, measured=None,config=None):
    """Time-major position/attitude samples with real task metadata."""
    config=config or cfg(relative_to_boundary=False,points=((-1.,0.,0.),(0.,0.,0.),(1.,0.,0.)))
    if p.ndim==2:p=p[:,None,:]
    H,N=p.shape[0]-1,p.shape[1]
    sim,ini,actor=system(N,p.dtype,H)
    ini=replace(ini,position_limit=p.new_full((N,),30.))
    closed=prepare(actor,ini,config)
    if q is None:q=p.new_tensor((1,0,0,0)).expand(H+1,N,4)
    if v is None:v=torch.zeros_like(p)
    if w is None:w=torch.zeros_like(p)
    R=quaternion_rotation(q)
    observations=torch.cat((p if measured is None else measured,v,R.flatten(2),w,p.new_zeros(H+1,N,4)),-1)
    initial=replace(closed.physical,position=p[0],velocity=v[0],omega=w[0],orientation=q[0])
    return SimpleNamespace(initial=initial, positions=p[1:],velocities=v[1:],post_orientations=q[1:],
        omegas=w[1:],observations=observations,actions=p.new_zeros(H,N,4),action_deltas=p.new_zeros(H,N,4),
        valid=torch.ones(H,N,dtype=torch.bool),end=closed),config

def shifted_ratios(tp,ts,H=500):
    t=torch.arange(H+1,dtype=tp.dtype)*.01
    # A single PASS event; monotone approach to STOP, without prescribed commands.
    passed=1.05+((t-tp)/.17).square()
    stopped=1.05+torch.nn.functional.softplus((ts-t)/.12)
    return passed[:,None],stopped[:,None]

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('tp,ts',[(.73,1.73),(1.13,2.23),(2.23,3.83)])
def test_time_shift_derivative(dtype,tp,ts):
    a=torch.tensor(tp+.00037,dtype=dtype,requires_grad=True)
    b=torch.tensor(ts+.00023,dtype=dtype,requires_grad=True)
    loss=temporal_proxy(*shifted_ratios(a,b))[0].sum()
    ga,gb=torch.autograd.grad(loss,(a,b))
    assert torch.isfinite(ga+gb) and gb>0 # delaying completion costs more
    assert abs(float(ga))<2. # no exploding time derivative
    if dtype==torch.float64:
        eps=1e-6
        fd=(temporal_proxy(*shifted_ratios(a.detach(),b.detach()+eps))[0]-
            temporal_proxy(*shifted_ratios(a.detach(),b.detach()-eps))[0])/(2*eps)
        torch.testing.assert_close(gb,fd.squeeze(),atol=2e-7,rtol=2e-6)

@pytest.mark.parametrize('dtype',DTYPES)
def test_proxy_binary_exact_and_departure(dtype):
    # Ratios 1 => satisfaction1, infinity =>0; exact hard clock limit.
    p=torch.full((501,1),torch.inf,dtype=dtype);p[80]=1
    s=torch.full_like(p,torch.inf);s[200:]=1
    value,_=temporal_proxy(p,s)
    torch.testing.assert_close(value,torch.tensor([2.],dtype=dtype))
    s[400]=torch.inf
    value,_=temporal_proxy(p,s)
    torch.testing.assert_close(value,torch.tensor([4.01],dtype=dtype))
    p[:]=torch.inf
    assert float(temporal_proxy(p,s)[0])==5.

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('seed',range(8))
def test_proxy_monotonic_derivatives(dtype,seed):
    gen=torch.Generator().manual_seed(seed)
    p=(torch.rand(51,5,dtype=dtype,generator=gen)*30+1.02).requires_grad_()
    s=(torch.rand(51,5,dtype=dtype,generator=gen)*20+1.03).requires_grad_()
    t,c=temporal_proxy(p,s)
    assert bool(((t>=0)&(t<=.5)).all())
    gp,gs=torch.autograd.grad(t.sum(),(p,s))
    assert bool((gp>=-1e-12).all() & (gs>=-1e-12).all())
    assert torch.isfinite(gp).all() and torch.isfinite(gs).all()
    # No cross-scene mixing.
    torch.testing.assert_close(t[:1],temporal_proxy(p[:,:1],s[:,:1])[0])

@pytest.mark.parametrize('dtype',DTYPES)
def test_no_repeated_weak_evidence(dtype):
    p=torch.full((501,1),10.,dtype=dtype);s=torch.ones_like(p)
    a,c=temporal_proxy(p,s)
    assert float(c.max())==pytest.approx(.1)
    # Repeating the same weak state never approaches full success.
    p[200:]=3
    _,c=temporal_proxy(p,s)
    assert float(c[-1])==pytest.approx(1/3)

@pytest.mark.parametrize('dtype',DTYPES)
def test_first_bad_visit_not_erased(dtype):
    p=torch.full((501,1),40.,dtype=dtype);p[100]=8.;p[200]=.5
    s=torch.ones_like(p)
    _,c=temporal_proxy(p,s,first_hit=torch.tensor([100]))
    assert float(c[-1])==pytest.approx(1/8)
    _,free=temporal_proxy(p,s)
    assert float(free[-1])==1.

@pytest.mark.parametrize('dtype',DTYPES)
def test_failure_cannot_claim_stopped_padding(dtype):
    p=torch.ones(501,1,dtype=dtype);s=p.clone();alive=torch.ones_like(p,dtype=torch.bool)
    alive[200:]=False
    t,_=temporal_proxy(p,s,alive=alive)
    assert float(t)==5.

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('deg',[0,45,90,105,179])
def test_attitude_valid_and_sign_invariant(dtype,deg):
    angle=torch.tensor(math.radians(deg),dtype=dtype,requires_grad=True)
    q=torch.stack((torch.cos(angle/2),torch.sin(angle/2),angle*0,angle*0))[None]
    ratio=attitude_ratio(q,torch.eye(3,dtype=dtype)[None],1.)
    torch.testing.assert_close(ratio,attitude_ratio(-q,torch.eye(3,dtype=dtype)[None],1.))
    g=torch.autograd.grad(ratio.sum(),angle)[0]
    assert torch.isfinite(g)
    if deg>0:assert g>0

@pytest.mark.parametrize('dtype',DTYPES)
def test_import_matches_translated_hover_multistep(dtype):
    torch.manual_seed(71)
    hover=ResponseMotorPolicy().to(dtype);goal=GoalMotorPolicy().to(dtype)
    import_motor_weights(goal,hover.state_dict())
    sim,ini,_=system(2,dtype,8)
    obs=measured_observation(ini);end=torch.tensor([[1.,2.,0.],[-1.,0.,.5]],dtype=dtype)
    I=torch.eye(3,dtype=dtype)[None].expand(2,-1,-1)
    hs=hover.initial_state(obs)
    gs=goal.task_state(obs,end-3,end,I,I,.1,is_stop=True)
    for k in range(8):
        adj=torch.cat((obs[:,:3]-end,obs[:,3:]),-1)
        h=hover(adj,hs);g=goal(obs,gs)
        torch.testing.assert_close(h.action,g.action,rtol=5e-6,atol=5e-8)
        torch.testing.assert_close(h.next_state.memory,g.next_state.memory,rtol=5e-6,atol=1e-7)
        hs,gs=h.next_state,g.next_state
        obs=measured_observation(sim.step(ini,h.action))
    assert len(list(goal.parameters()))==6
    assert goal.response_memory.input_size==26 and goal.response_memory.hidden_size==64

@pytest.mark.parametrize('dtype',DTYPES)
def test_switch_position_only_latches_and_does_not_reset(dtype):
    sim,ini,actor=system(2,dtype)
    closed=prepare(actor,ini,cfg());state=closed.policy
    state=replace(state,memory=torch.full_like(state.memory,.25))
    obs=measured_observation(closed.physical)
    obs=torch.cat((state.via+torch.tensor([[.05,0,0],[.11,0,0]],dtype=dtype),obs[:,3:]),-1)
    out=actor(obs,state)
    assert out.next_state.is_stop.tolist()==[True,False]
    # Incorrect pose does not prevent switching, and old memory contributes.
    other=actor(obs,replace(state,memory=torch.zeros_like(state.memory)))
    assert not torch.equal(out.next_state.memory,other.next_state.memory)
    far=torch.cat((state.via+2,obs[:,3:]),-1)
    assert actor(far,out.next_state).next_state.is_stop.tolist()==[True,False]

@pytest.mark.parametrize('dtype',DTYPES)
def test_sample_task_preserves_physics_sensor_velocity(dtype):
    sim,ini,actor=system(32,dtype,10,True)
    closed=prepare(actor,ini,cfg())
    s=closed.physical
    for name in ('mass','inertia','rotor_positions','thrust_coefficients','motor_time_rising','motor_time_falling',
                 'velocity','velocity_delay','noise_tape','rotation_tape','position_limit'):
        assert torch.equal(getattr(s,name),getattr(ini,name))
    assert not s.external_force.any() and not s.pulse_tape.any()
    assert torch.equal(s.previous_velocity[:,0],s.velocity)
    a=quaternion_rotation(s.orientation)[:,:,2]*(sim.thrust(s,s.motor).sum(-1)/s.mass)[:,None]
    torch.testing.assert_close(a,s.position.new_tensor((0,0,9.81)).expand_as(a),rtol=1e-5,atol=1e-5)
    assert not s.previous_action.any()
    assert torch.all(s.position.abs()<s.position_limit[:,None])

@pytest.mark.parametrize('dtype',DTYPES)
def test_pass_score_no_speed_requirement_and_same_time_pose(dtype):
    p=torch.tensor([[-1.,0,0],[.05,0,0],[1.,0,0]],dtype=dtype)
    wrong,c=synthetic(p)
    cval=typed_costs(wrong,c)[2]['pass_pose']
    q=p.new_tensor(GoalConfig().via_quaternion).expand(3,1,4)
    good,c=synthetic(p,q=q)
    base=typed_costs(good,c)[2]['pass_pose']
    assert float(base)==0 and float(cval)>1
    # Correct pose away from the first position hit does not satisfy the task.
    q2=q.clone();q2[1]=p.new_tensor((1,0,0,0))
    mixed,c=synthetic(p,q=q2)
    assert float(typed_costs(mixed,c)[2]['pass_pose'])>1
    for speed in (0.,.1,3.,8.):
        velocity=torch.zeros(3,1,3,dtype=dtype);velocity[1,:,0]=speed
        speedy,c=synthetic(p,q=q,v=velocity)
        torch.testing.assert_close(typed_costs(speedy,c)[2]['pass_pose'],base)

@pytest.mark.parametrize('dtype',DTYPES)
def test_position_gradient_cannot_hide_pose_by_distance(dtype):
    p=torch.tensor([[-1.,0,0],[.3,0,0],[1.,0,0]],dtype=dtype,requires_grad=True)
    q=p.new_tensor(GoalConfig().via_quaternion).expand(3,1,4).clone()
    q[1]=p.new_tensor((.9,math.sqrt(.19),0,0))
    tr,c=synthetic(p,q=q)
    # The single best joint witness can vary: first freeze it by making endpoints worse.
    q[0]=q[2]=p.new_tensor((1.,0,0,0))
    val=typed_costs(tr,c)[2]['pass_pose'].sum()
    g=torch.autograd.grad(val,p)[0]
    assert g[1,0]>=0 # descent does not move the selected point outward

@pytest.mark.parametrize('dtype',DTYPES)
def test_h500_real_physics_backward(dtype):
    torch.manual_seed(4)
    sim,ini,actor=system(2,dtype,500)
    tr=task_rollout(actor,sim,ini,500,cfg())
    costs,report,_=typed_costs(tr,cfg())
    grads=torch.autograd.grad(costs.mean(),list(actor.parameters()))
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().max()>0 for g in grads)
    assert tr.actions.shape==(500,2,4)
    assert report['physical_transitions']>0

@pytest.mark.parametrize('dtype',DTYPES)
def test_joint_temporal_gradcheck(dtype):
    if dtype==torch.float32:
        # A first-order audit is the appropriate check in float32.
        p=torch.tensor([[9.],[7.],[4.],[3.]],dtype=dtype,requires_grad=True)
        s=torch.tensor([[11.],[10.],[8.],[6.]],dtype=dtype,requires_grad=True)
        a,b=torch.autograd.grad(temporal_proxy(p,s)[0].sum(),(p,s))
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
    else:
        p=torch.tensor([[9.],[7.],[4.],[3.]],dtype=dtype,requires_grad=True)
        s=torch.tensor([[11.],[10.],[8.],[6.]],dtype=dtype,requires_grad=True)
        assert torch.autograd.gradcheck(lambda a,b:temporal_proxy(a,b)[0],(p,s),eps=1e-6,atol=1e-7)

@pytest.mark.parametrize('dtype',DTYPES)
def test_all_500_transitions_remain_connected(dtype):
    sim,ini,actor=system(1,dtype,500)
    ini=replace(ini,velocity=torch.zeros_like(ini.velocity))
    prepared=prepare(actor,ini,cfg())
    command=2*prepared.physical.motor-1
    with torch.no_grad():
        actor.readout.weight.zero_()
        actor.readout.bias.copy_(torch.atanh(command[0])+.001)
    tr=task_rollout(actor,sim,ini,500,cfg())
    assert tr.valid.all() and tr.valid.sum()==500
    costs,_,_=typed_costs(tr,cfg())
    g=torch.autograd.grad(costs.mean(),actor.readout.bias)[0]
    assert torch.isfinite(g).all() and g.abs().max()>0
    if dtype==torch.float64:
        original=actor.readout.bias.detach().clone()
        z=torch.ones(4,dtype=dtype)/2
        values=[];eps=2e-7
        with torch.no_grad():
            for sign in (-1,1):
                actor.readout.bias.copy_(original+sign*eps*z)
                t=task_rollout(actor,sim,ini,500,cfg())
                values.append(typed_costs(t,cfg())[0].mean())
            actor.readout.bias.copy_(original)
        fd=(values[1]-values[0])/(2*eps)
        torch.testing.assert_close(g@z,fd,rtol=5e-4,atol=.2)

@pytest.mark.parametrize('dtype',DTYPES)
def test_failed_approach_fallback_not_fixed_spawn(dtype):
    p=torch.tensor([[-1.,0,0],[-1.01,0,0],[-1.02,0,0]],dtype=dtype,requires_grad=True)
    tr,c=synthetic(p)
    value=typed_costs(tr,c)[2]['pass_pose'].sum()
    g=torch.autograd.grad(value,p)[0]
    assert g[0].abs().sum()==0 and g[1,0]<0

@pytest.mark.parametrize('dtype',DTYPES)
def test_stop_braking_penalty_unconditional_on_pass(dtype):
    p=torch.tensor([[-1.,0,0],[-.5,.4,0],[1.,0,0]],dtype=dtype)
    v=torch.zeros(3,1,3,dtype=dtype);v[-1,:,0]=2.
    trace,c=synthetic(p,v=v)
    cost=typed_costs(trace,c)[2]['stop']
    assert cost>0
    p[1]=p.new_tensor((.05,0,0))
    trace2,c=synthetic(p,v=v)
    torch.testing.assert_close(typed_costs(trace2,c)[2]['stop'],cost)

"""True short-window differential metric checks; no time-decay certificate."""
import copy
from dataclasses import replace
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_l2f import L2FParams, L2FSimulator
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import initialize, rollout
from response_contraction import (
    ContractionConfig, ContractionMetric, StateGeometry, contraction_loss,
    local_flow, sample_boundaries,
)


def setup(profile='raptor', n=3, horizon=20, dtype=torch.float64):
    torch.manual_seed(7)
    actor = ResponseMotorPolicy(ResponsePolicyConfig(hidden_dim=8, memory_dim=8)).to(dtype)
    sim = L2FSimulator(L2FParams(protocol=profile))
    closed = initialize(actor, sim.reset(n, seed=42, horizon=horizon, dtype=dtype))
    cfg = ContractionConfig(weight=.1, steps=3, samples=3, directions=2, hidden_dim=8, rank=2)
    metric = ContractionMetric(actor.config.memory_dim, cfg).to(dtype)
    return actor, sim, closed, metric, cfg


def test_metric_is_uniformly_bounded_and_differentiable():
    a, s, z, metric, cfg = setup()
    geo = StateGeometry(z, a.config.integral_limit)
    x = geo.pack(z)
    context = geo.context(z)
    M = metric.matrix(x, context)
    eig = torch.linalg.eigvalsh(M)
    assert float(eig.detach().min()) >= cfg.metric_min
    assert float(eig.detach().max()) <= cfg.metric_max
    v = torch.randn_like(x)
    torch.testing.assert_close(metric.energy(x, context, v), (v * (M @ v[..., None]).squeeze(-1)).sum(-1))
    metric.energy(x, context, v).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in metric.parameters())


@pytest.mark.parametrize('profile', ['l2f', 'raptor'])
def test_retraction_zero_and_legal_rotations(profile):
    a, sim, z, metric, cfg = setup(profile)
    geo = StateGeometry(z, a.config.integral_limit)
    delta = z.physical.position.new_zeros(3, geo.tangent_dim)
    moved = geo.retract(delta)
    torch.testing.assert_close(geo.pack(moved), geo.pack(z), rtol=0, atol=0)
    moved = geo.retract(torch.randn_like(delta)*1e-3)
    R = moved.physical.rotation
    torch.testing.assert_close(R.transpose(-1,-2)@R, torch.eye(3,dtype=R.dtype).expand_as(R), atol=1e-12, rtol=1e-12)
    Rprev = moved.policy.previous_rotation - geo.previous_rotation_noise
    torch.testing.assert_close(Rprev.transpose(-1,-2)@Rprev, torch.eye(3,dtype=R.dtype).expand_as(R), atol=1e-12, rtol=1e-12)
    assert moved.physical.noise_tape.data_ptr() == z.physical.noise_tape.data_ptr()
    assert torch.equal(moved.physical.mass, z.physical.mass)


@pytest.mark.parametrize('profile', ['l2f','raptor'])
def test_true_flow_jvp_matches_central_difference_and_rollout(profile):
    actor, sim, z, metric, cfg = setup(profile)
    geo = StateGeometry(z, actor.config.integral_limit)
    zero = z.physical.position.new_zeros(3, geo.tangent_dim)
    direction = torch.randn_like(zero); direction /= direction.norm(dim=-1,keepdim=True)
    fn = lambda d: local_flow(actor,sim,geo.retract(d),geo,cfg.steps)[0]
    output, tangent = torch.autograd.functional.jvp(fn,zero,direction,create_graph=True)
    epsilon=1e-6
    numeric=(fn(epsilon*direction)-fn(-epsilon*direction))/(2*epsilon)
    torch.testing.assert_close(tangent,numeric,rtol=2e-4,atol=2e-5)
    true=rollout(actor,sim,z,cfg.steps,time_decay=0)
    torch.testing.assert_close(output[-1],geo.pack(true.end),rtol=1e-12,atol=1e-12)
    grads=torch.autograd.grad(tangent.square().sum(),list(actor.parameters()),allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    assert sum(float(g.norm()) for g in grads if g is not None)>0


def test_joint_loss_trains_actor_and_metric():
    a,s,z,m,cfg=setup()
    loss, report=contraction_loss(a,s,m,z,cfg,seed=11)
    assert loss.requires_grad and report['certified'] is False
    loss.backward()
    ag=sum(float(p.grad.norm()) for p in a.parameters() if p.grad is not None)
    mg=sum(float(p.grad.norm()) for p in m.parameters() if p.grad is not None)
    assert ag>0 and mg>0
    assert report['evaluated_directions']==6


@pytest.mark.parametrize('kw', [dict(weight=-1),dict(steps=0),dict(samples=0),dict(directions=0),dict(rate=-1),dict(metric_min=0),dict(metric_min=3,metric_max=2),dict(rank=0)])
def test_invalid_config(kw):
    with pytest.raises(ValueError):
        ContractionConfig(**kw)


def test_heading_is_kept_in_geometry_but_not_a_new_task_target():
    a,s,z,m,cfg=setup(n=1)
    p=replace(z.physical, position=z.physical.position*0,velocity=z.physical.velocity*0,
              orientation=z.physical.orientation.new_tensor([[1.,0,0,0]]),omega=z.physical.omega*0)
    z=replace(z,physical=p)
    geo=StateGeometry(z,a.config.integral_limit)
    zero=p.position.new_zeros(1,geo.tangent_dim)
    direction=torch.zeros_like(zero);direction[0,8]=1.  # Pure right heading rotation.
    _,v=torch.autograd.functional.jvp(lambda d:geo.pack(geo.retract(d)),zero,direction)
    assert float(v.norm()) == pytest.approx(1.)
    assert float(geo.task_direction(v).norm()) == 0
    direction.zero_();direction[0,6]=1.
    _,tilt=torch.autograd.functional.jvp(lambda d:geo.pack(geo.retract(d)),zero,direction)
    assert float(geo.task_direction(tilt).norm()) == pytest.approx(1.)
    flipped=replace(z,physical=replace(z.physical,orientation=-z.physical.orientation))
    torch.testing.assert_close(geo.pack(flipped),geo.pack(z),rtol=0,atol=0)


@pytest.mark.parametrize('profile',['l2f','raptor'])
def test_history_noise_and_parameter_gradient_finite_difference(profile):
    a,s,z,m,cfg=setup(profile,n=1)
    z=rollout(a,s,z,2).end
    geo=StateGeometry(z,a.config.integral_limit)
    clean=geo.base.policy.previous_rotation-geo.previous_rotation_noise
    eye=torch.eye(3,dtype=clean.dtype)[None]
    torch.testing.assert_close(clean.transpose(-1,-2)@clean,eye,atol=1e-12,rtol=1e-12)
    loss,_=contraction_loss(a,s,m,z,cfg,seed=18)
    parameter=a.controller[-1].bias
    g=torch.autograd.grad(loss,parameter)[0]
    index=int(g.abs().argmax()); original=float(parameter[index].detach()); epsilon=1e-6
    values=[]
    try:
        for sign in (1,-1):
            with torch.no_grad():parameter[index]=original+sign*epsilon
            score,_=contraction_loss(a,s,m,z,cfg,seed=18,differentiable=False)
            values.append(float(score))
    finally:
        with torch.no_grad():parameter[index]=original
    numeric=(values[0]-values[1])/(2*epsilon)
    assert float(g[index]) == pytest.approx(numeric,rel=2e-4,abs=1e-5)


def test_loss_never_calls_decay_and_does_not_reinitialize_history(monkeypatch):
    import response_task
    a,s,z,m,cfg=setup()
    z=rollout(a,s,z,2).end
    def forbidden(*a,**kw):raise AssertionError('modified or missing history in true derivative')
    monkeypatch.setattr(response_task._GradientDecay,'apply',forbidden)
    monkeypatch.setattr(a,'initial_state',forbidden)
    loss,report=contraction_loss(a,s,m,z,cfg,seed=1)
    loss.backward()
    assert report['evaluated_directions']>0


def test_terminal_prefix_is_retained_not_falsely_certified():
    a,s,z,m,cfg=setup()
    p=z.physical.position.clone();v=z.physical.velocity.clone()
    p[0]=0;p[0,0]=z.physical.position_limit[0]-1e-8;v[0,0]=1.
    z=replace(z,physical=replace(z.physical,position=p,velocity=v))
    geo=StateGeometry(z,a.config.integral_limit)
    values,valid=local_flow(a,s,z,geo,3)
    assert valid[:,0].tolist()==[True,False,False]
    assert bool(valid[:,1:].any())
    loss,report=contraction_loss(a,s,m,z,cfg,seed=1)
    assert report['terminal_intervals']>=1 and report['certified'] is False
    assert report['true_forward_transitions']==int(valid.sum())*(cfg.directions+1)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.parameters())


def test_local_worst_direction_dominates_sampled_directions():
    from response_contraction import worst_direction
    a,s,z,m,cfg=setup(n=1)
    cfg=replace(cfg,steps=2,directions=4)
    m=ContractionMetric(a.config.memory_dim,cfg).double()
    _,sampled=contraction_loss(a,s,m,z,cfg,seed=12,differentiable=False)
    worst=worst_direction(a,s,m,z,cfg)
    assert worst['max_ratio'] >= sampled['max_ratio']-1e-9
    geo=StateGeometry(z,a.config.integral_limit)
    zero=z.physical.position.new_zeros(1,geo.tangent_dim)
    values,tangent=torch.autograd.functional.jvp(lambda d:local_flow(a,s,geo.retract(d),geo,cfg.steps)[0],zero,worst['direction'][None])
    context=geo.context(z)
    v0=m.energy(values[0],context,tangent[0])
    v1=m.energy(values[-1],context,tangent[-1])
    ratio=(v1+cfg.rate*cfg.steps*s.params.dt*geo.task_direction(tangent[0]).square().sum(-1))/v0
    assert float(ratio.detach())==pytest.approx(worst['max_ratio'],rel=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_true_jvp_and_joint_backward():
    a,s,z,m,cfg=setup(n=2,dtype=torch.float32)
    a=a.cuda();m=m.cuda()
    from response_task import ResponseClosedLoopState
    from dataclasses import fields
    z=ResponseClosedLoopState(z.physical.to('cuda',torch.float32),
        type(z.policy)(**{f.name:getattr(z.policy,f.name).cuda() for f in fields(z.policy)}))
    loss,_=contraction_loss(a,s,m,z,cfg,seed=4)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(a.parameters())+list(m.parameters()))


@pytest.mark.parametrize('profile',['l2f','raptor'])
def test_float32_default_width_jvp_and_joint_gradient(profile):
    torch.manual_seed(7)
    a=ResponseMotorPolicy().float()
    s=L2FSimulator(L2FParams(protocol=profile))
    z=initialize(a,s.reset(2,horizon=20,seed=42,dtype=torch.float32))
    cfg=ContractionConfig(weight=.1,steps=3,samples=2)
    m=ContractionMetric(64,cfg).float()
    loss,report=contraction_loss(a,s,m,z,cfg,seed=7)
    loss.backward()
    assert report['evaluated_directions']==4
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(a.parameters())+list(m.parameters()))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_unfused_gru_shares_weights_matches_native_values_and_vjp(dtype):
    from response_contraction import _UnfusedGRU, _differentiation_policy
    torch.manual_seed(71)
    cell=torch.nn.GRUCell(5,7).to(dtype)
    backend=_UnfusedGRU(cell)
    x=torch.randn(4,5,dtype=dtype,requires_grad=True)
    h=torch.randn(4,7,dtype=dtype,requires_grad=True)
    native,unfused=cell(x,h),backend(x,h)
    tolerance=1e-6 if dtype==torch.float32 else 1e-12
    torch.testing.assert_close(native,unfused,rtol=tolerance,atol=tolerance)
    vec=torch.randn_like(native)
    tensors=[x,h,*cell.parameters()]
    a=torch.autograd.grad((native*vec).sum(),tensors)
    b=torch.autograd.grad((unfused*vec).sum(),tensors)
    for ga,gb in zip(a,b):torch.testing.assert_close(ga,gb,rtol=tolerance,atol=tolerance)
    actor,sim,z,m,cfg=setup()
    old=actor.response_memory
    view=_differentiation_policy(actor)
    assert actor.response_memory is old
    assert view.response_memory.cell is old
    assert {id(p) for p in view.parameters()}=={id(p) for p in actor.parameters()}

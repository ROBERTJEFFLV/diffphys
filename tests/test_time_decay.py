"""Time Decay is an identity forward and a deliberate backward-only surrogate."""
from dataclasses import fields,replace
import copy
import math
import pytest
import torch

from env_raptor import RaptorSimulator
import response_task as task
from response_adjoints import collect_rollout,backward_actor
from response_groups import normalize_group_rows
from test_episode_termination import actor,flat_grads


@pytest.mark.parametrize('alpha',[0.,1.,4.])
def test_exponential_relative_time_rule(alpha):
    x=torch.tensor(2.,dtype=torch.float64,requires_grad=True);y=x
    for _ in range(25):y=task._GradientDecay.apply(y,math.exp(-alpha*.01))
    y.backward()
    assert y.item()==2.
    assert x.grad.item()==pytest.approx(math.exp(-alpha*.25))


def test_all_history_fields_are_decayed_metadata_is_shared():
    s=RaptorSimulator().reset(3,horizon=8,dtype=torch.float64);p=actor()
    closed=task.initialize(p,s);out=task._decay_closed_state(closed,.8)
    for group,dynamic in [('physical',task.PHYSICAL_DYNAMIC),('policy',task.POLICY_DYNAMIC)]:
        before,after=getattr(closed,group),getattr(out,group)
        for f in fields(before):
            a,b=getattr(before,f.name),getattr(after,f.name)
            assert torch.equal(a,b)
            if f.name not in dynamic:assert a is b
    assert 'previous_velocity' in task.PHYSICAL_DYNAMIC


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_noise_forward_loss_and_masks_identical_for_all_decay_settings(dtype):
    s=RaptorSimulator().reset(8,horizon=20,dtype=dtype);p=actor(dtype)
    traces=[task.rollout(p,RaptorSimulator(),s,20,time_decay=a) for a in (0.,1.,4.)]
    for other in traces[1:]:
        for key in ('actions','observations','positions','velocities','omegas','valid'):
            assert torch.equal(getattr(traces[0],key),getattr(other,key))
        assert torch.equal(task.task_loss(traces[0],task.TaskLossConfig()),task.task_loss(other,task.TaskLossConfig()))


@pytest.mark.parametrize('alpha',[0.,1.])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_grouped_full_graph_gradients_match_independent_group_vjps(alpha,dtype):
    s=RaptorSimulator().reset(64,horizon=12,dtype=dtype);p=actor(dtype);loss=task.TaskLossConfig()
    with torch.no_grad():p.readout.weight[:,:16].normal_(0,.001)
    r=collect_rollout(p,RaptorSimulator(),s,loss,horizon=12,time_decay=alpha)
    params=list(p.parameters());rows=[]
    for seed in r.group_coefficients:
        gs=torch.autograd.grad(r.costs,params,grad_outputs=.1*seed,retain_graph=True)
        rows.append(torch.cat([g.flatten() for g in gs]))
    expected,_=normalize_group_rows(torch.stack(rows),r.group_config.gradient_epsilon)
    backward_actor(p,RaptorSimulator(),r,loss)
    tol=5e-5 if dtype==torch.float32 else 1e-10
    torch.testing.assert_close(flat_grads(p),expected,rtol=tol,atol=tol)
    assert p.readout.weight.grad[:,:16].norm()>0
    assert p.response_memory.weight_hh.grad.norm()>0


@pytest.mark.parametrize('alpha',[-1.,float('nan'),float('inf')])
def test_invalid_decay(alpha):
    s=RaptorSimulator().reset(1,horizon=2,dtype=torch.float64)
    with pytest.raises(ValueError):task.rollout(actor(),RaptorSimulator(),s,2,time_decay=alpha)


@pytest.mark.parametrize('alpha',[0.,1.])
def test_delay_and_recurrent_graph_survive_fifty_step_metrics_boundary(alpha):
    from response_noise import DisturbanceConfig
    s=RaptorSimulator().reset(64,seed=34,horizon=70,dtype=torch.float64,
                             disturbances=DisturbanceConfig(pool=(0,0,0,0,1)))
    p=actor();other=copy.deepcopy(p);sim=RaptorSimulator();loss=task.TaskLossConfig()
    with torch.no_grad():
        s=replace(s,position=torch.zeros_like(s.position),velocity=torch.zeros_like(s.velocity),
                  previous_velocity=torch.zeros_like(s.velocity),omega=torch.zeros_like(s.omega))
    trace=task.rollout(other,sim,s,70,time_decay=alpha)
    assert trace.valid[50:].any() and s.velocity_delay.max()>0
    direct=task.step_costs(trace,loss).sum(0)
    record=collect_rollout(p,sim,s,loss,horizon=70,time_decay=alpha)
    torch.testing.assert_close(record.costs,direct,atol=1e-10,rtol=1e-10)
    weights=record.weights.detach()
    (weights*record.costs).sum().backward()
    (weights*direct).sum().backward()
    torch.testing.assert_close(flat_grads(p),flat_grads(other),atol=1e-9,rtol=1e-9)

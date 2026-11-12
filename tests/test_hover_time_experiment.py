"""Independent time examples and continuity of the original hover path."""
from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from diffphys.env_raptor import RaptorSimulator
from diffphys.response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from diffphys.response_noise import DisturbanceConfig
from diffphys.response_task import TaskLossConfig
from diffphys.response_groups import GroupBalanceConfig
from diffphys import response_training as training
from diffphys.hover_time_experiment import (HoverTimeConfig, settling_proxy,
                                           settling_metrics, experiment)


def test_proxy_counts_last_departure_and_failed_suffix():
    ratios=torch.tensor([[float('inf'),float('inf'),0.],
                         [0.,0.,0.], [0.,float('inf'),0.],
                         [0.,0.,0.], [0.,0.,0.]],dtype=torch.float64)
    alive=torch.ones_like(ratios,dtype=torch.bool)
    proxy=settling_proxy(ratios,alive,dt=1.)
    torch.testing.assert_close(proxy,torch.tensor([1.,3.,0.],dtype=torch.float64))
    alive[-1,2]=False
    assert settling_proxy(ratios,alive,dt=1.)[2]==4
    m=settling_metrics(ratios,alive,dt=1.)
    assert m['hover_success_fraction']==pytest.approx(2/3)
    assert m['hover_settling_time_seconds']==pytest.approx(8/3)


def test_failed_future_is_censored_and_has_no_extra_time_gradient():
    ratio=torch.tensor([[5.,5.],[4.,4.],[3.,3.],[2.,2.]],
                       dtype=torch.float64,requires_grad=True)
    alive=torch.ones_like(ratio,dtype=torch.bool);alive[-1,1]=False
    time=settling_proxy(ratio,alive,dt=1.)
    gradient,=torch.autograd.grad(time.sum(),(ratio,))
    assert time[1]==3
    assert not gradient[:,1].any()
    assert gradient[:,0].sum()>0


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_delaying_recovery_increases_proxy_and_matches_finite_difference(dtype):
    t=torch.arange(101,dtype=dtype)*.01
    delay=torch.tensor(.0347,dtype=dtype,requires_grad=True)
    def value(x):
        ratio=1+torch.nn.functional.softplus((.5+x-t)/.13)
        return settling_proxy(ratio[:,None],torch.ones(101,1,dtype=torch.bool),dt=.01).sum()
    gradient,=torch.autograd.grad(value(delay),(delay,))
    eps=1e-3 if dtype==torch.float32 else 1e-6
    fd=(value(delay.detach()+eps)-value(delay.detach()-eps))/(2*eps)
    assert gradient>0
    torch.testing.assert_close(gradient,fd,rtol=2e-4,atol=1e-5)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_zero_weight_preserves_real_costs_gradients_and_rollout(dtype):
    torch.manual_seed(91)
    actor=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).to(dtype)
    other=copy.deepcopy(actor)
    sim=RaptorSimulator();initial=sim.reset(32,seed=97,horizon=12,dtype=dtype,
                                          disturbances=DisturbanceConfig.clean())
    loss=TaskLossConfig(.01,.01,25.)
    group=GroupBalanceConfig(max_groups=1,min_scenarios=32)
    original=training.collect_rollout(actor,sim,initial,loss,horizon=12,time_decay=1.,group_config=group)
    training.backward_actor(actor,sim,original,loss)
    with experiment(HoverTimeConfig(weight=0)):
        timed=training.collect_rollout(other,sim,initial,loss,horizon=12,time_decay=1.,group_config=group)
        training.backward_actor(other,sim,timed,loss)
    assert torch.equal(original.costs,timed.costs)
    for a,b in zip(actor.parameters(),other.parameters()):
        assert torch.equal(a.grad,b.grad)
    assert timed.metrics['hover_objective']==original.metrics['task_objective']
    assert timed.metrics['task_objective']==original.metrics['task_objective']


def test_time_term_has_real_physics_gradient_and_restores_native_functions():
    torch.manual_seed(111)
    actor=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).double()
    sim=RaptorSimulator();initial=sim.reset(32,seed=113,horizon=20,dtype=torch.float64,
                                          disturbances=DisturbanceConfig.clean())
    initial=replace(initial,velocity=initial.velocity*.1)
    loss=TaskLossConfig(.01,.01,25.);group=GroupBalanceConfig(max_groups=1,min_scenarios=32)
    original_collect=training.collect_rollout;original_objective=training.TASK_OBJECTIVE_VERSION
    gradients=[]
    for weight in (0.,1.):
        with experiment(HoverTimeConfig(weight=weight)):
            record=training.collect_rollout(actor,sim,initial,loss,horizon=20,time_decay=0.,group_config=group)
            training.backward_actor(actor,sim,record,loss)
            gradients.append(torch.cat([p.grad.flatten() for p in actor.parameters()]).clone())
            assert record.metrics['hover_time_proxy_seconds']>0
            assert all(torch.isfinite(p.grad).all() for p in actor.parameters())
    assert (gradients[1]-gradients[0]).norm()>1e-10
    assert training.collect_rollout is original_collect
    assert training.TASK_OBJECTIVE_VERSION==original_objective


@pytest.mark.parametrize('changes',[{'weight':-1},{'position_m':0},{'speed_m_s':float('nan')},
                                   {'omega_rad_s':float('inf')}])
def test_invalid_experiment_scales_are_rejected(changes):
    with pytest.raises(ValueError):HoverTimeConfig(**changes)


def test_fresh_arms_share_initial_weights_and_checkpoints_reject_wrong_objective(tmp_path):
    from test_reference_training import args_for, run
    hashes=[]
    for weight in (0.,1.):
        work=tmp_path/str(weight)
        with experiment(HoverTimeConfig(weight=weight)):
            run(args_for(work,0))
            saved=torch.load(work/'latest.pt',weights_only=True)
            training.require_reference_checkpoint(saved)
            hashes.append(saved['model_sha256'])
            assert saved['optimizer']['state']=={}
            assert 'initialization' not in saved['progress']
            assert saved['binding']['protocol']['hover_time_experiment']['weight']==weight
        with pytest.raises(ValueError,match='objective'):
            training.require_reference_checkpoint(saved)
    assert hashes[0]==hashes[1]
    with experiment(HoverTimeConfig(weight=0.)):
        with pytest.raises(ValueError,match='configuration'):
            training.require_reference_checkpoint(saved)


def test_treatment_exact_resume_retains_adam_rng_and_sampling(tmp_path):
    from test_reference_training import args_for, run
    from tools.replay_response_update import compare_values
    with experiment(HoverTimeConfig(weight=1.)):
        run(args_for(tmp_path/'continuous',2))
        run(args_for(tmp_path/'split',1))
        run(args_for(tmp_path/'split',2,('--resume',str(tmp_path/'split/latest.pt'))))
    a=torch.load(tmp_path/'continuous/latest.pt',weights_only=True)
    b=torch.load(tmp_path/'split/latest.pt',weights_only=True)
    for key in ('model','optimizer','rng','next_update'):
        equal,error=compare_values(a[key],b[key]);assert equal,(key,error)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_h500_time_objective_gradients_and_adam_match_native_physics():
    from diffphys.compiled_physics import make_simulator
    from diffphys.env_raptor import RaptorParams
    from tools.replay_response_update import compare_values
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(7)
    actor=ResponseMotorPolicy().cuda()
    initial,_=training.sample_training_scenarios(32,0,device='cuda',horizon=500,sampling='coverage128')
    group=GroupBalanceConfig(max_groups=128,min_scenarios=1,layout='coverage128')
    loss=TaskLossConfig(.01,.01,25.);results=[]
    for simulator in (RaptorSimulator(),make_simulator(RaptorParams(),'cuda')):
        policy=copy.deepcopy(actor);optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
        with experiment(HoverTimeConfig(weight=1.)):
            record=training.collect_rollout(policy,simulator,initial,loss,horizon=500,
                                             time_decay=1.,group_config=group)
            training.backward_actor(policy,simulator,record,loss)
            training.safe_global_clip(policy.parameters(),10.)
        costs=record.costs.detach().clone();gradient=[p.grad.clone() for p in policy.parameters()]
        optimizer.step()
        results.append((costs,gradient,copy.deepcopy(policy.state_dict()),copy.deepcopy(optimizer.state_dict())))
        del record,policy,optimizer
    equal,error=compare_values(results[0],results[1]);assert equal,error

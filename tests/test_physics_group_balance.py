"""Physical grouping + detached score scaling; one ordinary task backward."""
from __future__ import annotations
from dataclasses import fields, replace
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_l2f import L2FParams, L2FSimulator
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import sample_scenarios, TaskLossConfig, risk_weights
from response_groups import GroupBalanceConfig, physics_group_layout, group_balanced_weights
from response_adjoints import backward_actor, collect_boundary_rollout
from response_training import train
from tools.train_response_control import parse_args


def bank(n=512, seed=7, dtype=torch.float64, device='cpu'):
    return sample_scenarios(n,seed=seed,dtype=dtype,device=torch.device(device),horizon=8)


@pytest.mark.parametrize('n', [32,33,63,64,95,100,127,128,255,256,511,512,513,1024])
def test_unique_complete_partition_at_least_32(n):
    state=bank(n)
    before=torch.get_rng_state().clone()
    indices, counts=physics_group_layout(state,GroupBalanceConfig(enabled=True))
    valid=indices < n
    selected=indices[valid]
    assert selected.numel()==n
    assert torch.equal(selected.sort().values,torch.arange(n))
    assert int(counts.min())>=32
    assert int(counts.max()-counts.min())<=1
    assert counts.numel()<=16
    assert torch.equal(counts,valid.sum(1))
    assert torch.equal(before,torch.get_rng_state())
    again=physics_group_layout(state,GroupBalanceConfig(enabled=True))
    assert torch.equal(indices,again[0])
    assert torch.equal(counts,again[1])
    if n==512:
        assert counts.tolist()==[32]*16


def test_too_few_scenes_never_repeated_or_silently_accepted():
    with pytest.raises(ValueError,match='32'):
        physics_group_layout(bank(31),GroupBalanceConfig(enabled=True))


@pytest.mark.parametrize('field',['thrust_to_weight','torque_to_inertia',
                                  'motor_time_rising','motor_time_falling'])
def test_actual_dynamics_features_define_the_groups(field):
    state=bank(64)
    state=replace(state,thrust_to_weight=torch.full_like(state.mass,2.),
        torque_to_inertia=torch.full_like(state.mass,100.),
        motor_time_rising=torch.full_like(state.motor_time_rising,.05),
        motor_time_falling=torch.full_like(state.motor_time_falling,.1))
    vals=torch.linspace(1.,2.,64,dtype=state.mass.dtype)
    vals=vals[torch.randperm(64,generator=torch.Generator().manual_seed(12))]
    value=vals if getattr(state,field).ndim==1 else vals[:,None].expand(64,4).clone()
    state=replace(state,**{field:value})
    indices,counts=physics_group_layout(state,GroupBalanceConfig(enabled=True))
    assert counts.tolist()==[32,32]
    assert vals[indices[0]].max()<=vals[indices[1]].min()


def test_mass_or_radius_alone_does_not_define_groups_and_equal_dynamics_not_split():
    state=bank(128)
    state=replace(state,thrust_to_weight=torch.full_like(state.mass,2.),
        torque_to_inertia=torch.full_like(state.mass,100.),
        motor_time_rising=torch.full_like(state.motor_time_rising,.05),
        motor_time_falling=torch.full_like(state.motor_time_falling,.1))
    indices,counts=physics_group_layout(state,GroupBalanceConfig(enabled=True))
    assert counts.tolist()==[128]
    assert torch.equal(indices[0],torch.arange(128))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('mode',['none','rms'])
@pytest.mark.parametrize('n',[100,101])
def test_vectorized_weights_equal_explicit_group_formula(dtype,mode,n):
    state=bank(n,dtype=dtype)
    costs=torch.linspace(.1,40.,n,dtype=dtype,requires_grad=True)
    cfg=GroupBalanceConfig(enabled=True,scale_mode=mode)
    base=risk_weights(costs,TaskLossConfig())
    weights,report=group_balanced_weights(costs,base,state,cfg)
    indices,counts=physics_group_layout(state,cfg)
    expected=torch.zeros_like(weights)
    for group in indices:
        group=group[group<n]
        rms=costs.detach()[group].square().mean().sqrt().clamp_min(cfg.scale_floor)
        scale=rms if mode=='rms' else 1.
        expected[group]=n*base[group]/(len(counts)*len(group)*scale)
    torch.testing.assert_close(weights,expected)
    assert not weights.requires_grad
    assert report['group_count']==len(counts)
    assert not report['values'].requires_grad
    # Only the costs are differentiated; no hidden derivative through RMS/statistics.
    gradient,=torch.autograd.grad((weights*costs).sum(),costs)
    torch.testing.assert_close(gradient,expected)


def test_identity_weights_when_disabled_and_when_equal_counts_no_normalization():
    state=bank(512)
    costs=torch.linspace(.1,12.,512,dtype=torch.float64)
    base=risk_weights(costs,TaskLossConfig())
    same,report=group_balanced_weights(costs,base,state,GroupBalanceConfig())
    assert same is base and report is None
    result,_=group_balanced_weights(costs,base,state,GroupBalanceConfig(enabled=True,scale_mode='none'))
    torch.testing.assert_close(result,base,atol=0,rtol=0)


def test_equal_group_coefficients_without_tail_and_unequal_counts():
    state=bank(100)
    costs=torch.linspace(.1,4.,100,dtype=torch.float64)
    cfg=GroupBalanceConfig(enabled=True,scale_mode='none')
    weights,_=group_balanced_weights(costs,torch.full_like(costs,.01),state,cfg)
    indices,counts=physics_group_layout(state,cfg)
    for group in indices:
        group=group[group<100]
        assert weights[group].sum().item()==pytest.approx(1/counts.numel())


def test_scale_floor_zero_costs_and_large_finite_float32_scores():
    state=bank(64,dtype=torch.float32)
    for costs in [torch.zeros(64),torch.full((64,),1e25),torch.full((64,),1e37)]:
        w,_=group_balanced_weights(costs,torch.full((64,),1/64),state,
                                  GroupBalanceConfig(enabled=True))
        assert torch.isfinite(w).all() and (w>0).all()
        if costs.max()==0:
            torch.testing.assert_close(w,torch.full((64,),1/64))
        else:
            assert (w*costs).sum().item()==pytest.approx(1.,rel=1e-6)


@pytest.mark.parametrize('bad',['nan','negative','tracked_weight','negative_cost'])
def test_bad_input_rejected_before_backward(bad):
    state=bank(64)
    costs=torch.ones(64,dtype=torch.float64)
    weights=torch.full_like(costs,1/64)
    if bad=='nan': costs[0]=float('nan')
    if bad=='negative': weights[0]=-1
    if bad=='tracked_weight': weights.requires_grad_(True)
    if bad=='negative_cost': costs[0]=-1
    with pytest.raises((ValueError,FloatingPointError)):
        group_balanced_weights(costs,weights,state,GroupBalanceConfig(enabled=True))


@pytest.mark.parametrize('field',['thrust_to_weight','torque_to_inertia',
                                  'motor_time_rising','motor_time_falling'])
def test_bad_physics_rejected(field):
    state=bank(64)
    val=getattr(state,field).clone(); val[0]=0
    with pytest.raises((ValueError,FloatingPointError)):
        physics_group_layout(replace(state,**{field:val}),GroupBalanceConfig(enabled=True))


@pytest.mark.parametrize('decay',[0.,1.])
def test_real_full_graph_one_vjp_original_flight_and_frozen_weight_gradient(decay):
    torch.set_num_threads(1); torch.manual_seed(7)
    policy=ResponseMotorPolicy(ResponsePolicyConfig(hidden_dim=8,memory_dim=8)).double()
    sim=L2FSimulator(L2FParams()); state=bank(64); lc=TaskLossConfig()
    kwargs=dict(horizon=8,window_steps=4,backprop_mode='full',time_decay=decay)
    plain=collect_boundary_rollout(policy,sim,state,lc,**kwargs)
    grouped=collect_boundary_rollout(policy,sim,state,lc,
        group_config=GroupBalanceConfig(enabled=True),**kwargs)
    assert torch.equal(plain.costs,grouped.costs)
    assert torch.equal(plain.weights,grouped.weights)
    assert torch.equal(plain.valid,grouped.valid)
    for t in plain.boundaries:
        for part in ('physical','policy'):
            for f in fields(getattr(plain.boundaries[t],part)):
                assert torch.equal(getattr(getattr(plain.boundaries[t],part),f.name),
                    getattr(getattr(grouped.boundaries[t],part),f.name))
    assert plain.metrics['task_objective']==grouped.metrics['task_objective']
    expected=torch.autograd.grad(.1*(grouped.costs*grouped.optimization_weights).sum(),
                                list(policy.parameters()),retain_graph=True)
    rng=torch.get_rng_state().clone()
    original=torch.autograd.grad
    with patch('torch.autograd.grad',wraps=original) as grad:
        backward_actor(policy,sim,grouped,lc)
        assert grad.call_count==1
        assert not grad.call_args.kwargs.get('is_grads_batched',False)
        assert not grad.call_args.kwargs.get('retain_graph',False)
    for p,g in zip(policy.parameters(),expected):
        torch.testing.assert_close(p.grad,g,atol=1e-11,rtol=1e-10)
    assert torch.equal(rng,torch.get_rng_state())


def test_windowed_recomputation_reuses_frozen_group_scales():
    torch.set_num_threads(1); torch.manual_seed(7)
    policy=ResponseMotorPolicy(ResponsePolicyConfig(hidden_dim=8,memory_dim=8)).double()
    sim=L2FSimulator(L2FParams()); state=bank(64); lc=TaskLossConfig()
    cfg=GroupBalanceConfig(enabled=True)
    expected=[]
    for mode in ['full','windowed']:
        record=collect_boundary_rollout(policy,sim,state,lc,horizon=8,window_steps=4,
            backprop_mode=mode,time_decay=1.,group_config=cfg)
        backward_actor(policy,sim,record,lc)
        expected.append([p.grad.clone() for p in policy.parameters()])
    for a,b in zip(*expected): torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)


@pytest.mark.parametrize('options',[
    ['--group-min-scenarios','16'],['--group-max-groups','3'],
    ['--group-scale-floor','0'],['--group-scale-floor','nan'],
    ['--group-balance','--scenarios','4'],['--contribution-clip','1'],
    ['--contribution-vjp-chunk-size','64'],
])
def test_invalid_config_and_removed_expensive_path_rejected(options):
    with pytest.raises(SystemExit): parse_args(options)


def args_for(path,updates,extra=()):
    return parse_args(['--device','cpu','--dtype','float64','--scenarios','16',
        '--eval-scenarios','2','--horizon','8','--window-steps','4','--hidden-dim','8',
        '--memory-dim','8','--updates',str(updates),'--max-seconds','60',
        '--development-every','1','--checkpoint-every','1','--group-balance',
        '--work-dir',str(path),*extra])


def run(args):
    pc=ResponsePolicyConfig(**{f.name:getattr(args,f.name) for f in fields(ResponsePolicyConfig)})
    lc=TaskLossConfig(**{f.name:getattr(args,f.name) for f in fields(TaskLossConfig)})
    return train(args,pc,lc)


def test_grouped_training_exact_resume_and_bound_configuration(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    run(args_for(a,2)); run(args_for(b,1))
    run(args_for(b,2,('--resume',str(b/'latest.pt'))))
    ca=torch.load(a/'latest.pt',weights_only=True); cb=torch.load(b/'latest.pt',weights_only=True)
    assert ca['model_sha256']==cb['model_sha256']
    assert torch.equal(ca['rng']['torch'],cb['rng']['torch'])
    for k,state in ca['optimizer']['state'].items():
        for key,v in state.items():
            torch.testing.assert_close(v,cb['optimizer']['state'][k][key],atol=0,rtol=0)
    assert ca['binding']['group_balance']['enabled'] is True
    row=json.loads((a/'history.jsonl').read_text().splitlines()[-1])
    assert row['group_balance']['group_count']==2
    assert row['group_balance']['values'][0][0]>=32
    assert 'optimization_objective' in row and 'task_objective' in row
    with pytest.raises(ValueError,match='configuration'):
        run(args_for(b,3,('--resume',str(b/'latest.pt'),'--group-scale-floor','2')))


def test_nonfinite_grouping_rolls_back_without_adam(tmp_path,monkeypatch):
    import response_training as module
    run(args_for(tmp_path/'zero',0))
    original=torch.load(tmp_path/'zero/latest.pt',weights_only=True)
    calls=[]
    def fail(*a,**kw): raise FloatingPointError('injected grouping failure')
    monkeypatch.setattr(module,'collect_boundary_rollout',fail)
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**kw:calls.append(True))
    with pytest.raises(FloatingPointError): run(args_for(tmp_path/'fail',1))
    checkpoint=torch.load(tmp_path/'fail/failure.pt',weights_only=True)
    assert checkpoint['model_sha256']==original['model_sha256']
    assert checkpoint['optimizer']['state']=={}
    assert checkpoint['progress']['updates']==0 and not calls


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_vectorized_groups_and_one_backward():
    state=bank(512,dtype=torch.float32,device='cuda')
    costs=torch.linspace(.1,12.,512,device='cuda',requires_grad=True)
    base=risk_weights(costs,TaskLossConfig())
    weights,report=group_balanced_weights(costs,base,state,GroupBalanceConfig(enabled=True))
    assert weights.device.type=='cuda' and report['values'].device.type=='cuda'
    assert report['group_count']==16
    (weights*costs).sum().backward()
    torch.testing.assert_close(costs.grad,weights)


def test_group_scores_do_not_claim_a_per_scene_gradient_cap():
    state=bank(32)
    theta=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64))
    sensitivity=torch.ones(32,dtype=torch.float64); sensitivity[0]=1e12
    costs=1+theta*sensitivity
    weights,_=group_balanced_weights(costs,torch.full_like(costs,1/32),state,
                                    GroupBalanceConfig(enabled=True))
    grad,=torch.autograd.grad((weights*costs).sum(),theta)
    assert grad.item()>1e10  # Equal finite scores do not imply small derivatives.


def test_standard_raptor_config_enables_group_scores():
    text=(Path(__file__).resolve().parents[1]/'configs/response_raptor_multi_airframe.args').read_text()
    args=parse_args(text.split())
    assert args.group_balance and args.group_min_scenarios==32
    assert args.group_max_groups==16 and args.scenarios==128
    assert args.group_scale_mode=='rms'
    assert not hasattr(args,'contribution_clip')

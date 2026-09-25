"""Physical grouping and normalization of true group parameter gradients."""
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
from response_groups import (GroupBalanceConfig, physics_group_layout,
    group_gradient_coefficients, normalize_group_rows, backward_group_gradients)
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


@pytest.mark.parametrize('n', [64, 65, 100, 512, 513])
def test_group_coefficients_preserve_cvar_and_group_mean(n):
    state = bank(n)
    cfg = GroupBalanceConfig(enabled=True)
    costs = torch.linspace(.1, 20, n, dtype=torch.float64)
    weights = risk_weights(costs, TaskLossConfig())
    seeds, report = group_gradient_coefficients(weights, state, cfg)
    indices, counts = physics_group_layout(state, cfg)
    expected = torch.zeros_like(seeds)
    for group, selected in enumerate(indices):
        selected = selected[selected < n]
        expected[group, selected] = n * weights[selected] / selected.numel()
    torch.testing.assert_close(seeds, expected)
    assert not seeds.requires_grad
    assert (seeds > 0).sum().item() == n
    assert 'cost_rms' not in report['columns'] and 'scale' not in report['columns']
    if bool((counts == counts[0]).all()):
        torch.testing.assert_close(seeds.mean(0), weights)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_gradient_outlier_not_cost_is_normalized(dtype):
    rows = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [-1e9, 0.]], dtype=dtype)
    result, report = normalize_group_rows(rows, 1e-12)
    torch.testing.assert_close(result, torch.tensor([.5, 0.], dtype=dtype))
    torch.testing.assert_close(report['values'][:, 1], torch.ones(4, dtype=torch.float64))
    assert report['values'][-1, 2] < 1e-8


def test_small_groups_are_scaled_up_not_only_clipped():
    rows = torch.tensor([[.01, 0.], [1., 0.], [2., 0.], [1e9, 0.]], dtype=torch.float64)
    result, report = normalize_group_rows(rows, 1e-12)
    torch.testing.assert_close(result, torch.tensor([1., 0.], dtype=rows.dtype))
    assert report['values'][0, 2] == 100.
    assert report['values'][3, 2] == 1e-9


@pytest.mark.parametrize('rows', [
    [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.], [3., 4.]],
    [[1e-30, 0.], [1., 0.], [1., 0.]],
    [[1e30, -1e30], [2e30, 0.], [0., 1e30]],
])
def test_zero_tiny_huge_groups_finite(rows):
    value = torch.tensor(rows, dtype=torch.float32)
    result, report = normalize_group_rows(value, 1e-12)
    assert torch.isfinite(result).all() and torch.isfinite(report['values']).all()
    assert not result.requires_grad
    assert torch.equal(report['values'][value.abs().sum(-1) == 0, 1],
                       torch.zeros_like(report['values'][value.abs().sum(-1) == 0, 1]))


def test_zero_groups_do_not_erase_other_groups_or_renormalize_vote_count():
    result, report = normalize_group_rows(torch.tensor([[0.,0.],[0.,0.],[3.,4.]]),1e-12)
    torch.testing.assert_close(result, torch.tensor([1.,4/3]))
    assert torch.equal(report['values'][:, 3], torch.full((3,),5.,dtype=torch.float64))


def test_large_float64_norm_avoids_square_overflow():
    result, report = normalize_group_rows(torch.tensor([[1e200,0.],[0.,1e200]],dtype=torch.float64),1e-12)
    assert torch.isfinite(result).all() and torch.isfinite(report['values']).all()


@pytest.mark.parametrize('bad', [float('inf'),float('nan')])
def test_nonfinite_gradient_is_not_sanitized(bad):
    with pytest.raises(FloatingPointError): normalize_group_rows(torch.tensor([[bad,0.],[1.,0.]]),1e-12)


@pytest.mark.parametrize('chunk', [1, 2, 3, 16])
def test_exact_weighted_group_gradients_match_closed_form(chunk):
    state=bank(128)
    p=torch.nn.Parameter(torch.tensor([.1,.2],dtype=torch.float64))
    unused=torch.nn.Parameter(torch.tensor([1.],dtype=p.dtype))
    x=torch.randn(128,2,generator=torch.Generator().manual_seed(4),dtype=p.dtype)
    costs=(x@p+1).square()
    cfg=GroupBalanceConfig(enabled=True,vjp_chunk_size=chunk)
    weights=risk_weights(costs,TaskLossConfig())
    coefficients,_=group_gradient_coefficients(weights,state,cfg)
    expected_rows=.1 * coefficients @ (2*(x@p.detach()+1)[:,None]*x)
    expected,_=normalize_group_rows(expected_rows,cfg.gradient_epsilon)
    with patch('torch.autograd.grad', wraps=torch.autograd.grad) as call:
        gradients,report=backward_group_gradients(costs,coefficients,[p,unused],cfg,gradient_scale=.1)
        assert call.call_count == (4+chunk-1)//chunk
    torch.testing.assert_close(gradients[0],expected,atol=1e-13,rtol=1e-12)
    assert gradients[1] is None and p.grad is None
    assert report['group_count']==4


def test_group_coefficients_do_not_depend_on_cost_offsets():
    state=bank(64)
    p=torch.nn.Parameter(torch.tensor(.1,dtype=torch.float64))
    weights=torch.full((64,),1/64,dtype=p.dtype)
    cfg=GroupBalanceConfig(enabled=True)
    seeds,_=group_gradient_coefficients(weights,state,cfg)
    results=[]
    for offset in (0.,1e6):
        costs=offset + (torch.arange(64,dtype=p.dtype)+1)*p
        results.append(backward_group_gradients(costs,seeds,[p],cfg,gradient_scale=.1)[0][0])
    torch.testing.assert_close(*results,atol=0,rtol=0)


@pytest.mark.parametrize('decay', [0.,1.])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_real_graph_batched_and_serial_agree_without_forward_change(decay,dtype):
    torch.set_num_threads(1); torch.manual_seed(7)
    policy=ResponseMotorPolicy(ResponsePolicyConfig(hidden_dim=8,memory_dim=8)).to(dtype=dtype)
    sim=L2FSimulator(L2FParams())
    state=bank(128,dtype=dtype); loss=TaskLossConfig()
    results=[]; records=[]
    for chunk in (1,3,16):
        cfg=GroupBalanceConfig(enabled=True,vjp_chunk_size=chunk)
        record=collect_boundary_rollout(policy,sim,state,loss,horizon=8,window_steps=4,
            backprop_mode='full',time_decay=decay,group_config=cfg)
        # Serial independent reference on the SAME retained graph.
        expected=[]
        for seed in record.group_coefficients:
            gs=torch.autograd.grad(record.costs,list(policy.parameters()),grad_outputs=.1*seed,
                                   retain_graph=True)
            expected.append(torch.cat([g.flatten() for g in gs]))
        expected,_=normalize_group_rows(torch.stack(expected),cfg.gradient_epsilon)
        rng=torch.get_rng_state().clone(); snapshots=copy.deepcopy(record.boundaries)
        info=backward_actor(policy,sim,record,loss)
        actual=torch.cat([p.grad.flatten() for p in policy.parameters()])
        tolerance=(dict(atol=1e-6,rtol=5e-5) if dtype==torch.float32 else dict(atol=1e-12,rtol=1e-10))
        torch.testing.assert_close(actual,expected,**tolerance)
        assert torch.equal(rng,torch.get_rng_state())
        for t,boundary in record.boundaries.items():
            for part in ('physical','policy'):
                for f in fields(getattr(boundary,part)):
                    assert torch.equal(getattr(getattr(boundary,part),f.name),
                                       getattr(getattr(snapshots[t],part),f.name))
        assert info['group_gradient']['group_count']==4
        assert 'optimization_objective' not in record.metrics
        results.append(actual.clone()); records.append(record)
    for actual in results[1:]: torch.testing.assert_close(actual,results[0],**tolerance)
    for record in records[1:]:
        assert torch.equal(record.costs,records[0].costs)
        assert torch.equal(record.weights,records[0].weights)
        assert torch.equal(record.valid,records[0].valid)


@pytest.mark.parametrize('options',[
    ['--group-min-scenarios','16'],['--group-max-groups','3'],
    ['--group-scale-mode','rms'],['--group-scale-floor','1'],
    ['--group-gradient-epsilon','0'],['--group-gradient-epsilon','nan'],
    ['--group-vjp-chunk-size','0'],['--group-vjp-chunk-size','65'],
    ['--group-balance','--scenarios','4'],['--contribution-clip','1'],
    ['--contribution-vjp-chunk-size','64'],['--group-balance','--agc','.01'],
    ['--group-balance','--backprop-mode','windowed'],
])
def test_invalid_old_or_conflicting_flags_rejected(options):
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


def test_grouped_training_exact_resume_and_new_binding(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    run(args_for(a,2)); run(args_for(b,1))
    run(args_for(b,2,('--resume',str(b/'latest.pt'))))
    ca=torch.load(a/'latest.pt',weights_only=True); cb=torch.load(b/'latest.pt',weights_only=True)
    assert ca['model_sha256']==cb['model_sha256']
    assert torch.equal(ca['rng']['torch'],cb['rng']['torch'])
    for k,state in ca['optimizer']['state'].items():
        for key,v in state.items():
            torch.testing.assert_close(v,cb['optimizer']['state'][k][key],atol=0,rtol=0)
    assert 'gradient-median' in ca['binding']['group_balance']['version']
    row=json.loads((a/'history.jsonl').read_text().splitlines()[-1])
    assert row['group_balance']['group_count']==2
    assert row['group_balance']['values'][0][0]>=32
    assert row['group_gradient']['group_count']==2
    assert row['group_gradient']['vjp_calls']==1
    assert 'optimization_objective' not in row
    assert 'task_objective' in row
    with pytest.raises(ValueError,match='configuration'):
        run(args_for(b,3,('--resume',str(b/'latest.pt'),'--group-vjp-chunk-size','1')))


def test_nonfinite_group_gradient_restores_actor_adam_rng(tmp_path,monkeypatch):
    import response_training as module
    run(args_for(tmp_path/'initial',1))
    original=torch.load(tmp_path/'initial/latest.pt',weights_only=True)
    from shutil import copytree
    copytree(tmp_path/'initial',tmp_path/'fail')
    real_backward=module.backward_actor
    calls=[]
    def fail(*a,**kw):
        real_backward(*a,**kw)
        raise FloatingPointError('injected group gradient failure')
    monkeypatch.setattr(module,'backward_actor',fail)
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**kw:calls.append(True))
    with pytest.raises(FloatingPointError):
        run(args_for(tmp_path/'fail',2,('--resume',str(tmp_path/'fail/latest.pt'))))
    checkpoint=torch.load(tmp_path/'fail/failure.pt',weights_only=True)
    assert checkpoint['model_sha256']==original['model_sha256']
    assert checkpoint['progress']['updates']==1 and not calls
    assert torch.equal(checkpoint['rng']['torch'],original['rng']['torch'])
    for pid,states in original['optimizer']['state'].items():
        for k,v in states.items():
            torch.testing.assert_close(v,checkpoint['optimizer']['state'][pid][k],rtol=0,atol=0)


def test_failed_group_does_not_publish_partial_gradients():
    p=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64)); p.grad=torch.tensor(7.,dtype=p.dtype)
    costs=torch.stack((p+1,p.sqrt()))
    with pytest.raises(FloatingPointError):
        backward_group_gradients(costs,torch.eye(2,dtype=p.dtype),[p],
            GroupBalanceConfig(enabled=True,vjp_chunk_size=1),gradient_scale=.1)
    assert p.grad.item()==7.


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_group_vjps_match_serial():
    torch.manual_seed(7)
    policy=ResponseMotorPolicy(ResponsePolicyConfig()).cuda()
    sim=L2FSimulator(L2FParams()); state=bank(128,dtype=torch.float32,device='cuda'); loss=TaskLossConfig()
    cfg=GroupBalanceConfig(enabled=True)
    record=collect_boundary_rollout(policy,sim,state,loss,horizon=8,window_steps=4,
        backprop_mode='full',time_decay=1.,group_config=cfg)
    rows=[]
    for seed in record.group_coefficients:
        gs=torch.autograd.grad(record.costs,list(policy.parameters()),grad_outputs=.1*seed,retain_graph=True)
        rows.append(torch.cat([g.flatten() for g in gs]))
    expected,_=normalize_group_rows(torch.stack(rows),cfg.gradient_epsilon)
    backward_actor(policy,sim,record,loss)
    torch.testing.assert_close(torch.cat([p.grad.flatten() for p in policy.parameters()]),expected,atol=2e-6,rtol=2e-4)


def test_standard_raptor_config_enables_gradients_not_scores():
    text=(Path(__file__).resolve().parents[1]/'configs/response_raptor_multi_airframe.args').read_text()
    args=parse_args(text.split())
    assert args.group_balance and args.group_min_scenarios==32
    assert args.group_max_groups==16 and args.scenarios==128
    assert args.group_vjp_chunk_size==16
    assert not hasattr(args,'group_scale_mode') and not hasattr(args,'contribution_clip')


def test_equal_forward_costs_with_one_sensitive_physical_group():
    state = bank(128)
    cfg = GroupBalanceConfig(enabled=True)
    p = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
    weights = torch.full((128,), 1/128, dtype=p.dtype)
    seeds, _ = group_gradient_coefficients(weights, state, cfg)
    derivatives = torch.where(seeds[-1] > 0, -1e9, 1.)
    costs = 1 + p*derivatives
    assert torch.equal(costs, torch.ones_like(costs))
    gradients, report = backward_group_gradients(costs, seeds, [p], cfg, gradient_scale=.1)
    torch.testing.assert_close(gradients[0], p.new_tensor(.05))
    torch.testing.assert_close(report['values'][:,1], torch.full((4,), .1, dtype=p.dtype))


def test_single_group_is_not_an_absolute_gradient_cap():
    rows = torch.tensor([[1e9, 0.]], dtype=torch.float64)
    result, _ = normalize_group_rows(rows, 1e-12)
    torch.testing.assert_close(result, rows[0])

"""Physical grouping and normalization of true group parameter gradients."""
from __future__ import annotations
from dataclasses import replace
import copy
from pathlib import Path
import sys
import warnings
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from response_task import sample_scenarios, TaskLossConfig, risk_weights
from response_groups import (GroupBalanceConfig, physics_group_layout,
    group_gradient_coefficients, normalize_group_rows, backward_group_gradients)


def bank(n=512, seed=7, dtype=torch.float64, device='cpu'):
    return sample_scenarios(n,seed=seed,dtype=dtype,device=torch.device(device),horizon=8)


@pytest.mark.parametrize('n', [32,33,63,64,95,100,127,128,255,256,511,512,513,1024])
def test_unique_complete_partition_at_least_32(n):
    state=bank(n)
    before=torch.get_rng_state().clone()
    indices, counts=physics_group_layout(state,GroupBalanceConfig())
    valid=indices < n
    selected=indices[valid]
    assert selected.numel()==n
    assert torch.equal(selected.sort().values,torch.arange(n))
    assert int(counts.min())>=32
    assert int(counts.max()-counts.min())<=1
    assert counts.numel()<=16
    assert torch.equal(counts,valid.sum(1))
    assert torch.equal(before,torch.get_rng_state())
    again=physics_group_layout(state,GroupBalanceConfig())
    assert torch.equal(indices,again[0])
    assert torch.equal(counts,again[1])
    if n==512:
        assert counts.tolist()==[32]*16


def test_too_few_scenes_never_repeated_or_silently_accepted():
    with pytest.raises(ValueError,match='32'):
        physics_group_layout(bank(31),GroupBalanceConfig())


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
    indices,counts=physics_group_layout(state,GroupBalanceConfig())
    assert counts.tolist()==[32,32]
    assert vals[indices[0]].max()<=vals[indices[1]].min()


def test_mass_or_radius_alone_does_not_define_groups_and_equal_dynamics_not_split():
    state=bank(128)
    state=replace(state,thrust_to_weight=torch.full_like(state.mass,2.),
        torque_to_inertia=torch.full_like(state.mass,100.),
        motor_time_rising=torch.full_like(state.motor_time_rising,.05),
        motor_time_falling=torch.full_like(state.motor_time_falling,.1))
    indices,counts=physics_group_layout(state,GroupBalanceConfig())
    assert counts.tolist()==[128]
    assert torch.equal(indices[0],torch.arange(128))


@pytest.mark.parametrize('n', [64, 65, 100, 512, 513])
def test_group_coefficients_preserve_cvar_and_group_mean(n):
    state = bank(n)
    cfg = GroupBalanceConfig()
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
    cfg=GroupBalanceConfig(vjp_chunk_size=chunk)
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


def test_group_vjp_avoids_legacy_activation_fallbacks():
    # Preserve the historical nonlinear-operator regression even though the
    # new Actor no longer has an external SiLU encoder.
    p = torch.nn.Parameter(torch.tensor([.1, -.3], dtype=torch.float64))
    x = torch.arange(24, dtype=p.dtype).reshape(12, 2) / 24
    costs = torch.nn.functional.silu(x @ p).tanh()
    seeds = torch.eye(3, dtype=p.dtype).repeat_interleave(4, dim=1)
    expected_rows = torch.stack([
        torch.autograd.grad(costs, p, grad_outputs=.1*seed, retain_graph=True)[0]
        for seed in seeds
    ])
    expected, _ = normalize_group_rows(expected_rows, 1e-12)
    torch._C._debug_only_display_vmap_fallback_warnings(True)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            gradients, _ = backward_group_gradients(
                costs, seeds, [p], GroupBalanceConfig(), gradient_scale=.1)
    finally:
        torch._C._debug_only_display_vmap_fallback_warnings(False)
    torch.testing.assert_close(gradients[0], expected, atol=1e-13, rtol=1e-12)
    assert not any('LegacyBatchedFallback' in str(w.message) for w in captured)


@pytest.mark.parametrize('chunk', [1, 2, 3])
def test_all_unused_group_parameters_keep_none_and_existing_adam_moments(chunk):
    p = torch.nn.Parameter(torch.tensor([.2, .4], dtype=torch.float64))
    optimizer = torch.optim.Adam([p], lr=3e-4)
    p.square().sum().backward()
    optimizer.step()
    before = p.detach().clone()
    moments = copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    external = torch.ones(6, dtype=p.dtype, requires_grad=True)
    costs = external.square()
    seeds = torch.eye(3, dtype=p.dtype).repeat_interleave(2, dim=1)
    gradients, report = backward_group_gradients(
        costs, seeds, [p], GroupBalanceConfig( vjp_chunk_size=chunk),
        gradient_scale=.1)
    assert gradients == [None]
    assert torch.count_nonzero(report['values']) == 0
    p.grad = gradients[0]
    optimizer.step()
    assert torch.equal(p, before)
    for key, value in moments['state'][0].items():
        torch.testing.assert_close(optimizer.state_dict()['state'][0][key], value, atol=0, rtol=0)


def test_group_coefficients_do_not_depend_on_cost_offsets():
    state=bank(64)
    p=torch.nn.Parameter(torch.tensor(.1,dtype=torch.float64))
    weights=torch.full((64,),1/64,dtype=p.dtype)
    cfg=GroupBalanceConfig()
    seeds,_=group_gradient_coefficients(weights,state,cfg)
    results=[]
    for offset in (0.,1e6):
        costs=offset + (torch.arange(64,dtype=p.dtype)+1)*p
        results.append(backward_group_gradients(costs,seeds,[p],cfg,gradient_scale=.1)[0][0])
    torch.testing.assert_close(*results,atol=0,rtol=0)


@pytest.mark.parametrize('chunk', [1, 2])
def test_failed_group_does_not_publish_partial_gradients(chunk):
    p=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64)); p.grad=torch.tensor(7.,dtype=p.dtype)
    costs=torch.stack((p+1,p.sqrt()))
    with pytest.raises(FloatingPointError):
        backward_group_gradients(costs,torch.eye(2,dtype=p.dtype),[p],
            GroupBalanceConfig(vjp_chunk_size=chunk),gradient_scale=.1)
    assert p.grad.item()==7.


def test_equal_forward_costs_with_one_sensitive_physical_group():
    state = bank(128)
    cfg = GroupBalanceConfig()
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

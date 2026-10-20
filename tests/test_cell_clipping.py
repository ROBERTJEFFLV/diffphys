"""Independent contracts for fixed physical-cell, shrink-only Actor gradients."""

from loss_fixtures import test_loss
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from response_groups import (GroupBalanceConfig, normalize_group_rows,
                             physics_group_layout, group_gradient_coefficients,
                             backward_group_gradients)
from response_sampling import sample_coverage, physics_cell_ids
from response_task import TaskLossConfig, uniform_scene_weights


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_small_gradients_never_amplified_even_when_majority_explodes(dtype):
    rows = torch.tensor([[.195, 0.], [2617.46, 0.], [1.0734e10, 0.]], dtype=dtype)
    result, report = normalize_group_rows(rows, 1e-12, max_norm=1.)
    values = report['values']
    assert values[0, 2].item() == 1.
    assert bool((values[:, 2] <= 1).all())
    assert bool((values[:, 1] <= 1. + 1e-12).all())
    torch.testing.assert_close(result, rows.new_tensor([(.195 + 2)/3, 0.]))


@pytest.mark.parametrize('cap', [.01, 1., 10.])
def test_clipping_preserves_small_values_directions_and_fixed_denominator(cap):
    rows = torch.tensor([[0., 0.], [cap/10, 0.], [3*cap, 4*cap]], dtype=torch.float64)
    expected = (rows[1] + torch.tensor([.6*cap, .8*cap])) / 3
    result, report = normalize_group_rows(rows, 1e-12, max_norm=cap)
    torch.testing.assert_close(result, expected)
    assert report['values'][0, 2] == 1.
    assert report['values'][1, 2] == 1.
    assert result.norm() <= cap


def cell_config(**extra):
    return GroupBalanceConfig(max_groups=128, min_scenarios=1,
                              layout='coverage128', **extra)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_all_128_groups_exactly_match_existing_sampler_cells(dtype):
    initial, _ = sample_coverage(2048, [31000007,31000008,31000009,31000010], dtype=dtype)
    ids = physics_cell_ids(initial)
    indices, counts = physics_group_layout(initial, cell_config())
    assert counts.tolist() == [16]*128
    assert torch.equal(ids[indices], torch.arange(128)[:,None].expand(128,16))
    assert torch.equal(indices.flatten().sort().values, torch.arange(2048))
    weights = uniform_scene_weights(torch.arange(2048, dtype=dtype))
    coefficients, report = group_gradient_coefficients(weights, initial, cell_config())
    torch.testing.assert_close(coefficients.mean(0), weights)
    assert report['group_ids'] == list(range(128))
    assert torch.equal(report['scene_group_ids'], ids)


def test_cell_groups_refuse_missing_or_unequal_quotas():
    from response_training import sample_pool
    with pytest.raises(ValueError, match='quota|128'):
        physics_group_layout(sample_pool(32, [7,8,9,10], horizon=1), cell_config())


def test_all_128_group_vjps_match_analytic_mean_clip_mean_and_use_eight_calls():
    initial, _ = sample_coverage(128, [7,8,9,10], dtype=torch.float64)
    cfg = cell_config(vjp_chunk_size=16, clip_norm=.3)
    parameter = torch.nn.Parameter(torch.tensor([.1,-.2], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    x = torch.arange(256, dtype=torch.float64).reshape(128,2)/100
    costs = (x @ parameter + 1).square()
    weights = uniform_scene_weights(costs)
    coeff, _ = group_gradient_coefficients(weights, initial, cfg)
    row = .1 * coeff @ (2*(x @ parameter.detach() + 1)[:,None]*x)
    expected = (row * torch.minimum(torch.ones(128), .3/row.norm(dim=1))[:,None]).mean(0)
    with patch('torch.autograd.grad', wraps=torch.autograd.grad) as call:
        actual, report = backward_group_gradients(costs, coeff, [parameter,unused], cfg, gradient_scale=.1)
    torch.testing.assert_close(actual[0], expected, rtol=1e-12, atol=1e-13)
    assert actual[1] is None
    assert call.call_count == report['vjp_calls'] == 8
    assert report['group_count'] == 128
    assert parameter.grad is None


def test_uncapped_128_cell_gru_gradients_recover_original_pooled_objective():
    from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
    from response_task import rollout, task_loss
    from response_adjoints import collect_rollout, backward_actor
    from env_raptor import RaptorSimulator
    import copy
    torch.manual_seed(123)
    initial, _ = sample_coverage(128, [7,8,9,10], dtype=torch.float64)
    a = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    b = copy.deepcopy(a)
    config = cell_config(clip_norm=1e9)
    simulator = RaptorSimulator()
    loss_config = test_loss()
    record = collect_rollout(a, simulator, initial, loss_config, horizon=6, time_decay=1., group_config=config)
    backward_actor(a, simulator, record, loss_config, gradient_scale=.1)
    trace = rollout(b, simulator, initial, 6, time_decay=1.)
    (.1*task_loss(trace,loss_config)).backward()
    for actual,expected in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(actual.grad,expected.grad,rtol=1e-11,atol=1e-12)


@pytest.mark.parametrize('bad_cap', [0., -1., float('nan'), float('inf')])
def test_invalid_fixed_caps_are_rejected(bad_cap):
    with pytest.raises(ValueError):
        normalize_group_rows(torch.ones(3,2),1e-12,max_norm=bad_cap)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_fixed_cells_and_eight_chunk_gru_backward():
    from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
    from response_adjoints import collect_rollout, backward_actor
    from response_training import sample_training_scenarios
    from env_raptor import RaptorSimulator
    initial,_=sample_training_scenarios(32,0,horizon=5,sampling='coverage128',device='cuda')
    policy=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).cuda()
    rec=collect_rollout(policy,RaptorSimulator(),initial,test_loss(),horizon=5,group_config=cell_config())
    report=backward_actor(policy,RaptorSimulator(),rec,test_loss())['group_gradient']
    assert report['vjp_calls']==1 and report['group_count']==128
    assert bool((report['values'][:,2]<=1).all())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())

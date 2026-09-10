from __future__ import annotations

import importlib
import importlib.util

import pytest
import torch

from response_critic import critic_features
from response_phase1 import terminal_state_gradient
from response_task import TaskLossConfig, initialize, rollout, task_loss
from test_response_control import fixture


def diagnostic():
    assert importlib.util.find_spec('tools.diagnose_response_update'), 'missing update autopsy'
    return importlib.import_module('tools.diagnose_response_update')


def test_terminal_probe_uses_actor_scene_weights_without_normalizing_them():
    policy, sim, initial = fixture()
    closed = rollout(policy, sim, initial, 2).end
    target = torch.nn.Linear(critic_features(closed, 2, 6).shape[-1], 1, bias=False).double()
    with torch.no_grad():
        target.weight.zero_(); target.weight[0, 0] = 5.
    target.requires_grad_(False)
    weights = torch.tensor([.5, 1.], dtype=torch.float64)
    report = terminal_state_gradient(target, closed, 2, 6, mean_risk=False, weights=weights)
    # feature[0] = position_x / 5; dV/dposition_x = scene_weight.
    assert report['terminal_state_gradient_norm'] == pytest.approx(float(weights.norm()))
    assert all(p.grad is None for p in policy.parameters())
    assert all(p.grad is None for p in target.parameters())


def test_diagnostic_full_gradient_matches_pooled_cvar_with_and_without_chunks():
    d = diagnostic()
    policy, sim, initial = fixture()
    config = TaskLossConfig(steady_steps=2, tail_fraction=.5)
    policy.zero_grad(set_to_none=True)
    task_loss(rollout(policy, sim, initial, 6), config).backward()
    expected = d.parameter_gradient(policy).clone()
    for chunk in (1, 2):
        actual, evidence = d.exact_gradient(policy, sim, initial, 6, config, chunk_size=chunk)
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
        assert evidence['finite'] and evidence['weights_sum'] == pytest.approx(1.5)


def test_boundary_vjp_uses_only_preceding_window_and_preserves_suffix_time():
    d = diagnostic()
    policy, sim, initial = fixture()
    config = TaskLossConfig(steady_steps=2)
    weights = torch.tensor([.5, 1.], dtype=torch.float64)
    start = initialize(policy, initial)
    target = torch.nn.Linear(critic_features(start, 0, 6).shape[-1], 1).double().requires_grad_(False)
    result, tensors = d.boundary_comparison(policy, target, sim, start, 0, 2, 6, config, weights)
    # Direct graph reference with suffix policy parameters frozen: the only
    # Actor path allowed is through the first two steps' terminal state.
    trace = rollout(policy, sim, initial, 2)
    frozen = {name: p.detach() for name, p in policy.named_parameters()}
    continuation = rollout(policy, sim, trace.end, 4, parameters=frozen)
    from response_task import step_costs
    objective = (weights * step_costs(continuation, config, start=2, horizon=6).sum(0)).sum()
    grads = torch.autograd.grad(objective, list(policy.parameters()), allow_unused=True)
    expected = d.flatten_gradients(grads, list(policy.parameters()))
    torch.testing.assert_close(tensors['g_true'], expected, rtol=1e-8, atol=1e-9)
    assert result['state']['finite'] and result['parameter']['finite']


def test_gradient_comparison_keeps_float32_large_finite_norms_finite():
    d = diagnostic()
    a = torch.tensor([1.e25, -2.e25])
    result = d.compare_vectors(a, a * 2)
    assert result['finite'] and result['cosine'] == pytest.approx(1.)
    assert result['norm_ratio'] == pytest.approx(.5)

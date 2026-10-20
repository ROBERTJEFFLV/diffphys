"""Fixed failure bookkeeping; no rollout, physics or optimizer redesign."""

from loss_fixtures import test_loss
from dataclasses import asdict, replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from response_task import TaskLossConfig, rollout, step_costs, scenario_costs, uniform_scene_weights, step_cost_components
from loss_fixtures import parse_loss_args as parse_args
from test_episode_termination import actor, CountingSimulator, scheduled, flat_grads


def no_failure_cost(config):
    return replace(config, dead_cost=0.0, terminal_cost=0.0)


@pytest.mark.parametrize('name', ['dead_cost', 'terminal_cost'])
@pytest.mark.parametrize('value', [-1.0, float('nan'), float('inf')])
def test_failure_costs_reject_invalid_values(name, value):
    with pytest.raises(ValueError, match='failure costs'):
        test_loss(**{name: value})


def test_default_costs_and_automatic_cli_binding():
    config = test_loss()
    assert config.dead_cost == 3.0
    assert config.terminal_cost == 200.0
    args = parse_args([])
    assert (args.dead_cost, args.terminal_cost) == (3.0, 200.0)
    args = parse_args(['--dead-cost', '0', '--terminal-cost', '0'])
    assert (args.dead_cost, args.terminal_cost) == (0.0, 0.0)
    assert asdict(config)['terminal_cost'] == 200.0


@torch.no_grad()
def test_h500_exact_failure_cost_and_final_step_failure():
    h = 500
    lengths = [1, 19, 49, 50, 51, 100, 250, 400, 450, 499, 500, 501]
    trace = rollout(actor(), CountingSimulator(), scheduled(lengths, h), h)
    config = test_loss()
    base = step_cost_components(trace, no_failure_cost(config))
    features = step_cost_components(trace, config)
    for name in ('position', 'attitude_delta', 'action_delta'):
        torch.testing.assert_close(features[name], base[name], rtol=0, atol=0)
    actual = torch.stack((features['dead'].sum(0), features['terminal'].sum(0)), -1)
    expected = actual.new_tensor([
        [(h-x)*3.0/h, 200.0/h] if x <= h else [0.0, 0.0]
        for x in lengths
    ])
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual[1].sum().item() == pytest.approx(3.286)
    assert actual[-2].sum().item() == pytest.approx(.4)  # Dies on step 500.
    assert actual[-1].sum().item() == 0  # Time limit without failure.
    assert all(not value[~trace.valid].any() for value in features.values())
    assert (features['terminal'].ne(0).sum(0) == expected[:, 1].ne(0).long()).all()


@pytest.mark.parametrize('horizon', [8, 20])
def test_failure_penalty_has_one_horizon_division(horizon):
    trace = rollout(actor(), CountingSimulator(), scheduled([1, horizon, horizon+1], horizon), horizon)
    config = test_loss()
    parts = step_cost_components(trace, config)
    penalty = (parts['dead'] + parts['terminal']).sum(0)
    expected = penalty.new_tensor([((horizon-1)*3+200)/horizon, 200/horizon, 0])
    torch.testing.assert_close(penalty, expected)


def test_constants_do_not_add_state_gradient_or_post_terminal_calls():
    policy = actor(); simulator = CountingSimulator()
    trace = rollout(policy, simulator, scheduled([1, 8, 21], 20), 20)
    config = test_loss()
    calls = len(simulator.ids)
    parameters = list(policy.parameters())
    new_costs = scenario_costs(trace, config)
    old_costs = scenario_costs(trace, no_failure_cost(config))
    # Uniform weights cannot redirect a gradient when failure bookkeeping changes.
    weights = uniform_scene_weights(new_costs)
    a = torch.autograd.grad((weights*new_costs).sum(), parameters, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad((weights*old_costs).sum(), parameters, allow_unused=True)
    for x, y in zip(a, b):
        if x is None or y is None:
            assert x is None and y is None
        else:
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert len(simulator.ids) == calls == 29


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_loss_device_dtype_and_finite_gradient():
    initial = scheduled([1, 3, 9], 8, torch.float32).to('cuda', torch.float32)
    policy = actor(torch.float32).cuda()
    trace = rollout(policy, CountingSimulator(), initial, 8)
    costs = step_costs(trace, test_loss())
    assert costs.device.type == 'cuda' and costs.dtype == torch.float32
    costs.sum().backward()
    assert torch.isfinite(flat_grads(policy)).all()

"""Uniform planned-H normalization, direct new costs, and exact new-objective resume."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from response_task import step_costs, TASK_OBJECTIVE_VERSION
from loss_fixtures import test_loss, parse_loss_args as parse_args
from test_reference_training import args_for, run

FIELDS = ('pre_positions', 'pre_orientations', 'post_orientations', 'action_deltas', 'positions')


def constant_trace(horizon, dtype=torch.float64):
    q = torch.full((horizon, 1, 4), .5, dtype=dtype, requires_grad=True)
    return SimpleNamespace(
        pre_positions=torch.full((horizon, 1, 3), .25, dtype=dtype, requires_grad=True),
        pre_orientations=q, positions=torch.zeros(horizon, 1, 3, dtype=dtype),
        post_orientations=q.detach().clone().requires_grad_(),
        actions=torch.zeros(horizon, 1, 4, dtype=dtype),
        action_deltas=torch.full((horizon, 1, 4), .25, dtype=dtype, requires_grad=True),
        valid=torch.ones(horizon, 1, dtype=torch.bool),
        initial=SimpleNamespace(position_limit=torch.tensor([10.0], dtype=dtype)),
    )


def test_production_config_preserves_sampling_group_and_optimizer_settings():
    root = Path(__file__).resolve().parents[1]
    args = parse_args(['@' + str(root / 'configs/response_raptor_multi_airframe.args')])
    assert args.scenarios == 512 and args.horizon == 500
    assert args.group_max_groups == 128 and args.group_min_scenarios == 16
    assert args.lr == 3e-4 and args.time_decay == 1 and args.gradient_scale == .1


@pytest.mark.parametrize('horizon', [1, 8, 100, 101, 500, 751])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_every_step_has_identical_cost_and_direct_error_gradient(horizon, dtype):
    trace = constant_trace(horizon, dtype)
    costs = step_costs(trace, test_loss())
    # Hand-evaluated values: p^2=3/16, da^2=1/4, constant attitude costs zero.
    raw_cost = (7/16)**.5-.5 + (5/16)**.5-.25
    torch.testing.assert_close(costs, torch.full_like(costs, raw_cost/horizon))
    p, q, post_q, a = torch.autograd.grad(costs.sum(), (trace.pre_positions, trace.pre_orientations,
                                                     trace.post_orientations, trace.action_deltas))
    torch.testing.assert_close(p, torch.full_like(p, .25/(7/16)**.5/horizon))
    torch.testing.assert_close(a, torch.full_like(a, .25/(5/16)**.5/horizon))
    torch.testing.assert_close(q, torch.zeros_like(q))
    torch.testing.assert_close(post_q, torch.zeros_like(post_q))
    assert torch.equal(costs[0], costs[-1])


@pytest.mark.parametrize('horizon', [101, 500, 751])
def test_relocating_an_identical_position_error_does_not_reweight_time(horizon):
    trace = constant_trace(horizon)
    trace.pre_positions = torch.zeros_like(trace.pre_positions)
    trace.pre_positions[0, 0, 0] = .5
    early = step_costs(trace, test_loss()).sum()
    trace.pre_positions = trace.pre_positions.roll(horizon-1, dims=0)
    late = step_costs(trace, test_loss()).sum()
    torch.testing.assert_close(early, late, rtol=1e-13, atol=1e-13)


def test_h500_uniform_chunks_preserve_the_full_cost_and_gradient():
    trace = constant_trace(500)
    full = step_costs(trace, test_loss())
    chunks = []
    for start in range(0, 500, 37):
        stop = min(500, start+37)
        part = SimpleNamespace(**{name: getattr(trace, name)[start:stop] for name in (*FIELDS, 'actions')},
                               valid=trace.valid[start:stop], initial=trace.initial)
        chunks.append(step_costs(part, test_loss(), start=start, horizon=500))
    chunked = torch.cat(chunks)
    torch.testing.assert_close(full, chunked, rtol=0, atol=0)
    tensors = (trace.pre_positions, trace.pre_orientations, trace.post_orientations, trace.action_deltas)
    a = torch.autograd.grad(full.sum(), tensors, retain_graph=True)
    b = torch.autograd.grad(chunked.sum(), tensors)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize('flag', ['--steady-weight', '--steady-steps', '--tail-weight', '--huber-delta'])
def test_removed_loss_flags_are_rejected_before_creating_a_run(tmp_path, flag):
    path = tmp_path/'must_not_create'
    with pytest.raises(SystemExit):
        args_for(path, updates=0, extra=(flag, '1'))
    assert not path.exists()


def test_same_source_new_objective_resume_preserves_model_and_adam(tmp_path):
    full, split = tmp_path/'full', tmp_path/'split'
    run(args_for(full, 2)); run(args_for(split, 1))
    run(args_for(split, 2, ('--resume', str(split/'latest.pt'))))
    a = torch.load(full/'latest.pt', weights_only=True)
    b = torch.load(split/'latest.pt', weights_only=True)
    assert a['binding']['protocol']['task_objective'] == TASK_OBJECTIVE_VERSION
    assert a['model_sha256'] == b['model_sha256']
    assert a['next_update'] == b['next_update'] == 2
    for index, state in a['optimizer']['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b['optimizer']['state'][index][key], rtol=0, atol=0)
    assert torch.equal(a['rng']['torch'], b['rng']['torch'])

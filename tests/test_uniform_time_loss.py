"""Uniform time weights for new training; explicit historical scoring stays intact."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from response_task import TaskLossConfig, step_costs, weighted_task_features
from tools.train_response_control import parse_args
from test_reference_training import args_for, run


FIELDS = {
    'positions': (3, 'position_weight', 1.0),
    'velocities': (3, 'velocity_weight', 1.0),
    'omegas': (3, 'omega_weight', 1.0),
    'actions': (4, 'action_weight', 0.5),
    'action_deltas': (4, 'action_delta_weight', 0.5),
    'omega_deltas': (3, 'omega_delta_weight', 1.0),
}


def constant_trace(horizon, dtype=torch.float64):
    # Isolate loss weighting from the dynamics Jacobian. All errors are in the
    # quadratic part of Huber and the one scene remains inside its boundary.
    values = {name: torch.full((horizon, 1, width), 0.25, dtype=dtype,
                              requires_grad=True)
              for name, (width, _, _) in FIELDS.items()}
    return SimpleNamespace(**values,
        valid=torch.ones(horizon, 1, dtype=torch.bool),
        initial=SimpleNamespace(position_limit=torch.tensor([10.0], dtype=dtype)))


def test_defaults_and_production_config_use_uniform_time_weights():
    assert TaskLossConfig().steady_weight == 0.0
    assert parse_args([]).steady_weight == 0.0
    root = Path(__file__).resolve().parents[1]
    args = parse_args(['@' + str(root / 'configs/response_raptor_multi_airframe.args'),
                       '--scenarios', '512'])
    assert args.steady_weight == 0.0
    # Per-bank convention stays intact; the later requested fixed-cell gradient
    # groups now match the coverage sampler. This must not reweight time.
    assert args.scenarios == 512
    assert args.group_max_groups == 128
    assert args.group_min_scenarios == 16


@pytest.mark.parametrize('horizon', [1, 8, 100, 101, 500, 751])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_every_step_has_identical_cost_and_direct_error_gradient(horizon, dtype):
    trace = constant_trace(horizon, dtype)
    config = TaskLossConfig()
    costs = step_costs(trace, config)
    raw_cost = sum(width * getattr(config, weight) * (0.25 * scale)**2
                   for width, weight, scale in FIELDS.values())
    torch.testing.assert_close(costs, torch.full_like(costs, raw_cost / horizon))
    gradients = torch.autograd.grad(costs.sum(), [getattr(trace, name) for name in FIELDS])
    for gradient, (_, weight, scale) in zip(gradients, FIELDS.values()):
        expected = 2 * getattr(config, weight) * scale**2 * 0.25 / horizon
        torch.testing.assert_close(gradient, torch.full_like(gradient, expected))
    assert torch.equal(costs[0], costs[-1])


@pytest.mark.parametrize('horizon', [101, 500, 751])
def test_relocating_an_identical_error_does_not_change_cost(horizon):
    trace = constant_trace(horizon)
    trace.positions = torch.zeros_like(trace.positions)
    trace.positions[0, 0, 0] = 0.5
    early = step_costs(trace, TaskLossConfig()).sum()
    trace.positions = trace.positions.roll(horizon - 1, dims=0)
    late = step_costs(trace, TaskLossConfig()).sum()
    torch.testing.assert_close(early, late, rtol=1e-13, atol=1e-13)


def test_h500_uniform_chunks_preserve_the_full_cost_and_gradient():
    trace = constant_trace(500)
    config = TaskLossConfig()
    full = step_costs(trace, config)
    chunks = []
    # Include the previous 400-step boundary and an irregular final chunk.
    for start in range(0, 500, 37):
        stop = min(500, start + 37)
        part = SimpleNamespace(**{name: getattr(trace, name)[start:stop] for name in FIELDS},
                               valid=trace.valid[start:stop], initial=trace.initial)
        chunks.append(step_costs(part, config, start=start, horizon=500))
    chunked = torch.cat(chunks)
    torch.testing.assert_close(full, chunked, rtol=0, atol=0)
    tensors = [getattr(trace, name) for name in FIELDS]
    a = torch.autograd.grad(full.sum(), tensors, retain_graph=True)
    b = torch.autograd.grad(chunked.sum(), tensors)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_historical_config_still_reproduces_the_old_eleven_to_one_score():
    trace = constant_trace(500)
    legacy = TaskLossConfig(**{**asdict(TaskLossConfig()), 'steady_weight': 2.0})
    old = step_costs(trace, legacy)
    new = step_costs(trace, TaskLossConfig())
    torch.testing.assert_close(old[:400], new[:400], rtol=0, atol=0)
    torch.testing.assert_close(old[400:], 11 * new[400:], rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(old.sum(), 3 * new.sum(), rtol=1e-13, atol=1e-13)
    # Failure bookkeeping is deliberately not reweighted by the steady window.
    torch.testing.assert_close(weighted_task_features(trace, legacy)[..., 20:],
                               weighted_task_features(trace, TaskLossConfig())[..., 20:],
                               rtol=0, atol=0)


@pytest.mark.parametrize('weight', [0.1, 2.0])
def test_new_training_rejects_legacy_time_weight_before_creating_a_run(tmp_path, weight):
    path = tmp_path / 'must_not_create'
    args = args_for(path, updates=0, extra=('--steady-weight', str(weight)))
    with pytest.raises(ValueError, match='uniform per-step'):
        run(args)
    assert not path.exists()


def test_same_source_uniform_resume_preserves_model_and_adam(tmp_path):
    full, split = tmp_path / 'full', tmp_path / 'split'
    run(args_for(full, 2))
    run(args_for(split, 1))
    run(args_for(split, 2, ('--resume', str(split / 'latest.pt'))))
    a = torch.load(full / 'latest.pt', weights_only=True)
    b = torch.load(split / 'latest.pt', weights_only=True)
    assert a['binding']['protocol']['loss']['steady_weight'] == 0.0
    assert b['binding']['protocol']['loss']['steady_weight'] == 0.0
    assert a['model_sha256'] == b['model_sha256']
    assert a['next_update'] == b['next_update'] == 2
    for index, state in a['optimizer']['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b['optimizer']['state'][index][key], rtol=0, atol=0)
    assert torch.equal(a['rng']['torch'], b['rng']['torch'])

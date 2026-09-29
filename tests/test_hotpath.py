"""Execution-only changes must retain losses, full BPTT and audit semantics."""
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from env_raptor import RaptorSimulator
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
import response_task as task
from response_adjoints import collect_rollout, backward_actor
from response_groups import GroupBalanceConfig, normalize_group_rows
from response_training import sample_training_scenarios


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('decay', [0., 1.])
@pytest.mark.parametrize('terminate', [False, True])
def test_lean_rollout_preserves_all_task_tensors_and_full_parameter_gradients(dtype, decay, terminate):
    torch.manual_seed(29)
    sim = RaptorSimulator()
    state = sim.reset(16, seed=3, horizon=55, dtype=dtype)
    if terminate:
        limits = state.position_limit.clone()
        limits[::3] = .001
        state = replace(state, position_limit=limits)
    p = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).to(dtype=dtype)
    reference = task.rollout(p, sim, state, 55, time_decay=decay)
    actual = task.rollout(p, sim, state, 55, time_decay=decay, record_observations=False)
    assert actual.observations is None
    for key in ('actions', 'positions', 'velocities', 'omegas', 'action_deltas', 'omega_deltas', 'valid'):
        assert torch.equal(getattr(reference, key), getattr(actual, key)), key
    a, b = [task.task_loss(t, task.TaskLossConfig()) for t in (reference, actual)]
    assert torch.equal(a, b)
    ga = torch.autograd.grad(a, tuple(p.parameters()))
    gb = torch.autograd.grad(b, tuple(p.parameters()))
    for x, y in zip(ga, gb):
        assert torch.equal(x, y)


def test_training_does_not_construct_unused_observation_tape_or_repeat_huber():
    torch.manual_seed(7)
    state, _ = sample_training_scenarios(32, 0, horizon=8, sampling='coverage128')
    p = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4))
    cfg = GroupBalanceConfig(max_groups=128, min_scenarios=1, layout='coverage128')
    with patch.object(task, 'observation', wraps=task.observation) as obs, \
         patch.object(task, 'weighted_task_features', wraps=task.weighted_task_features) as features:
        record = collect_rollout(p, RaptorSimulator(), state, task.TaskLossConfig(),
                                 horizon=8, group_config=cfg)
    # One initial-state observation, then exactly one per live control step.
    assert obs.call_count == 9
    assert features.call_count == 1
    backward_actor(p, RaptorSimulator(), record, task.TaskLossConfig())


def test_reused_statistics_are_bitwise_equal_and_not_attached_to_graph():
    state = RaptorSimulator().reset(8, seed=4, horizon=6, dtype=torch.float64)
    p = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    trace = task.rollout(p, RaptorSimulator(), state, 6)
    loss = task.TaskLossConfig()
    squares = task.weighted_task_features(trace, loss, horizon=6).square()
    a, b = [task.FlightStatistics(state, 6, loss) for _ in range(2)]
    a.add(trace, 0)
    b.add(trace, 0, feature_squares=squares.detach())
    costs = squares.sum((0, 2)).detach()
    weights = task.risk_weights(costs, loss)
    assert a.finish(costs, weights) == b.finish(costs, weights)
    assert not b.components.requires_grad
    with pytest.raises(ValueError, match='feature'):
        b.add(trace, 0, feature_squares=squares[..., :1])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_mixed_dtype_accumulate_equals_explicit_fp64_cast(dtype):
    generator = torch.Generator().manual_seed(71)
    a = torch.randn(128, 32, generator=generator, dtype=torch.float64)
    b = a.clone()
    for _ in range(50):
        value = torch.randn(a.shape, generator=generator, dtype=dtype)
        a.add_(value.double())
        b.add_(value)
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable: no GPU throughput validation')
def test_cuda_mixed_dtype_accumulate_equals_explicit_fp64_cast():
    a = torch.randn(128, 16068, device='cuda', dtype=torch.float64)
    b = a.clone()
    for _ in range(20):
        value = torch.randn(a.shape, device='cuda', dtype=torch.float32)
        a.add_(value.double())
        b.add_(value)
    assert torch.equal(a, b)


def test_preflight_single_backend_reports_no_unperformed_comparison(tmp_path, capsys):
    import json
    from tools.benchmark_hotpath import main
    report = tmp_path / 'preflight.json'
    assert main(['--device', 'cpu', '--backends', 'eager', '--scenes', '128',
                 '--horizon', '2', '--memory-dim', '4', '--warmup', '1',
                 '--repeats', '1', '--report', str(report)]) == 0
    result = json.loads(report.read_text())
    assert result['passed'] is None
    assert len(result['samples']['eager']) == 2
    assert result['samples']['eager'][0]['warmup']
    assert not result['samples']['eager'][1]['warmup']
    assert result['execution']['eager']['physics_backend'] == 'eager'


@pytest.mark.parametrize('option', ['--time-decay', '--clip-norm'])
def test_preflight_rejects_nonfinite_settings(tmp_path, option):
    from tools.benchmark_hotpath import main
    with pytest.raises(SystemExit):
        main(['--device', 'cpu', option, 'nan', '--report', str(tmp_path / 'x.json')])


def test_backend_comparison_does_not_hide_bad_raw_group_gradients():
    from tools.benchmark_hotpath import compare
    a = {'raw_groups': torch.ones(2, 3), 'clipped_gradient': torch.ones(3)}
    b = {k: v.clone() for k, v in a.items()}
    b['raw_groups'][0, 0] = 100.
    check = compare(a, b, rtol=1e-5, atol=1e-6)
    assert check['clipped_gradient_close']
    assert not check['raw_groups_close'] and not check['passed']

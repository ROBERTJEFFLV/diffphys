"""Compiled execution preserves the native model, physics and grouped adjoints."""
from __future__ import annotations

import copy
from dataclasses import asdict

import pytest
import torch

from env_raptor import RaptorParams, RaptorSimulator
from pass_stop import GoalMotorPolicy, GoalConfig, PassStopLossConfig
from response_adjoints import collect_rollout, backward_actor
from response_groups import GroupBalanceConfig
from response_noise import DisturbanceConfig


def test_cpu_execution_keeps_native_modules_and_parameters():
    from compiled_execution import compile_actor
    from compiled_physics import make_simulator
    actor = GoalMotorPolicy()
    before = copy.deepcopy(actor.state_dict())
    assert compile_actor(actor, 'cpu') is actor
    assert type(make_simulator(RaptorParams(), 'cpu')) is RaptorSimulator
    assert type(actor.response_memory) is torch.nn.GRUCell
    assert type(actor.readout) is torch.nn.Linear
    assert len(list(actor.parameters())) == 6
    for name, value in actor.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('dtype,decay,horizon', [(torch.float32,0.,12), (torch.float64,0.,500), (torch.float32,1.,12)])
def test_compiled_goal_rollout_gradients_and_adam_match_reference(dtype, decay, horizon):
    from compiled_execution import compile_actor
    from compiled_physics import make_simulator
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(17)
    actor = GoalMotorPolicy().to(device='cuda', dtype=dtype)
    initial = RaptorSimulator().reset(64, device='cuda', dtype=dtype, seed=19,
                                      horizon=horizon, disturbances=DisturbanceConfig())
    config = PassStopLossConfig(.01, .01, 25., mission=asdict(GoalConfig()))
    group = GroupBalanceConfig(max_groups=2, min_scenarios=32)
    results = []
    for compiled in (False, True):
        policy = copy.deepcopy(actor)
        sim = make_simulator(RaptorParams(), 'cuda') if compiled else RaptorSimulator()
        if compiled:
            compile_actor(policy, 'cuda')
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        record = collect_rollout(policy, sim, initial, config, horizon=horizon,
                                  time_decay=decay, group_config=group)
        if compiled:
            assert 'CompiledFunctionBackward' in type(record.costs.grad_fn).__name__
        backward_actor(policy, sim, record, config)
        costs = record.costs.detach().clone()
        gradients = [p.grad.clone() for p in policy.parameters()]
        optimizer.step()
        results.append((costs, gradients, copy.deepcopy(policy.state_dict())))
        del record, policy, optimizer
    tol = 3e-4 if dtype == torch.float32 else 2e-10
    for left, right in zip(results[0][:2], results[1][:2]):
        torch.testing.assert_close(left, right, rtol=tol, atol=tol)
    for name in results[0][2]:
        torch.testing.assert_close(results[0][2][name], results[1][2][name], rtol=tol, atol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@torch.no_grad()
def test_compiled_h500_evaluation_preserves_event_scans():
    from pass_stop import task_rollout, typed_costs
    torch.set_num_threads(1)
    torch.manual_seed(17)
    actor = GoalMotorPolicy().cuda()
    initial = RaptorSimulator().reset(256, device='cuda', seed=19, horizon=500)
    config = PassStopLossConfig(.01, .01, 25., mission=asdict(GoalConfig()))
    trace = task_rollout(actor, RaptorSimulator(), initial, 500, config)
    expected = typed_costs(trace, config)
    actual = typed_costs(trace, config, compiled=True)
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    for name in ('region_enter_fraction', 'region_exit_fraction', 'region_samples',
                 'pass_fraction', 'success_fraction', 'physical_transitions'):
        assert actual[1][name] == expected[1][name], name

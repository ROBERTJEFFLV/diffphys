"""Position is the only episode boundary; speed remains a differentiable cost."""
from dataclasses import replace
import copy

import pytest
import torch

from env_l2f import L2FParams, L2FSimulator
from response_task import TaskLossConfig, rollout, reference_episode_metrics, weighted_task_features
from response_adjoints import collect_boundary_rollout, backward_actor
from test_episode_termination import actor, flat_grads


@pytest.mark.parametrize('profile', ['raptor', 'l2f'])
def test_only_position_terminates(profile):
    sim = L2FSimulator(L2FParams(protocol=profile))
    s = sim.reset(2, seed=7, dtype=torch.float64, horizon=8)
    s = replace(s, position=torch.zeros_like(s.position),
                velocity=2*s.velocity_limit[:, None].expand(-1, 3),
                omega=2*s.omega_limit[:, None].expand(-1, 3))
    assert not sim.terminated(s).any()
    p = s.position.clone()
    p[0, 0] = s.position_limit[0]
    p[1, 1] = -s.position_limit[1] - 1e-6
    assert sim.terminated(replace(s, position=p)).tolist() == [False, True]


def test_speed_exceedance_continues_cost_evaluation_and_backward():
    sim = L2FSimulator()
    s = sim.reset(2, seed=7, dtype=torch.float64, horizon=8)
    s = replace(s, position=torch.zeros_like(s.position),
                position_limit=torch.full_like(s.position_limit, 100),
                velocity=torch.full_like(s.velocity, 2.1),
                omega=torch.full_like(s.omega, 35.1))
    a = actor(); b = copy.deepcopy(a); loss = TaskLossConfig()
    trace = rollout(a, sim, s, 8)
    assert trace.valid.all()
    assert trace.end.physical.step_index.tolist() == [8, 8]
    assert reference_episode_metrics(trace)['raptor_share_terminated'] == 0
    assert not weighted_task_features(trace, loss)[..., -2:].any()
    records = []
    for policy, mode in [(a, 'full'), (b, 'windowed')]:
        record = collect_boundary_rollout(policy, sim, s, loss, horizon=8,
                                         window_steps=4, backprop_mode=mode)
        backward_actor(policy, sim, record, loss)
        assert torch.isfinite(flat_grads(policy)).all()
        assert flat_grads(policy).norm() > 0
        records.append(record)
    torch.testing.assert_close(records[0].costs, records[1].costs, rtol=0, atol=0)
    torch.testing.assert_close(flat_grads(a), flat_grads(b), rtol=1e-9, atol=1e-9)

"""A small airframe must use the same0.1m ball in planning, Actor and scoring."""
from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from env_raptor import quaternion_rotation
from pass_stop import GoalConfig, GoalMotorPolicy, PassStopLossConfig, prepare, typed_costs
from response_noise import measured_observation
from response_sampling import sample_coverage
from response_task import _select_rows


def sampled(dtype, *, plane=True):
    raw, _ = sample_coverage(128, [31000007, 31000008, 31000009, 31000010], dtype=dtype)
    goal = GoalConfig(plane_task=plane, plane_crossing_only=plane, random_via=True,
                      random_initial_state=True, velocities=((0., 0., 0.), (1.5, 0., 0.), (0., 0., 0.)))
    loss = PassStopLossConfig(.01, .01, 25., mission=asdict(goal))
    actor = GoalMotorPolicy().to(dtype)
    return raw, actor, loss


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('plane', [True, False])
def test_sampled_radius_has_floor_and_actor_receives_the_same_value(dtype, plane):
    raw, actor, loss = sampled(dtype, plane=plane)
    closed = prepare(actor, raw, loss)
    small = raw.arm_length < .05
    assert small.any() and (~small).any()
    torch.testing.assert_close(closed.policy.radius[small], raw.arm_length.new_full((int(small.sum()),), .1), rtol=0, atol=0)
    torch.testing.assert_close(closed.policy.radius[~small], 2*raw.arm_length[~small], rtol=0, atol=0)
    obs = measured_observation(closed.physical)
    features = actor.goal_features(obs, closed.policy, closed.policy.phase)
    torch.testing.assert_close(features[:, -1], closed.policy.radius, rtol=0, atol=0)
    assert ((closed.policy.start-closed.policy.via).norm(dim=-1) > closed.policy.radius).all()
    assert ((closed.policy.stop-closed.policy.via).norm(dim=-1) > closed.policy.radius+.1).all()
    assert (closed.policy.via.abs()+closed.policy.radius[:, None] < closed.physical.position_limit[:, None]).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_old_small_airframe_gap_now_triggers_phase_and_hard_plane_scoring(dtype):
    raw, actor, loss = sampled(dtype)
    index = (raw.arm_length < .04).nonzero().flatten()[:1]
    assert len(index) == 1
    closed = prepare(actor, _select_rows(raw, index), loss)
    state = closed.policy
    heading = torch.nn.functional.normalize(state.target_velocities[:, 1], dim=-1)
    lateral = state.via_rotation[:, :, 2]
    offset = .086
    p = state.via[None]+raw.arm_length.new_tensor([-.2, 0., .2])[:, None, None]*heading[None]+offset*lateral[None]
    assert float(2*closed.physical.arm_length[0]) < offset < .1
    q = state.via_quaternion[None].expand(3, -1, -1)
    v = (1.6*heading)[None].expand(3, -1, -1)
    w = torch.zeros_like(v)
    R = quaternion_rotation(q)
    obs = torch.cat((p, v, R.flatten(2), w, p.new_zeros(3, 1, 4)), -1)
    start = replace(state, previous_position=p[0], phase=torch.zeros_like(state.phase))
    entered = actor(obs[1], start).next_state
    assert entered.phase.item() == 1
    assert actor(obs[2], entered).next_state.phase.item() == 2
    initial = replace(closed.physical, position=p[0], velocity=v[0], orientation=q[0], omega=w[0])
    trace = SimpleNamespace(initial=initial, positions=p[1:], velocities=v[1:], post_orientations=q[1:],
        omegas=w[1:], observations=obs, actions=p.new_zeros(2, 1, 4), action_deltas=p.new_zeros(2, 1, 4),
        valid=torch.ones(2, 1, dtype=torch.bool), end=closed)
    _, metrics, components = typed_costs(trace, loss)
    assert metrics['plane_crossing_fraction'] == metrics['pass_fraction'] == 1.
    assert components['pass_missing'].item() == 0.


def test_previous_radius_protocol_is_not_silently_loaded_as_the_new_task():
    from response_training import require_reference_checkpoint
    path = Path(__file__).resolve().parents[1]/'frozen/20261026-u1456/latest.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    with pytest.raises(ValueError, match='task/Actor mismatch'):
        require_reference_checkpoint(saved)

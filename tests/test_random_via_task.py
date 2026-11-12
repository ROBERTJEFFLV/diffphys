"""Requested random starts and strict PASS targets, using the production task."""
from __future__ import annotations

from dataclasses import asdict, fields, replace
import math
import random

import pytest
import torch

from env_raptor import RaptorSimulator, quaternion_rotation
from pass_stop import GoalConfig, GoalMotorPolicy, PassStopLossConfig, prepare, typed_costs
from response_noise import DisturbanceConfig, attach_disturbances
from response_sampling import sample_coverage
from response_task import IMMUTABLE_TAPES
from test_pass_stop import synthetic


def mission(**changes):
    return GoalConfig(random_initial_state=True, random_via=True,
                      velocities=((0., 0., 0.), (1.5, 0., 0.), (0., 0., 0.)), **changes)


def config(goal):
    return PassStopLossConfig(.01, .01, 25., mission=asdict(goal))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_original_random_initial_state_and_actual_airframes_are_retained(dtype):
    raw, _ = sample_coverage(128, [31000007, 31000008, 31000009, 31000010], dtype=dtype)
    goal = mission()
    actor = GoalMotorPolicy().to(dtype)
    before = {f.name: getattr(raw, f.name).clone() for f in fields(raw)}
    actual = prepare(actor, raw, config(goal))
    for name in ('velocity', 'orientation', 'omega', 'motor', 'previous_velocity',
                 'previous_action', 'mass', 'inertia', 'thrust_coefficients', 'arm_length',
                 'position_limit', 'noise_row', 'velocity_delay'):
        torch.testing.assert_close(getattr(actual.physical, name), before[name], rtol=0, atol=0)
    torch.testing.assert_close(actual.physical.position,
                               raw.position_limit[:, None]*raw.position.new_tensor((-.7, 0., 0.)), rtol=0, atol=0)
    torch.testing.assert_close(actual.policy.stop,
                               raw.position_limit[:, None]*raw.position.new_tensor((.7, 0., 0.)), rtol=0, atol=0)
    assert actual.physical.noise_tape is raw.noise_tape
    assert actual.physical.rotation_tape is raw.rotation_tape
    assert not actual.physical.external_force.any() and not actual.physical.pulse_tape.any()
    # Random physical pose is not leaked through the public START target pose.
    torch.testing.assert_close(actual.policy.start_rotation,
                               torch.eye(3, dtype=dtype).expand(128, -1, -1), rtol=0, atol=0)
    for f in fields(raw):
        torch.testing.assert_close(getattr(raw, f.name), before[f.name], rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_every_pass_goal_has_requested_tilt_margin_speed_and_fence_margin(dtype):
    raw, _ = sample_coverage(128, [31000007, 31000008, 31000009, 31000010], dtype=dtype)
    closed = prepare(GoalMotorPolicy().to(dtype), raw, config(mission()))
    s = closed.policy
    assert bool((s.via.abs() <= (raw.position_limit-7*raw.arm_length)[:, None]).all())
    tilt = torch.rad2deg(torch.acos(s.via_rotation[:, 2, 2].clamp(-1., 1.)))
    assert bool(((tilt >= 61) & (tilt <= 119)).all())
    assert float(tilt.std()) > 10
    torch.testing.assert_close(s.target_velocities[:, 1].norm(dim=-1), tilt.new_full((128,), 1.5), rtol=0, atol=0)
    assert bool(((s.start-s.via).norm(dim=-1) > torch.maximum(s.radius, s.radius.new_full((128,), .2))).all())
    assert bool(((s.stop-s.via).norm(dim=-1) > torch.maximum(s.radius+.1, s.radius.new_full((128,), .2))).all())


def test_task_sampling_is_replayable_row_stable_and_independent_of_hidden_state_and_noise():
    raw, _ = sample_coverage(128, [31000007, 31000008, 31000009, 31000010], dtype=torch.float64)
    actor = GoalMotorPolicy().double()
    cfg = config(mission())
    torch_rng, python_rng = torch.get_rng_state(), random.getstate()
    a = prepare(actor, raw, cfg).policy
    assert torch.equal(torch.get_rng_state(), torch_rng) and random.getstate() == python_rng
    order = torch.randperm(128)
    permuted = replace(raw, **{f.name: getattr(raw, f.name).index_select(0, order)
                               for f in fields(raw) if f.name not in IMMUTABLE_TAPES})
    b = prepare(actor, permuted, cfg).policy
    for name in ('via', 'via_rotation', 'target_velocities'):
        torch.testing.assert_close(getattr(b, name), getattr(a, name)[order], rtol=0, atol=0)
    noisy = attach_disturbances(raw, DisturbanceConfig(), seed=456, horizon=3)
    noisy = replace(noisy, motor=raw.motor+.1, velocity=-raw.velocity,
                    orientation=-raw.orientation, omega=-raw.omega)
    c = prepare(actor, noisy, cfg).policy
    torch.testing.assert_close(c.via, a.via, rtol=0, atol=0)
    torch.testing.assert_close(c.via_rotation, a.via_rotation, rtol=0, atol=0)
    changed = prepare(actor, raw, config(mission(task_seed=8))).policy
    assert not torch.equal(changed.via, a.via)
    assert not torch.equal(changed.via_rotation, a.via_rotation)


@pytest.mark.parametrize('speed', [1.49, 1.5, 2.5])
def test_speed_floor_has_no_slack_and_does_not_penalize_faster_passages(speed):
    p = torch.tensor([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]], dtype=torch.float64)
    goal = GoalConfig(relative_to_boundary=False, points=((-1., 0., 0.), (0., 0., 0.), (1., 0., 0.)),
                      velocities=((0., 0., 0.), (1.5, 0., 0.), (0., 0., 0.)))
    q = p.new_tensor(goal.via_quaternion).expand(3, 1, 4)
    velocity = p.new_tensor((speed, 0., 0.)).expand(3, 1, 3).clone().requires_grad_()
    trace, cfg = synthetic(p, q=q, v=velocity, config=config(goal))
    cost = typed_costs(trace, cfg)[2]['pass_speed']
    if speed < 1.5:
        assert float(cost.detach()) > 0
        gradient, = torch.autograd.grad(cost.sum(), velocity)
        assert float((gradient*velocity).sum().detach()) < 0
    else:
        assert float(cost.detach()) == 0


def test_full_pose_loss_reads_each_rows_sampled_quaternion():
    from pass_stop import RandomGoalState

    p = torch.tensor([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]], dtype=torch.float64)
    p = p[:, None].expand(-1, 2, -1)
    angles = torch.tensor([75., 110.], dtype=p.dtype, requires_grad=True)
    half = angles*math.pi/360
    q = torch.stack((half.cos(), half.sin(), half*0, half*0), -1)[None].expand(3, -1, -1)
    trace, cfg = synthetic(p, q=q)
    target = q[0].detach().clone()
    state = RandomGoalState(**vars(trace.end.policy), via_quaternion=target)
    trace.end = replace(trace.end, policy=state)
    assert not typed_costs(trace, cfg)[2]['pass_attitude'].any()
    target[1] = target[0]
    costs = typed_costs(trace, cfg)[2]['pass_attitude']
    assert costs[0] == 0 and costs[1] > 0
    grad, = torch.autograd.grad(costs.sum(), angles)
    assert grad[0] == 0 and grad[1] > 0


@pytest.mark.parametrize('changes', [
    {'via_tilt_range_deg': (59., 120.)}, {'via_tilt_range_deg': (60., 121.)},
    {'via_boundary_margin_arms': 6.}, {'pose_tolerance_deg': 30.},
    {'task_seed': True},
])
def test_random_task_rejects_weakened_or_inconsistent_constraints(changes):
    with pytest.raises(ValueError):
        mission(**changes)


def test_random_task_rejects_speed_floor_below_requested_minimum():
    with pytest.raises(ValueError):
        GoalConfig(random_via=True, velocities=((0., 0., 0.), (1.49, 0., 0.), (0., 0., 0.)))


def test_impossible_local_geometry_is_reported_without_dropping_a_scene():
    raw = RaptorSimulator().reset(2, seed=73)
    small = replace(raw, arm_length=raw.arm_length.new_full((2,), .0001),
                    position_limit=raw.position_limit.new_full((2,), .003))
    with pytest.raises(ValueError, match='geometry'):
        prepare(GoalMotorPolicy(), small, config(mission()))


def test_random_goals_keep_native_group_backward_and_exact_replay_commands():
    import numpy as np
    from response_adjoints import collect_rollout, backward_actor
    from response_groups import GroupBalanceConfig
    from tools.short_eval_replay import capture_short_eval

    raw, _ = sample_coverage(128, [31000007, 31000008, 31000009, 31000010], dtype=torch.float64)
    actor = GoalMotorPolicy().double()
    cfg = config(mission())
    groups = GroupBalanceConfig(layout='coverage128', max_groups=128, min_scenarios=1)
    record = collect_rollout(actor, RaptorSimulator(), raw, cfg, horizon=3,
                             time_decay=0, group_config=groups)
    result = backward_actor(actor, RaptorSimulator(), record, cfg)
    assert result['group_gradient']['group_count'] == 128
    assert result['group_gradient']['graph_traversals'] == 1
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
    assert any(bool(p.grad.abs().any()) for p in actor.parameters())
    report, arrays = capture_short_eval(actor, RaptorSimulator(), raw, 3, cfg)
    expected = prepare(actor, raw, cfg)
    np.testing.assert_array_equal(arrays['position'][0], expected.physical.position.numpy())
    np.testing.assert_array_equal(arrays['velocity'][0], raw.velocity.numpy())
    np.testing.assert_array_equal(arrays['orientation'][0], raw.orientation.numpy())
    np.testing.assert_array_equal(arrays['pass_rotation'], expected.policy.via_rotation.numpy())
    np.testing.assert_array_equal(arrays['mission_points'][:, 1], expected.policy.via.numpy())
    assert report['via_target_tilt_deg_min'] >= 61
    assert report['via_target_tilt_deg_max'] <= 119
    assert report['via_boundary_clearance_arms_min'] >= 7


def test_new_preset_is_fresh_and_random_task_resume_is_exact(tmp_path):
    from tools.train_response_control import parse_args
    from test_pass_stop_integration import cli, equal_tree

    args = parse_args(['@configs/random_via_flight.args'])
    assert args.resume is None and args.init_checkpoint is None
    assert args.horizon == 500 and args.scenarios == 768 and args.train_sampling == 'coverage128'
    assert args.mission.random_initial_state and args.mission.random_via
    extra = ['--mission-config', 'configs/random_via_pass_stop.json']
    first, continuous = tmp_path/'first', tmp_path/'continuous'
    cli(first, 1, extra=extra)
    cli(continuous, 2, extra=extra)
    cli(first, 2, extra=[*extra, '--resume', str(first/'latest.pt')])
    a = torch.load(first/'latest.pt', weights_only=True)
    b = torch.load(continuous/'latest.pt', weights_only=True)
    assert equal_tree(a['model'], b['model']) and equal_tree(a['optimizer'], b['optimizer'])
    assert equal_tree(a['rng'], b['rng'])
    assert a['binding']['protocol']['loss']['mission']['random_via'] is True
    assert a['binding']['protocol']['task_objective'].endswith('random-initial-via-v1')

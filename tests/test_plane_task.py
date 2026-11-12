"""An unoriented plane frees body spin while preserving directed native thrust."""
from __future__ import annotations

from dataclasses import asdict, fields, replace
import math

import pytest
import torch

from env_raptor import RaptorSimulator, quaternion_rotation
from pass_stop import GoalConfig, GoalMotorPolicy, PassStopLossConfig, prepare, typed_costs
from response_sampling import sample_coverage
from response_task import IMMUTABLE_TAPES
from test_pass_stop import synthetic


def plane_config():
    goal = GoalConfig(plane_task=True, random_via=True, random_initial_state=True,
                      velocities=((0., 0., 0.), (1.5, 0., 0.), (0., 0., 0.)))
    return PassStopLossConfig(.01, .01, 25., mission=asdict(goal))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_cube_surface_endpoints_interior_pass_and_directed_plane_targets(dtype):
    raw, _ = sample_coverage(128, [31000007, 31000008, 31000009, 31000010], dtype=dtype)
    closed = prepare(GoalMotorPolicy().to(dtype), raw, plane_config())
    s = closed.policy
    L = raw.arm_length
    torch.testing.assert_close(closed.physical.position_limit, 50*L, rtol=0, atol=0)
    for point in (s.start, s.stop):
        torch.testing.assert_close(point.abs().amax(-1), 40*L, rtol=2e-6, atol=2e-7)
        assert bool((point.abs() <= 40*L[:, None]+1e-7).all())
    assert bool((s.via.abs() <= 30*L[:, None]).all())
    heading = torch.nn.functional.normalize(s.stop-s.via, dim=-1)
    velocity = s.target_velocities[:, 1]
    torch.testing.assert_close(torch.nn.functional.normalize(velocity, dim=-1), heading,
                               rtol=2e-6, atol=2e-7)
    torch.testing.assert_close((s.via_rotation[:, :, 2]*heading).sum(-1),
                               torch.zeros_like(L), rtol=0, atol=5e-7)
    assert bool((s.via_rotation[:, 2, 2] >= -1e-7).all())
    R = s.via_rotation
    roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    pitch = torch.asin((-R[:, 2, 0]).clamp(-1., 1.))
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    assert bool((torch.stack((roll, pitch, yaw), -1).abs() <= math.pi/2+1e-5).all())
    for name in ('velocity', 'orientation', 'omega', 'motor', 'previous_velocity',
                 'mass', 'inertia', 'arm_length', 'noise_row'):
        torch.testing.assert_close(getattr(closed.physical, name), getattr(raw, name), rtol=0, atol=0)
    torch.testing.assert_close(raw.position_limit, 30*L, rtol=2e-6, atol=2e-7)
    assert closed.physical.noise_tape is raw.noise_tape


def test_plane_sampling_is_stable_under_row_permutation():
    raw, _ = sample_coverage(128, [31000007, 31000008, 31000009, 31000010], dtype=torch.float64)
    actor = GoalMotorPolicy().double()
    a = prepare(actor, raw, plane_config()).policy
    order = torch.arange(127, -1, -1)
    shuffled = replace(raw, **{f.name: getattr(raw, f.name).index_select(0, order)
                               for f in fields(raw) if f.name not in IMMUTABLE_TAPES})
    b = prepare(actor, shuffled, plane_config()).policy
    for name in ('start', 'stop', 'via', 'via_rotation', 'target_velocities'):
        torch.testing.assert_close(getattr(b, name), getattr(a, name)[order], rtol=0, atol=0)


@pytest.mark.parametrize('roll', [45., -135., 225.])
def test_unoriented_plane_identifies_flipped_normals_and_allows_body_spin(roll):
    from plane_tasks import plane_attitude_ratio

    angle = torch.tensor(math.radians(roll)/2, dtype=torch.float64)
    q = torch.stack((angle.cos(), angle.sin(), angle*0, angle*0))[None]
    normal = q.new_tensor((0., -math.sqrt(.5), math.sqrt(.5)))
    assert float(plane_attitude_ratio(q, normal, 1.)) < 1e-10
    spin = .73
    # Postmultiply by a body-z rotation, which preserves the actual body plane.
    a, b, _, _ = q[0]
    turned = torch.stack((a*math.cos(spin), b*math.cos(spin), -b*math.sin(spin), a*math.sin(spin)))[None]
    assert float(plane_attitude_ratio(turned, normal, 1.)) < 1e-10


def test_plane_equivalence_does_not_flip_the_physical_thrust_law():
    angles = torch.tensor([45., -135.], dtype=torch.float64)*math.pi/360
    q = torch.stack((angles.cos(), angles.sin(), angles*0, angles*0), -1)
    z = quaternion_rotation(q)[:, :, 2]
    torch.testing.assert_close(z[0], -z[1], rtol=1e-14, atol=1e-14)
    assert z[0, 2] > 0 and z[1, 2] < 0


@pytest.mark.parametrize('velocity,accepted', [((0., 0., 1.5), True), ((0., 0., 3.), True),
                                             ((0., 0., -2.), False), ((2., 0., 0.), False)])
def test_velocity_direction_towards_stop_is_signed(velocity, accepted):
    from plane_tasks import directed_velocity_ratio

    v = torch.tensor([velocity], dtype=torch.float64, requires_grad=True)
    direction = v.new_tensor((0., 0., 1.))
    ratio = directed_velocity_ratio(v, direction, v.new_tensor([1.5]), 1.)
    assert bool((ratio <= 1).item()) is accepted
    gradient, = torch.autograd.grad(ratio.sum(), v)
    assert bool(torch.isfinite(gradient).all())


def test_zero_speed_and_orthogonal_plane_have_finite_useful_gradients():
    from plane_tasks import directed_velocity_ratio, plane_attitude_ratio

    v = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    ratio = directed_velocity_ratio(v, v.new_tensor((0., 0., 1.)), v.new_tensor([1.5]), 1.)
    grad, = torch.autograd.grad(ratio.sum(), v)
    assert bool(torch.isfinite(grad).all()) and grad[0, 2] < 0
    angle = torch.tensor(math.pi/4, dtype=v.dtype, requires_grad=True)
    q = torch.stack((angle.cos(), angle.sin(), angle*0, angle*0))[None]
    err = plane_attitude_ratio(q, v.new_tensor((0., 0., 1.)), 1.)
    g, = torch.autograd.grad(err.sum(), angle)
    assert torch.isfinite(g) and g != 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('departure_deg,accepted', [(0., True), (.5, True),
                                                 (1.5, False), (90., False),
                                                 (-.5, True), (-90., False)])
def test_velocity_is_perpendicular_to_body_z_not_parallel(dtype, departure_deg, accepted):
    from plane_tasks import velocity_z_orthogonality_ratio

    angle = math.radians(departure_deg)
    q = torch.tensor([[1., 0., 0., 0.]], dtype=dtype, requires_grad=True)
    v = torch.tensor([[2*math.cos(angle), 0., 2*math.sin(angle)]],
                     dtype=dtype, requires_grad=True)
    ratio = velocity_z_orthogonality_ratio(q, v, v.new_tensor([1.5]), 1.)
    assert bool((ratio <= 1).item()) is accepted
    gradients = torch.autograd.grad(ratio.sum(), (q, v))
    assert all(bool(torch.isfinite(g).all()) for g in gradients)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_vertical_and_nearly_vertical_headings_have_valid_perpendicular_targets(dtype):
    from plane_tasks import perpendicular_body_normal

    heading = torch.tensor(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                            (0., 0., -1.), (1e-20, 0., 1.),
                            (.2, -.4, -.8)), dtype=dtype)
    heading = torch.nn.functional.normalize(heading, dim=-1)
    normal = perpendicular_body_normal(heading)
    assert bool(torch.isfinite(normal).all())
    assert bool((normal[:, 2] >= 0).all())
    torch.testing.assert_close(normal.norm(dim=-1), torch.ones(6, dtype=dtype),
                               rtol=0, atol=2e-7)
    torch.testing.assert_close((normal*heading).sum(-1), torch.zeros(6, dtype=dtype),
                               rtol=0, atol=2e-7)


@pytest.mark.parametrize('roll', [0., 180.])
@pytest.mark.parametrize('velocity', [1.49, 1.5, 3., -2.])
def test_plane_loss_frees_normal_sign_but_keeps_speed_floor_and_direction(roll, velocity):
    p = torch.tensor([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]], dtype=torch.float64)
    goal = GoalConfig(relative_to_boundary=False,
                      points=((-1., 0., 0.), (0., 0., 0.), (1., 0., 0.)),
                      via_quaternion=(1., 0., 0., 0.),
                      velocities=((0., 0., 0.), (1.5, 0., 0.), (0., 0., 0.)))
    cfg = PassStopLossConfig(.01, .01, 25., mission=asdict(goal))
    half = math.radians(roll)/2
    q = p.new_tensor((math.cos(half), math.sin(half), 0., 0.)).expand(3, 1, 4)
    v = p.new_tensor((velocity, 0., 0.)).expand(3, 1, 3)
    trace, _ = synthetic(p, q=q, v=v, config=cfg)
    cfg = replace(cfg, mission=asdict(replace(goal, plane_task=True, random_via=True)))
    _, _, parts = typed_costs(trace, cfg)
    assert float(parts['pass_attitude']) == pytest.approx(0., abs=1e-20)
    assert (float(parts['pass_speed']) == 0) is (abs(velocity) >= 1.5)
    # Below the floor, direction is incomplete and cannot be made cheaper by braking.
    assert (float(parts['pass_direction']) == 0) is (velocity >= 1.5)
    assert float(parts['pass_orthogonality']) == pytest.approx(0., abs=1e-20)


def test_player_reports_plane_error_instead_of_full_attitude_error():
    import numpy as np
    from tools.play_response_short import goal_status

    half = math.radians(-135)/2
    q = np.array((math.cos(half), math.sin(half), 0., 0.))
    normal = np.array((0., -math.sqrt(.5), math.sqrt(.5)))
    rotation = quaternion_rotation(torch.tensor((math.cos(math.pi/8), math.sin(math.pi/8), 0., 0.))).numpy()
    arrays = {'valid':np.ones((1, 1), dtype=bool), 'command_is_stop':np.zeros((1, 1), dtype=bool),
              'command_phase':np.ones((1, 1), dtype=int), 'region_radius_m':np.array([.2]),
              'arm_length_m':np.array([.1]), 'velocity':(1.5*normal)[None, None],
              'orientation':q[None, None], 'pass_rotation':rotation[None],
              'mission_points':np.stack((-normal, np.zeros(3), normal))[None],
              'command_min_speed_m_s':np.array([[1.5]])}
    status = goal_status({'mission':{'plane_task':True, 'radius_m':.1}}, arrays, 0, 0)
    assert status['attitude_mode'] == 'unoriented-plane'
    assert status['plane_error_deg'] == pytest.approx(0., abs=1e-5)
    assert status['velocity_direction_error_deg'] == pytest.approx(0., abs=1e-5)
    assert status['velocity_plane_error_deg'] == pytest.approx(0., abs=1e-5)
    assert status['target_velocity_plane_angle_deg'] == pytest.approx(90., abs=1e-5)

    # A policy's incorrect current pose must not change the displayed target.
    arrays['orientation'][0, 0] = (1., 0., 0., 0.)
    status = goal_status({'mission':{'plane_task':True, 'radius_m':.1}}, arrays, 0, 0)
    assert status['velocity_plane_error_deg'] == pytest.approx(45., abs=1e-5)
    assert status['target_velocity_plane_angle_deg'] == pytest.approx(90., abs=1e-5)

    arrays['mission_points'][0, 2] = (1., 0., 0.)
    status = goal_status({'mission':{'plane_task':True, 'radius_m':.1}}, arrays, 0, 0)
    assert status['target_velocity_plane_angle_deg'] == pytest.approx(0., abs=1e-5)


def test_player_distinguishes_target_and_actual_velocity_body_z_angle():
    import numpy as np
    from tools.play_response_short import goal_status

    arrays = {'valid':np.ones((1, 1), dtype=bool), 'command_is_stop':np.zeros((1, 1), dtype=bool),
              'arm_length_m':np.array([.1]), 'velocity':np.array([[[2., 0., 0.]]]),
              'orientation':np.array([[[1., 0., 0., 0.]]]), 'pass_rotation':np.eye(3)[None],
              'mission_points':np.array([[[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]]])}
    meta = {'mission':{'plane_task':True, 'radius_m':.1},
            'metrics':{'pass_velocity_body_z_constraint':'orthogonal'}}
    status = goal_status(meta, arrays, 0, 0)
    assert status['target_velocity_z_angle_deg'] == pytest.approx(90.)
    assert status['velocity_z_error_deg'] == pytest.approx(0.)
    arrays['orientation'][0, 0] = (math.sqrt(.5), 0., math.sqrt(.5), 0.)
    status = goal_status(meta, arrays, 0, 0)
    assert status['target_velocity_z_angle_deg'] == pytest.approx(90.)
    assert status['velocity_z_error_deg'] == pytest.approx(90.)
    arrays['velocity'][0, 0] = 0.
    assert goal_status(meta, arrays, 0, 0)['velocity_z_error_deg'] is None


@pytest.mark.parametrize('body_pitch,passes', [(0., True), (90., False)])
def test_first_crossing_accepts_perpendicular_and_rejects_parallel_body_z(body_pitch, passes):
    p = torch.tensor([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]], dtype=torch.float64)
    goal = GoalConfig(relative_to_boundary=False,
                      points=((-1., 0., 0.), (0., 0., 0.), (1., 0., 0.)),
                      via_quaternion=(1., 0., 0., 0.),
                      velocities=((0., 0., 0.), (1.5, 0., 0.), (0., 0., 0.)))
    cfg = PassStopLossConfig(.01, .01, 25., mission=asdict(goal))
    angle = math.radians(body_pitch)/2
    q = p.new_tensor((math.cos(angle), 0., math.sin(angle), 0.)).expand(3, 1, 4)
    v = p.new_tensor((2., 0., 0.)).expand(3, 1, 3)
    trace, _ = synthetic(p, q=q, v=v, config=cfg)
    active = replace(goal, plane_task=True, random_via=True, plane_crossing_only=True)
    _, report, parts = typed_costs(trace, replace(cfg, mission=asdict(active)))
    assert (report['pass_fraction'] == 1.) is passes
    assert (parts['pass_orthogonality'].item() == 0.) is passes
    assert report['pass_velocity_body_z_constraint'] == 'orthogonal'


def test_velocity_must_be_orthogonal_to_actual_body_z_within_tolerance():
    p = torch.tensor([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]], dtype=torch.float64)
    goal = GoalConfig(relative_to_boundary=False,
                      points=((-1., 0., 0.), (0., 0., 0.), (1., 0., 0.)),
                      via_quaternion=(1., 0., 0., 0.),
                      velocities=((0., 0., 0.), (1.5, 0., 0.), (0., 0., 0.)))
    cfg = PassStopLossConfig(.01, .01, 25., mission=asdict(goal))
    a = math.radians(.75)
    q = p.new_tensor((math.cos(a/2), 0., math.sin(a/2), 0.)).expand(3, 1, 4)
    v = p.new_tensor((2*math.cos(a), 0., 2*math.sin(a))).expand(3, 1, 3)
    trace, _ = synthetic(p, q=q, v=v, config=cfg)
    cfg = replace(cfg, mission=asdict(replace(goal, plane_task=True, random_via=True)))
    parts = typed_costs(trace, cfg)[2]
    assert parts['pass_attitude'].item() == 0
    assert parts['pass_direction'].item() == 0
    assert parts['pass_orthogonality'].item() > 0


def test_plane_task_preserves_native_128_group_backward_and_replay_fence():
    import numpy as np
    from response_adjoints import collect_rollout, backward_actor
    from response_groups import GroupBalanceConfig
    from tools.short_eval_replay import capture_short_eval

    raw, _ = sample_coverage(128, [31000007, 31000008, 31000009, 31000010], dtype=torch.float64)
    actor = GoalMotorPolicy().double()
    cfg = plane_config()
    record = collect_rollout(actor, RaptorSimulator(), raw, cfg, horizon=3, time_decay=0,
                             group_config=GroupBalanceConfig(layout='coverage128', max_groups=128, min_scenarios=1))
    result = backward_actor(actor, RaptorSimulator(), record, cfg)
    assert result['group_gradient']['graph_traversals'] == 1
    assert result['group_gradient']['group_count'] == 128
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
    assert any(bool(p.grad.abs().any()) for p in actor.parameters())
    report, arrays = capture_short_eval(actor, RaptorSimulator(), raw, 3, cfg)
    np.testing.assert_array_equal(arrays['position_limit'], (50*raw.arm_length).numpy())
    assert bool((arrays['command_min_speed_m_s'][~arrays['command_is_stop']] == 1.5).all())
    assert report['pass_attitude_mode'] == 'unoriented-plane'


def test_first_plane_intersection_does_not_constrain_pose_or_direction_elsewhere_in_ball():
    p = torch.tensor([[-1., 0., 0.], [-.02, 0., 0.], [0., 0., 0.],
                      [.02, 0., 0.], [1., 0., 0.]], dtype=torch.float64)
    goal = GoalConfig(relative_to_boundary=False,
                      points=((-1., 0., 0.), (0., 0., 0.), (1., 0., 0.)),
                      via_quaternion=(1., 0., 0., 0.),
                      velocities=((0., 0., 0.), (1.5, 0., 0.), (0., 0., 0.)))
    cfg = PassStopLossConfig(.01, .01, 25., mission=asdict(goal))
    q = p.new_tensor((math.sqrt(.5), 0., math.sqrt(.5), 0.)).expand(5, 1, 4).clone()
    q[2, 0] = p.new_tensor((1., 0., 0., 0.))
    v = p.new_tensor((0., 0., 2.)).expand(5, 1, 3).clone()
    v[2, 0] = p.new_tensor((2., 0., 0.))
    trace, _ = synthetic(p, q=q, v=v, config=cfg)
    continuous = replace(goal, plane_task=True, random_via=True)
    strict = typed_costs(trace, replace(cfg, mission=asdict(continuous)))[2]
    assert strict['pass_attitude'].item() > 0 and strict['pass_direction'].item() > 0
    crossing = replace(continuous, plane_crossing_only=True)
    _, report, parts = typed_costs(trace, replace(cfg, mission=asdict(crossing)))
    for name in ('pass_position', 'pass_attitude', 'pass_speed', 'pass_direction', 'pass_orthogonality'):
        assert parts[name].item() == pytest.approx(0., abs=1e-20)
    assert report['plane_crossing_fraction'] == 1.


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_plane_intersection_fraction_keeps_position_gradients(dtype):
    from plane_tasks import segment_plane_geometry

    a = torch.tensor([[0., 0., -1.]], dtype=dtype, requires_grad=True)
    b = torch.tensor([[0., 0., 1.]], dtype=dtype, requires_grad=True)
    hit, fraction, forward = segment_plane_geometry(a, b, torch.zeros_like(a),
        a.new_tensor([[0., 0., 1.]]), a.new_tensor([.1]))
    assert hit.item() and forward.item()
    torch.testing.assert_close(fraction, fraction.new_tensor([.5]), rtol=0, atol=0)
    ga, gb = torch.autograd.grad(fraction.sum(), (a, b))
    torch.testing.assert_close(ga, a.new_tensor([[0., 0., -.25]]), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(gb, a.new_tensor([[0., 0., -.25]]), rtol=1e-6, atol=1e-7)


def test_missing_plane_crossing_and_backward_first_crossing_cannot_pass():
    goal = GoalConfig(relative_to_boundary=False,
                      points=((-1., 0., 0.), (0., 0., 0.), (1., 0., 0.)),
                      via_quaternion=(1., 0., 0., 0.),
                      velocities=((0., 0., 0.), (1.5, 0., 0.), (0., 0., 0.)))
    cfg = PassStopLossConfig(.01, .01, 25., mission=asdict(goal))
    # It visits the sphere close to its centre but never crosses the x=0 gate.
    p = torch.tensor([[-.01, -1., 0.], [-.01, 0., 0.], [-.01, 1., 0.]], dtype=torch.float64)
    v = p.new_tensor((2., 0., 0.)).expand(3, 1, 3)
    trace, _ = synthetic(p, q=p.new_tensor((1., 0., 0., 0.)).expand(3, 1, 4), v=v, config=cfg)
    event = replace(goal, plane_task=True, random_via=True, plane_crossing_only=True)
    active = replace(cfg, mission=asdict(event))
    _, report, parts = typed_costs(trace, active)
    assert report['plane_crossing_fraction'] == 0 and report['pass_fraction'] == 0
    assert parts['pass_missing'].item() > 0
    # First crossing is backwards; a later forward crossing cannot erase it.
    p = p.new_tensor([[1., 0., 0.], [.02, 0., 0.], [-.02, 0., 0.],
                      [.02, 0., 0.], [1., 0., 0.]])
    v = p.new_tensor([[-2., 0., 0.], [-2., 0., 0.], [-2., 0., 0.],
                      [3., 0., 0.], [3., 0., 0.]])[:, None]
    trace, _ = synthetic(p, q=p.new_tensor((1., 0., 0., 0.)).expand(5, 1, 4), v=v, config=cfg)
    _, report, parts = typed_costs(trace, active)
    assert report['plane_crossing_fraction'] == 1 and report['pass_fraction'] == 0
    assert parts['pass_direction'].item() > 0

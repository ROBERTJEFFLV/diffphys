from dataclasses import fields, replace
import inspect
import math

import pytest
import torch

from diffphys.agile_mission import (
    AgileMissionConfig, GRAVITY, HOVER_EQUILIBRIUM_LIMIT, MID_SPEED_COEFFICIENT,
    _actuators, _screen_reference, flatness_attitude, reference_arc_length, reference_kinematics,
    rk4_motor_setpoint, sample_agile_mission, smootherstep9,
)
from diffphys.env_raptor import RaptorParams, RaptorSimulator
from diffphys.response_noise import DisturbanceConfig


def clean_bank(count=8, seed=314, horizon=500):
    return RaptorSimulator().reset(count, seed=seed, dtype=torch.float64,
                                   horizon=horizon, disturbances=DisturbanceConfig.clean())


def test_s9_derivatives_and_three_point_boundary_values():
    x = torch.tensor((0., .19, .5, .81, 1.), dtype=torch.float64, requires_grad=True)
    derivatives = smootherstep9(x)
    # Differentiate the polynomial directly, independently of the symmetric
    # implementation's derivative formula and its midpoint branch.
    value = 126*x**5-420*x**6+540*x**7-315*x**8+70*x**9
    for order in range(1, 5):
        value, = torch.autograd.grad(value.sum(), x, create_graph=True)
        torch.testing.assert_close(derivatives[order], value, atol=2e-10, rtol=1e-11)
        assert torch.equal(derivatives[order][[0, -1]], torch.zeros(2, dtype=x.dtype))
    center = torch.tensor(((.1, -.2, .3),), dtype=x.dtype)
    span = torch.tensor((.7,), dtype=x.dtype)
    bend = torch.tensor(((0., .2, .1),), dtype=x.dtype)
    direction = torch.tensor(((1., 0., 0.),), dtype=x.dtype)
    duration = torch.tensor((2.,), dtype=x.dtype)
    times = torch.tensor(((0.,), (1.,), (2.,), (3.,)), dtype=x.dtype)
    p, v, a, jerk, snap = reference_kinematics(center, span, bend, direction, duration, times)
    torch.testing.assert_close(p[0], center-span[:, None]*direction)
    torch.testing.assert_close(p[1], center+bend)
    torch.testing.assert_close(p[2], center+span[:, None]*direction)
    assert torch.equal(p[2], p[3])
    expected_v = MID_SPEED_COEFFICIENT*span/duration
    torch.testing.assert_close(v[1], expected_v[:, None]*direction)
    torch.testing.assert_close(a[1], -2*(expected_v/span).square()[:, None]*bend)
    for derivative in (v, a, jerk, snap):
        assert torch.equal(derivative[[0, 2, 3]], torch.zeros_like(derivative[[0, 2, 3]]))


def test_flatness_angular_velocity_and_acceleration_match_time_derivatives():
    center = torch.zeros(1, 3, dtype=torch.float64)
    span = torch.tensor((.2,), dtype=torch.float64)
    bend = torch.tensor(((0., .07, .03),), dtype=torch.float64)
    direction = torch.tensor(((math.cos(.4), math.sin(.4), 0.),), dtype=torch.float64)
    # Make the displacement exactly transverse to direction.
    bend[:, :2] = .07*torch.tensor((-math.sin(.4), math.cos(.4)), dtype=torch.float64)
    duration = torch.tensor((1.4,), dtype=torch.float64)
    def frame(t):
        _, _, a, j, snap = reference_kinematics(center, span, bend, direction, duration, t)
        return flatness_attitude(a, j, snap, direction)
    t = torch.tensor((.63,), dtype=torch.float64)
    rotation, omega, omega_dot, valid = frame(t)
    eps = 1e-5
    plus, wp, _, _ = frame(t+eps)
    minus, wm, _, _ = frame(t-eps)
    rd = (plus-minus)/(2*eps)
    skew = rotation.transpose(-1, -2) @ rd
    measured = torch.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), -1)
    assert bool(valid.all())
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation,
                               torch.eye(3, dtype=rotation.dtype)[None], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(omega, measured, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(omega_dot, (wp-wm)/(2*eps), atol=1e-6, rtol=1e-6)


def test_exact_arc_length_matches_spatial_quadrature_and_degenerate_limits():
    # Integrate the spatial tangent independently of both the closed form and
    # its S9 timing. Include a straight line, a point, and zero forward span.
    span = torch.tensor((.7, .7, 0., 0., 1e-7), dtype=torch.float64)
    bend = torch.zeros(5, 3, dtype=torch.float64)
    bend[:, 1] = bend.new_tensor((.2, 0., 0., .3, .2))
    q = torch.linspace(0, 1, 20_001, dtype=torch.float64)
    forward = span[:, None]*span.new_tensor((1., 0., 0.))
    tangent = 2*forward[None] + 4*(1-2*q)[:, None, None]*bend[None]
    numerical = torch.trapezoid(tangent.norm(dim=-1), q, dim=0)
    actual = reference_arc_length(span, bend)
    torch.testing.assert_close(actual, numerical, atol=2e-9, rtol=1e-8)
    assert actual[1] == 1.4
    assert actual[2] == 0.
    assert actual[3] == .6


def test_full_geometry_is_retained_until_the_first_passing_duration(monkeypatch):
    import diffphys.agile_mission as module

    original = clean_bank(1, horizon=800)
    config = AgileMissionConfig()
    native_screen = module._screen_reference
    attempts = []

    def record_trial(actuators, center, span, bend, direction, duration, bound, config):
        ok, reason = native_screen(actuators, center, span, bend, direction,
                                   duration, bound, config)
        attempts.append((duration.clone(), span.clone(), bend.clone(), ok.clone(), reason.clone()))
        return ok, reason

    monkeypatch.setattr(module, "_screen_reference", record_trial)
    _, mission = sample_agile_mission(original, config, seed=44, horizon=800)
    assert len(attempts) > 1  # This aircraft cannot fly the first candidate.
    assert bool(mission.feasible.all())
    assert torch.equal(mission.reference_scale, torch.ones_like(mission.reference_scale))
    assert mission.report["reference_capped_count"] == 0
    assert mission.report["time_stretched_count"] == 1
    assert all(not bool(ok.any()) for _, _, _, ok, _ in attempts[:-1])
    assert bool(attempts[-1][3].all())
    assert bool((attempts[-2][4] != 0).all())
    assert bool((attempts[-2][0] < mission.duration).all())
    torch.testing.assert_close(attempts[-1][0], mission.duration, rtol=0, atol=0)
    assert all(bool((a[0] < b[0]).all()) for a, b in zip(attempts, attempts[1:]))
    for _, span, bend, _, _ in attempts:
        torch.testing.assert_close(span, config.half_span_fraction*mission.position_scale)
        torch.testing.assert_close(bend, mission.bend, rtol=0, atol=0)
    # The chosen path is independently rechecked by the unwrapped physical
    # actuator audit, rather than accepting the sampler's feasibility flag.
    ok, _ = native_screen(_actuators(original), mission.center, mission.half_span,
                          mission.bend, mission.direction, mission.duration,
                          mission.position_scale, config)
    assert bool(ok.all())


def test_rk4_motor_inverse_matches_original_physics_for_rising_and_falling():
    original = clean_bank(4)
    now = torch.full_like(original.motor, .45)
    following = now+now.new_tensor((.01, -.01, .005, -.005))
    setpoint = rk4_motor_setpoint(now, following, original.motor_time_rising,
                                 original.motor_time_falling)
    state = replace(original, motor=now)
    advanced = RaptorSimulator(RaptorParams()).step(state, 2*setpoint-1)
    torch.testing.assert_close(advanced.motor, following, atol=2e-15, rtol=2e-15)


def test_sampler_preserves_airframes_tapes_rng_and_recomputes_actual_targets():
    original = clean_bank(12)
    config = AgileMissionConfig()
    rng = torch.random.get_rng_state().clone()
    initial, mission = sample_agile_mission(original, config, seed=91, horizon=500)
    repeated, duplicate = sample_agile_mission(original, config, seed=91, horizon=500)
    assert torch.equal(rng, torch.random.get_rng_state())
    changed = {"position", "velocity", "omega", "orientation", "motor", "previous_action", "previous_velocity"}
    for field in fields(original):
        if field.name not in changed:
            assert getattr(initial, field.name) is getattr(original, field.name), field.name
    for field in fields(mission):
        value = getattr(mission, field.name)
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, getattr(duplicate, field.name)), field.name
    for field in fields(initial):
        assert torch.equal(getattr(initial, field.name), getattr(repeated, field.name)), field.name
    assert len(initial.mass) == len(original.mass)
    assert mission.report["aircraft_resampled"] == 0
    assert bool((mission.duration+mission.hold_seconds <= 5+1e-12).all())
    assert bool((mission.duration < 4.).any())
    assert not bool(mission.is_hover.any())
    assert bool((mission.reference_scale <= 1.).all())
    assert set(mission.reference_scale.tolist()) <= set(config.shrink_factors)
    torch.testing.assert_close(mission.half_span,
                               mission.reference_scale*config.half_span_fraction*mission.position_scale)
    assert "difficulty" not in inspect.signature(sample_agile_mission).parameters
    assert not any("difficulty" in field.name for field in fields(mission))
    assert not any("difficulty" in key for key in mission.report)
    assert bool(mission.feasible.any())
    mid = mission.nominal_reference(mission.duration/2)
    moving = ~mission.is_hover
    torch.testing.assert_close(mid.position, mission.waypoint, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(mid.velocity, mission.velocity, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(mid.rotation[moving], mission.rotation[moving], atol=1e-12, rtol=1e-12)
    # Use far more samples than the actuator audit for an independent geometry
    # check; containment is guaranteed by the Bezier control hull, not the mesh.
    curve = mission.nominal_reference(torch.linspace(0, 5, 1001, dtype=torch.float64)[:, None])
    assert bool((curve.position.abs() <= (1-config.boundary_margin_fraction)*mission.position_scale[None, :, None]+1e-12).all())
    assert torch.equal(initial.previous_action, torch.zeros_like(initial.previous_action))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_initial_velocity_preserves_draws_fills_zero_rows_and_keeps_sensor_tapes(dtype):
    original = RaptorSimulator().reset(4, seed=314, dtype=dtype, horizon=800,
                                       disturbances=DisturbanceConfig(force_enabled=False))
    velocity = original.velocity.clone()
    velocity[[0, 2]] = 0.
    original = replace(original, velocity=velocity)
    rng = torch.random.get_rng_state().clone()
    initial, mission = sample_agile_mission(original, AgileMissionConfig(), seed=91, horizon=800)
    repeated, _ = sample_agile_mission(original, AgileMissionConfig(), seed=91, horizon=800)
    # Geometry choices and feasibility backoff cannot consume the velocity
    # stream or silently replace already sampled nonzero kinematics.
    turned, _ = sample_agile_mission(original, AgileMissionConfig(yaw_degrees=0.),
                                     seed=91, horizon=800)
    redrawn, _ = sample_agile_mission(original, AgileMissionConfig(), seed=92, horizon=800)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(initial.velocity, repeated.velocity)
    assert torch.equal(initial.velocity, turned.velocity)
    assert torch.equal(initial.velocity[[1, 3]], original.velocity[[1, 3]])
    assert torch.equal(redrawn.velocity[[1, 3]], original.velocity[[1, 3]])
    assert not torch.equal(initial.velocity[[0, 2]], redrawn.velocity[[0, 2]])
    assert bool((initial.velocity.norm(dim=-1) > 0).all())
    assert bool((initial.velocity[[0, 2]].abs() <= 2.5).all())
    assert initial.velocity.dtype == dtype
    assert mission.report["initial_zero_velocity_replaced_count"] == 2
    assert mission.report["initial_speed_m_s"]["min"] > 0
    expected_history = initial.velocity[:, None, :].expand_as(initial.previous_velocity)
    assert torch.equal(initial.previous_velocity, expected_history)
    assert bool((original.noise_tape != 0).any())
    for name in ("noise_tape", "rotation_tape", "pulse_tape", "pulse_active_tape",
                 "noise_row", "noise_std", "velocity_delay", "guidance"):
        assert getattr(initial, name) is getattr(original, name), name
    assert torch.equal(initial.previous_action, torch.zeros_like(initial.previous_action))
    # Added speed is a physical initial tracking error; the first reference
    # point is still the same stationary endpoint, with no fourth waypoint.
    start = mission.nominal_reference(0.)
    torch.testing.assert_close(start.position, initial.position)
    assert torch.equal(start.velocity, torch.zeros_like(start.velocity))


def test_constant_force_hover_is_a_true_source_dynamics_equilibrium():
    original = clean_bank(4)
    force = original.mass[:, None]*original.velocity.new_tensor((.6, -.4, .3))
    original = replace(original, external_force=force)
    initial, mission = sample_agile_mission(original, AgileMissionConfig(hover_fraction=1.),
                                           seed=4, horizon=500)
    assert bool(mission.feasible.all())
    assert bool(mission.is_hover.all())
    held = RaptorSimulator().step(initial, 2*initial.motor-1)
    torch.testing.assert_close(held.position, initial.position, atol=1e-14, rtol=0.)
    torch.testing.assert_close(held.velocity, initial.velocity, atol=1e-14, rtol=0.)
    torch.testing.assert_close(held.omega, initial.omega, atol=1e-12, rtol=0.)
    torch.testing.assert_close(held.rotation, mission.hover_rotation, atol=1e-12, rtol=0.)
    torch.testing.assert_close(held.motor, initial.motor, atol=1e-14, rtol=0.)


def test_impossible_hover_is_flagged_and_kept_not_resampled():
    original = clean_bank(3)
    original = replace(original, external_force=original.mass[:, None]*original.velocity.new_tensor((100*GRAVITY, 0., 0.)))
    initial, mission = sample_agile_mission(original, AgileMissionConfig(hover_fraction=1.),
                                           seed=4, horizon=500)
    assert not bool(mission.feasible.any())
    assert bool(((mission.limit_reason & HOVER_EQUILIBRIUM_LIMIT) != 0).all())
    assert len(initial.mass) == 3
    assert initial.external_force is original.external_force
    assert mission.report["infeasible_count"] == 3
    assert bool(torch.isfinite(initial.motor).all())
    assert bool((initial.motor >= original.motor_min[:, None]).all())
    assert bool((initial.motor <= original.motor_max[:, None]).all())


@pytest.mark.parametrize("fallback", (False, True))
def test_hover_and_fallback_public_goals_do_not_encode_hidden_constant_force(fallback):
    original = clean_bank(3)
    if fallback:
        original = replace(original, inertia=original.inertia*1e10)
    disturbed = replace(original, external_force=original.mass[:, None]*original.velocity.new_tensor((.6, -.4, .3)))
    config = AgileMissionConfig(hover_fraction=0. if fallback else 1.)
    _, clean = sample_agile_mission(original, config, seed=3, horizon=500)
    _, changed = sample_agile_mission(disturbed, config, seed=3, horizon=500)
    for name in ("start", "waypoint", "goal", "rotation", "velocity", "direction", "duration",
                 "half_span", "center", "bend"):
        assert torch.equal(getattr(clean, name), getattr(changed, name)), name
    assert not torch.equal(clean.hover_rotation, changed.hover_rotation)
    torch.testing.assert_close(changed.rotation[:, 2, :],
                               changed.rotation.new_tensor((0., 0., 1.)).expand(3, 3),
                               atol=1e-12, rtol=1e-12)


def test_motor_torque_limits_cap_reference_without_hiding_airframes():
    original = clean_bank(2)
    original = replace(original, inertia=original.inertia*1e10)
    initial, mission = sample_agile_mission(original, AgileMissionConfig(hover_fraction=0.),
                                           seed=8, horizon=500)
    assert len(initial.mass) == 2
    assert initial.inertia is original.inertia
    assert not bool(mission.feasible.any())
    assert not bool(mission.is_hover.any())  # Still counted as failed attempted agile tasks.
    assert bool((mission.reference_scale == 0).all())
    assert mission.report["fallback_hover_count"] == 2
    assert mission.report["reference_capped_count"] == 2
    assert mission.report["initial_rejection_counts"]["individual_rotor_thrust"] == 2


def test_reported_feasible_motion_passes_the_same_full_actuator_grid():
    original = clean_bank(6)
    config = AgileMissionConfig(hover_fraction=0.)
    _, mission = sample_agile_mission(original, config, seed=44, horizon=500)
    ok, _ = _screen_reference(_actuators(original), mission.center, mission.half_span,
                              mission.bend, mission.direction, mission.duration,
                              mission.position_scale, config)
    assert bool(ok[mission.feasible].all())
    assert bool(mission.feasible.any())


@pytest.mark.parametrize("kwargs", (
    {"tilt_degrees": 90.}, {"thrust_ratio": 0.}, {"boundary_margin_fraction": 1.},
    {"hold_seconds": 0.}, {"hover_fraction": -1.}, {"shrink_factors": (1., .5)},
    {"center_fraction": (0., 0., .9)}, {"screen_batch_size": 0},
    {"duration_search_factor": 1.}, {"duration_search_factor": 2.01},
    {"duration_search_factor": float("nan")},
))
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        AgileMissionConfig(**kwargs)


def test_short_horizon_rejected_and_selection_keeps_metadata_types():
    original = clean_bank(3)
    with pytest.raises(ValueError, match="horizon"):
        sample_agile_mission(original, AgileMissionConfig(), seed=1, horizon=100)
    _, mission = sample_agile_mission(original, AgileMissionConfig(hover_fraction=1.), seed=1,
                                      horizon=500)
    selected = mission.select(torch.tensor((2, 0))).to("cpu", torch.float32)
    assert selected.start.dtype == torch.float32
    assert selected.feasible.dtype == torch.bool
    assert selected.limit_reason.dtype == torch.int64
    assert len(selected.start) == 2

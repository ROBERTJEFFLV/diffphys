"""Continuous clock-based targets, complete-row costs and physical gradients."""
from dataclasses import fields, replace
import inspect
import math

import pytest
import torch

import diffphys.agile_objective as objective
from diffphys.agile_mission import AgileMission
from diffphys.agile_objective import (
    AGILE_STEP_COMPONENTS, AGILE_TERMINAL_COMPONENTS, AgileLossConfig,
    agile_metrics, agile_step, agile_terminal, make_goal_packet,
    prepare_agile_schedule,
)
from diffphys.env_raptor import RaptorSimulator
from diffphys.response_noise import DisturbanceConfig


DT = .01
HORIZON = 100
DTYPE = torch.float64


def scene(count=1, *, bend=(0., .03, -.01), durations=None, dtype=DTYPE):
    """Small exact geometric missions, avoiding an actuator screen in loss tests."""
    physical = RaptorSimulator().reset(
        count, seed=13, horizon=HORIZON, dtype=dtype,
        disturbances=DisturbanceConfig.clean(),
    )
    center = torch.zeros(count, 3, dtype=dtype)
    direction = center.new_tensor((1., 0., 0.)).expand(count, 3).clone()
    half_span = torch.full((count,), .3, dtype=dtype)
    bend = center.new_tensor(bend).expand(count, 3).clone()
    rotation = torch.eye(3, dtype=dtype).expand(count, 3, 3).clone()
    duration = torch.full((count,), .8, dtype=dtype) if durations is None else center.new_tensor(durations)
    mission = AgileMission(
        start=center-half_span[:, None]*direction,
        waypoint=center+bend, goal=center+half_span[:, None]*direction,
        rotation=rotation.clone(), velocity=torch.zeros_like(center),
        direction=direction, duration=duration,
        position_scale=torch.ones(count, dtype=dtype),
        reference_scale=torch.ones(count, dtype=dtype),
        feasible=torch.ones(count, dtype=torch.bool),
        is_hover=torch.zeros(count, dtype=torch.bool), report={},
        half_span=half_span, hover_rotation=rotation, center=center, bend=bend,
        limit_reason=torch.zeros(count, dtype=torch.long),
        hover_feasible=torch.ones(count, dtype=torch.bool),
    )
    middle = mission.nominal_reference(duration/2.)
    mission = replace(mission, rotation=middle.rotation, velocity=middle.velocity)
    physical = replace(
        physical, position=mission.start.clone(), velocity=center.clone(),
        omega=center.clone(), orientation=center.new_tensor((1., 0., 0., 0.)).expand(count, 4).clone(),
        previous_action=torch.zeros(count, 4, dtype=dtype),
        position_limit=torch.ones(count, dtype=dtype),
        step_index=torch.zeros(count, dtype=torch.long),
    )
    return physical, mission


def step_cost(before, after, mission, *, step=40, active=None, config=AgileLossConfig(), schedule=None):
    if active is None:
        active = torch.ones(len(mission.start), dtype=torch.bool)
    return agile_step(before, after, active, mission, config, step=step,
                      horizon=HORIZON, dt=DT, schedule=schedule)


def test_schedule_is_detached_normalized_and_matches_each_midpoint():
    _, mission = scene(2, durations=(.8, .6))
    mission = replace(mission, center=mission.center.requires_grad_())
    schedule = prepare_agile_schedule(mission, AgileLossConfig(), horizon=HORIZON, dt=DT)
    assert schedule.position.shape == (HORIZON+1, 2, 3)
    assert schedule.rotation.shape == (HORIZON+1, 2, 3, 3)
    for field in fields(schedule):
        value = getattr(schedule, field.name)
        if isinstance(value, torch.Tensor):
            assert not value.requires_grad
            assert torch.isfinite(value).all()
    rows = torch.arange(2)
    midsteps = torch.tensor((40, 30))
    assert torch.equal(schedule.midpoint_weight.argmax(0)+1, midsteps)
    torch.testing.assert_close(schedule.midpoint_weight.sum(0), torch.ones(2, dtype=DTYPE))
    torch.testing.assert_close(schedule.position[midsteps, rows], mission.waypoint)
    torch.testing.assert_close(schedule.velocity[midsteps, rows], mission.velocity)
    torch.testing.assert_close(schedule.rotation[midsteps, rows], mission.rotation)
    torch.testing.assert_close(schedule.phase[midsteps, rows], torch.full((2,), .5, dtype=DTYPE))
    torch.testing.assert_close(schedule.position[-1], mission.goal)
    assert torch.equal(schedule.velocity[-1], torch.zeros_like(mission.velocity))
    assert torch.equal(schedule.phase[-1], torch.ones(2, dtype=DTYPE))
    assert not hasattr(schedule, "rest_weight")


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_motion_can_end_exactly_at_horizon_without_a_reserved_hold_interval(dtype):
    _, mission = scene(durations=(HORIZON*DT,), dtype=dtype)
    schedule = prepare_agile_schedule(mission, AgileLossConfig(), horizon=HORIZON, dt=DT)
    torch.testing.assert_close(schedule.position[HORIZON], mission.goal)
    assert torch.equal(schedule.velocity[HORIZON], torch.zeros_like(mission.velocity))
    assert torch.equal(schedule.phase[HORIZON], torch.ones(1, dtype=dtype))
    assert not any("hold" in field.name for field in fields(schedule))


def test_final_point_reference_and_goal_packet_continue_after_motion_and_horizon():
    _, mission = scene(2, durations=(.6, 1.))
    observation = torch.zeros(2, 22, dtype=DTYPE)
    for time in (1., 1.01, 3., 1234.):
        reference = mission.nominal_reference(time)
        torch.testing.assert_close(reference.position, mission.goal, rtol=0, atol=0)
        for value in (reference.velocity, reference.acceleration, reference.jerk,
                      reference.snap, reference.omega, reference.omega_dot):
            assert torch.equal(value, torch.zeros_like(value))
        packet = make_goal_packet(observation, mission, time, HORIZON*DT-time)
        assert packet.shape == (2, 23)
        for section in (slice(0, 3), slice(3, 6), slice(6, 9)):
            torch.testing.assert_close(packet[:, section], mission.goal, rtol=0, atol=0)
        torch.testing.assert_close(packet[:, 9:18], reference.rotation.flatten(1), rtol=0, atol=0)
        assert torch.equal(packet[:, 18:21], torch.zeros_like(mission.velocity))
        assert torch.equal(packet[:, 21], torch.ones(2, dtype=DTYPE))
        assert torch.equal(packet[:, 22], torch.zeros(2, dtype=DTYPE))


@pytest.mark.parametrize("scheduled", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_packet_switches_each_row_to_only_the_endpoint_using_time_not_arrival(scheduled, dtype):
    _, mission = scene(2, durations=(.6, 1.), dtype=dtype)
    config = AgileLossConfig()
    schedule = prepare_agile_schedule(mission, config, horizon=HORIZON, dt=DT) if scheduled else None
    observation = torch.zeros(2, 22, dtype=dtype)
    for step in (59, 60, 61, HORIZON):
        time = step*DT
        kwargs = {"schedule": schedule, "step": step} if scheduled else {}
        packet = make_goal_packet(observation, mission, time, HORIZON*DT-time, **kwargs)
        finished = time >= mission.duration
        for row in range(2):
            if finished[row]:
                for section in (slice(0, 3), slice(3, 6), slice(6, 9)):
                    torch.testing.assert_close(packet[row, section], mission.goal[row], rtol=0, atol=0)
                expected_rotation = mission.nominal_reference(time).rotation[row]
                torch.testing.assert_close(packet[row, 9:18], expected_rotation.flatten(), rtol=0, atol=0)
                assert torch.equal(packet[row, 18:21], torch.zeros(3, dtype=dtype))
            else:
                torch.testing.assert_close(packet[row, 3:6], mission.waypoint[row], rtol=0, atol=0)
                torch.testing.assert_close(packet[row, 9:18], mission.rotation[row].flatten(), rtol=0, atol=0)
                torch.testing.assert_close(packet[row, 18:21], mission.velocity[row], rtol=0, atol=0)
        # Changing observations cannot latch arrival or undo the clock target.
        altered = make_goal_packet(observation+1000, mission, time, HORIZON*DT-time, **kwargs)
        assert torch.equal(packet, altered)


@pytest.mark.parametrize("scheduled", (False, True))
def test_endpoint_packet_never_exposes_hidden_force_equilibrium(scheduled):
    _, mission = scene()
    observation = torch.zeros(1, 22, dtype=DTYPE)
    config = AgileLossConfig()
    schedule = prepare_agile_schedule(mission, config, horizon=HORIZON, dt=DT) if scheduled else None
    kwargs = {"schedule": schedule, "step": HORIZON} if scheduled else {}
    hidden_changed = replace(mission, hover_rotation=mission.hover_rotation+1000.)
    first = make_goal_packet(observation, mission, HORIZON*DT, 0., **kwargs)
    second = make_goal_packet(observation, hidden_changed, HORIZON*DT, 0., **kwargs)
    assert torch.equal(first, second)
    expected = torch.eye(3, dtype=DTYPE).flatten()[None]
    torch.testing.assert_close(first[:, 9:18], expected, rtol=0, atol=0)


def test_goal_packet_is_clock_only_and_independent_of_observation_or_truth():
    physical, mission = scene(2, durations=(.8, .6))
    schedule = prepare_agile_schedule(mission, AgileLossConfig(), horizon=HORIZON, dt=DT)
    measured = torch.zeros(2, 22, dtype=DTYPE)
    altered = torch.arange(44, dtype=DTYPE).reshape(2, 22)*100
    packet = make_goal_packet(measured, mission, .3, .7, schedule=schedule, step=30)
    other = make_goal_packet(altered, mission, .3, .7, schedule=schedule, step=30)
    direct = make_goal_packet(measured, mission, .3, .7)
    torch.testing.assert_close(packet, other, atol=0., rtol=0.)
    torch.testing.assert_close(packet, direct, atol=0., rtol=0.)
    assert packet.shape == (2, 23)
    torch.testing.assert_close(packet[:, :3], schedule.position[30])
    torch.testing.assert_close(packet[:, 3:6], mission.waypoint)
    torch.testing.assert_close(packet[:, 6:9], mission.goal)
    torch.testing.assert_close(packet[:, 9:18], mission.rotation.flatten(1))
    torch.testing.assert_close(packet[:, 18:21], mission.velocity)
    torch.testing.assert_close(packet[:, 21], torch.tensor((.375, .5), dtype=DTYPE))
    assert torch.equal(packet[:, 22], torch.full((2,), .7, dtype=DTYPE))
    # Crossing from either side, or a stale frozen simulator step index, cannot
    # latch a different task state: only the explicitly supplied clock is used.
    after = replace(physical, position=mission.waypoint+.01)
    left = replace(physical, position=mission.waypoint-.01)
    right = replace(physical, position=mission.waypoint+.01, step_index=physical.step_index+90)
    first, _ = step_cost(left, after, mission, step=30, schedule=schedule)
    second, _ = step_cost(right, after, mission, step=30, schedule=schedule)
    for name in AGILE_STEP_COMPONENTS:
        torch.testing.assert_close(first[name], second[name], atol=0., rtol=0.)
    later = make_goal_packet(altered, mission, .8, -.1, schedule=schedule, step=80)
    assert torch.equal(later[:, 21], torch.ones(2, dtype=DTYPE))
    assert torch.equal(later[:, 22], torch.zeros(2, dtype=DTYPE))
    assert not torch.equal(packet[:, :3], later[:, :3])


def test_midpoint_window_tracks_moving_reference_instead_of_braking_at_waypoint():
    physical, mission = scene()
    schedule = prepare_agile_schedule(mission, AgileLossConfig(), horizon=HORIZON, dt=DT)
    for step in (38, 42):
        after = replace(physical, position=schedule.position[step], velocity=schedule.velocity[step])
        components, _ = step_cost(physical, after, mission, step=step, schedule=schedule)
        assert (after.position-mission.waypoint).norm() > .01
        assert schedule.midpoint_weight[step-1].item() > 0.
        assert components["midpoint_position"].item() == 0.
        assert components["midpoint_velocity"].item() == 0.


@pytest.mark.parametrize("step", (40, 95))
def test_complete_loss_gradient_matches_finite_difference(step):
    before, mission = scene()
    config = AgileLossConfig()
    schedule = prepare_agile_schedule(mission, config, horizon=HORIZON, dt=DT)

    def loss(parameters):
        angle = parameters[6]
        zero = angle.new_zeros(())
        q = torch.stack((torch.cos(angle/2), zero, zero, torch.sin(angle/2)))[None]
        after = replace(before, position=parameters[:3][None], velocity=parameters[3:6][None],
                        orientation=q, omega=parameters[7:10][None],
                        previous_action=parameters[10:14][None])
        components, _ = step_cost(before, after, mission, step=step, config=config, schedule=schedule)
        return sum(x.sum() for x in components.values())+sum(
            x.sum() for x in agile_terminal(after, mission, config).values())

    x = torch.tensor((.93, .12, -.07, .4, -.2, .1, .23, .3, -.4, .2,
                      .1, -.15, .2, -.05), dtype=DTYPE, requires_grad=True)
    derivative, = torch.autograd.grad(loss(x), x)
    numerical = []
    for index in range(x.numel()):
        offset = torch.zeros_like(x)
        offset[index] = 1e-6
        numerical.append((loss(x.detach()+offset)-loss(x.detach()-offset))/(2e-6))
    assert torch.isfinite(derivative).all()
    assert bool((derivative.abs() > 0).all())
    torch.testing.assert_close(derivative, torch.stack(numerical), rtol=3e-5, atol=2e-8)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_identical_targets_have_zero_finite_gradient(dtype):
    before, mission = scene(bend=(0., 0., 0.), dtype=dtype)
    p = mission.goal.clone().requires_grad_()
    v = torch.zeros_like(p, requires_grad=True)
    q = before.orientation.clone().requires_grad_()
    omega = torch.zeros_like(p, requires_grad=True)
    command = before.previous_action.clone().requires_grad_()
    after = replace(before, position=p, velocity=v, orientation=q,
                    omega=omega, previous_action=command)
    components, _ = step_cost(before, after, mission, step=HORIZON)
    terminal = agile_terminal(after, mission, AgileLossConfig())
    loss = sum(x.sum() for x in (*components.values(), *terminal.values()))
    assert loss.item() == 0.
    derivatives = torch.autograd.grad(loss, (p, v, q, omega, command))
    for gradient in derivatives:
        assert torch.isfinite(gradient).all()
        assert torch.equal(gradient, torch.zeros_like(gradient))


def test_quaternion_sign_preserves_all_attitude_losses():
    before, mission = scene()
    q = torch.tensor(((math.cos(.2), math.sin(.2), 0., 0.),), dtype=DTYPE)
    first_state = replace(before, orientation=q)
    second_state = replace(before, orientation=-q)
    for step in (40, 95):
        first, _ = step_cost(before, first_state, mission, step=step)
        second, _ = step_cost(before, second_state, mission, step=step)
        torch.testing.assert_close(first["midpoint_attitude"], second["midpoint_attitude"], atol=0., rtol=0.)
    first = agile_terminal(first_state, mission, AgileLossConfig())
    second = agile_terminal(second_state, mission, AgileLossConfig())
    torch.testing.assert_close(first["final_attitude"], second["final_attitude"], atol=0., rtol=0.)


def test_terminal_speed_and_rate_have_gradients_without_arrival():
    before, mission = scene()
    velocity = torch.tensor(((.4, -.2, .1),), dtype=DTYPE, requires_grad=True)
    omega = torch.tensor(((.3, -.4, .2),), dtype=DTYPE, requires_grad=True)
    # Far from the end, with an inactive row, braking costs still apply.
    after = replace(before, position=mission.start, velocity=velocity, omega=omega)
    components, _ = step_cost(before, after, mission, step=95, active=torch.tensor((False,)))
    terminal = agile_terminal(after, mission, AgileLossConfig())
    for name, values, tensor in (("tracking_velocity", components, velocity),
                                 ("final_velocity", terminal, velocity),
                                 ("final_omega", terminal, omega)):
        assert values[name].item() > 0.
        gradient, = torch.autograd.grad(values[name].sum(), tensor, retain_graph=True)
        assert torch.isfinite(gradient).all()
        assert bool((gradient*tensor > 0).all())


def test_late_tracking_has_no_extra_hold_cost_or_terminal_weight_window():
    before, mission = scene()
    after = replace(before, position=mission.goal+.03,
                    velocity=torch.tensor(((.4, -.2, .1),), dtype=DTYPE),
                    omega=torch.tensor(((.3, -.4, .2),), dtype=DTYPE))
    config = AgileLossConfig()
    no_terminal = replace(config, terminal_position_weight=0., terminal_velocity_weight=0.,
                          terminal_attitude_weight=0., terminal_omega_weight=0., final_weight=0.)
    for step in (81, 90, HORIZON):
        components, metrics = step_cost(before, after, mission, step=step, config=config)
        other, _ = step_cost(before, after, mission, step=step, config=no_terminal)
        assert components["tracking_position"].item() > 0.
        assert components["tracking_velocity"].item() > 0.
        assert not any(name.startswith(("rest_", "final_", "hold_")) for name in components)
        assert not any(name.startswith(("rest_", "hold_")) for name in metrics)
        for name in components:
            torch.testing.assert_close(components[name], other[name], rtol=0, atol=0)
    assert agile_terminal(after, mission, config)["final_velocity"].item() > 0.
    assert all(value.item() == 0. for value in agile_terminal(after, mission, no_terminal).values())


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("speed", (0., .4))
def test_terminal_velocity_derivative_matches_closed_form_at_rest_and_in_motion(dtype, speed):
    physical, mission = scene(dtype=dtype)
    config = AgileLossConfig()
    velocity = torch.tensor(((speed, -.5*speed, .25*speed),), dtype=dtype, requires_grad=True)
    after = replace(physical, velocity=velocity)
    cost = agile_terminal(after, mission, config)["final_velocity"]
    gradient, = torch.autograd.grad(cost.sum(), velocity)
    scale = config.terminal_velocity_scale_m_s
    radius = torch.sqrt((velocity.detach()/scale).square().sum(-1, keepdim=True)
                        +config.smooth_epsilon**2)
    expected = config.final_weight*config.terminal_velocity_weight*velocity.detach()/(scale**2*radius)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, expected)
    if speed == 0.:
        assert cost.item() == 0.
        assert torch.equal(gradient, torch.zeros_like(gradient))
    else:
        assert cost.item() > 0.
        assert bool((gradient*velocity > 0).all())


@pytest.mark.parametrize("sign", (-1., 1.))
def test_boundary_warning_has_inward_gradient_before_original_hard_boundary(sign):
    before, mission = scene()
    p = torch.tensor(((sign*.95, 0., 0.),), dtype=DTYPE, requires_grad=True)
    after = replace(before, position=p)
    components, _ = step_cost(before, after, mission)
    gradient, = torch.autograd.grad(components["boundary"].sum(), p)
    assert not RaptorSimulator.terminated(after).item()
    assert components["boundary"].item() > 0.
    assert gradient[0, 0]*sign > 0.
    assert gradient[0, 1:].abs().sum() == 0.
    at_limit = replace(after, position=p.detach().new_tensor(((sign, 0., 0.),)))
    outside = replace(after, position=p.detach().new_tensor(((sign*1.001, 0., 0.),)))
    assert not RaptorSimulator.terminated(at_limit).item()
    assert RaptorSimulator.terminated(outside).item()


def test_frozen_failed_rows_retain_every_remaining_clock_cost_and_gradient():
    before, mission = scene()
    position = torch.tensor(((1.2, .2, -.1),), dtype=DTYPE, requires_grad=True)
    frozen = replace(before, position=position, step_index=before.step_index+10,
                     velocity=torch.tensor(((.5, 0., 0.),), dtype=DTYPE))
    assert RaptorSimulator.terminated(frozen).item()
    schedule = prepare_agile_schedule(mission, AgileLossConfig(), horizon=HORIZON, dt=DT)
    remaining = []
    for step in range(11, HORIZON+1):
        failed, _ = step_cost(frozen, frozen, mission, step=step,
                              active=torch.tensor((False,)), schedule=schedule)
        active, _ = step_cost(frozen, frozen, mission, step=step,
                              active=torch.tensor((True,)), schedule=schedule)
        for name in AGILE_STEP_COMPONENTS:
            torch.testing.assert_close(failed[name], active[name], atol=0., rtol=0.)
        remaining.append(sum(value.sum() for value in failed.values()))
    final = sum(x.sum() for x in agile_terminal(frozen, mission, AgileLossConfig()).values())
    assert bool((torch.stack(remaining) > 0).all())
    total = sum(remaining)+final
    assert total > remaining[0]+final
    derivative, = torch.autograd.grad(total, position)
    assert torch.isfinite(derivative).all()
    assert derivative[0, 0] > 0.


def test_metrics_and_loss_keep_all_original_rows_including_hover_and_infeasible():
    physical, mission = scene(4)
    mission = replace(mission, feasible=torch.tensor((True, True, False, False)),
                      is_hover=torch.tensor((True, False, True, False)),
                      reference_scale=torch.tensor((.1, .2, 0., 0.), dtype=DTYPE))
    offset = torch.arange(4, dtype=DTYPE)[:, None]*torch.tensor(((1., 0., 0.),), dtype=DTYPE)
    position = (mission.goal+offset).requires_grad_()
    physical = replace(physical, position=position, velocity=offset, omega=2*offset)
    config = AgileLossConfig()
    final = agile_terminal(physical, mission, config)
    assert set(final) == set(AGILE_TERMINAL_COMPONENTS)
    assert all(value.shape == (4,) for value in final.values())
    derivative, = torch.autograd.grad(final["final_position"].mean(), position)
    assert bool((derivative[1:, 0] > 0).all())
    squares = torch.tensor((1., 4., 9., 16.), dtype=DTYPE)
    measurements = {name: squares for name in (
        "midpoint_position_sq", "midpoint_velocity_sq", "midpoint_attitude_sq",
        "reference_position_sq", "reference_velocity_sq",
    )}
    measurements["midpoint_speed"] = squares.sqrt()
    report = agile_metrics(physical, mission, config, measurements=measurements)
    assert report["scenario_count"] == 4
    assert report["final_position_rms_m"] == pytest.approx(math.sqrt(14/4))
    assert report["final_velocity_rms_m_s"] == pytest.approx(math.sqrt(14/4))
    assert report["final_omega_rms_rad_s"] == pytest.approx(2*math.sqrt(14/4))
    assert report["reference_feasible_fraction"] == .5
    assert report["boundary_failure_rate"] == .75
    assert report["reference_scale_mean"] == pytest.approx(.075)
    assert report["midpoint_position_rms_m"] == pytest.approx(math.sqrt(30/4))
    assert report["reference_velocity_rms_m_s"] == pytest.approx(math.sqrt(30/4))
    assert report["midpoint_speed_mean_m_s"] == 2.5
    assert not any(word in key for key in report
                   for word in ("success", "passed", "dwell", "difficulty", "rest_", "hold_"))


def test_no_event_progress_or_success_tolerance_in_continuous_objective_api():
    assert not hasattr(objective, "initial_progress")
    assert not hasattr(objective, "AgileProgress")
    assert "progress" not in inspect.signature(agile_step).parameters
    assert "next_observation" not in inspect.signature(agile_step).parameters
    assert not any(word in field.name for field in fields(AgileLossConfig)
                   for word in ("tolerance", "success", "crossing", "dwell", "hold", "rest_"))


@pytest.mark.parametrize("kwargs", (
    {"boundary_weight": -1.}, {"tracking_position_weight": True},
    {"smooth_epsilon": 0.}, {"terminal_velocity_scale_m_s": "1"},
    {"midpoint_window_fraction": 1.}, {"terminal_attitude_scale_rad": math.pi},
    {"final_weight": float("nan")}, {"terminal_omega_scale_rad_s": float("inf")},
))
def test_invalid_continuous_loss_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        AgileLossConfig(**kwargs)


def test_incompatible_time_schedule_and_invalid_packet_dimensions_are_rejected():
    physical, mission = scene()
    config = AgileLossConfig()
    schedule = prepare_agile_schedule(mission, config, horizon=HORIZON, dt=DT)
    with pytest.raises(ValueError, match="mission motion"):
        prepare_agile_schedule(replace(mission, duration=mission.duration+1.),
                               config, horizon=HORIZON, dt=DT)
    with pytest.raises(ValueError, match="horizon/dt mismatch"):
        step_cost(physical, physical, mission, schedule=replace(schedule, dt=DT*2))
    with pytest.raises(ValueError, match="valid step"):
        make_goal_packet(torch.zeros(1, 22, dtype=DTYPE), mission, .1, .9, schedule=schedule)
    with pytest.raises(ValueError, match="observation"):
        make_goal_packet(torch.zeros(1, 21, dtype=DTYPE), mission, .1, .9)

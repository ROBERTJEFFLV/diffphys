from __future__ import annotations

import torch

from env_l2f import L2FParams, L2FSimulator
from full_space_shooting import so3_exp
from full_space_shooting import FullSpaceProblem, solve_joint_sqp_step
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy
from structured_stability import default_phase_space_metric
from structured_rollout import (
    LEGACY_BOUNDARY_CODEC_VERSION,
    build_action_probe_bank,
    clone_closed_loop,
    functional_probe_actions,
    functional_trajectory_actions,
    functional_trajectory_diagnostics,
    ParameterVectorSpec,
    StructuredBoundaryCodec,
    StructuredClosedLoopState,
    initialize_exact_endpoints,
    make_segment_map,
    phase_space_contraction_risk_residual,
    rollout_structured_segment,
    structured_observation,
    terminal_risk_residual,
)


def _fixture():
    simulator = L2FSimulator(
        L2FParams(
            dt=0.01,
            max_initial_position=0.01,
            max_initial_velocity=0.01,
            max_initial_angle=0.01,
            max_initial_omega=0.01,
        )
    )
    physical = simulator.reset(2, device="cpu", dtype=torch.float64)
    policy = StructuredRecurrentPolicy(
        StructuredPolicyConfig(hidden_dim=6, identifier_dim=5, slow_cadence=2)
    ).double()
    observation = torch.cat(
        (
            physical.position,
            physical.velocity,
            physical.rotation.reshape(2, 9),
            physical.omega,
            torch.zeros(2, 3, dtype=torch.float64),
            physical.previous_action,
        ),
        dim=-1,
    )
    recurrent = policy.initial_state(observation)
    return simulator, policy, StructuredClosedLoopState(physical, recurrent)


def test_complete_boundary_round_trip_preserves_deployable_state() -> None:
    _, policy, closed = _fixture()
    codec = StructuredBoundaryCodec(closed.physical, policy)
    packed = codec.pack(closed)
    restored = codec.unpack(packed)
    torch.testing.assert_close(restored.physical.position, closed.physical.position)
    torch.testing.assert_close(restored.physical.rotation, closed.physical.rotation, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(restored.policy.hidden, closed.policy.hidden)
    torch.testing.assert_close(restored.policy.identifier, closed.policy.identifier)
    torch.testing.assert_close(restored.policy.motor_estimate, closed.policy.motor_estimate)
    torch.testing.assert_close(restored.policy.slow_body_z, closed.policy.slow_body_z)
    torch.testing.assert_close(restored.policy.context_sum, closed.policy.context_sum)
    for name in (
        "capability_log_mean", "capability_log_scale", "capability_ucb",
        "capability_ucb_target", "contextual_gain", "contextual_gain_target",
        "contextual_blend", "boot_progress",
    ):
        torch.testing.assert_close(getattr(restored.policy, name), getattr(closed.policy, name))
    assert codec.layout.rotation_slice.stop - codec.layout.rotation_slice.start == 3
    assert codec.slices["slow_body_z"].stop - codec.slices["slow_body_z"].start == 2


def test_boundary_codec_v2_preserves_cadence_latches_and_v1_is_explicitly_legacy() -> None:
    _, policy, closed = _fixture()
    closed.policy.disturbance_response_count.fill_(7.0)
    closed.policy.identification_failed.fill_(True)
    closed.policy.slow_counter = 19

    codec = StructuredBoundaryCodec(closed.physical, policy)
    restored = codec.unpack(codec.pack(closed))
    assert codec.codec_version == 2
    assert restored.policy.slow_counter == 19
    torch.testing.assert_close(
        restored.policy.disturbance_response_count,
        closed.policy.disturbance_response_count,
    )
    torch.testing.assert_close(
        restored.policy.identification_failed,
        closed.policy.identification_failed,
    )

    # Version 1 remains readable with its documented zero/phase semantics;
    # old boundaries are not silently interpreted as complete v2 state.
    legacy = StructuredBoundaryCodec(
        closed.physical, policy, codec_version=LEGACY_BOUNDARY_CODEC_VERSION
    )
    legacy_state = legacy.unpack(legacy.pack(closed))
    assert legacy.codec_version == LEGACY_BOUNDARY_CODEC_VERSION
    assert legacy_state.policy.slow_counter == 0
    assert legacy_state.policy.identification_failed is None
    torch.testing.assert_close(
        legacy_state.policy.disturbance_response_count,
        torch.zeros_like(closed.policy.disturbance_response_count),
    )


def test_boundary_pack_unpack_matches_uninterrupted_rollout_at_cadence_calls() -> None:
    torch.manual_seed(123)
    simulator = L2FSimulator(
        L2FParams(
            dt=0.01,
            max_initial_position=0.01,
            max_initial_velocity=0.01,
            max_initial_angle=0.01,
            max_initial_omega=0.01,
        )
    )
    physical = simulator.reset(1, device="cpu", dtype=torch.float64)
    policy = StructuredRecurrentPolicy(
        StructuredPolicyConfig(
            hidden_dim=4,
            identifier_dim=4,
            slow_cadence=25,
            motor_observer_bank_size=15,
            motor_observer_mode="fixed_multi_tau_v1",
            motor_tau_grid_version=1,
        )
    ).double()
    initial = StructuredClosedLoopState(
        physical,
        policy.initial_state(
            torch.cat(
                (
                    physical.position,
                    physical.velocity,
                    physical.rotation.reshape(1, 9),
                    physical.omega,
                    torch.zeros(1, 3, dtype=torch.float64),
                    physical.previous_action,
                ),
                dim=-1,
            )
        ),
    )
    # Exercise the fields which are otherwise zero at initialization.
    initial.policy.disturbance_response_count.fill_(3.0)
    initial.policy.disturbance_residual_sum.copy_(
        torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64)
    )
    initial.policy.disturbance_thrust_sum.copy_(
        torch.tensor([[0.4, 0.5, -0.2]], dtype=torch.float64)
    )
    initial.policy.identification_failed.fill_(True)
    codec = StructuredBoundaryCodec(initial.physical, policy)

    def run_one(state: StructuredClosedLoopState):
        end, trace = rollout_structured_segment(
            policy, simulator, state, steps=1, collect=True
        )
        return (
            end,
            trace["action"][0, 0].detach().clone(),
            trace["identification_published"][0, 0].detach().clone(),
            trace["identification_publication_available"][0, 0].detach().clone(),
        )

    reference_states = {}
    reference_actions = []
    reference_publications = []
    reference_availability = []
    reference = clone_closed_loop(initial)
    for call in range(1, 301):
        reference, action, published, available = run_one(reference)
        reference_actions.append(action)
        reference_publications.append(published)
        reference_availability.append(available)
        if call in (50, 75, 300):
            reference_states[call] = clone_closed_loop(reference)

    segmented = clone_closed_loop(initial)
    previous_call = 0
    for call in (50, 75, 300):
        actions = []
        segmented, trace = rollout_structured_segment(
            policy, simulator, segmented,
            steps=call - previous_call,
            collect=True,
        )
        actions.extend(trace["action"][:, 0])
        torch.testing.assert_close(
            torch.stack(actions),
            torch.stack(reference_actions[previous_call:call]),
            atol=2.0e-9,
            rtol=2.0e-9,
        )
        torch.testing.assert_close(
            trace["identification_published"][:, 0],
            torch.stack(reference_publications[previous_call:call]),
        )
        torch.testing.assert_close(
            trace["identification_publication_available"][:, 0],
            torch.stack(reference_availability[previous_call:call]),
        )
        segmented = codec.unpack(codec.pack(segmented))
        expected = reference_states[call]
        for name in ("position", "velocity", "rotation", "omega", "motor", "previous_action"):
            torch.testing.assert_close(
                getattr(segmented.physical, name),
                getattr(expected.physical, name),
                atol=2.0e-9,
                rtol=2.0e-9,
            )
        for name in (
            "hidden", "identifier", "motor_estimate", "integral", "motor_bank",
            "slow_trim", "slow_body_z", "capability", "prev_velocity", "prev_omega",
            "context_sum", "capability_log_mean", "capability_log_scale",
            "capability_ucb", "capability_ucb_target", "contextual_gain",
            "contextual_gain_target", "contextual_blend", "boot_progress",
            "disturbance_accel", "disturbance_residual_sum", "disturbance_thrust_sum",
            "previous_executed_action", "disturbance_response_count",
            "excitation_history", "previous_motor_estimate", "identification_failed",
        ):
            torch.testing.assert_close(
                getattr(segmented.policy, name),
                getattr(expected.policy, name),
                atol=2.0e-9,
                rtol=2.0e-9,
            )
        assert segmented.policy.slow_counter == expected.policy.slow_counter
        previous_call = call


def test_multi_tau_boundary_codec_round_trip_preserves_observer_bank() -> None:
    simulator, _, closed = _fixture()
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(
        hidden_dim=6, identifier_dim=5, slow_cadence=2,
        motor_observer_bank_size=15,
        motor_observer_mode="fixed_multi_tau_v1", motor_tau_grid_version=1,
    )).double()
    observation = structured_observation(closed)
    banked_closed = StructuredClosedLoopState(
        closed.physical, policy.initial_state(observation)
    )
    codec = StructuredBoundaryCodec(banked_closed.physical, policy)
    packed = codec.pack(banked_closed)
    restored = codec.unpack(packed)
    assert codec.slices["motor_bank"].stop - codec.slices["motor_bank"].start == 60
    torch.testing.assert_close(
        restored.policy.motor_bank, banked_closed.policy.motor_bank,
        atol=1e-10, rtol=1e-10,
    )


def test_parameter_vector_functional_segment_has_parameter_credit() -> None:
    simulator, policy, closed = _fixture()
    codec = StructuredBoundaryCodec(closed.physical, policy)
    spec = ParameterVectorSpec.from_module(policy)
    theta = spec.flatten(policy).detach().requires_grad_(True)
    segment = make_segment_map(policy, simulator, codec, spec, steps=2)
    end = segment(codec.pack(closed), theta)
    gradient = torch.autograd.grad(end.square().mean(), theta)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


def test_exact_endpoint_initialization_has_expected_shape() -> None:
    simulator, policy, closed = _fixture()
    codec = StructuredBoundaryCodec(closed.physical, policy)
    spec = ParameterVectorSpec.from_module(policy)
    theta = spec.flatten(policy).detach()
    segment = make_segment_map(policy, simulator, codec, spec, steps=2)
    endpoints = initialize_exact_endpoints(segment, codec.pack(closed), theta, 2)
    assert endpoints.shape == (2, 2, codec.state_dim)
    restored = codec.unpack(endpoints[-1])
    rotation = restored.physical.rotation
    identity = torch.eye(3, dtype=rotation.dtype).expand_as(rotation)
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-10, rtol=1e-10)


def test_collected_rollout_reports_allocator_diagnostics() -> None:
    simulator, policy, closed = _fixture()
    end, trace = rollout_structured_segment(policy, simulator, closed, steps=2, collect=True)
    assert trace["action"].shape == (2, 2, 4)
    assert trace["allocator_condition"].shape == (2, 2)
    assert trace["identification_failed"].shape == (2, 2)
    assert trace["effectiveness_log_interval_width"].shape == (2, 2, 3)
    assert torch.isfinite(trace["wrench_residual"]).all()
    assert structured_observation(end).shape == (2, 25)


def test_l2f_full_space_step_updates_policy_then_exactly_restores_defects() -> None:
    torch.manual_seed(2)
    simulator = L2FSimulator(
        L2FParams(
            dt=0.01,
            max_initial_position=0.01,
            max_initial_velocity=0.01,
            max_initial_angle=0.01,
            max_initial_omega=0.01,
        )
    )
    physical = simulator.reset(1, device="cpu", dtype=torch.float64)
    policy = StructuredRecurrentPolicy(
        StructuredPolicyConfig(hidden_dim=2, identifier_dim=2, slow_cadence=1)
    ).double()
    observation = torch.cat(
        (
            physical.position,
            physical.velocity,
            physical.rotation.reshape(1, 9),
            physical.omega,
            torch.zeros(1, 3, dtype=torch.float64),
            physical.previous_action,
        ),
        dim=-1,
    )
    closed = StructuredClosedLoopState(physical, policy.initial_state(observation))
    codec = StructuredBoundaryCodec(physical, policy)
    spec = ParameterVectorSpec.from_module(policy)
    theta = spec.flatten(policy).detach()
    segment = make_segment_map(policy, simulator, codec, spec, steps=1)
    initial = codec.pack(closed)
    endpoints = initialize_exact_endpoints(segment, initial, theta, 2)
    problem = FullSpaceProblem(
        initial,
        endpoints,
        theta,
        segment,
        codec.layout,
        task_residual=lambda starts, ends, value: terminal_risk_residual(codec, ends),
    )
    step = solve_joint_sqp_step(
        problem,
        damping=100.0,
        penalty=1.0,
        cg_iterations=4,
        max_backtracks=6,
        parameter_radius=1.0,
    )
    assert step.accepted
    assert step.parameter_step_norm > 0.0
    restored = initialize_exact_endpoints(segment, initial, step.theta, 2)
    restored_problem = FullSpaceProblem(
        initial,
        restored,
        step.theta,
        segment,
        codec.layout,
        task_residual=problem.task_residual,
    )
    assert float(torch.linalg.vector_norm(restored_problem.defects())) < 1.0e-10


def test_parameter_spec_respects_trainable_allowlist_and_partial_assignment() -> None:
    _, policy, _ = _fixture()
    spec = ParameterVectorSpec.from_module(
        policy, allow_prefixes=("contextual_gain_head.",)
    )
    assert spec.names
    assert all(name.startswith("contextual_gain_head.") for name in spec.names)
    before = policy.capability_head.weight.detach().clone()
    vector = spec.flatten(policy) + 0.01
    spec.assign_(policy, vector)
    torch.testing.assert_close(policy.capability_head.weight, before)
    assert spec.trust_scale(policy).shape == vector.shape


def test_fixed_probe_actions_preserve_intermediate_cadence_phase() -> None:
    simulator, policy, closed = _fixture()
    codec = StructuredBoundaryCodec(closed.physical, policy)
    spec = ParameterVectorSpec.from_module(policy)
    theta = spec.flatten(policy).detach()
    segment = make_segment_map(policy, simulator, codec, spec, steps=2)
    initial = codec.pack(closed)
    endpoints = initialize_exact_endpoints(segment, initial, theta, 2)
    probes = build_action_probe_bank(
        policy, simulator, codec, spec, theta, initial, endpoints, steps=2
    )
    assert probes.slow_counters == (0, 1, 2, 3)
    actions = functional_probe_actions(policy, codec, spec, theta, probes)
    assert actions.shape == (4, 2, 4)


def test_combined_trajectory_diagnostics_matches_action_rollout() -> None:
    simulator, policy, closed = _fixture()
    codec = StructuredBoundaryCodec(closed.physical, policy)
    spec = ParameterVectorSpec.from_module(policy)
    theta = spec.flatten(policy).detach()
    segment = make_segment_map(policy, simulator, codec, spec, steps=2)
    initial = codec.pack(closed)
    endpoints = initialize_exact_endpoints(segment, initial, theta, 2)
    actions = functional_trajectory_actions(
        policy, simulator, codec, spec, theta, initial, endpoints, steps=2
    )
    combined_actions, failures = functional_trajectory_diagnostics(
        policy, simulator, codec, spec, theta, initial, endpoints, steps=2
    )
    torch.testing.assert_close(combined_actions, actions)
    assert failures.shape == (2, 2, 2)


def test_boot_completed_codec_fixes_timer_outside_shooting_variables() -> None:
    _, policy, closed = _fixture()
    codec = StructuredBoundaryCodec(closed.physical, policy, boot_completed=True)
    try:
        codec.pack(closed)
    except ValueError as error:
        assert "unfinished" in str(error)
    else:
        raise AssertionError("unfinished burn-in was accepted")
    total = policy.config.burn_in_steps + policy.config.contextual_blend_steps
    closed.policy.boot_progress.fill_(float(total))
    packed = codec.pack(closed)
    restored = codec.unpack(packed)
    torch.testing.assert_close(
        restored.policy.boot_progress,
        torch.full_like(restored.policy.boot_progress, float(total)),
    )


def test_phase_space_contraction_risk_has_policy_credit() -> None:
    simulator, policy, closed = _fixture()
    # This objective is a post-migration/full-space test.  Put the fixture at
    # the first available publication instead of testing the intentionally
    # parameter-independent conservative prior used before call50.
    closed.policy.boot_progress.fill_(float(policy.config.identification_publish_start))
    closed.policy.slow_counter = policy.config.identification_publish_start
    codec = StructuredBoundaryCodec(closed.physical, policy)
    spec = ParameterVectorSpec.from_module(policy)
    theta = spec.flatten(policy).detach().requires_grad_(True)
    segment = make_segment_map(policy, simulator, codec, spec, steps=2)
    initial = codec.pack(closed)
    endpoints = initialize_exact_endpoints(segment, initial, theta.detach(), 2)
    starts = torch.cat((initial.unsqueeze(0), endpoints[:-1]), dim=0)
    predicted = torch.stack([segment(value, theta) for value in starts])
    metric = default_phase_space_metric(
        dt=policy.config.dt, sample_steps=2, dtype=torch.float64
    )
    residual = phase_space_contraction_risk_residual(
        codec, starts, predicted, metric=metric.matrix,
        retention=metric.linear_model_retention,
    )
    gradient = torch.autograd.grad(0.5 * residual.square().sum(), theta)[0]
    assert torch.isfinite(residual).all()
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0

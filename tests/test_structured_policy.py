from __future__ import annotations

from dataclasses import replace

import torch

from env_l2f import L2FParams, L2FSimulator
from policy_observation import PolicyObservationState, build_policy_observation

from structured_distillation import (
    DistillationTargets,
    action_distillation_loss,
    build_rollout_target,
    student_action_jacobian,
    structured_distillation_loss,
    teacher_action_jacobian,
)
from structured_policy import (
    CAPABILITY_HI,
    EXCITATION_LAGS,
    IDENTIFICATION_PROBE_PERIOD,
    MultiTauMotorObserverBank,
    DampedConstrainedAllocator,
    FastFeedbackInterface,
    StructuredPolicyConfig,
    StructuredRecurrentPolicy,
    equilibrium_from_capability_disturbance,
    effective_wrench_mixer,
    identification_probe_patterns,
    motor_observer_tau_grid,
    reference_fast_gain,
)
from structured_allocator import ActiveSetBoxQPAllocator


def test_structured_policy_forward_and_slow_cadence() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=12, identifier_dim=5))
    observation = torch.randn(3, 25)
    output = policy.forward_with_aux(observation)
    assert output.action.shape == (3, 4)
    assert output.auxiliary["error"].shape == (3, 15)
    assert output.auxiliary["capability"].shape == (3, 6)
    previous_trim = output.next_state.slow_trim.clone()
    for _ in range(23):
        output = policy.forward_with_aux(observation, output.next_state)
    torch.testing.assert_close(output.next_state.slow_trim, previous_trim)
    output = policy.forward_with_aux(observation, output.next_state)
    assert output.next_state.slow_counter == 25


def test_call_zero_has_no_response_or_identifier_publication() -> None:
    policy = StructuredRecurrentPolicy(
        StructuredPolicyConfig(hidden_dim=8, identifier_dim=6)
    )
    observation = torch.zeros(2, 25)
    observation[:, (6, 10, 14)] = 1.0
    initial = policy.initial_state(observation)
    output0 = policy.forward_with_aux(observation, initial)
    torch.testing.assert_close(output0.auxiliary["identifier"], initial.identifier)
    assert bool(torch.isnan(output0.auxiliary["disturbance_limit"]).all())

    output1 = policy.forward_with_aux(observation, output0.next_state)
    assert not torch.equal(output1.auxiliary["identifier"], initial.identifier)


def test_default_availability_keeps_call25_hidden_and_publishes_call50() -> None:
    policy = StructuredRecurrentPolicy(
        StructuredPolicyConfig(
            hidden_dim=4, identifier_dim=4, burn_in_steps=25,
            contextual_blend_steps=25, slow_cadence=25,
        )
    )
    observation = torch.zeros(1, 25)
    observation[:, (6, 10, 14)] = 1.0
    state = policy.initial_state(observation)
    call0 = policy.forward_with_aux(observation, state)
    assert bool(torch.isnan(call0.auxiliary["disturbance_limit"]).all())
    state = call0.next_state
    for call_index in range(1, 126):
        output = policy.forward_with_aux(observation, state)
        state = output.next_state
        if call_index == 25:
            assert not bool(output.auxiliary["identification_publication_available"].any())
            assert not bool(output.auxiliary["identification_published"].any())
            assert bool(torch.isnan(output.auxiliary["disturbance_limit"]).all())
            torch.testing.assert_close(
                output.auxiliary["allocation_capability"][:, :3],
                output.auxiliary["allocation_capability"].new_tensor(
                    CAPABILITY_HI[:3]
                ).expand(1, 3),
            )
            assert float(output.auxiliary["contextual_gain_weight"].max()) == 0.0
        if call_index == 100:
            assert bool(output.auxiliary["identification_publication_available"].all())
            assert bool(output.auxiliary["identification_published"].all())
            assert bool(torch.isfinite(output.auxiliary["disturbance_limit"]).all())
        if call_index == 125:
            assert bool(output.auxiliary["identification_publication_available"].all())
            assert bool(output.auxiliary["identification_published"].all())


def test_absolute_position_and_integral_are_absent_from_slow_context() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=8, identifier_dim=8))
    observation = torch.zeros(1, 25)
    observation[:, (6, 10, 14)] = 1.0
    altered = observation.clone()
    altered[:, :3] = torch.tensor((4.0, -2.0, 1.0))
    altered[:, 18:21] = torch.tensor((0.2, -0.1, 0.3))
    first = policy.forward_with_aux(observation)
    second = policy.forward_with_aux(altered)
    torch.testing.assert_close(first.next_state.context_sum, second.next_state.context_sum)


def test_analytic_equilibrium_is_independent_of_fast_error_and_hidden() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=8, identifier_dim=8))
    capability = torch.tensor((3.0, 100.0, 0.2, 1.7, 0.1, 0.2)).expand(2, 6)
    disturbance = torch.tensor((0.2, -0.1, 0.05)).expand(2, 3)
    body_z, trim, feasible = equilibrium_from_capability_disturbance(
        capability, disturbance
    )
    torch.testing.assert_close(trim[0], trim[1])
    torch.testing.assert_close(body_z[0], body_z[1])
    assert bool(feasible.all())
    assert not hasattr(policy, "trim_head")
    assert not hasattr(policy, "body_z_head")


def test_disturbance_residual_uses_the_orientation_that_generated_delta_v() -> None:
    torch.manual_seed(1707)
    simulator = L2FSimulator(L2FParams(dt=0.01))
    physical = simulator.reset(
        8, device="cpu", sample_dynamics=True,
        sampled_dynamics_level="broad", broad_sampler="physical-fit",
    )
    action = torch.full((8, 4), 0.2)
    next_physical = simulator.step(physical, action, grad_decay=1.0)
    observation, _ = build_policy_observation(
        next_physical,
        PolicyObservationState(torch.zeros(8, 3)),
        mode="integral25", integral_input_frame="body",
    )
    policy = StructuredRecurrentPolicy(
        StructuredPolicyConfig(hidden_dim=8, identifier_dim=8)
    )
    policy_state = policy.initial_state(observation)
    capability = torch.stack((
        physical.thrust_to_weight, physical.alpha_roll_max, physical.eta_yaw,
        physical.jz_over_jxy, physical.motor_time_rising,
        physical.motor_time_falling,
    ), dim=-1)
    policy_state = replace(
        policy_state,
        capability=capability,
        motor_estimate=next_physical.motor,
        prev_velocity=physical.velocity,
        prev_omega=physical.omega,
        slow_counter=1,
    )
    output = policy.forward_with_aux(observation, policy_state)
    expected = physical.external_force / physical.mass[:, None]
    torch.testing.assert_close(
        output.auxiliary["disturbance_residual"], expected,
        atol=2.0e-5, rtol=2.0e-5,
    )


def test_motor_path_is_differentiable_and_bounded() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=10, identifier_dim=4))
    observation = torch.randn(2, 25, requires_grad=True)
    action, state = policy(observation)
    assert bool((action.abs() <= 1.0).all())
    (action.square().mean() + state.motor_estimate.square().mean()).backward()
    assert observation.grad is not None
    assert bool(torch.isfinite(observation.grad).all())


def test_multi_tau_bank_has_registered_fifteen_mode_grid() -> None:
    pairs = motor_observer_tau_grid()
    assert len(pairs) == 15
    assert len({rise for rise, _ in pairs}) == 5
    assert all(0.03 <= fall <= 0.35 for _, fall in pairs)
    bank = MultiTauMotorObserverBank()
    assert bank.modes == 15
    torch.testing.assert_close(
        bank.tau_rise.cpu(), torch.tensor([rise for rise, _ in pairs])
    )
    torch.testing.assert_close(
        bank.tau_fall.cpu(), torch.tensor([fall for _, fall in pairs])
    )


def test_multi_tau_bank_advances_only_with_applied_action() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(
        hidden_dim=8, identifier_dim=6, motor_observer_bank_size=15,
        motor_observer_mode="fixed_multi_tau_v1", motor_tau_grid_version=1,
    ))
    observation = torch.zeros(1, 25)
    observation[:, (6, 10, 14)] = 1.0
    candidate = policy.forward_with_aux(observation)
    applied = torch.full((1, 4), 0.4)
    executed = policy.forward_with_aux(
        observation, candidate.next_state, applied_action=applied
    )
    expected = policy.motor_observer_bank(
        candidate.next_state.motor_bank, applied, policy.config.dt
    )
    torch.testing.assert_close(executed.next_state.motor_bank, expected)
    proposal = policy.motor_observer_bank(
        candidate.next_state.motor_bank, candidate.action, policy.config.dt
    )
    assert float(torch.linalg.vector_norm(expected - proposal)) > 1.0e-3


def test_zero_initialized_bank_adapter_preserves_legacy_policy_action() -> None:
    torch.manual_seed(1707)
    legacy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=8, identifier_dim=6))
    banked = StructuredRecurrentPolicy(StructuredPolicyConfig(
        hidden_dim=8, identifier_dim=6, motor_observer_bank_size=15,
        motor_observer_mode="fixed_multi_tau_v1", motor_tau_grid_version=1,
    ))
    banked.load_state_dict(legacy.state_dict(), strict=False)
    assert torch.equal(banked.bank_adapter.weight, torch.zeros_like(banked.bank_adapter.weight))
    observation = torch.randn(2, 25)
    observation[:, (6, 10, 14)] = 1.0
    legacy_state = legacy.initial_state(observation)
    banked_state = banked.initial_state(observation)
    for _ in range(3):
        legacy_output = legacy.forward_with_aux(observation, legacy_state)
        banked_output = banked.forward_with_aux(observation, banked_state)
        torch.testing.assert_close(
            banked_output.action, legacy_output.action, atol=0.0, rtol=0.0
        )
        legacy_state = legacy_output.next_state
        banked_state = banked_output.next_state


def test_second_order_residual_is_zero_to_first_order_at_zero_error() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=8, identifier_dim=3))
    # The residual head is intentionally zero initialized and the explicit
    # squared-norm gate guarantees both value and first derivative vanish.
    error = torch.zeros(2, 15, requires_grad=True)
    residual = torch.tanh(policy.residual_head(torch.zeros(2, 23).detach().requires_grad_(True)))
    gated = residual * error.square().sum(-1, keepdim=True) / (1.0 + error.square().sum(-1, keepdim=True))
    assert torch.equal(gated, torch.zeros_like(gated))
    assert torch.equal(torch.autograd.grad(gated.sum(), error)[0], torch.zeros_like(error))


def test_allocator_is_smooth_bounded_and_reports_condition() -> None:
    allocator = DampedConstrainedAllocator(rate_limit=0.5)
    wrench = torch.randn(4, 4, requires_grad=True)
    previous = torch.zeros(4, 4)
    action, diagnostics = allocator(wrench, previous, dt=0.01)
    assert bool((action.abs() < 1.0).all())
    assert bool((diagnostics.condition_number > 0.0).all())
    (action.square().mean() + diagnostics.wrench_residual.mean()).backward()
    assert wrench.grad is not None
    assert bool(torch.isfinite(wrench.grad).all())


def test_allocator_delta_preserves_trim_at_zero_wrench_delta() -> None:
    allocator = DampedConstrainedAllocator()
    trim = torch.tensor(((-0.4, -0.1, 0.2, 0.6),))
    mixer = torch.eye(4).unsqueeze(0)
    desired = torch.bmm(mixer, trim.unsqueeze(-1)).squeeze(-1)
    action, diagnostics = allocator(desired, mixer=mixer, trim=trim)
    torch.testing.assert_close(action, trim, atol=2e-6, rtol=0.0)
    assert float(diagnostics.headroom_violation.max()) == 0.0


def test_structured_policy_can_select_exact_box_qp_allocator() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(
        hidden_dim=8, identifier_dim=6, allocator_solver="box_qp",
    ))
    assert isinstance(policy.allocator, ActiveSetBoxQPAllocator)
    observation = torch.zeros(2, 25)
    observation[:, (6, 10, 14)] = 1.0
    output = policy.forward_with_aux(observation)
    assert bool(torch.isfinite(output.action).all())
    assert bool((output.action.abs() <= 1.0).all())
    assert float(output.auxiliary["allocator"].primal_violation.max()) <= 1.0e-6
    assert float(output.auxiliary["allocator"].kkt_residual.max()) <= 1.0e-4


def test_effective_mixer_matches_env_signs() -> None:
    capability = torch.tensor(((3.0, 100.0, 0.2, 1.7, 0.1, 0.2),))
    mixer = effective_wrench_mixer(capability)[0]
    assert mixer[1, 1] > 0 and mixer[1, 3] < 0
    assert mixer[2, 0] < 0 and mixer[2, 2] > 0
    assert mixer[3, 0] > 0 and mixer[3, 1] < 0 and mixer[3, 2] > 0 and mixer[3, 3] < 0


def test_fast_feedback_defaults_frozen_and_distillation_api() -> None:
    feedback = FastFeedbackInterface()
    assert feedback.frozen and not feedback.verified
    assert not feedback.install_verified_gain(torch.zeros(4, 15))
    gain = torch.zeros(4, 15)
    gain[0, 0] = 0.1
    assert feedback.install_verified_gain(gain)
    observation = torch.randn(2, 25)
    teacher = lambda value: torch.tanh(value[:, :4])
    targets = DistillationTargets(action=teacher(observation).detach())
    student = teacher(observation) + 0.1
    losses = structured_distillation_loss(student, targets)
    torch.testing.assert_close(losses.total, action_distillation_loss(student, targets.action))
    jacobian = teacher_action_jacobian(teacher, observation)
    assert jacobian.shape == (2, 4, 25)
    student_jacobian = student_action_jacobian(lambda value, state=None: teacher(value), observation)
    assert student_jacobian.shape == (2, 4, 25)
    rollout = build_rollout_target(teacher, observation.unsqueeze(0).expand(3, -1, -1),
                                   lambda state, action: state + 0.0 * action.sum(-1, keepdim=True))
    assert rollout.shape == (4, 2, 25)


def test_reference_gain_is_nonzero_finite_and_installable() -> None:
    feedback = FastFeedbackInterface()
    gain = reference_fast_gain(dtype=torch.float64)
    assert torch.isfinite(gain).all()
    assert float(torch.linalg.vector_norm(gain)) > 0.0
    assert feedback.install_verified_gain(gain.float())


def test_capability_posterior_and_initial_ucb_are_explicit() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=8, identifier_dim=6))
    observation = torch.zeros(2, 25)
    observation[:, (6, 10, 14)] = 1.0
    output = policy.forward_with_aux(observation)
    state = output.next_state
    assert state.capability_log_mean.shape == (2, 6)
    assert state.capability_log_scale.shape == (2, 6)
    assert state.capability_ucb.shape == (2, 6)
    expected_hi = observation.new_tensor(CAPABILITY_HI).expand(2, 6)
    torch.testing.assert_close(state.capability_ucb[:, :3], expected_hi[:, :3])
    torch.testing.assert_close(state.capability_ucb[:, 3:], state.capability[:, 3:])
    torch.testing.assert_close(output.auxiliary["allocation_capability"][:, :3], expected_hi[:, :3])
    assert "K_ref" in dict(policy.named_buffers())
    assert not any(name == "K_ref" for name, _ in policy.named_parameters())
    assert not any(parameter.requires_grad for parameter in policy.residual_head.parameters())


def test_capability_scale_starts_inside_trainable_z_space_bound() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=8, identifier_dim=6))
    identifier = torch.zeros(2, 6)
    _, _, log_sigma, _ = policy._capability_statistics(identifier)
    assert float(log_sigma.exp().max()) <= 2.0
    log_sigma.sum().backward()
    assert policy.capability_log_scale_head.bias.grad is not None
    assert float(policy.capability_log_scale_head.bias.grad.abs().sum()) > 0.0


def test_contextual_gain_is_held_bounded_and_blended_after_burn_in() -> None:
    config = StructuredPolicyConfig(hidden_dim=8, identifier_dim=6, contextual_gain_rho=0.1)
    policy = StructuredRecurrentPolicy(config)
    with torch.no_grad():
        policy.contextual_gain_head.bias.fill_(1.0)
    observation = torch.zeros(1, 25)
    observation[:, (6, 10, 14)] = 1.0
    output = policy.forward_with_aux(observation)
    initial_gain = output.next_state.contextual_gain.clone()
    # Call25 only updates persistent identifier memory; no control output is
    # allowed to consume it before the registered call50 publication.
    for _ in range(23):
        output = policy.forward_with_aux(observation, output.next_state)
    torch.testing.assert_close(output.next_state.contextual_gain, initial_gain)
    output = policy.forward_with_aux(observation, output.next_state)
    assert output.next_state.slow_counter == 25
    assert float(output.auxiliary["contextual_induced_norm"].max()) <= config.contextual_gain_rho + 1e-6
    assert float(output.auxiliary["contextual_blend"].max()) == 0.0
    for _ in range(76):
        output = policy.forward_with_aux(observation, output.next_state)
    assert output.next_state.slow_counter == 101
    assert 0.0 < float(output.auxiliary["contextual_blend"].max()) <= 1.0


def test_allocator_reports_trust_limited_burn_in_action() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=8, identifier_dim=6))
    observation = torch.zeros(1, 25)
    observation[:, (6, 10, 14)] = 1.0
    observation[:, 0] = 10.0
    output = policy.forward_with_aux(observation)
    assert bool(torch.isfinite(output.auxiliary["allocator"].trust_limited).all())
    assert bool((output.action.abs() <= 1.0).all())


def test_experimental_probe_is_collective_and_its_scalar_lags_are_independent() -> None:
    """Every publication block is zero-DC and lagged design stays full rank."""

    config = StructuredPolicyConfig(
        hidden_dim=8,
        identifier_dim=6,
        burn_in_steps=25,
        contextual_blend_steps=25,
        burn_in_probe_amplitude=0.005, burn_in_rate_limit=50.0,
    )
    policy = StructuredRecurrentPolicy(config)
    observation = torch.zeros(1, 25)
    observation[:, (6, 10, 14)] = 1.0
    state = policy.initial_state(observation)
    probes = []
    for _ in range(25 + IDENTIFICATION_PROBE_PERIOD):
        output = policy.forward_with_aux(observation, state)
        probes.append(output.auxiliary["identification_probe_action"])
        state = output.next_state
    cycle = torch.cat(probes, dim=0)[25:]
    registered = identification_probe_patterns(device=cycle.device, dtype=cycle.dtype)
    torch.testing.assert_close(cycle, config.burn_in_probe_amplitude * registered)
    for block in cycle.reshape(2, 25, 4):
        torch.testing.assert_close(
            block.sum(dim=0), torch.zeros_like(block[0]), atol=1e-7, rtol=0.0
        )
    # The collective component (the mean of the four motors) is also
    # explicitly zero mean, so the probe does not inject a net DC thrust.
    torch.testing.assert_close(
        cycle.mean(dim=-1).mean(), torch.zeros((), dtype=cycle.dtype), atol=1e-7, rtol=0.0
    )
    assert bool((cycle.abs() <= config.burn_in_probe_amplitude + 1e-7).all())
    design = torch.cat(
        tuple(cycle[12 - lag:IDENTIFICATION_PROBE_PERIOD - lag]
              for lag in EXCITATION_LAGS),
        dim=-1,
    )
    singular_values = torch.linalg.svdvals(design)
    assert int(torch.linalg.matrix_rank(design, atol=1e-6)) == 3
    assert torch.allclose(cycle, cycle[:, :1].expand_as(cycle), atol=1e-7)
    assert float(singular_values.max() / singular_values[2]) < 30.0


def test_identification_probe_is_zero_after_transition_window() -> None:
    config = StructuredPolicyConfig(
        hidden_dim=8,
        identifier_dim=6,
        burn_in_steps=2,
        contextual_blend_steps=3,
        identification_publish_start=5,
        slow_cadence=1,
        burn_in_probe_amplitude=0.005, burn_in_rate_limit=50.0,
    )
    policy = StructuredRecurrentPolicy(config)
    observation = torch.zeros(1, 25)
    observation[:, (6, 10, 14)] = 1.0
    state = policy.initial_state(observation)
    for _ in range(config.burn_in_steps + config.contextual_blend_steps):
        output = policy.forward_with_aux(observation, state)
        state = output.next_state
    # At boot_progress == burn_in + blend the transition mask is off.  This
    # checks the exact boundary, not merely a later steady-state call.
    output = policy.forward_with_aux(observation, state)
    assert not bool(output.auxiliary["burn_in"].any())
    torch.testing.assert_close(
        output.auxiliary["identification_probe_action"],
        torch.zeros_like(output.auxiliary["identification_probe_action"]),
        atol=0.0,
        rtol=0.0,
    )


def test_identification_probe_action_respects_burn_in_cap_and_rate_limit() -> None:
    config = StructuredPolicyConfig(
        hidden_dim=8,
        identifier_dim=6,
        burn_in_steps=8,
        contextual_blend_steps=1,
        identification_publish_start=9,
        slow_cadence=1,
        burn_in_action_cap=0.01,
        burn_in_rate_limit=0.05,
        burn_in_probe_amplitude=0.005,
    )
    policy = StructuredRecurrentPolicy(config)
    observation = torch.zeros(1, 25)
    observation[:, (6, 10, 14)] = 1.0
    # A large state error makes the unconstrained feedback command nontrivial.
    observation[:, :3] = torch.tensor((10.0, -10.0, 5.0))
    previous_action = torch.zeros(1, 4)
    observation[:, 21:25] = previous_action
    output = policy.forward_with_aux(observation)
    trim = output.auxiliary["trim_action"]
    assert bool((output.auxiliary["action_delta_cap"] == config.burn_in_action_cap).all())
    assert bool((output.auxiliary["rate_limit"] == config.burn_in_rate_limit).all())
    assert bool((output.action - trim).abs().max() <= config.burn_in_action_cap + 1e-7)
    assert bool((output.action - previous_action).abs().max() <=
                config.burn_in_rate_limit * config.dt + 1e-7)
    assert bool((output.auxiliary["identification_probe_action"].abs() <=
                 config.burn_in_probe_amplitude + 1e-7).all())


def test_applied_action_drives_observer_state_not_candidate_action() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=8, identifier_dim=6))
    observation = torch.zeros(1, 25)
    observation[:, (6, 10, 14)] = 1.0
    candidate = policy.forward_with_aux(observation)
    applied = torch.full((1, 4), 0.4)
    executed = policy.forward_with_aux(
        observation, candidate.next_state, applied_action=applied
    )
    expected = policy.motor_observer(
        candidate.next_state.motor_estimate,
        applied,
        policy.config.dt,
        candidate.next_state.capability[:, 4:5].expand(1, 4),
        candidate.next_state.capability[:, 5:6].expand(1, 4),
    )
    torch.testing.assert_close(executed.next_state.motor_estimate, expected)
    torch.testing.assert_close(executed.auxiliary["applied_action"], applied)


def test_excitation_history_uses_applied_action_not_unexecuted_candidate() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=8, identifier_dim=6))
    observation = torch.zeros(1, 25)
    observation[:, (6, 10, 14)] = 1.0
    candidate = policy.forward_with_aux(observation)
    applied = torch.full((1, 4), 0.4)
    executed = policy.forward_with_aux(
        observation, candidate.next_state, applied_action=applied
    )
    motor_before_command = candidate.next_state.motor_estimate
    expected = ((applied - motor_before_command) / 0.10).clamp(-5.0, 5.0)
    candidate_expected = ((candidate.action - motor_before_command) / 0.10).clamp(-5.0, 5.0)
    torch.testing.assert_close(
        executed.next_state.excitation_history[:, 0], expected, atol=1e-7, rtol=0.0
    )
    # The externally applied command is deliberately different from the
    # policy proposal; using the proposal would produce the other history.
    assert float(torch.linalg.vector_norm(expected - candidate_expected)) > 1.0e-3
    assert float(torch.linalg.vector_norm(
        executed.next_state.excitation_history[:, 0] - candidate_expected
    )) > 1.0e-3


def test_boot_transition_finishes_instead_of_becoming_a_permanent_guard() -> None:
    config = StructuredPolicyConfig(
        hidden_dim=8, identifier_dim=6, burn_in_steps=2,
        contextual_blend_steps=3, slow_cadence=1,
        identification_publish_start=5,
    )
    policy = StructuredRecurrentPolicy(config)
    observation = torch.zeros(1, 25)
    observation[:, (6, 10, 14)] = 1.0
    state = policy.initial_state(observation)
    output = None
    for _ in range(6):
        output = policy.forward_with_aux(observation, state)
        state = output.next_state
    assert output is not None
    torch.testing.assert_close(state.boot_progress, torch.full_like(state.boot_progress, 5.0))
    assert float(output.auxiliary["action_delta_cap"].min()) > 1.0e5
    assert 0.0 < float(output.auxiliary["contextual_blend"].min()) <= 1.0
    for _ in range(2):
        output = policy.forward_with_aux(observation, state)
        state = output.next_state
    assert float(output.auxiliary["contextual_blend"].min()) == 1.0


def test_valid_zero_conformal_score_replaces_smoke_beta() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=8, identifier_dim=6))
    identifier = torch.zeros(2, 6)
    mean_before, _, _, ucb_before = policy._capability_statistics(identifier)
    assert bool((ucb_before[:, :3] >= mean_before[:, :3]).all())
    policy.install_capability_conformal_q(torch.zeros(6), sample_count=512)
    mean_after, _, _, ucb_after = policy._capability_statistics(identifier)
    torch.testing.assert_close(ucb_after, mean_after)


def test_uncertain_capability_suppresses_contextual_gain_weight() -> None:
    config = StructuredPolicyConfig(
        hidden_dim=8, identifier_dim=6, slow_cadence=1,
        burn_in_steps=0, contextual_blend_steps=1,
        identification_publish_start=1,
    )
    policy = StructuredRecurrentPolicy(config)
    with torch.no_grad():
        policy.contextual_gain_head.bias.fill_(1.0)
    observation = torch.zeros(1, 25)
    observation[:, (6, 10, 14)] = 1.0
    uncertain0 = policy.forward_with_aux(observation)
    uncertain = policy.forward_with_aux(observation, uncertain0.next_state)
    assert float(uncertain.auxiliary["capability_confidence"].max()) == 0.0
    assert float(uncertain.auxiliary["contextual_gain_weight"].max()) == 0.0
    policy.install_capability_conformal_q(torch.zeros(6), sample_count=512)
    calibrated0 = policy.forward_with_aux(observation)
    calibrated = policy.forward_with_aux(observation, calibrated0.next_state)
    assert float(calibrated.auxiliary["capability_confidence"].min()) == 1.0
    assert float(calibrated.auxiliary["contextual_gain_weight"].min()) == 1.0


def test_contextual_gain_head_is_capability_only() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=8, identifier_dim=6))
    assert policy.contextual_gain_head.in_features == 6
    observation = torch.zeros(1, 25)
    observation[:, (6, 10, 14)] = 1.0
    state = policy.initial_state(observation)
    first = policy.forward_with_aux(observation, state)
    altered = state.detach()
    altered.identifier.add_(10.0)
    second = policy.forward_with_aux(observation, altered)
    torch.testing.assert_close(
        first.auxiliary["contextual_gain"], second.auxiliary["contextual_gain"]
    )


def test_t50_identification_failure_is_reported_from_width_gate() -> None:
    config = StructuredPolicyConfig(
        hidden_dim=8, identifier_dim=6, slow_cadence=1,
        burn_in_steps=1, contextual_blend_steps=1,
        identification_publish_start=2,
    )
    policy = StructuredRecurrentPolicy(config)
    observation = torch.zeros(1, 25)
    observation[:, (6, 10, 14)] = 1.0
    state = policy.initial_state(observation)
    # Force a wide posterior interval and advance through t50.
    with torch.no_grad():
        policy.capability_log_scale_head.bias.fill_(10.0)
    for _ in range(3):
        output = policy.forward_with_aux(observation, state)
        state = output.next_state
    assert bool(output.auxiliary["identification_failed"].all())
    assert bool(state.identification_failed.all())
    for _ in range(3):
        output = policy.forward_with_aux(observation, state)
        state = output.next_state
    assert bool(output.auxiliary["identification_failed"].all())
    assert bool(state.identification_failed.all())


def test_applied_saturation_controls_integral_anti_windup() -> None:
    policy = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=8, identifier_dim=6))
    observation = torch.zeros(1, 25)
    observation[:, (6, 10, 14)] = 1.0
    observation[:, 0] = 0.2
    initial = policy.initial_state(observation)
    free = policy.forward_with_aux(observation, initial, applied_action=torch.zeros(1, 4))
    saturated = policy.forward_with_aux(observation, initial, applied_action=torch.ones(1, 4))
    assert float(free.next_state.integral.abs().sum()) > 0.0
    torch.testing.assert_close(saturated.next_state.integral, initial.integral)

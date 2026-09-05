"""Teacher-to-structured-policy distillation losses.

All helpers are model-agnostic.  A legacy ``MotorGRUPolicy`` can be passed as
the teacher, while the student is normally ``StructuredRecurrentPolicy``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Optional

import torch
from torch.nn import functional as F

from env_l2f import L2FParams, L2FSimulator, L2FState
from equilibrium_control import (
    EquilibriumTarget,
    analytic_equilibrium_target,
    materialize_equilibrium_state,
)
from policy_observation import PolicyObservationState, build_policy_observation, initial_observation_state, update_position_integral
from structured_policy import (
    CAPABILITY_HI,
    CAPABILITY_LO,
    EXCITATION_SCALE,
    StructuredPolicyState,
    StructuredRecurrentPolicy,
)


@dataclass(frozen=True)
class DistillationTargets:
    action: torch.Tensor
    action_jacobian: Optional[torch.Tensor] = None
    one_step_state: Optional[torch.Tensor] = None
    rollout: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class DistillationLoss:
    total: torch.Tensor
    action: torch.Tensor
    jacobian: torch.Tensor
    one_step: torch.Tensor
    rollout: torch.Tensor


DAGGER_BETA_SCHEDULE = (0.5, 0.25, 0.1, 0.0, 0.0)
from probe_contract_v5 import ProbeState, apply_probe


@dataclass(frozen=True)
class DAggerScenarioBank:
    state: L2FState
    tw_bin: torch.Tensor
    log_alpha_bin: torch.Tensor
    stratum: tuple[str, ...]

    @property
    def count(self) -> int:
        return int(self.state.position.shape[0])


@dataclass(frozen=True)
class DAggerEpisode:
    observations: torch.Tensor
    teacher_actions: torch.Tensor
    teacher_same_latent_intercepts: torch.Tensor
    teacher_fast_delta_actions: torch.Tensor
    motor_trim_target: torch.Tensor
    body_z_target: torch.Tensor
    disturbance_accel_target: torch.Tensor
    student_actions: torch.Tensor
    teacher_hidden: torch.Tensor
    student_hidden: torch.Tensor
    executed_actions: torch.Tensor
    intervention_mask: torch.Tensor
    finite: torch.Tensor
    position_norm: torch.Tensor
    velocity_norm: torch.Tensor
    omega_norm: torch.Tensor
    allocator_headroom: torch.Tensor
    allocator_residual: torch.Tensor
    action_rate: torch.Tensor
    identification_norm_t50: torch.Tensor
    effectiveness_log_interval_width_t50: torch.Tensor
    identification_failure_t50: torch.Tensor
    capability_z_mean: torch.Tensor
    capability_z_log_scale: torch.Tensor
    student_trim: torch.Tensor
    student_body_z: torch.Tensor
    student_disturbance_accel: torch.Tensor
    equilibrium_feasible: torch.Tensor
    equilibrium_action_error: torch.Tensor
    equilibrium_one_step_accel: torch.Tensor
    equilibrium_one_step_omega: torch.Tensor
    beta: float
    intervention: tuple[str, ...]
    tw_bin: torch.Tensor
    log_alpha_bin: torch.Tensor
    capability_target_z: torch.Tensor
    equilibrium_template: L2FState
    simulator_dt: float
    simulator_gravity: float
    executed_probe_residual: Optional[torch.Tensor] = None
    executed_probe_aborted: Optional[torch.Tensor] = None
    motor_observer_rms_error: Optional[torch.Tensor] = None

    @property
    def max_position_norm(self) -> torch.Tensor:
        return self.position_norm.amax(dim=0)

    @property
    def max_velocity_norm(self) -> torch.Tensor:
        return self.velocity_norm.amax(dim=0)

    @property
    def max_omega_norm(self) -> torch.Tensor:
        return self.omega_norm.amax(dim=0)


@dataclass(frozen=True)
class CapabilityContract:
    capability_z_mean: torch.Tensor
    capability_z_log_scale: torch.Tensor
    capability_ucb: torch.Tensor
    contextual_gain: torch.Tensor


def build_dagger_scenario_bank(
    count: int = 64,
    *,
    seed: int = 7,
    dt: float = 0.01,
    per_cell: int = 4,
) -> DAggerScenarioBank:
    """Create a deterministic 4x4 thrust/roll-authority stratified bank."""

    if count != 16 * per_cell or per_cell < 1:
        raise ValueError("count must equal 16 * per_cell for the 4x4 authority bank")
    simulator = L2FSimulator(L2FParams(dt=dt))
    # A large vectorized draw makes the cell quotas deterministic without
    # rejection loops whose runtime varies with the sampler implementation.
    candidate_count = max(4096, count * 128)
    # A scenario seed must not overwrite the model/optimizer RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        candidates = simulator.reset(
            candidate_count,
            device="cpu",
            dtype=torch.float32,
            sample_dynamics=True,
            sampled_dynamics_level="broad",
            broad_sampler="physical-fit",
            balanced_dynamics_sampling=False,
            sample_external_force=True,
        )
    tw_edges = torch.linspace(1.45, 5.50, 5)
    alpha_edges = torch.logspace(torch.log10(torch.tensor(35.0)), torch.log10(torch.tensor(2200.0)), 5)
    tw_bin = torch.bucketize(candidates.thrust_to_weight.cpu(), tw_edges[1:-1])
    log_alpha = candidates.alpha_roll_max.clamp_min(1.0e-6).log10().cpu()
    log_alpha_edges = alpha_edges.log10()
    alpha_bin = torch.bucketize(log_alpha, log_alpha_edges[1:-1])
    selected = []
    selected_tw = []
    selected_alpha = []
    for tw_index in range(4):
        for alpha_index in range(4):
            matches = torch.nonzero((tw_bin == tw_index) & (alpha_bin == alpha_index), as_tuple=False).flatten()
            if matches.numel() < per_cell:
                raise RuntimeError("physical-fit sampler did not fill the 4x4 authority bank")
            chosen = matches[:per_cell]
            selected.append(chosen)
            selected_tw.append(torch.full((per_cell,), tw_index, dtype=torch.long))
            selected_alpha.append(torch.full((per_cell,), alpha_index, dtype=torch.long))
    indices = torch.cat(selected)
    state = L2FState(**{name: getattr(candidates, name).index_select(0, indices) for name in candidates.__dataclass_fields__})
    tw_values = torch.cat(selected_tw)
    alpha_values = torch.cat(selected_alpha)
    strata = tuple(f"tw{int(tw)}_logalpha{int(alpha)}" for tw, alpha in zip(tw_values.tolist(), alpha_values.tolist()))
    return DAggerScenarioBank(state=state, tw_bin=tw_values, log_alpha_bin=alpha_values, stratum=strata)


def _raw_capability(state: L2FState) -> torch.Tensor:
    return torch.stack((state.thrust_to_weight, state.alpha_roll_max, state.eta_yaw,
                        state.jz_over_jxy, state.motor_time_rising, state.motor_time_falling), dim=-1)


def normalize_log_capability(capability: torch.Tensor) -> torch.Tensor:
    """Map physical positive capability to the policy's [-1,1] log chart."""

    log_lo = capability.new_tensor(CAPABILITY_LO).log()
    log_hi = capability.new_tensor(CAPABILITY_HI).log()
    center = 0.5 * (log_lo + log_hi)
    half = 0.5 * (log_hi - log_lo)
    return (capability.clamp_min(1.0e-8).log() - center) / half


def _clone_l2f(state: L2FState) -> L2FState:
    return L2FState(**{name: getattr(state, name).detach().clone() for name in state.__dataclass_fields__})


def _teacher_action_hidden(teacher: Callable, observation: torch.Tensor, hidden: torch.Tensor):
    result = teacher(observation, hidden)
    return (result[0], result[1]) if isinstance(result, (tuple, list)) else (result, hidden)


def _q2_action_hidden_intercept(
    teacher: Callable, observation: torch.Tensor, hidden: torch.Tensor,
    state: L2FState, simulator: L2FSimulator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Q2 action, post-forward hidden and same-hidden equilibrium intercept."""
    forward = getattr(teacher, "forward_with_aux", None)
    action_from_latent = getattr(teacher, "_action_from_latent", None)
    if forward is None or action_from_latent is None:
        action, next_hidden = _teacher_action_hidden(teacher, observation, hidden)
        return action, next_hidden, action.detach()
    result = forward(observation, hidden)
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        action, next_hidden = result, hidden
    else:
        action, next_hidden = result[0], result[1]
    slope = float(getattr(teacher, "negative_slope", 0.01))
    latent = F.leaky_relu(next_hidden, negative_slope=slope)
    target = analytic_equilibrium_target(state, gravity=simulator.params.gravity)
    equilibrium_observation = observation.clone()
    equilibrium_observation[:, 15:18] = 0.0
    equilibrium_observation[:, 21:25] = target.motor_trim
    intercept, _ = action_from_latent(equilibrium_observation, latent)
    return action, next_hidden, intercept


def _observation_from_student_integral(
    observation: torch.Tensor,
    student_state: object,
    fallback: PolicyObservationState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the student's deployable integral as the sole observation source.

    The legacy observation builder owns an auxiliary integral accumulator.  A
    structured policy also carries its anti-windup integral; mixing the two
    creates a hidden second integral path.  Prefer the policy state whenever it
    exposes one, while retaining the fallback for model-agnostic test doubles.
    """
    integral = getattr(student_state, "integral", None)
    if not torch.is_tensor(integral):
        return observation, fallback.integral_position
    rotation = observation[:, 6:15].reshape(-1, 3, 3)
    integral_body = torch.bmm(rotation.transpose(1, 2), integral.unsqueeze(-1)).squeeze(-1)
    return torch.cat((observation[:, :18], integral_body, observation[:, 21:25]), dim=-1), integral


def _advance_student_with_executed_action(
    student: StructuredRecurrentPolicy,
    observation: torch.Tensor,
    state: object,
    candidate: object,
    executed_action: torch.Tensor,
    *,
    dt: float,
    executed_probe_residual: torch.Tensor | None = None,
    executed_probe_aborted: torch.Tensor | None = None,
) -> object:
    """Advance action-dependent student state using the command actually sent.

    Re-evaluate from the SAME immutable input state with the executed action.
    This advances exactly one call and shares the deployment transition,
    including first-publication observer reconstruction. Test doubles retain
    the field-based compatibility path below.
    """
    if isinstance(student, StructuredRecurrentPolicy) and isinstance(state, StructuredPolicyState):
        return student.forward_with_aux(
            observation, state, dt=dt, applied_action=executed_action,
            applied_probe_residual=executed_probe_residual,
            applied_probe_aborted=executed_probe_aborted,
        ).next_state
    next_state = getattr(candidate, "next_state", candidate)
    observer = getattr(student, "motor_observer", None)
    previous_motor = getattr(state, "motor_estimate", None)
    capability = getattr(next_state, "capability", None)
    if observer is None or not torch.is_tensor(previous_motor):
        return next_state
    if torch.is_tensor(capability) and capability.shape[-1] >= 6:
        tau_rise = capability[:, 4:5].expand_as(previous_motor)
        tau_fall = capability[:, 5:6].expand_as(previous_motor)
    else:
        tau_rise = tau_fall = None
    motor = observer(previous_motor, executed_action, dt, tau_rise, tau_fall)
    updates = {"motor_estimate": motor,
               "previous_motor_estimate": previous_motor,
               "previous_executed_action": observation[:, 21:25]}
    motor_bank = getattr(student, "motor_observer_bank", None)
    previous_bank = getattr(state, "motor_bank", None)
    if motor_bank is not None and torch.is_tensor(previous_bank):
        updates["motor_bank"] = motor_bank(previous_bank, executed_action, dt)
    # Candidate evaluation happens before DAgger intervention is selected.
    # Patch only action-dependent state here; forwarding the same observation a
    # second time would consume a second call index and duplicate its response.
    integrator = getattr(student, "integrator", None)
    previous_integral = getattr(state, "integral", None)
    if integrator is not None and torch.is_tensor(previous_integral):
        authority = (1.0 - executed_action.abs().mean(dim=-1)).clamp(
            0.0, 1.0
        ).unsqueeze(-1)
        updates["integral"] = integrator(
            previous_integral, observation[:, 0:3], dt, authority
        )
    excitation_history = getattr(state, "excitation_history", None)
    if torch.is_tensor(excitation_history):
        innovation = ((executed_action - previous_motor) / EXCITATION_SCALE).clamp(-5.0, 5.0)
        updates["excitation_history"] = torch.cat(
            (innovation.unsqueeze(1), excitation_history[:, :-1]), dim=1
        )
    history = getattr(state, "identification_actions", None)
    if torch.is_tensor(history) and getattr(state, "slow_counter", 0) < student.config.identification_publish_start:
        updates["identification_actions"] = torch.cat((history[:, 1:], executed_action[:, None]), 1)
    try:
        return replace(next_state, **updates)
    except TypeError:
        return next_state


@torch.no_grad()
def collect_dagger_episode(
    teacher: Callable,
    student: StructuredRecurrentPolicy,
    simulator: L2FSimulator,
    bank: DAggerScenarioBank,
    *,
    beta: float,
    horizon: int = 126,
    episode_seed: int = 0,
    teacher_probe: bool = False,
    teacher_observation_settings: dict | None = None,
) -> DAggerEpisode:
    """Share the actual trajectory; each controller owns its integral state."""

    if not 0.0 <= beta <= 1.0 or horizon < 101:
        raise ValueError(
            "beta must be in [0,1] and horizon must include the first publication"
        )
    state = _clone_l2f(bank.state)
    batch = bank.count
    equilibrium_target = analytic_equilibrium_target(
        state, gravity=simulator.params.gravity
    )
    body_z_target = equilibrium_target.body_z.detach()
    motor_trim_target = equilibrium_target.motor_trim.detach()
    disturbance_accel_target = (
        state.external_force / state.mass[:, None].clamp_min(1.0e-12)
    ).detach()
    observation_state = initial_observation_state(batch, device=state.position.device, dtype=state.position.dtype)
    settings = dict(mode="integral25", integral_input_frame="body",
                    integral_input_multiplier=1.0, noise_max=0.0,
                    integral_limit=0.5, integral_leak=0.0,
                    integral_clamp_mode="box")
    settings.update(getattr(teacher, "q2_observation_settings", {}))
    if teacher_observation_settings is not None:
        settings.update(teacher_observation_settings)
    observation_options = {name: settings[name] for name in (
        "mode", "integral_input_frame", "integral_input_multiplier", "noise_max")}
    teacher_hidden = teacher.initial_hidden(batch, device=state.position.device, dtype=state.position.dtype)
    initial_obs, _ = build_policy_observation(state, observation_state, **observation_options)
    student_state: StructuredPolicyState = student.initial_state(initial_obs)
    observations, teacher_actions = [], []
    teacher_same_latent_intercepts, teacher_fast_delta_actions, student_actions = [], [], []
    teacher_hiddens, student_hiddens, executed_actions = [], [], []
    interventions = []
    finite_values, position_values, velocity_values, omega_values = [], [], [], []
    headroom_values, residual_values, action_rate_values = [], [], []
    identification_values = []
    identification_failure_values = []
    effectiveness_width_values = []
    capability_means, capability_scales = [], []
    trim_values, body_z_values, disturbance_values = [], [], []
    equilibrium_feasible_values, equilibrium_action_errors = [], []
    equilibrium_accel_values, equilibrium_omega_values = [], []
    generator = torch.Generator(device=state.position.device)
    generator.manual_seed(int(episode_seed))
    intervention = torch.rand(batch, device=state.position.device, generator=generator) < float(beta)
    previous_executed = state.previous_action
    teacher_probe_state = ProbeState.initial(previous_executed)
    executed_probe_rows, executed_abort_rows, observer_errors = [], [], []
    for step in range(horizon):
        observation, observed_position = build_policy_observation(
            state, observation_state, **observation_options
        )
        # Q2 owns its original observation integral, updated on the actual
        # trajectory. A student anti-windup state must not change its teacher.
        teacher_action, teacher_hidden_next, teacher_intercept = _q2_action_hidden_intercept(
            teacher, observation, teacher_hidden, state, simulator
        )
        observation, _ = _observation_from_student_integral(observation, student_state, observation_state)
        student_output = student.forward_with_aux(observation, student_state)
        student_action = student_output.action
        observed_motor = student_output.auxiliary.get("motor_estimate")
        if observed_motor is None:
            observer_errors.append(torch.full_like(state.motor[:, 0], float("inf")))
        else:
            observer_errors.append((observed_motor - state.motor).square().mean(-1).sqrt().detach())
        teacher_executed = teacher_action
        if teacher_probe:
            teacher_executed, teacher_probe_state, _ = apply_probe(
                teacher_action, teacher_probe_state, step, position=state.position,
                velocity=state.velocity, omega=state.omega, body_z=state.rotation[:, :, 2])
        executed = torch.where(intervention[:, None], teacher_executed, student_action)
        if hasattr(student_output.next_state, "probe_residual"):
            candidate_probe = student_output.next_state.probe_residual
            candidate_abort = student_output.next_state.probe_aborted
            executed_probe = torch.where(intervention[:, None], teacher_probe_state.residual, candidate_probe)
            executed_abort = torch.where(intervention, teacher_probe_state.aborted, candidate_abort)
        else:
            executed_probe = torch.zeros_like(executed[:, :1])
            executed_abort = torch.zeros(batch, dtype=torch.bool, device=executed.device)
        student_state_next = _advance_student_with_executed_action(
            student, observation, student_state, student_output, executed,
            dt=simulator.params.dt, executed_probe_residual=executed_probe,
            executed_probe_aborted=executed_abort,
        )
        executed_probe_rows.append(executed_probe.detach())
        executed_abort_rows.append(executed_abort.detach())
        observations.append(observation.detach())
        teacher_actions.append(teacher_action.detach())
        teacher_same_latent_intercepts.append(teacher_intercept.detach())
        teacher_fast_delta_actions.append(
            (teacher_action - motor_trim_target).detach()
        )
        student_actions.append(student_action.detach())
        teacher_hiddens.append(teacher_hidden.detach())
        student_hiddens.append(student_state.hidden.detach())
        executed_actions.append(executed.detach())
        interventions.append(intervention.detach())
        finite_values.append((torch.isfinite(state.position).all(-1) & torch.isfinite(state.velocity).all(-1) & torch.isfinite(state.omega).all(-1) & torch.isfinite(executed).all(-1)).detach())
        position_values.append(torch.linalg.vector_norm(state.position, dim=-1).detach())
        velocity_values.append(torch.linalg.vector_norm(state.velocity, dim=-1).detach())
        omega_values.append(torch.linalg.vector_norm(state.omega, dim=-1).detach())
        allocator = student_output.auxiliary.get("allocator")
        if allocator is None:
            headroom_values.append(torch.ones(batch, device=state.position.device))
            residual_values.append(torch.zeros(batch, device=state.position.device))
        else:
            headroom_values.append(allocator.minimum_headroom.detach())
            residual_values.append(allocator.wrench_residual.detach())
        action_rate_values.append(torch.linalg.vector_norm(executed - previous_executed, dim=-1).detach())
        identifier = student_output.auxiliary.get("identifier")
        if identifier is None:
            identifier = getattr(student_state, "identifier", None)
        if not torch.is_tensor(identifier):
            identifier = torch.zeros(batch, 1, device=state.position.device, dtype=state.position.dtype)
        identification_values.append(torch.linalg.vector_norm(identifier, dim=-1).detach())
        failed = student_output.auxiliary.get("t50_identification_failed")
        if failed is None:
            failed = student_output.auxiliary.get("identification_failed")
        if not torch.is_tensor(failed):
            failed = torch.zeros(batch, device=state.position.device, dtype=torch.bool)
        identification_failure_values.append(failed.detach().bool())
        width = student_output.auxiliary.get("effectiveness_log_interval_width")
        if not torch.is_tensor(width):
            width = torch.zeros(batch, 3, device=state.position.device, dtype=state.position.dtype)
        effectiveness_width_values.append(width.detach())
        previous_executed = executed
        contract = capability_contract(student_output.auxiliary)
        capability_means.append(contract.capability_z_mean.detach())
        capability_scales.append(contract.capability_z_log_scale.detach())
        predicted_trim = student_output.auxiliary.get("trim_action", student_action)
        predicted_body_z = student_output.auxiliary.get(
            "body_z", body_z_target
        )
        predicted_disturbance = student_output.auxiliary.get(
            "disturbance_accel",
            torch.zeros(batch, 3, device=state.position.device, dtype=state.position.dtype),
        )
        predicted_feasible = student_output.auxiliary.get(
            "equilibrium_feasible",
            torch.ones(batch, dtype=torch.bool, device=state.position.device),
        )
        trim_values.append(predicted_trim.detach())
        body_z_values.append(predicted_body_z.detach())
        disturbance_values.append(predicted_disturbance.detach())
        equilibrium_feasible_values.append(predicted_feasible.detach().bool())
        predicted_target = EquilibriumTarget(
            body_z=predicted_body_z,
            total_thrust=torch.zeros(batch, device=state.position.device,
                                     dtype=state.position.dtype),
            motor_trim=predicted_trim,
            feasible=predicted_feasible.bool(),
        )
        equilibrium_state = materialize_equilibrium_state(state, predicted_target)
        equilibrium_next = simulator.step(
            equilibrium_state, predicted_trim, grad_decay=1.0
        )
        equilibrium_accel_values.append(
            (torch.linalg.vector_norm(equilibrium_next.velocity, dim=-1)
             / (simulator.params.dt * simulator.params.gravity)).detach()
        )
        equilibrium_omega_values.append(
            (torch.linalg.vector_norm(equilibrium_next.omega, dim=-1) / 0.5).detach()
        )
        if isinstance(student_state, StructuredPolicyState):
            equilibrium_observation, _ = build_policy_observation(
                equilibrium_state,
                PolicyObservationState(torch.zeros_like(student_state.integral)),
                mode="integral25", integral_input_frame="body",
            )
            equilibrium_policy_state = replace(
                student_state_next,
                motor_estimate=predicted_trim,
                integral=torch.zeros_like(student_state.integral),
                slow_trim=predicted_trim,
                slow_body_z=predicted_body_z,
                capability=student_output.auxiliary.get(
                    "capability", student_state_next.capability
                ),
                disturbance_accel=predicted_disturbance,
                prev_velocity=torch.zeros_like(student_state.prev_velocity),
                prev_omega=torch.zeros_like(student_state.prev_omega),
                disturbance_residual_sum=torch.zeros_like(
                    student_state.disturbance_residual_sum
                ),
                disturbance_thrust_sum=torch.zeros_like(
                    student_state.disturbance_thrust_sum
                ),
                disturbance_response_count=torch.zeros_like(
                    student_state.disturbance_response_count
                ),
                previous_executed_action=predicted_trim,
                context_sum=torch.zeros_like(student_state.context_sum),
                slow_counter=0,
            )
            equilibrium_policy_output = student.forward_with_aux(
                equilibrium_observation, equilibrium_policy_state,
                applied_action=predicted_trim,
            )
            equilibrium_action_errors.append(
                torch.linalg.vector_norm(
                    equilibrium_policy_output.action - predicted_trim, dim=-1
                ).detach()
            )
        else:
            equilibrium_action_errors.append(
                torch.zeros(batch, device=state.position.device,
                            dtype=state.position.dtype)
            )
        observation_state = update_position_integral(
            observation_state, observed_position, dt=simulator.params.dt,
            integral_limit=settings["integral_limit"], integral_leak=settings["integral_leak"],
            integral_clamp_mode=settings["integral_clamp_mode"],
        )
        state = simulator.step(state, executed, grad_decay=1.0)
        teacher_hidden, student_state = teacher_hidden_next, student_state_next
    capability = normalize_log_capability(_raw_capability(bank.state))
    stacked_position = torch.stack(position_values)
    stacked_velocity = torch.stack(velocity_values)
    stacked_omega = torch.stack(omega_values)
    identification_norm = torch.stack(identification_values)
    # Episode rows are indexed by policy call: row 0 is call0, row 50 is
    # call50.  ``collect_dagger_episode`` requires that call to be present.
    t50_index = _call_index(identification_norm.shape[0], int(getattr(getattr(student, "config", None), "identification_publish_start", 100)))
    identification_t50 = identification_norm[t50_index]
    identification_failure_t50 = torch.stack(identification_failure_values)[t50_index]
    effectiveness_width_t50 = torch.stack(effectiveness_width_values)[t50_index]
    return DAggerEpisode(
        observations=torch.stack(observations), teacher_actions=torch.stack(teacher_actions),
        teacher_same_latent_intercepts=torch.stack(teacher_same_latent_intercepts),
        teacher_fast_delta_actions=torch.stack(teacher_fast_delta_actions),
        motor_trim_target=motor_trim_target,
        body_z_target=body_z_target,
        disturbance_accel_target=disturbance_accel_target,
        student_actions=torch.stack(student_actions), teacher_hidden=torch.stack(teacher_hiddens),
        student_hidden=torch.stack(student_hiddens), executed_actions=torch.stack(executed_actions),
        intervention_mask=torch.stack(interventions), finite=torch.stack(finite_values),
        position_norm=stacked_position, velocity_norm=stacked_velocity,
        omega_norm=stacked_omega, allocator_headroom=torch.stack(headroom_values),
        allocator_residual=torch.stack(residual_values), action_rate=torch.stack(action_rate_values),
        identification_norm_t50=identification_t50,
        effectiveness_log_interval_width_t50=effectiveness_width_t50,
        identification_failure_t50=identification_failure_t50,
        motor_observer_rms_error=torch.stack(observer_errors),
        capability_z_mean=torch.stack(capability_means),
        capability_z_log_scale=torch.stack(capability_scales),
        student_trim=torch.stack(trim_values),
        student_body_z=torch.stack(body_z_values),
        student_disturbance_accel=torch.stack(disturbance_values),
        equilibrium_feasible=torch.stack(equilibrium_feasible_values),
        equilibrium_action_error=torch.stack(equilibrium_action_errors),
        equilibrium_one_step_accel=torch.stack(equilibrium_accel_values),
        equilibrium_one_step_omega=torch.stack(equilibrium_omega_values),
        beta=float(beta), intervention=tuple("teacher" if bool(value) else "student" for value in intervention.tolist()), tw_bin=bank.tw_bin,
        log_alpha_bin=bank.log_alpha_bin, capability_target_z=capability,
        equilibrium_template=_clone_l2f(bank.state),
        simulator_dt=float(simulator.params.dt),
        simulator_gravity=float(simulator.params.gravity),
        executed_probe_residual=torch.stack(executed_probe_rows),
        executed_probe_aborted=torch.stack(executed_abort_rows),
    )


def _last_time_value(value: torch.Tensor) -> torch.Tensor:
    """Return a final-time [scenario,...] value from either [scenario,...] or
    [time,scenario,...] episode representations.

    The small adapter tests use both forms; accepting both keeps the conformal
    helper usable with serialized episode summaries without silently treating a
    scenario axis as time.
    """
    return value[-1] if value.ndim >= 3 else value


def _call_index(length: int, call_index: int) -> int:
    """Return an exact observation/call index from an episode row array."""

    if length < 1 or call_index < 0 or call_index >= length:
        raise ValueError(
            f"episode has no row for call index {call_index}; "
            "collect through that call under the current cadence contract"
        )
    return int(call_index)


def phase_a_equilibrium_gate(
    episode: DAggerEpisode,
    *,
    phase_steps: tuple[int, ...] = (100, 125),
    trim_rms_max: float = 1.3e-3,
    trim_absolute_max: float = 5.0e-3,
    body_z_p99_degrees: float = 1.0,
    body_z_max_degrees: float = 3.0,
    disturbance_rms_over_g: float = 0.02,
    disturbance_p99_over_g: float = 0.05,
    one_step_accel_rms_over_g: float = 0.02,
    one_step_omega_rms_over_scale: float = 0.02,
    equilibrium_action_error_max: float = 1.0e-6,
    capability_z_rms_max: float = 0.20,
    effectiveness_z_rms_max: float = 0.15,
    capability_axis_z_rms_max: float = 0.25,
    require_identification_width: bool = True,
    motor_observer_p95_max: float = 1e-3,
    motor_observer_absolute_max: float = 5e-3,
) -> tuple[dict, bool]:
    """Strict held-out gate for the Phase-A physical equilibrium estimator."""

    if not phase_steps or any(int(step) <= 0 for step in phase_steps):
        raise ValueError("phase steps must be positive call indices")
    rows = []
    identification_failure_count = int(
        episode.identification_failure_t50.to(torch.bool).sum().item()
    )
    all_passed = identification_failure_count == 0 or not require_identification_width
    true_trim = episode.motor_trim_target
    true_body_z = F.normalize(episode.body_z_target, dim=-1)
    true_disturbance = episode.disturbance_accel_target
    for phase_step in phase_steps:
        index = _call_index(episode.student_trim.shape[0], int(phase_step))
        predicted_trim = episode.student_trim[index]
        trim_error = predicted_trim - true_trim
        trim_rms = float(trim_error.square().mean().sqrt())
        trim_max = float(trim_error.abs().amax())
        predicted_body_z = F.normalize(episode.student_body_z[index], dim=-1)
        cosine = (predicted_body_z * true_body_z).sum(dim=-1).clamp(-1.0, 1.0)
        angles = torch.rad2deg(torch.acos(cosine))
        angle_p99 = float(torch.quantile(angles, 0.99))
        angle_max = float(angles.max())
        disturbance_error = torch.linalg.vector_norm(
            episode.student_disturbance_accel[index] - true_disturbance, dim=-1
        ) / 9.80665
        disturbance_rms = float(disturbance_error.square().mean().sqrt())
        disturbance_p99 = float(torch.quantile(disturbance_error, 0.99))
        accel_rms = float(
            episode.equilibrium_one_step_accel[index].square().mean().sqrt()
        )
        omega_rms = float(
            episode.equilibrium_one_step_omega[index].square().mean().sqrt()
        )
        action_error = float(episode.equilibrium_action_error[index].max())
        feasible_fraction = float(episode.equilibrium_feasible[index].float().mean())
        # Missing mean/observer evidence must not silently become zero error.
        motor_errors = getattr(episode, "motor_observer_rms_error", None)
        motor_p95 = float(motor_errors[index].quantile(.95)) if motor_errors is not None else None
        motor_max = float(motor_errors[index].max()) if motor_errors is not None else None
        observer_passed = (motor_p95 is not None and math.isfinite(motor_p95)
                           and motor_max is not None and math.isfinite(motor_max)
                           and motor_p95 <= motor_observer_p95_max and motor_max <= motor_observer_absolute_max)
        capability_mean = getattr(episode, "capability_z_mean", None)
        capability_target = getattr(episode, "capability_target_z", None)
        capability_log_scale = getattr(episode, "capability_z_log_scale", None)
        capability_evidence_present = capability_mean is not None and capability_target is not None and capability_log_scale is not None
        if not capability_evidence_present:
            capability_error = trim_error.new_zeros(trim_error.shape[0], 6)
            capability_scale = trim_error.new_ones(trim_error.shape[0], 6)
        else:
            capability_error = capability_mean[index] - capability_target
            capability_scale = capability_log_scale[index].exp()
        capability_z_rms = float(capability_error.square().mean().sqrt())
        effectiveness_z_rms = float(
            capability_error[:, :3].square().mean().sqrt()
        )
        capability_axis_z_rms = capability_error.square().mean(dim=0).sqrt()
        passed = bool(
            observer_passed and capability_evidence_present
            and trim_rms <= trim_rms_max
            and trim_max <= trim_absolute_max
            and angle_p99 <= body_z_p99_degrees
            and angle_max <= body_z_max_degrees
            and disturbance_rms <= disturbance_rms_over_g
            and disturbance_p99 <= disturbance_p99_over_g
            and accel_rms <= one_step_accel_rms_over_g
            and omega_rms <= one_step_omega_rms_over_scale
            and action_error <= equilibrium_action_error_max
            and feasible_fraction == 1.0
            and capability_z_rms <= capability_z_rms_max
            and effectiveness_z_rms <= effectiveness_z_rms_max
            and float(capability_axis_z_rms.max()) <= capability_axis_z_rms_max
        )
        rows.append({
            "phase_step": int(phase_step),
            "motor_observer_p95": motor_p95, "motor_observer_max": motor_max,
            "motor_observer_passed": observer_passed,
            "capability_evidence_present": capability_evidence_present,
            "trim_rms": trim_rms,
            "trim_max": trim_max,
            "body_z_angle_p99_degrees": angle_p99,
            "body_z_angle_max_degrees": angle_max,
            "disturbance_error_rms_over_g": disturbance_rms,
            "disturbance_error_p99_over_g": disturbance_p99,
            "one_step_accel_rms_over_g": accel_rms,
            "one_step_omega_rms_over_scale": omega_rms,
            "equilibrium_action_error_max": action_error,
            "predicted_equilibrium_feasible_fraction": feasible_fraction,
            "capability_z_rms": capability_z_rms,
            "effectiveness_z_rms": effectiveness_z_rms,
            "capability_axis_z_rms": [float(value) for value in capability_axis_z_rms],
            "effectiveness_sigma_mean": float(capability_scale[:, :3].mean()),
            "effectiveness_sigma_p99": float(
                torch.quantile(capability_scale[:, :3].reshape(-1), 0.99)
            ),
            "passed": passed,
        })
        all_passed = all_passed and passed
    report = {
        "phase_steps": list(phase_steps),
        "rows": rows,
        "thresholds": {
            "motor_observer_p95": motor_observer_p95_max,
            "motor_observer_max": motor_observer_absolute_max,
            "trim_rms": trim_rms_max,
            "trim_max": trim_absolute_max,
            "body_z_p99_degrees": body_z_p99_degrees,
            "body_z_max_degrees": body_z_max_degrees,
            "disturbance_rms_over_g": disturbance_rms_over_g,
            "disturbance_p99_over_g": disturbance_p99_over_g,
            "one_step_accel_rms_over_g": one_step_accel_rms_over_g,
            "one_step_omega_rms_over_scale": one_step_omega_rms_over_scale,
            "equilibrium_action_error_max": equilibrium_action_error_max,
            "capability_z_rms": capability_z_rms_max,
            "effectiveness_z_rms": effectiveness_z_rms_max,
            "capability_axis_z_rms": capability_axis_z_rms_max,
        },
        "structural_no_learned_trim_or_body_z_head": True,
        "identification_failure_t50_count": identification_failure_count,
        "identification_failure_t50_required": 0,
        "identification_width_required": bool(require_identification_width),
        "gate_passed": bool(all_passed),
    }
    return report, bool(all_passed)


def _phase_prefixes(phase: str) -> tuple[str, ...]:
    if phase not in ("A", "A1", "A2", "B", "C"):
        raise ValueError("phase must be A1, A2, B, or C")
    # MotorObserver and AntiWindupIntegral currently have no learned tensors,
    # but naming them here documents the staged contract and future-proofs the
    # freeze gate if their constants become learned calibration parameters.
    return {
        # ``A`` remains a mean-only compatibility alias.  Mean identification
        # and uncertainty fitting must be separate optimizations: allowing the
        # scale NLL to reshape the identifier lets a broad posterior substitute
        # for a scenario-dependent mean.
        "A": ("identifier.", "motor_observer.", "integrator.",
              "capability_head.", "bank_adapter."),
        "A1": ("identifier.", "motor_observer.", "integrator.",
               "capability_head.", "bank_adapter."),
        "A2": ("capability_log_scale_head.",),
        "B": ("contextual_gain_head.",),
        "C": ("encoder.", "gru.", "residual_head."),
    }[phase]


def distillation_phase_components(phase: str) -> tuple[str, ...]:
    """Human-readable staged components, including non-parametric interfaces."""
    if phase not in ("A", "A1", "A2", "B", "C"):
        raise ValueError("phase must be A1, A2, B, or C")
    return {
        "A": ("identifier", "capability", "explicit_disturbance_observer",
              "analytic_equilibrium", "observer"),
        "A1": ("identifier", "capability_mean", "explicit_disturbance_observer",
               "analytic_equilibrium", "observer"),
        "A2": ("capability_uncertainty",),
        "B": ("contextual_gain",),
        "C": ("encoder", "gru", "residual"),
    }[phase]


def capability_contract(auxiliary: dict[str, torch.Tensor], *, ucb_z: float = 1.645) -> CapabilityContract:
    """Normalize the temporary capability uncertainty/context contract."""

    capability = auxiliary.get("capability")
    if capability is None:
        raise KeyError("student auxiliary output must contain capability")
    mean = auxiliary.get("capability_z_mean", normalize_log_capability(capability))
    scale = auxiliary.get(
        "capability_z_log_scale",
        auxiliary.get(
            "capability_log_scale",
            -1.5 + 0.1 * torch.tanh(mean.detach()),
        ),
    )
    ucb = auxiliary.get("capability_ucb", mean + float(ucb_z) * scale.exp())
    gain = auxiliary.get(
        "contextual_gain",
        auxiliary.get("fast_feedback", mean.new_zeros(mean.shape[:-1] + (4, 15))),
    )
    return CapabilityContract(mean, scale, ucb, gain)


def set_distillation_phase(student: torch.nn.Module, phase: str) -> tuple[str, ...]:
    """Freeze the structured policy outside the requested staged interface."""

    for parameter in student.parameters():
        parameter.requires_grad_(False)
    prefixes = _phase_prefixes(phase)
    active = []
    for name, parameter in student.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(True)
            active.append(name)
    return tuple(active)


def capability_heteroscedastic_nll(
    log_mean: torch.Tensor,
    log_scale: torch.Tensor,
    target_log: torch.Tensor,
    *,
    minimum_scale: float = -5.0,
    maximum_scale: float = math.log(2.0),
) -> torch.Tensor:
    if log_mean.shape != log_scale.shape or log_mean.shape != target_log.shape:
        raise ValueError("capability tensors must have the same shape")
    scale = log_scale.clamp(float(minimum_scale), float(maximum_scale))
    standardized = (target_log - log_mean) * torch.exp(-scale)
    return 0.5 * (standardized.square() + 2.0 * scale).mean()


def one_sided_conformal_quantile(scores: torch.Tensor, *, miscoverage: float = 0.1) -> torch.Tensor:
    """Finite-sample one-sided conformal quantile (calibration split only)."""

    if scores.numel() < 1 or not 0.0 < miscoverage < 1.0:
        raise ValueError("scores must be non-empty and miscoverage in (0,1)")
    flat = scores.detach().reshape(-1).sort().values
    rank = min(flat.numel() - 1, int(torch.ceil(torch.tensor((flat.numel() + 1) * (1.0 - miscoverage))).item()) - 1)
    return flat[rank]


def one_sided_conformal_coverage(
    log_mean: torch.Tensor, log_scale: torch.Tensor, target_log: torch.Tensor, q: torch.Tensor,
) -> torch.Tensor:
    """Validation coverage under a calibration-only one-sided UCB threshold."""

    if log_mean.shape != log_scale.shape or log_mean.shape != target_log.shape:
        raise ValueError("conformal tensors must have matching shapes")
    upper = log_mean + log_scale.exp() * q
    return (target_log <= upper).all(dim=-1).to(log_mean.dtype).mean()


def fit_effectiveness_conformal_q(
    calibration_episode: DAggerEpisode,
    *,
    validation_episode: Optional[DAggerEpisode] = None,
    miscoverage: float = 0.01,
    phase_steps: tuple[int, ...] = (100, 125),
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Fit one conservative joint UCB score over four roll-authority strata.

    Each scenario contributes one score: the maximum standardized upper error
    over all requested phases and all three monotone effectiveness dimensions.
    A quantile is fit inside each predeclared log-roll-authority stratum and the
    deployed multiplier is the maximum of the four quantiles.  Thus the phase
    checks are joint rather than three separately calibrated 99% statements.
    """

    if not phase_steps:
        raise ValueError("phase_steps must be non-empty")
    if any(int(step) <= 0 for step in phase_steps):
        raise ValueError("phase steps must be positive call indices")
    validation = calibration_episode if validation_episode is None else validation_episode
    rows: list[dict[str, object]] = []
    q_values = []
    calibration_indices = [
        _call_index(calibration_episode.capability_z_mean.shape[0], int(step))
        for step in phase_steps
    ]
    validation_indices = [
        _call_index(validation.capability_z_mean.shape[0], int(step))
        for step in phase_steps
    ]
    for risk_stratum in range(4):
        cal_mask = calibration_episode.log_alpha_bin == risk_stratum
        val_mask = validation.log_alpha_bin == risk_stratum
        calibration_mean = calibration_episode.capability_z_mean[
            calibration_indices, :, :3
        ][:, cal_mask]
        calibration_sigma = calibration_episode.capability_z_log_scale[
            calibration_indices, :, :3
        ][:, cal_mask].exp()
        calibration_target = calibration_episode.capability_target_z[
            cal_mask, :3
        ].unsqueeze(0)
        joint_score = (
            (calibration_target - calibration_mean)
            / calibration_sigma.clamp_min(1.0e-8)
        ).amax(dim=(0, 2))
        q = one_sided_conformal_quantile(
            joint_score, miscoverage=miscoverage
        ).clamp_min(0.0)
        validation_mean = validation.capability_z_mean[
            validation_indices, :, :3
        ][:, val_mask]
        validation_sigma = validation.capability_z_log_scale[
            validation_indices, :, :3
        ][:, val_mask].exp()
        validation_target = validation.capability_target_z[
            val_mask, :3
        ].unsqueeze(0)
        covered = (
            validation_target <= validation_mean + q * validation_sigma
        ).all(dim=(0, 2)).to(validation_mean.dtype)
        rows.append(
            {
                "phase_steps": [int(step) for step in phase_steps],
                "risk_stratum": int(risk_stratum),
                "calibration_samples": int(cal_mask.sum()),
                "validation_samples": int(val_mask.sum()),
                "q": float(q),
                "validation_joint_phase_dimension_coverage": float(covered.mean()),
                "validation_coverage": float(covered.mean()),
                "validation_interpretation": "shift diagnostic only; not coverage proof",
            }
        )
        q_values.append(q)
    deployed = torch.stack(q_values).max()
    q_vector = calibration_episode.capability_z_mean.new_zeros(6)
    q_vector[:3] = deployed
    return q_vector, rows


def conformal_coverage_by_stratum(
    episode: DAggerEpisode, *, validation_episode: Optional[DAggerEpisode] = None,
    miscoverage: float = 0.1,
) -> list[dict[str, float | int | str]]:
    """Calibrate/validate UCB separately in each 4x4 authority cell.

    Passing ``validation_episode`` keeps calibration and validation banks
    independent.  Omitting it retains the small smoke-test split semantics.
    """

    rows = []
    target = _last_time_value(episode.capability_target_z)
    mean = _last_time_value(episode.capability_z_mean)
    scale = _last_time_value(episode.capability_z_log_scale)
    validation_source = episode if validation_episode is None else validation_episode
    validation_target = _last_time_value(validation_source.capability_target_z)
    validation_mean = _last_time_value(validation_source.capability_z_mean)
    validation_scale = _last_time_value(validation_source.capability_z_log_scale)
    for tw in range(4):
        for alpha in range(4):
            calibration = torch.nonzero(
                (episode.tw_bin == tw) & (episode.log_alpha_bin == alpha), as_tuple=False
            ).flatten()
            if validation_episode is None:
                midpoint = max(1, calibration.numel() // 2)
                calibration, validation = calibration[:midpoint], calibration[midpoint:]
                validation_target_cell = validation_target[validation]
                validation_mean_cell = validation_mean[validation]
                validation_scale_cell = validation_scale[validation]
            else:
                validation = torch.nonzero(
                    (validation_source.tw_bin == tw)
                    & (validation_source.log_alpha_bin == alpha), as_tuple=False
                ).flatten()
                validation_target_cell = validation_target[validation]
                validation_mean_cell = validation_mean[validation]
                validation_scale_cell = validation_scale[validation]
            scores = ((target[calibration] - mean[calibration]) / scale[calibration].exp()).amax(dim=-1)
            q = one_sided_conformal_quantile(scores, miscoverage=miscoverage)
            coverage = one_sided_conformal_coverage(
                validation_mean_cell, validation_scale_cell, validation_target_cell, q
            )
            rows.append({
                "tw_bin": tw, "log_alpha_bin": alpha,
                "calibration_effective_samples": int(calibration.numel()),
                "validation_effective_samples": int(validation.numel()),
                "conformal_q": float(q), "validation_coverage": float(coverage),
            })
    return rows


def dagger_window_loss(
    student: StructuredRecurrentPolicy,
    episode: DAggerEpisode,
    *,
    prefix: int = 25,
    action_weight: float = 1.0,
    capability_weight: float = 1.0,
    capability_mean_weight: float = 1.0,
    delta_action_weight: float = 0.1,
    phase: str = "A",
    full_action_weight: float = 0.05,
    body_z_weight: float = 0.25,
    disturbance_weight: float = 1.0,
    one_step_equilibrium_weight: float = 0.25,
    motor_observer_weight: float = 1000.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reconstruct recurrent state then use phase-specific analytic targets.

    The Q2 same-latent intercept is recorded for diagnosis only.  Phase A fits
    analytic motor trim/body-z/capability; phases B/C fit fast action relative
    to analytic motor trim.
    """

    if prefix < 1 or prefix >= episode.observations.shape[0]:
        raise ValueError("prefix must be inside episode")
    if phase not in ("A", "A1", "A2", "C"):
        if phase == "B":
            raise ValueError(
                "phase B requires the analytic-equilibrium local-JVP trainer; "
                "full-trajectory DAgger must not train the contextual gain"
            )
        raise ValueError("phase must be A1, A2, B, or C")
    phase_a = phase in ("A", "A1", "A2")
    state = student.initial_state(episode.observations[0])
    action_loss = episode.observations.sum() * 0.0
    capability_loss = action_loss
    capability_mean_loss = action_loss
    delta_action_loss = action_loss
    full_action_loss = action_loss
    body_z_loss = action_loss
    disturbance_loss = action_loss
    one_step_equilibrium_loss = action_loss
    motor_observer_loss = action_loss
    # Simulator motor truth is a detached training target only. It never
    # enters the observation, identifier context, or production observer.
    target_motor = episode.equilibrium_template.motor.detach()
    true_rise = episode.equilibrium_template.motor_time_rising[:, None].detach()
    true_fall = episode.equilibrium_template.motor_time_falling[:, None].detach()
    equilibrium_simulator = L2FSimulator(L2FParams(
        dt=float(episode.simulator_dt), gravity=float(episode.simulator_gravity)
    ))
    training_count = 0
    full_action_count = 0
    slow_cadence = int(getattr(getattr(student, "config", None), "slow_cadence", 25))
    publish_start = int(getattr(
        getattr(student, "config", None), "identification_publish_start", 100
    ))
    for step in range(episode.observations.shape[0]):
        if step:
            command = episode.executed_actions[step - 1].detach()
            tau = torch.where(command >= target_motor, true_rise, true_fall)
            target_motor = target_motor + (episode.simulator_dt / tau).clamp_max(1.0) * (command - target_motor)
        try:
            output = student.forward_with_aux(
                episode.observations[step], state,
                applied_action=episode.executed_actions[step],
                applied_probe_residual=(None if episode.executed_probe_residual is None else episode.executed_probe_residual[step]),
                applied_probe_aborted=(None if episode.executed_probe_aborted is None else episode.executed_probe_aborted[step]),
            )
        except TypeError:
            output = student.forward_with_aux(episode.observations[step], state)
        state = output.next_state
        if phase == "C" and step < prefix:
            if step + 1 == prefix:
                state = state.detach()
            continue
        full_action_loss = full_action_loss + F.smooth_l1_loss(
            output.action, episode.teacher_actions[step]
        )
        full_action_count += 1
        # t25 updates recurrent memory but is explicitly unavailable as a
        # capability output.  Supervision starts at call50 and then applies
        # only at actual sample-and-hold publications.
        publication_step = (
            step >= publish_start
            and (step - publish_start) % slow_cadence == 0
        )
        if phase_a and not publication_step:
            continue
        training_count += 1
        if phase in ("A", "A1") and isinstance(student, StructuredRecurrentPolicy):
            motor_observer_loss = motor_observer_loss + F.mse_loss(
                output.auxiliary["motor_estimate"], target_motor
            )
        trim_action = output.auxiliary.get("trim_action")
        if trim_action is None:
            trim_action = output.action
        if phase_a:
            action_loss = action_loss + F.smooth_l1_loss(
                trim_action, episode.motor_trim_target
            )
            body_z = output.auxiliary.get("body_z")
            if body_z is None:
                body_z = episode.body_z_target
            predicted = F.normalize(body_z, dim=-1)
            target = F.normalize(episode.body_z_target, dim=-1)
            body_z_loss = body_z_loss + (
                1.0 - (predicted * target).sum(dim=-1)
            ).clamp_min(0.0).mean()
            disturbance = output.auxiliary.get("disturbance_accel")
            if disturbance is None:
                raise KeyError("Phase A requires explicit disturbance_accel")
            disturbance_loss = disturbance_loss + F.smooth_l1_loss(
                disturbance / 9.80665,
                episode.disturbance_accel_target / 9.80665,
            )
            feasible = output.auxiliary.get(
                "equilibrium_feasible",
                torch.ones(trim_action.shape[0], dtype=torch.bool,
                           device=trim_action.device),
            )
            predicted_target = EquilibriumTarget(
                body_z=body_z,
                total_thrust=torch.zeros(trim_action.shape[0], device=trim_action.device,
                                         dtype=trim_action.dtype),
                motor_trim=trim_action,
                feasible=feasible.bool(),
            )
            equilibrium_state = materialize_equilibrium_state(
                episode.equilibrium_template, predicted_target
            )
            equilibrium_next = equilibrium_simulator.step(
                equilibrium_state, trim_action, grad_decay=1.0
            )
            one_step_equilibrium_loss = one_step_equilibrium_loss + (
                (equilibrium_next.velocity / (
                    episode.simulator_dt * episode.simulator_gravity
                )).square().sum(dim=-1).mean()
                + (equilibrium_next.omega / 0.5).square().sum(dim=-1).mean()
            )
        else:  # Phase C: diagnostic decomposition; total uses full action.
            delta_action_loss = delta_action_loss + F.smooth_l1_loss(
                output.action - trim_action, episode.teacher_fast_delta_actions[step]
            )
        contract = capability_contract(output.auxiliary)
        target = episode.capability_target_z.expand_as(contract.capability_z_mean)
        capability_loss = capability_loss + capability_heteroscedastic_nll(
            contract.capability_z_mean, contract.capability_z_log_scale, target
        )
        # Mean accuracy is an explicit identification objective.  Relying on
        # the heteroscedastic NLL alone lets an uninformative mean minimize the
        # loss by reporting sigma equal to its population error, which is
        # calibrated uncertainty but not a learned per-scenario capability.
        capability_mean_loss = capability_mean_loss + F.smooth_l1_loss(
            contract.capability_z_mean, target
        )
        if phase_a and publication_step:
            # The first graph spans call0..call50.  Later graphs begin from the
            # numerically persistent but detached prior publication state.
            state = state.detach()
        elif phase == "C" and (step - prefix + 1) % 25 == 0:
            state = state.detach()
    if training_count < 1 or full_action_count < 1:
        raise ValueError("episode does not contain a supervised policy window")
    count = float(training_count)
    action_loss, capability_loss = action_loss / count, capability_loss / count
    capability_mean_loss = capability_mean_loss / count
    full_action_loss = full_action_loss / float(full_action_count)
    delta_action_loss = delta_action_loss / count
    body_z_loss = body_z_loss / count
    disturbance_loss = disturbance_loss / count
    one_step_equilibrium_loss = one_step_equilibrium_loss / count
    motor_observer_loss = motor_observer_loss / count
    effective_full_action_weight = 0.0 if phase_a else float(action_weight)
    total = (float(action_weight) * action_loss
             + float(capability_weight) * capability_loss
             + float(capability_mean_weight) * capability_mean_loss
             + (0.0 if phase == "C" else float(delta_action_weight)) * delta_action_loss
             + (float(body_z_weight) * body_z_loss if phase_a else 0.0)
             + (float(disturbance_weight) * disturbance_loss if phase_a else 0.0)
             + (float(one_step_equilibrium_weight) * one_step_equilibrium_loss
                if phase_a else 0.0)
             + effective_full_action_weight * full_action_loss)
    if phase in ("A", "A1"):
        total = total + float(motor_observer_weight) * motor_observer_loss
    return total, {
        "action": float(action_loss.detach()),
        "analytic_trim": float(action_loss.detach()) if phase_a else 0.0,
        "same_latent_intercept_diagnostic": float(
            F.smooth_l1_loss(
                episode.teacher_same_latent_intercepts[prefix:],
                episode.teacher_actions[prefix:],
            ).detach()
        ),
        "full_action": float(full_action_loss.detach()),
        "capability_nll": float(capability_loss.detach()),
        "capability_mean": float(capability_mean_loss.detach()),
        "delta_action": float(delta_action_loss.detach()),
        "body_z": float(body_z_loss.detach()),
        "disturbance": float(disturbance_loss.detach()),
        "one_step_equilibrium": float(one_step_equilibrium_loss.detach()),
        "motor_observer": float(motor_observer_loss.detach()),
        "beta": episode.beta,
    }


def layered_dagger_gate(
    episodes: list[DAggerEpisode],
    *,
    action_rms_threshold: float = 1.3e-3,
    omega_max: float = 5.0,
    action_rate_max: float = 2.0,
) -> tuple[list[dict[str, float | int | str]], bool]:
    """Return per-cell action gates and intervention provenance."""

    rows = []
    passed = True
    for episode in episodes:
        error = (episode.student_actions - episode.teacher_actions).square().mean(dim=(0, 2)).sqrt()
        for index, value in enumerate(error.tolist()):
            finite = bool(episode.finite[:, index].all())
            max_position = float(episode.max_position_norm[index])
            max_velocity = float(episode.max_velocity_norm[index])
            max_omega = float(episode.max_omega_norm[index])
            max_action_rate = float(episode.action_rate[:, index].max())
            teacher_steps = int(episode.intervention_mask[:, index].sum())
            width_finite = bool(torch.isfinite(episode.effectiveness_log_interval_width_t50[index]).all())
            row = {
                "scenario_index": index,
                "tw_bin": int(episode.tw_bin[index]),
                "log_alpha_bin": int(episode.log_alpha_bin[index]),
                "authority_stratum": f"tw{int(episode.tw_bin[index])}_logalpha{int(episode.log_alpha_bin[index])}",
                "beta": episode.beta,
                "intervention": episode.intervention[index],
                "teacher_execution_steps": teacher_steps,
                "action_rms": float(value),
                "finite": int(finite), "max_position": max_position,
                "max_velocity": max_velocity, "max_omega": max_omega,
                "max_action_rate": max_action_rate,
                "identification_norm_t50": float(episode.identification_norm_t50[index]),
                "effectiveness_log_interval_width_t50": [
                    float(item) for item in episode.effectiveness_log_interval_width_t50[index]
                ],
                "identification_failure_t50": int(episode.identification_failure_t50[index]),
                "gate_passed": int(
                    finite and width_finite and not bool(episode.identification_failure_t50[index])
                    and value <= action_rms_threshold and max_omega <= omega_max
                    and max_action_rate <= action_rate_max
                ),
            }
            rows.append(row)
            passed = passed and bool(row["gate_passed"])
    return rows, passed


def _student_action(student_output: object) -> torch.Tensor:
    if torch.is_tensor(student_output):
        return student_output
    if hasattr(student_output, "action"):
        return student_output.action
    if isinstance(student_output, (tuple, list)) and student_output:
        return student_output[0]
    raise TypeError("student output must be a tensor, tuple, or policy output")


def _teacher_action(teacher: Callable, observation: torch.Tensor,
                    hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
    result = teacher(observation) if hidden is None else teacher(observation, hidden)
    if isinstance(result, (tuple, list)):
        return result[0]
    return result


def _teacher_action_hidden(teacher: Callable, observation: torch.Tensor,
                           hidden: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    result = teacher(observation) if hidden is None else teacher(observation, hidden)
    if isinstance(result, (tuple, list)):
        return result[0], (result[1] if len(result) > 1 else hidden)
    return result, hidden


def teacher_action_jacobian(teacher: Callable, observation: torch.Tensor,
                            hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return per-sample ``d action / d observation`` without cross-sample terms."""
    if observation.ndim != 2:
        raise ValueError("observation must be [batch,features]")
    rows = []
    for index in range(observation.shape[0]):
        sample = observation[index:index + 1].detach().requires_grad_(True)
        hidden_sample = None if hidden is None else hidden[index:index + 1].detach()

        def fn(value: torch.Tensor) -> torch.Tensor:
            return _teacher_action(teacher, value, hidden_sample).squeeze(0)

        rows.append(torch.autograd.functional.jacobian(fn, sample, create_graph=False).squeeze(1))
    return torch.stack(rows, dim=0)


def student_action_jacobian(student: Callable, observation: torch.Tensor,
                            state: Optional[object] = None) -> torch.Tensor:
    """Per-sample Jacobian helper for a structured student policy."""
    if observation.ndim != 2:
        raise ValueError("observation must be [batch,features]")
    rows = []
    for index in range(observation.shape[0]):
        sample = observation[index:index + 1].detach().requires_grad_(True)
        sample_state = None
        if state is not None:
            sample_state = state
            if hasattr(state, "detach"):
                sample_state = state.detach()
                for name in vars(sample_state):
                    value = getattr(sample_state, name, None)
                    if torch.is_tensor(value) and value.shape[0] == observation.shape[0]:
                        setattr(sample_state, name, value[index:index + 1])

        def fn(value: torch.Tensor) -> torch.Tensor:
            result = student(value, sample_state)
            return _student_action(result).squeeze(0)

        rows.append(torch.autograd.functional.jacobian(fn, sample, create_graph=False).squeeze(1))
    return torch.stack(rows, dim=0)


def collect_teacher_targets(teacher: Callable, observations: torch.Tensor,
                            hidden: Optional[torch.Tensor] = None,
                            *, action_jacobian: bool = False) -> DistillationTargets:
    with torch.no_grad():
        action = _teacher_action(teacher, observations, hidden).detach()
    jacobian = teacher_action_jacobian(teacher, observations, hidden) if action_jacobian else None
    return DistillationTargets(action=action, action_jacobian=jacobian)


def action_distillation_loss(student_action: torch.Tensor,
                             teacher_action: torch.Tensor) -> torch.Tensor:
    if student_action.shape != teacher_action.shape:
        raise ValueError("student and teacher actions must have the same shape")
    return F.smooth_l1_loss(student_action, teacher_action)


def jacobian_distillation_loss(student_jacobian: torch.Tensor,
                               teacher_jacobian: torch.Tensor) -> torch.Tensor:
    if student_jacobian.shape != teacher_jacobian.shape:
        raise ValueError("student and teacher Jacobians must have the same shape")
    return F.smooth_l1_loss(student_jacobian, teacher_jacobian)


def one_step_distillation_loss(student_next: torch.Tensor,
                               teacher_next: torch.Tensor) -> torch.Tensor:
    if student_next.shape != teacher_next.shape:
        raise ValueError("one-step states must have the same shape")
    return F.smooth_l1_loss(student_next, teacher_next)


def rollout_distillation_loss(student_rollout: torch.Tensor,
                              teacher_rollout: torch.Tensor) -> torch.Tensor:
    if student_rollout.shape != teacher_rollout.shape:
        raise ValueError("rollouts must have the same shape")
    return F.smooth_l1_loss(student_rollout, teacher_rollout)


def structured_distillation_loss(
    student_output: object,
    teacher_targets: DistillationTargets,
    *,
    student_jacobian: Optional[torch.Tensor] = None,
    student_one_step: Optional[torch.Tensor] = None,
    student_rollout: Optional[torch.Tensor] = None,
    action_weight: float = 1.0,
    jacobian_weight: float = 0.25,
    one_step_weight: float = 0.5,
    rollout_weight: float = 0.5,
) -> DistillationLoss:
    """Combine action, local Jacobian, one-step, and rollout supervision."""
    student_action = _student_action(student_output)
    zero = student_action.sum() * 0.0
    action = action_distillation_loss(student_action, teacher_targets.action)
    jacobian = zero
    if student_jacobian is not None and teacher_targets.action_jacobian is not None:
        jacobian = jacobian_distillation_loss(student_jacobian, teacher_targets.action_jacobian)
    one_step = zero
    if student_one_step is not None and teacher_targets.one_step_state is not None:
        one_step = one_step_distillation_loss(student_one_step, teacher_targets.one_step_state)
    rollout = zero
    if student_rollout is not None and teacher_targets.rollout is not None:
        rollout = rollout_distillation_loss(student_rollout, teacher_targets.rollout)
    total = (float(action_weight) * action + float(jacobian_weight) * jacobian
             + float(one_step_weight) * one_step + float(rollout_weight) * rollout)
    return DistillationLoss(total, action, jacobian, one_step, rollout)


def build_one_step_target(teacher: Callable, observation: torch.Tensor,
                          step_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                          hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Generate teacher one-step state targets with a caller-supplied simulator."""
    action = _teacher_action(teacher, observation, hidden)
    return step_fn(observation, action).detach()


def build_rollout_target(teacher: Callable, observations: torch.Tensor,
                         step_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                         *, horizon: Optional[int] = None,
                         hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Roll a teacher over an observation/state sequence using ``step_fn``."""
    if observations.ndim == 2:
        observations = observations.unsqueeze(1)
    if observations.ndim != 3:
        raise ValueError("observations must have shape [time,batch,features]")
    count = observations.shape[0] if horizon is None else min(int(horizon), observations.shape[0])
    current = observations[0]
    states = [current]
    for index in range(count):
        action, hidden = _teacher_action_hidden(teacher, observations[index], hidden)
        current = step_fn(current, action)
        states.append(current)
    return torch.stack(states)

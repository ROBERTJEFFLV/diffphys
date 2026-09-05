"""Local, cap-only contextual-gain distillation prototype.

This module deliberately stays separate from the recurrent DAgger trainer.  It
fits only ``contextual_gain_head`` against directional derivatives measured on
the same Q2 hidden state and leaves ``K_ref`` and the residual head untouched.
The returned diagnostics are useful for a Phase-B smoke check; they are not a
migration certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import torch
from torch.nn import functional as F

from equilibrium_control import analytic_equilibrium_target, rotation_from_body_z
from env_l2f import L2FParams, L2FSimulator, L2FState
from policy_observation import (
    build_policy_observation,
    initial_observation_state,
    update_position_integral,
)
from structured_distillation import (
    DAggerScenarioBank,
    _clone_l2f,
    build_dagger_scenario_bank,
)
from structured_policy import StructuredPolicyState, StructuredRecurrentPolicy, effective_wrench_mixer


LOCAL_RADII = (0.1, 0.05, 0.025)
EFFECTIVENESS_DIMS = 3
ERROR_DIM = 15
# Phase B may use the looser pre-calibration surrogate threshold only as a
# trainability screen.  The final, calibrated deployment finite-difference
# check uses the stricter 20% normalized-JVP threshold.
LOCAL_ACTION_PARITY_ANNULUS = 0.05
LOCAL_ACTION_PARITY_THRESHOLD = 0.20
LOCAL_JVP_P95_THRESHOLD = 0.20
LOCAL_PRECAL_SURROGATE_P95_THRESHOLD = 0.50
LOCAL_TAYLOR_R2_THRESHOLD = 0.10
LOCAL_ALLOCATOR_RESIDUAL_THRESHOLD = 1.0


@dataclass
class LocalSnapshot:
    step: int
    observation: torch.Tensor
    equilibrium_observation: torch.Tensor
    teacher_hidden: torch.Tensor
    student_state: StructuredPolicyState
    motor_trim_target: torch.Tensor
    body_z_target: torch.Tensor
    capability: torch.Tensor
    tw_bin: torch.Tensor
    log_alpha_bin: torch.Tensor


@dataclass
class LocalDerivativeBatch:
    radii: torch.Tensor
    directions: torch.Tensor
    teacher_directional_derivative: torch.Tensor
    student_directional_derivative: torch.Tensor
    teacher_center_action: torch.Tensor
    teacher_action: torch.Tensor
    equilibrium_observation: torch.Tensor
    snapshot_steps: torch.Tensor
    tw_bin: torch.Tensor
    log_alpha_bin: torch.Tensor
    direction_count: int


def _clone_policy_state(state: StructuredPolicyState) -> StructuredPolicyState:
    return state.detach()


def _teacher_action(teacher: Any, observation: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    result = teacher.forward_with_aux(observation, hidden)
    return result[0] if isinstance(result, (tuple, list)) else result


def _student_equilibrium_state(
    student: StructuredRecurrentPolicy,
    state: StructuredPolicyState,
    target: Any,
    *,
    trim: torch.Tensor,
) -> StructuredPolicyState:
    """Keep learned recurrent history, but evaluate the local plant at hover."""
    capability = state.capability
    if capability is None:
        capability = trim.new_tensor((3.2, 100.0, 0.2, 1.7, 0.1, 0.15)).expand(trim.shape[0], 6)
    body_z = target.body_z
    # Pre-settle the projected cap-only contextual K.  A one-step call would
    # otherwise expose only 1/25 of the gain because the deployable policy
    # ramps contextual K at its slow cadence.
    log_mean = state.capability_log_mean
    if log_mean is None:
        log_mean = capability.clamp_min(1.0e-8).log()
    log_lo = log_mean.new_tensor((1.45, 35.0, 0.02, 1.45, 0.025, 0.03)).log()
    log_hi = log_mean.new_tensor((5.50, 2200.0, 1.00, 1.95, 0.18, 0.35)).log()
    gain_context = ((log_mean - log_lo) / (log_hi - log_lo)) * 2.0 - 1.0
    raw_gain = torch.tanh(student.contextual_gain_head(gain_context)).reshape(
        -1, 4, 15
    ) * student.config.contextual_gain_scale
    mixer = effective_wrench_mixer(capability)
    projection = student._project_contextual_gain(raw_gain, mixer)
    # Current policy returns projected gain plus pre/post induced norms.  The
    # small compatibility branch keeps this adapter usable with the earlier
    # two-value prototype while never changing the fitted variable set.
    projected_gain = projection[0] if isinstance(projection, (tuple, list)) else projection
    # Keep boot progress past burn-in and hold this projected gain fixed during
    # the local finite-difference calls.
    kwargs = {
        "motor_estimate": trim,
        "integral": torch.zeros_like(target.body_z),
        "slow_trim": trim,
        "slow_body_z": body_z,
        "capability": capability,
        # The settled local evaluation is not itself a publication boundary;
        # use an inter-boundary phase under the corrected counter semantics.
        "slow_counter": 1,
        "contextual_gain": projected_gain,
        "contextual_gain_target": projected_gain,
        "contextual_blend": trim.new_ones((trim.shape[0], 1)),
        "boot_progress": trim.new_full((trim.shape[0], 1), 100.0),
    }
    return replace(state, **kwargs)


def _equilibrium_observation(
    observation: torch.Tensor, target: Any,
) -> torch.Tensor:
    eq = observation.clone()
    eq[:, 0:6] = 0.0
    eq[:, 6:15] = rotation_from_body_z(target.body_z).reshape(-1, 9)
    eq[:, 15:18] = 0.0
    eq[:, 18:21] = 0.0
    eq[:, 21:25] = target.motor_trim
    return eq


def perturb_normalized_error(
    equilibrium_observation: torch.Tensor,
    direction: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """Map normalized [position,velocity,tilt,omega,motor] error to observation."""
    if direction.shape[-1] != ERROR_DIM:
        raise ValueError("direction must have 15 normalized error coordinates")
    if direction.ndim == 1:
        direction = direction.expand(equilibrium_observation.shape[0], -1)
    result = equilibrium_observation.clone()
    value = float(radius) * direction
    result[:, 0:3] += 0.1 * value[:, 0:3]
    result[:, 3:6] += 0.1 * value[:, 3:6]
    # Apply the two tilt coordinates through a local SO(3) exponential chart.
    # This keeps the perturbed equilibrium observation on the rotation
    # manifold instead of injecting nine unconstrained matrix coordinates.
    tangent = torch.zeros(
        result.shape[0], 3, device=result.device, dtype=result.dtype
    )
    # Policy tilt q=(R^T z)_xy maps to local tangent [q_y,-q_x,0].
    tangent[:, 0] = 0.1 * value[:, 7]
    tangent[:, 1] = -0.1 * value[:, 6]
    skew = torch.zeros(
        result.shape[0], 3, 3, device=result.device, dtype=result.dtype
    )
    skew[:, 0, 1], skew[:, 1, 0] = -tangent[:, 2], tangent[:, 2]
    skew[:, 0, 2], skew[:, 2, 0] = tangent[:, 1], -tangent[:, 1]
    skew[:, 1, 2], skew[:, 2, 1] = -tangent[:, 0], tangent[:, 0]
    theta = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1.0e-8)
    a = torch.sin(theta) / theta
    b = (1.0 - torch.cos(theta)) / theta.square()
    exp_tangent = torch.eye(3, device=result.device, dtype=result.dtype).expand_as(skew)
    exp_tangent = exp_tangent + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * torch.bmm(skew, skew)
    base_rotation = result[:, 6:15].reshape(-1, 3, 3)
    result[:, 6:15] = torch.bmm(base_rotation, exp_tangent).reshape(-1, 9)
    result[:, 15:18] += 0.5 * value[:, 8:11]
    result[:, 21:25] += 0.1 * value[:, 11:15]
    return result


def perturb_student_error_state(
    state: StructuredPolicyState,
    direction: torch.Tensor,
    radius: float,
) -> StructuredPolicyState:
    """Apply the motor-error coordinates to the recurrent state as well.

    The observation's final four entries are the previous command, while the
    student's local error uses ``state.motor_estimate - trim``.  Perturbing
    only the observation would therefore leave four Jacobian columns
    disconnected from the student state.  This helper applies the same
    normalized motor displacement to both; hidden, identifier, integral, and
    ``previous_executed_action`` remain fixed for the directional probe.
    """
    if direction.ndim == 1:
        direction = direction.expand(state.motor_estimate.shape[0], -1)
    if direction.shape[-1] != ERROR_DIM:
        raise ValueError("direction must have shape [...,15]")
    displacement = 0.1 * float(radius) * direction[:, 11:15]
    return replace(state, motor_estimate=state.motor_estimate + displacement)


@torch.no_grad()
def collect_common_history(
    teacher: Any,
    student: StructuredRecurrentPolicy,
    simulator: L2FSimulator,
    bank: DAggerScenarioBank,
    *,
    snapshot_steps: Sequence[int] = (50, 75),
) -> list[LocalSnapshot]:
    """Roll one actual teacher trajectory and save pre-action hidden states."""
    state = _clone_l2f(bank.state)
    batch = bank.count
    obs_state = initial_observation_state(
        batch, device=state.position.device, dtype=state.position.dtype
    )
    observation, _ = build_policy_observation(
        state, obs_state, mode="integral25", integral_input_frame="body"
    )
    student_state = student.initial_state(observation)
    teacher_hidden = teacher.initial_hidden(
        batch, device=state.position.device, dtype=state.position.dtype
    )
    snapshots: list[LocalSnapshot] = []
    wanted = set(int(value) for value in snapshot_steps)
    # ``step`` is the policy call index.  Call0 observes the initial state;
    # after each call exactly one physics transition is applied.
    for step in range(0, max(wanted) + 1):
        observation, observed_position = build_policy_observation(
            state, obs_state, mode="integral25", integral_input_frame="body"
        )
        target = analytic_equilibrium_target(state, gravity=simulator.params.gravity)
        teacher_hidden_pre = teacher_hidden
        if step in wanted:
            snapshots.append(LocalSnapshot(
                step=step,
                observation=observation.detach().clone(),
                equilibrium_observation=_equilibrium_observation(observation, target).detach(),
                teacher_hidden=teacher_hidden_pre.detach().clone(),
                student_state=_clone_policy_state(student_state),
                motor_trim_target=target.motor_trim.detach().clone(),
                body_z_target=target.body_z.detach().clone(),
                capability=torch.stack((state.thrust_to_weight, state.alpha_roll_max,
                                       state.eta_yaw, state.jz_over_jxy,
                                       state.motor_time_rising, state.motor_time_falling), -1).detach(),
                tw_bin=bank.tw_bin.detach().clone(),
                log_alpha_bin=bank.log_alpha_bin.detach().clone(),
            ))
        teacher_result = teacher.forward_with_aux(observation, teacher_hidden)
        teacher_action, teacher_hidden = teacher_result[0], teacher_result[1]
        student_result = student.forward_with_aux(
            observation, student_state, applied_action=teacher_action
        )
        student_state = student_result.next_state
        obs_state = update_position_integral(
            obs_state, observed_position, dt=simulator.params.dt,
            integral_limit=0.5, integral_leak=0.0,
        )
        state = simulator.step(state, teacher_action, grad_decay=1.0)
    return snapshots


@torch.no_grad()
def collect_equilibrium_history(
    teacher: Any,
    student: StructuredRecurrentPolicy,
    simulator: L2FSimulator,
    bank: DAggerScenarioBank,
    *,
    snapshot_steps: Sequence[int] = (50, 75),
) -> list[LocalSnapshot]:
    """Burn in both recurrent policies on a fixed analytic equilibrium.

    These are the training snapshots for local-JVP fitting.  In particular,
    the teacher hidden state is not taken from a moving Q2 trajectory and then
    paired with a different equilibrium observation; both hidden states are
    generated by the same fixed equilibrium observation history.  The moving
    trajectory from :func:`collect_common_history` remains useful only as a
    shift diagnostic.
    """
    if not snapshot_steps or min(int(value) for value in snapshot_steps) < 1:
        raise ValueError("snapshot_steps must contain positive steps")
    state = _clone_l2f(bank.state)
    batch = bank.count
    obs_state = initial_observation_state(
        batch, device=state.position.device, dtype=state.position.dtype
    )
    observation, _ = build_policy_observation(
        state, obs_state, mode="integral25", integral_input_frame="body"
    )
    target = analytic_equilibrium_target(state, gravity=simulator.params.gravity)
    equilibrium_observation = _equilibrium_observation(observation, target).detach()
    student_state = student.initial_state(equilibrium_observation)
    teacher_hidden = teacher.initial_hidden(
        batch, device=state.position.device, dtype=state.position.dtype
    )
    snapshots: list[LocalSnapshot] = []
    wanted = set(int(value) for value in snapshot_steps)
    # ``step`` is the policy call index, so call50 is the state after exactly
    # 50 completed transitions and is the t50 diagnostic boundary.
    for step in range(0, max(wanted) + 1):
        teacher_hidden_pre = teacher_hidden
        if step in wanted:
            snapshots.append(LocalSnapshot(
                step=step,
                observation=equilibrium_observation.clone(),
                equilibrium_observation=equilibrium_observation.clone(),
                teacher_hidden=teacher_hidden_pre.detach().clone(),
                student_state=_clone_policy_state(student_state),
                motor_trim_target=target.motor_trim.detach().clone(),
                body_z_target=target.body_z.detach().clone(),
                capability=torch.stack((state.thrust_to_weight, state.alpha_roll_max,
                                       state.eta_yaw, state.jz_over_jxy,
                                       state.motor_time_rising, state.motor_time_falling), -1).detach(),
                tw_bin=bank.tw_bin.detach().clone(),
                log_alpha_bin=bank.log_alpha_bin.detach().clone(),
            ))
        teacher_result = teacher.forward_with_aux(
            equilibrium_observation, teacher_hidden
        )
        teacher_action, teacher_hidden = teacher_result[0], teacher_result[1]
        student_result = student.forward_with_aux(
            # The equilibrium burn-in is an analytic plant state, so the
            # student's observer must see the analytic trim.  Q2's action is
            # retained only as a hidden-state/label diagnostic here.
            equilibrium_observation, student_state,
            applied_action=target.motor_trim,
        )
        student_state = student_result.next_state
    return snapshots


def _derivative_for_policy(
    policy: Any,
    observation: torch.Tensor,
    hidden_or_state: Any,
    direction: torch.Tensor,
    radius: float,
    *,
    teacher: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if direction.ndim == 1:
        direction = direction.expand(observation.shape[0], -1)
    plus = perturb_normalized_error(observation, direction, radius)
    minus = perturb_normalized_error(observation, direction, -float(radius))
    if teacher:
        plus_action = _teacher_action(policy, plus, hidden_or_state)
        minus_action = _teacher_action(policy, minus, hidden_or_state)
        center = _teacher_action(policy, observation, hidden_or_state)
    else:
        plus_state = perturb_student_error_state(hidden_or_state, direction, radius)
        minus_state = perturb_student_error_state(hidden_or_state, direction, -float(radius))
        plus_action = policy.forward_with_aux(plus, plus_state).action
        minus_action = policy.forward_with_aux(minus, minus_state).action
        center = policy.forward_with_aux(observation, hidden_or_state).action
    derivative = (plus_action - minus_action) / (2.0 * float(radius))
    return derivative, plus_action, center


@torch.no_grad()
def build_local_derivative_batch(
    teacher: Any,
    student: StructuredRecurrentPolicy,
    snapshots: Sequence[LocalSnapshot],
    *,
    radii: Sequence[float] = LOCAL_RADII,
    seed: int = 7,
) -> LocalDerivativeBatch:
    generator = torch.Generator(device=snapshots[0].observation.device)
    generator.manual_seed(int(seed))
    radii_tensor = snapshots[0].observation.new_tensor(tuple(float(r) for r in radii))
    direction_count = ERROR_DIM
    directions = torch.randn(
        len(snapshots), len(radii), direction_count, ERROR_DIM,
        generator=generator, device=snapshots[0].observation.device,
        dtype=snapshots[0].observation.dtype,
    )
    # QR gives 15 random orthonormal directions per snapshot/radius; the
    # heldout call uses a different seed and therefore an independent basis.
    directions = torch.stack([
        torch.linalg.qr(item.transpose(0, 1), mode="reduced").Q.transpose(0, 1)
        for item in directions.reshape(-1, direction_count, ERROR_DIM)
    ]).reshape(len(snapshots), len(radii), direction_count, ERROR_DIM)
    teacher_derivatives, student_derivatives = [], []
    teacher_centers, teacher_actions, equilibrium_observations = [], [], []
    steps, tw_bins, alpha_bins = [], [], []
    for snapshot_index, snapshot in enumerate(snapshots):
        target = type("Target", (), {"body_z": snapshot.body_z_target})
        student_state = _student_equilibrium_state(
            student, snapshot.student_state, target, trim=snapshot.motor_trim_target
        )
        local_teacher, local_student = [], []
        local_center, local_eq = [], []
        for radius_index, radius in enumerate(radii):
            radius_teacher, radius_student = [], []
            for direction in directions[len(teacher_derivatives), radius_index]:
                td, _, _ = _derivative_for_policy(
                    teacher, snapshot.equilibrium_observation, snapshot.teacher_hidden,
                    direction, radius, teacher=True
                )
                sd, _, _ = _derivative_for_policy(
                    student, snapshot.equilibrium_observation, student_state,
                    direction, radius, teacher=False
                )
                radius_teacher.append(td)
                radius_student.append(sd)
            local_teacher.append(torch.stack(radius_teacher))
            local_student.append(torch.stack(radius_student))
            center = _teacher_action(
                teacher, snapshot.equilibrium_observation, snapshot.teacher_hidden
            )
            local_center.append(center)
            local_eq.append(snapshot.equilibrium_observation)
        teacher_derivatives.append(torch.stack(local_teacher))
        student_derivatives.append(torch.stack(local_student))
        teacher_actions.append(torch.stack(local_center))
        teacher_centers.append(torch.stack(local_center))
        equilibrium_observations.append(torch.stack(local_eq))
        steps.append(snapshot.step)
        tw_bins.append(snapshot.tw_bin)
        alpha_bins.append(snapshot.log_alpha_bin)
    return LocalDerivativeBatch(
        radii=radii_tensor,
        directions=directions,
        teacher_directional_derivative=torch.stack(teacher_derivatives),
        student_directional_derivative=torch.stack(student_derivatives),
        teacher_center_action=torch.stack(teacher_centers),
        teacher_action=torch.stack(teacher_actions),
        equilibrium_observation=torch.stack(equilibrium_observations),
        snapshot_steps=torch.tensor(steps, device=radii_tensor.device),
        tw_bin=torch.stack(tw_bins),
        log_alpha_bin=torch.stack(alpha_bins),
        direction_count=direction_count,
    )


def fit_contextual_gain_local(
    student: StructuredRecurrentPolicy,
    teacher: Any,
    snapshots: Sequence[LocalSnapshot],
    *,
    iterations: int = 2,
    learning_rate: float = 3.0e-4,
    radii: Sequence[float] = LOCAL_RADII,
    seed: int = 7,
    training_args=None,
    development_callback=None,
) -> tuple[LocalDerivativeBatch, list[float]]:
    """Fit only cap-conditioned contextual K to Q2 directional derivatives."""
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in student.contextual_gain_head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        student.contextual_gain_head.parameters(), lr=learning_rate, weight_decay=1.0e-5
    )
    history: list[float] = []
    session = None
    if training_args is not None:
        from structured_training_runtime import TrainingSession
        session = TrainingSession(training_args, student, optimizer, stage="phase_b")
        student._training_session = session
        history = [row["loss"] for row in session.progress["history"]]
        if training_args.final_evaluation:
            session.begin_final([training_args.seed + 120001])
            return build_local_derivative_batch(teacher, student, snapshots, radii=radii, seed=seed), history
        session.save()
    # Targets are fixed teacher derivatives. Recompute student JVP estimates
    # each iteration so updates affect the actual projected K path.
    with torch.no_grad():
        reference_batch = build_local_derivative_batch(
            teacher, student, snapshots, radii=radii, seed=seed
        )
        targets = reference_batch.teacher_directional_derivative.detach()
        # A calibrated-but-wide Phase-A posterior can intentionally make the
        # deployable contextual gain weight exactly zero.  In that state an
        # action-space loss has identically zero gradient, so silently
        # reporting a successful K fit would be misleading.  Use an explicit
        # pre-gate Jacobian surrogate in this smoke case: recover the teacher
        # action Jacobian from the orthonormal frame and fit the candidate
        # contextual matrix relative to K_ref.  The final deployment JVP
        # diagnostics still expose the gate (and therefore remain false until
        # confidence is genuinely available).
        # Keep activation at [snapshot, scenario] granularity.  A single
        # confident scenario must not switch all low-confidence scenarios to
        # the deployment-FD branch.
        probe_active_masks = []
        for snapshot in snapshots:
            target = type("Target", (), {"body_z": snapshot.body_z_target})
            probe_state = _student_equilibrium_state(
                student, snapshot.student_state, target, trim=snapshot.motor_trim_target
            )
            probe_output = student.forward_with_aux(
                snapshot.equilibrium_observation, probe_state
            )
            probe_active_masks.append(
                (probe_output.auxiliary["contextual_gain_weight"].squeeze(-1) > 1.0e-8)
            )
        pre_gate_targets = []
        for snapshot_index in range(len(snapshots)):
            jacobians = []
            for radius_index in range(len(radii)):
                jacobians.append(torch.einsum(
                    "dba,de->bae",
                    targets[snapshot_index, radius_index],
                    reference_batch.directions[snapshot_index, radius_index],
                ))
            # These are action/error Jacobians.  The loss below maps the
            # candidate wrench gain through the same damped allocator inverse
            # before comparing, so no action/wrench unit mix-up is hidden in
            # the pre-gate fallback.
            pre_gate_targets.append(torch.stack(jacobians).mean(0))
        pre_gate_targets = torch.stack(pre_gate_targets)
    for _ in range(session.updates if session is not None else 0, max(1, int(iterations))):
        if session is not None and session.should_stop():
            break
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for snapshot_index, snapshot in enumerate(snapshots):
            target = type("Target", (), {"body_z": snapshot.body_z_target})
            state = _student_equilibrium_state(
                student, snapshot.student_state, target, trim=snapshot.motor_trim_target
            )
            active_mask = probe_active_masks[snapshot_index]
            inactive_mask = ~active_mask
            if bool(inactive_mask.any().item()):
                capability = state.capability
                mixer = effective_wrench_mixer(capability)
                gram = mixer.transpose(1, 2) @ mixer
                eye = torch.eye(4, device=mixer.device, dtype=mixer.dtype).expand_as(gram)
                allocator_inverse = torch.linalg.solve(
                    gram + student.config.allocator_damping * eye,
                    mixer.transpose(1, 2),
                )
                reference_gain = student.K_ref.to(
                    device=state.contextual_gain.device,
                    dtype=state.contextual_gain.dtype,
                ).expand_as(state.contextual_gain)
                predicted_action_jacobian = torch.bmm(
                    allocator_inverse, reference_gain + state.contextual_gain
                )
                losses.append(F.mse_loss(
                    predicted_action_jacobian[inactive_mask],
                    pre_gate_targets[snapshot_index][inactive_mask],
                ))
            if bool(active_mask.any().item()):
                for radius_index, radius in enumerate(radii):
                    for direction_index, direction in enumerate(
                        reference_batch.directions[snapshot_index, radius_index]
                    ):
                        plus = perturb_normalized_error(
                            snapshot.equilibrium_observation, direction, radius
                        )
                        minus = perturb_normalized_error(
                            snapshot.equilibrium_observation, direction, -float(radius)
                        )
                        plus_state = perturb_student_error_state(state, direction, radius)
                        minus_state = perturb_student_error_state(state, direction, -float(radius))
                        plus_action = student.forward_with_aux(plus, plus_state).action
                        minus_action = student.forward_with_aux(minus, minus_state).action
                        derivative = (plus_action - minus_action) / (2.0 * float(radius))
                        losses.append(F.mse_loss(
                            derivative[active_mask],
                            targets[snapshot_index, radius_index, direction_index][active_mask],
                        ))
        loss = torch.stack(losses).mean()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(student.contextual_gain_head.parameters(), 10.0)
        if not bool(torch.isfinite(gradient_norm)):
            if session is not None:
                session.save()
            raise RuntimeError("non-finite local-gain gradient")
        optimizer.step()
        history.append(float(loss.detach()))
        if session is not None:
            session.record_update({"loss": float(loss.detach()), "gradient_norm": float(gradient_norm)})
            if development_callback is not None and session.development_due():
                score, passed, metrics = development_callback()
                session.record_development(score=score, passed=passed, metrics=metrics)
    final = build_local_derivative_batch(teacher, student, snapshots, radii=radii, seed=seed)
    return final, history


@torch.no_grad()
def projection_diagnostics(
    student: StructuredRecurrentPolicy,
    snapshots: Sequence[LocalSnapshot],
    *,
    rho: float | None = None,
) -> dict[str, float]:
    induced_values = []
    base_induced_values = []
    projected_gain_max = []
    bound = float(student.config.contextual_gain_rho if rho is None else rho)
    for snapshot_index, snapshot in enumerate(snapshots):
        target = type("Target", (), {"body_z": snapshot.body_z_target})
        state = _student_equilibrium_state(
            student, snapshot.student_state, target, trim=snapshot.motor_trim_target
        )
        output = student.forward_with_aux(snapshot.equilibrium_observation, state)
        capability = output.auxiliary["allocation_capability"]
        mixer = effective_wrench_mixer(capability)
        # ``next_state.contextual_gain`` is the pre-projection candidate in
        # the policy state.  Reapply the exact policy projection here so the
        # reported bound is on the deployed gain, not on an unconstrained
        # optimizer variable.
        candidate = output.next_state.contextual_gain
        projected = student._project_contextual_gain(candidate, mixer)
        delta = projected[0] if isinstance(projected, (tuple, list)) else projected
        gram = mixer.transpose(1, 2) @ mixer
        eye = torch.eye(4, device=mixer.device, dtype=mixer.dtype).expand_as(gram)
        pinv = torch.linalg.solve(gram + student.config.allocator_damping * eye, mixer.transpose(1, 2))
        induced = torch.linalg.matrix_norm(torch.bmm(pinv, delta), ord=2, dim=(-2, -1))
        base_induced = torch.linalg.matrix_norm(
            torch.bmm(pinv, student.K_ref.to(mixer).expand_as(delta)),
            ord=2, dim=(-2, -1),
        )
        induced_values.append(induced)
        base_induced_values.append(base_induced)
        projected_gain_max.append(delta.abs().amax(dim=(-2, -1)))
    values = torch.cat(induced_values)
    base_values = torch.cat(base_induced_values)
    # Mirror the policy's actual relative spectral budget, including the
    # absolute numerical floor used when K_ref is nearly singular.  Comparing
    # against rho*||K_ref|| alone would silently under-report the deployable
    # bound in that case.
    jfloor = float(getattr(student.config, "contextual_gain_jfloor", 0.0))
    budget_reference = base_values.clamp_min(jfloor)
    budgets = float(bound) * budget_reference
    ratios = values / budget_reference.clamp_min(1.0e-8)
    return {
        "induced_spectral_max": float(values.max()),
        "induced_spectral_mean": float(values.mean()),
        "projection_budget_max": float(budgets.max()),
        "projection_budget_min": float(budgets.min()),
        "relative_induced_spectral_max": float(ratios.max()),
        "projection_jfloor": jfloor,
        "rho": bound,
        "projection_bound_passed": bool((values <= budgets + 1.0e-6).all()),
        "projected_gain_abs_max": float(torch.cat(projected_gain_max).max()),
        "post_induced_spectral_norm_max": float(values.max()),
        "K_ref_induced_spectral_max": float(base_values.max()),
    }


@torch.no_grad()
def local_derivative_diagnostics(
    teacher: Any,
    student: StructuredRecurrentPolicy,
    snapshots: Sequence[LocalSnapshot],
    batch: LocalDerivativeBatch,
) -> dict[str, Any]:
    error = batch.student_directional_derivative - batch.teacher_directional_derivative
    denominator = batch.teacher_directional_derivative.norm(dim=-1).clamp_min(1.0e-8)
    normalized = error.norm(dim=-1) / denominator
    remainder_values = []
    equilibrium_rms, equilibrium_max = [], []
    for snapshot_index, snapshot in enumerate(snapshots):
        target = type("Target", (), {"body_z": snapshot.body_z_target})
        state = _student_equilibrium_state(
            student, snapshot.student_state, target, trim=snapshot.motor_trim_target
        )
        center = student.forward_with_aux(snapshot.equilibrium_observation, state).action
        equilibrium_error = center - snapshot.motor_trim_target
        equilibrium_rms.append(torch.sqrt(equilibrium_error.square().mean()))
        equilibrium_max.append(equilibrium_error.abs().max())
        for radius_index, radius in enumerate(batch.radii.tolist()):
            for direction_index, direction in enumerate(
                batch.directions[snapshot_index, radius_index]
            ):
                plus = perturb_normalized_error(
                    snapshot.equilibrium_observation, direction, radius
                )
                plus_state = perturb_student_error_state(state, direction, radius)
                action = student.forward_with_aux(plus, plus_state).action
                linear = (
                    center
                    + batch.student_directional_derivative[
                        snapshot_index, radius_index, direction_index
                    ] * radius
                )
                remainder_values.append(
                    (action - linear).norm(dim=-1) / max(radius * radius, 1.0e-12)
                )
    remainder = torch.cat(remainder_values)
    jvp_cell_rows = []
    for tw in range(4):
        for alpha in range(4):
            cell_values = []
            for snapshot_index, snapshot in enumerate(snapshots):
                cell = (batch.tw_bin[snapshot_index] == tw) & (
                    batch.log_alpha_bin[snapshot_index] == alpha
                )
                if bool(cell.any()):
                    cell_values.append(normalized[snapshot_index, :, :, cell].reshape(-1))
            values = torch.cat(cell_values) if cell_values else normalized.new_zeros(0)
            jvp_cell_rows.append({
                "tw_bin": tw,
                "log_alpha_bin": alpha,
                "samples": int(values.numel()),
                "p95": float(torch.quantile(values, 0.95)) if values.numel() else None,
                "max": float(values.max()) if values.numel() else None,
                "coverage_at_registered_threshold": (
                    float((values <= LOCAL_JVP_P95_THRESHOLD).float().mean())
                    if values.numel() else None
                ),
            })
    return {
        "equilibrium_action_trim_rms": float(torch.stack(equilibrium_rms).mean()),
        "equilibrium_action_trim_max": float(torch.stack(equilibrium_max).max()),
        "jvp_normalized_error_mean": float(normalized.mean()),
        "jvp_normalized_error_max": float(normalized.max()),
        "jvp_normalized_error_p95": float(torch.quantile(normalized.reshape(-1), 0.95)),
        "jvp_registered_p95_threshold": LOCAL_JVP_P95_THRESHOLD,
        "jvp_coverage_by_authority_cell": jvp_cell_rows,
        "taylor_remainder_over_radius2_mean": float(remainder.mean()),
        "taylor_remainder_over_radius2_max": float(remainder.max()),
        "radii": [float(value) for value in batch.radii],
    }


@torch.no_grad()
def local_mode_diagnostics(
    teacher: Any,
    student: StructuredRecurrentPolicy,
    snapshots: Sequence[LocalSnapshot],
    batch: LocalDerivativeBatch,
) -> dict[str, Any]:
    """Report deployment-FD versus pre-gate-surrogate errors per authority cell."""
    rows = []
    all_active, all_inactive = [], []
    for snapshot_index, snapshot in enumerate(snapshots):
        target = type("Target", (), {"body_z": snapshot.body_z_target})
        state = _student_equilibrium_state(
            student, snapshot.student_state, target, trim=snapshot.motor_trim_target
        )
        output = student.forward_with_aux(snapshot.equilibrium_observation, state)
        active = output.auxiliary["contextual_gain_weight"].squeeze(-1) > 1.0e-8
        direction_error = (
            batch.student_directional_derivative[snapshot_index]
            - batch.teacher_directional_derivative[snapshot_index]
        )
        direction_denominator = batch.teacher_directional_derivative[snapshot_index].norm(
            dim=-1
        ).clamp_min(1.0e-8)
        deployment_error = (direction_error.norm(dim=-1) / direction_denominator).mean(1).mean(0)

        # Recover an action Jacobian from the teacher's orthonormal directional
        # frame and compare it to the projected candidate through the damped
        # allocator inverse (same units as the teacher action).
        teacher_jacobians = []
        for radius_index in range(batch.radii.numel()):
            teacher_jacobians.append(torch.einsum(
                "dba,de->bae",
                batch.teacher_directional_derivative[snapshot_index, radius_index],
                batch.directions[snapshot_index, radius_index],
            ))
        teacher_jacobian = torch.stack(teacher_jacobians).mean(0)
        mixer = effective_wrench_mixer(state.capability)
        gram = mixer.transpose(1, 2) @ mixer
        eye = torch.eye(4, device=mixer.device, dtype=mixer.dtype).expand_as(gram)
        inverse = torch.linalg.solve(
            gram + student.config.allocator_damping * eye, mixer.transpose(1, 2)
        )
        projected = student._project_contextual_gain(state.contextual_gain, mixer)
        candidate = projected[0] if isinstance(projected, (tuple, list)) else projected
        reference = student.K_ref.to(candidate).expand_as(candidate)
        surrogate_jacobian = torch.bmm(inverse, reference + candidate)
        surrogate_error = (
            (surrogate_jacobian - teacher_jacobian).norm(dim=(-2, -1))
            / teacher_jacobian.norm(dim=(-2, -1)).clamp_min(1.0e-8)
        )
        all_active.append(deployment_error[active])
        all_inactive.append(surrogate_error[~active])
        for tw in range(4):
            for alpha in range(4):
                cell = (snapshot.tw_bin == tw) & (snapshot.log_alpha_bin == alpha)
                cell_active = cell & active
                cell_inactive = cell & ~active
                rows.append({
                    "snapshot_step": snapshot.step,
                    "tw_bin": tw,
                    "log_alpha_bin": alpha,
                    "active_fraction": float(active[cell].float().mean()) if bool(cell.any()) else 0.0,
                    "active_deployment_jvp_error_mean": (
                        float(deployment_error[cell_active].mean()) if bool(cell_active.any()) else None
                    ),
                    "inactive_surrogate_error_mean": (
                        float(surrogate_error[cell_inactive].mean()) if bool(cell_inactive.any()) else None
                    ),
                    "active_samples": int(cell_active.sum()),
                    "inactive_samples": int(cell_inactive.sum()),
                })
    active_values = torch.cat([v for v in all_active if v.numel()]) if any(v.numel() for v in all_active) else torch.zeros(0)
    inactive_values = torch.cat([v for v in all_inactive if v.numel()]) if any(v.numel() for v in all_inactive) else torch.zeros(0)
    active_count = sum(int(v.numel()) for v in all_active)
    inactive_count = sum(int(v.numel()) for v in all_inactive)
    selected_parts = [v for v in (*all_active, *all_inactive) if v.numel()]
    selected_values = (
        torch.cat(selected_parts) if selected_parts else torch.zeros(0)
    )
    return {
        "active_fraction": float(active_count / max(active_count + inactive_count, 1)),
        "active_deployment_jvp_error_mean": float(active_values.mean()) if active_count else None,
        "inactive_surrogate_error_mean": float(inactive_values.mean()) if inactive_count else None,
        "selected_gate_error_p95": (
            float(torch.quantile(selected_values, 0.95))
            if selected_values.numel() else None
        ),
        "selected_gate_samples": int(selected_values.numel()),
        "precalibration_registered_p95_threshold": LOCAL_PRECAL_SURROGATE_P95_THRESHOLD,
        "by_authority_cell": rows,
    }


@torch.no_grad()
def lifted_h250_diagnostics(
    student: StructuredRecurrentPolicy,
    simulator: L2FSimulator,
    bank: DAggerScenarioBank,
    *,
    horizon: int = 250,
    segment_length: int = 25,
) -> dict[str, Any]:
    if horizon < 1 or segment_length < 1 or horizon % segment_length:
        raise ValueError("horizon must be a positive multiple of segment_length")
    state = _clone_l2f(bank.state)
    obs_state = initial_observation_state(
        bank.count, device=state.position.device, dtype=state.position.dtype
    )
    observation, _ = build_policy_observation(
        state, obs_state,
        mode="integral25", integral_input_frame="body",
    )
    recurrent = student.initial_state(observation)
    rows = []
    global_max_omega = torch.zeros(bank.count, device=state.position.device)
    global_finite = torch.ones(bank.count, dtype=torch.bool, device=state.position.device)
    for segment_index in range(horizon // segment_length):
        segment_max_omega = torch.zeros(bank.count, device=state.position.device)
        segment_finite = torch.ones(bank.count, dtype=torch.bool, device=state.position.device)
        for _ in range(segment_length):
            observation, observed_position = build_policy_observation(
                state, obs_state, mode="integral25", integral_input_frame="body"
            )
            output = student.forward_with_aux(observation, recurrent)
            recurrent = output.next_state
            state = simulator.step(state, output.action, grad_decay=1.0)
            obs_state = update_position_integral(
                obs_state, observed_position, dt=simulator.params.dt,
                integral_limit=0.5, integral_leak=0.0,
            )
            segment_max_omega = torch.maximum(
                segment_max_omega, torch.linalg.vector_norm(state.omega, dim=-1)
            )
            segment_finite &= (
                torch.isfinite(state.position).all(-1)
                & torch.isfinite(state.velocity).all(-1)
                & torch.isfinite(state.rotation).flatten(1).all(-1)
                & torch.isfinite(state.omega).all(-1)
                & torch.isfinite(output.action).all(-1)
            )
        global_max_omega = torch.maximum(global_max_omega, segment_max_omega)
        global_finite &= segment_finite
        rows.append({
            "segment": segment_index + 1,
            "start_step": segment_index * segment_length + 1,
            "end_step": (segment_index + 1) * segment_length,
            "finite_fraction": float(segment_finite.float().mean()),
            "max_omega": float(segment_max_omega.max()),
            "finite": bool(segment_finite.all()),
        })
    return {
        "horizon": horizon,
        "segment_length": segment_length,
        "segments": rows,
        "finite_fraction": float(global_finite.float().mean()),
        "max_omega": float(global_max_omega.max()),
        "finite": bool(global_finite.all()),
    }


@torch.no_grad()
def beta0_action_parity_diagnostics(
    teacher: Any,
    student: StructuredRecurrentPolicy,
    simulator: L2FSimulator,
    bank: DAggerScenarioBank,
    *,
    horizon: int = 25,
    annulus_radius: float = LOCAL_ACTION_PARITY_ANNULUS,
    annulus_r_max: float | None = None,
    parity_threshold: float = LOCAL_ACTION_PARITY_THRESHOLD,
) -> dict[str, Any]:
    """Compare beta-0 student actions with Q2 on an away-from-equilibrium annulus."""
    state = _clone_l2f(bank.state)
    obs_state = initial_observation_state(
        bank.count, device=state.position.device, dtype=state.position.dtype
    )
    observation, _ = build_policy_observation(
        state, obs_state, mode="integral25", integral_input_frame="body"
    )
    student_state = student.initial_state(observation)
    teacher_hidden = teacher.initial_hidden(
        bank.count, device=state.position.device, dtype=state.position.dtype
    )
    parity_values, trim_values, headroom_values, residual_values = [], [], [], []
    away_values = []
    for _ in range(horizon):
        observation, observed_position = build_policy_observation(
            state, obs_state, mode="integral25", integral_input_frame="body"
        )
        target = analytic_equilibrium_target(state, gravity=simulator.params.gravity)
        teacher_result = teacher.forward_with_aux(observation, teacher_hidden)
        teacher_action, teacher_hidden = teacher_result[0], teacher_result[1]
        output = student.forward_with_aux(observation, student_state)
        student_state = output.next_state
        # Use the exact normalized 15-D feature chart used by both the policy
        # residual and the pre-registered Phase-C oracle.  A raw p/v/omega
        # norm has different units, omits tilt/motor-observer error, and would
        # silently validate a different annulus.
        annulus_error = torch.linalg.vector_norm(
            output.auxiliary["feedback_features"], dim=-1
        )
        away = annulus_error >= float(annulus_radius)
        if annulus_r_max is not None:
            away = away & (annulus_error <= float(annulus_r_max))
        # Oracle thresholds are motor-component RMS, not four-vector L2.
        parity = (output.action - teacher_action).square().mean(dim=-1).sqrt()
        trim_error = (output.action - target.motor_trim).square().mean(dim=-1).sqrt()
        allocator = output.auxiliary["allocator"]
        parity_values.append(parity)
        trim_values.append(trim_error)
        headroom_values.append(allocator.minimum_headroom)
        residual_values.append(allocator.wrench_residual)
        away_values.append(away)
        obs_state = update_position_integral(
            obs_state, observed_position, dt=simulator.params.dt,
            integral_limit=0.5, integral_leak=0.0,
        )
        state = simulator.step(state, output.action, grad_decay=1.0)
    if parity_values:
        parity_all = torch.cat(parity_values)
        trim_all = torch.cat(trim_values)
        headroom_all = torch.cat(headroom_values)
        residual_all = torch.cat(residual_values)
        away_all = torch.cat(away_values)
    else:
        dtype = bank.state.position.dtype
        parity_all = trim_all = headroom_all = residual_all = bank.state.position.new_zeros(0)
        away_all = torch.zeros(0, dtype=torch.bool, device=state.position.device)
    rows = []
    # Re-evaluate cells from the fixed bank axis; each row is a genuine
    # scenario cell summary rather than a batch-global parity scalar.
    for tw in range(4):
        for alpha in range(4):
            cell = (bank.tw_bin == tw) & (bank.log_alpha_bin == alpha)
            # Scenario-level cell summaries are conservative: the rollout
            # tensors are flattened time/scenario, so repeat the cell mask.
            cell_mask = cell.repeat(horizon)
            rows.append({
                "tw_bin": tw,
                "log_alpha_bin": alpha,
                "away_samples": int((cell_mask & away_all).sum()),
                "parity_coverage": (
                    float((parity_all[cell_mask & away_all] <= parity_threshold).float().mean())
                    if bool((cell_mask & away_all).any()) else None
                ),
                "parity_p95": (
                    float(torch.quantile(parity_all[cell_mask & away_all], 0.95))
                    if bool((cell_mask & away_all).any()) else None
                ),
                "trim_error_rms": (
                    float(trim_all[cell_mask & away_all].square().mean().sqrt())
                    if bool((cell_mask & away_all).any()) else None
                ),
                "min_headroom": (
                    float(headroom_all[cell_mask & away_all].min())
                    if bool((cell_mask & away_all).any()) else None
                ),
                "allocator_residual_p95": (
                    float(torch.quantile(residual_all[cell_mask & away_all], 0.95))
                    if bool((cell_mask & away_all).any()) else None
                ),
                "allocator_residual_p99": (
                    float(torch.quantile(residual_all[cell_mask & away_all], 0.99))
                    if bool((cell_mask & away_all).any()) else None
                ),
            })
    return {
        "horizon": int(horizon),
        "annulus_radius": float(annulus_radius),
        "parity_threshold": float(parity_threshold),
        # Global parity and allocator statistics are defined on the same
        # away-annulus mask as the release gate.  Including all near-
        # equilibrium samples here would make the reported coverage and RMS
        # disagree with the registered oracle contract.
        "away_samples": int(away_all.sum()),
        "annulus_r_min": float(annulus_radius),
        "annulus_r_max": None if annulus_r_max is None else float(annulus_r_max),
        "parity_rms": float(parity_all[away_all].square().mean().sqrt()) if bool(away_all.any()) else None,
        "parity_p95": float(torch.quantile(parity_all[away_all], 0.95)) if bool(away_all.any()) else None,
        "parity_max": float(parity_all[away_all].max()) if bool(away_all.any()) else None,
        "parity_coverage": float((parity_all[away_all] <= parity_threshold).float().mean()) if bool(away_all.any()) else None,
        "trim_error_rms": float(trim_all[away_all].square().mean().sqrt()) if bool(away_all.any()) else None,
        "allocator_min_headroom": float(headroom_all[away_all].min()) if bool(away_all.any()) else None,
        "allocator_residual_p95": float(torch.quantile(residual_all[away_all], 0.95)) if bool(away_all.any()) else None,
        "allocator_residual_p99": float(torch.quantile(residual_all[away_all], 0.99)) if bool(away_all.any()) else None,
        "by_authority_cell": rows,
    }


@torch.no_grad()
def h25_diagnostics(
    student: StructuredRecurrentPolicy,
    simulator: L2FSimulator,
    bank: DAggerScenarioBank,
    *,
    horizon: int = 25,
) -> dict[str, float | int]:
    result = lifted_h250_diagnostics(
        student, simulator, bank, horizon=horizon, segment_length=25
    )
    return {key: value for key, value in result.items() if key != "segments"}


__all__ = [
    "LOCAL_RADII", "LocalSnapshot", "LocalDerivativeBatch", "build_dagger_scenario_bank",
    "collect_common_history", "perturb_normalized_error", "build_local_derivative_batch",
    "collect_equilibrium_history",
    "perturb_student_error_state",
    "fit_contextual_gain_local", "projection_diagnostics", "local_derivative_diagnostics",
    "local_mode_diagnostics",
    "h25_diagnostics", "lifted_h250_diagnostics",
    "beta0_action_parity_diagnostics",
]

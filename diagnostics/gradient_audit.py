from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from diagnostics.formal_rollout import clone_state, load_q2_policy
from env_l2f import (
    L2FLossConfig,
    L2FParams,
    L2FSimulator,
    L2FState,
    apply_gradient_decay,
)
from model import MotorGRUPolicy
from policy_observation import PolicyObservationState, build_policy_observation, initial_observation_state, update_position_integral
from training_objectives import independent_cvar_tail_loss


AuditMode = Literal["legacy_detach", "full_bptt", "no_detach_decay", "checkpoint_recompute"]
PARAMETER_GROUPS = {
    "encoder": ("encoder.",),
    "gru_input": ("gru.weight_ih", "gru.bias_ih"),
    "gru_recurrent": ("gru.weight_hh", "gru.bias_hh"),
    "main_head": ("motor_head.",),
    "integral_head": ("integral_residual_head.",),
    "damping_head": ("damping_residual_head.",),
    "motor_state_head": ("motor_state_head.",),
}


@dataclass
class ModeAudit:
    mode: str
    summary: dict[str, Any]
    parameter_rows: list[dict[str, Any]]
    boundary_rows: list[dict[str, Any]]
    segment_rows: list[dict[str, Any]]
    gradient: torch.Tensor
    final_signature: torch.Tensor


def _dynamic(state: L2FState, hidden: torch.Tensor, integral: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return (
        state.position, state.velocity, state.rotation, state.omega, state.motor,
        state.previous_action, hidden, integral,
    )


def _state_from_dynamic(template: L2FState, values: tuple[torch.Tensor, ...]) -> L2FState:
    return L2FState(
        position=values[0], velocity=values[1], rotation=values[2], omega=values[3],
        motor=values[4], previous_action=values[5],
        **{name: getattr(template, name) for name in L2FState.__dataclass_fields__
           if name not in ("position", "velocity", "rotation", "omega", "motor", "previous_action")},
    )


def _detach_dynamic(values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().requires_grad_(value.requires_grad) for value in values)


def _segment(
    policy: MotorGRUPolicy,
    sim: L2FSimulator,
    template: L2FState,
    values: tuple[torch.Tensor, ...],
    *,
    segment_steps: int,
    episode_offset: int,
    episode_steps: int,
    segment_count: int,
    use_decay: bool,
) -> tuple[torch.Tensor, ...]:
    state = _state_from_dynamic(template, values)
    hidden = values[6]
    observation_state = PolicyObservationState(values[7])
    config = L2FLossConfig()
    tracking = state.position.sum() * 0.0
    clf = tracking
    outward = tracking
    du = tracking
    ddu = tracking
    sat = tracking
    motor_aux_sum = tracking
    motor_aux_count = 0
    tail: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    omegas: list[torch.Tensor] = []
    previous_potential = sim.tracking_potential(state, config)
    previous_action_delta: torch.Tensor | None = None
    first_action: torch.Tensor | None = None
    state_decay = 0.5 ** 0.01 if use_decay else 1.0
    hidden_decay = 0.7 ** 0.01 if use_decay else 1.0
    for local_step in range(segment_steps):
        previous_action = state.previous_action
        motor_target = state.motor.detach()
        observation, observed_position = build_policy_observation(
            state, observation_state, mode="integral25", integral_input_frame="body",
            integral_input_multiplier=1.0,
        )
        action, hidden, auxiliary = policy.forward_with_aux(observation, hidden)
        if first_action is None:
            first_action = action
        hidden = apply_gradient_decay(hidden, hidden_decay)
        observation_state = update_position_integral(
            observation_state, observed_position, dt=0.01, integral_limit=0.5, integral_leak=0.0,
        )
        action_delta = action - previous_action
        state = sim.step(state, action, grad_decay=state_decay)
        potential = sim.tracking_potential(state, config)
        tracking = tracking + potential.mean()
        clf_target = 0.99 * previous_potential.detach()
        clf = clf + F.relu(potential - clf_target).square().mean()
        outward = outward + sim.outward_velocity_loss(state, config)
        du = du + action_delta.square().mean()
        sat = sat + F.relu(action.abs() - 0.9).square().mean()
        if previous_action_delta is not None:
            ddu = ddu + (action_delta - previous_action_delta).square().mean()
        sample_step = episode_offset + local_step
        if sample_step >= 15:
            motor_aux_sum = motor_aux_sum + F.smooth_l1_loss(
                auxiliary["motor_state"], motor_target, reduction="mean"
            )
            motor_aux_count += 1
        tail.append(potential)
        positions.append(state.position)
        omegas.append(state.omega)
        previous_potential = potential
        previous_action_delta = action_delta
    assert first_action is not None
    segment_loss = (
        tracking / segment_steps + 0.5 * clf / segment_steps + 0.1 * outward / segment_steps
        + torch.stack(tail[-50:]).mean() + 0.003 * du / segment_steps
        + 0.0003 * ddu / max(segment_steps - 1, 1) + 0.03 * sat / segment_steps
    )
    if motor_aux_count:
        # The outer loop averages segment gradients. Multiplying by the segment
        # count reproduces one episode-time mean after the 15-step burn-in.
        segment_loss = segment_loss + 0.03 * motor_aux_sum * segment_count / max(episode_steps - 15, 1)
    dynamic = _dynamic(state, hidden, observation_state.integral_position)
    return (*dynamic, segment_loss, torch.stack(positions), torch.stack(omegas), first_action)


def _flat_grad(grads: tuple[torch.Tensor | None, ...], parameters: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([
        (torch.zeros_like(parameter) if grad is None else grad).reshape(-1)
        for grad, parameter in zip(grads, parameters)
    ])


def _norm(values: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None]) -> float:
    present = [value.reshape(-1).double() for value in values if value is not None]
    return float(torch.linalg.vector_norm(torch.cat(present)).item()) if present else 0.0


def _parameter_group(name: str) -> str:
    for group, prefixes in PARAMETER_GROUPS.items():
        if name.startswith(prefixes):
            return group
    return "other"


def audit_mode(
    checkpoint_path: str | Path,
    initial_state: L2FState,
    *,
    mode: AuditMode,
    horizon: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> ModeAudit:
    policy, _ = load_q2_policy(checkpoint_path, device=device, dtype=dtype)
    policy.train()
    sim = L2FSimulator(L2FParams(dt=0.01))
    template = clone_state(initial_state)
    hidden = policy.initial_hidden(initial_state.position.shape[0], device=device, dtype=dtype)
    integral = initial_observation_state(initial_state.position.shape[0], device=device, dtype=dtype).integral_position
    values = _dynamic(clone_state(initial_state), hidden, integral)
    segments = horizon // 250
    if horizon % 250:
        raise ValueError("gradient audit horizon must be divisible by 250")
    use_decay = mode in ("legacy_detach", "no_detach_decay")
    use_checkpoint = mode == "checkpoint_recompute"
    segment_losses: list[torch.Tensor] = []
    position_histories: list[torch.Tensor] = []
    omega_histories: list[torch.Tensor] = []
    first_actions: list[torch.Tensor] = []
    boundaries: list[tuple[torch.Tensor, ...]] = []
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        torch.cuda.synchronize(torch.device(device))
    started = time.perf_counter()
    for segment in range(segments):
        def function(*inputs: torch.Tensor, segment_index: int = segment) -> tuple[torch.Tensor, ...]:
            return _segment(
                policy, sim, template, tuple(inputs), segment_steps=250,
                episode_offset=segment_index * 250, episode_steps=horizon,
                segment_count=segments, use_decay=use_decay,
            )
        outputs = checkpoint(function, *values, use_reentrant=False) if use_checkpoint else function(*values)
        values = tuple(outputs[:8])
        segment_losses.append(outputs[8])
        position_histories.append(outputs[9])
        omega_histories.append(outputs[10])
        first_actions.append(outputs[11])
        boundaries.append(values)
        if mode == "legacy_detach" and segment + 1 < segments:
            values = _detach_dynamic(values)
    early = independent_cvar_tail_loss(
        position_histories[0], omega_histories[0], window_steps=100,
        w_position_cvar=0.001, w_omega_cvar=0.001,
    ).loss
    final = independent_cvar_tail_loss(
        position_histories[-1], omega_histories[-1], window_steps=100,
        w_position_cvar=0.001, w_omega_cvar=0.001,
    ).loss
    total_loss = torch.stack(segment_losses).mean() + 0.25 * early + final
    parameters = tuple(policy.parameters())
    segment_rows: list[dict[str, Any]] = []
    for index, loss in enumerate(segment_losses):
        contribution = torch.autograd.grad(loss / segments, parameters, retain_graph=True, allow_unused=True)
        flat = _flat_grad(contribution, parameters)
        segment_rows.append({
            "mode": mode, "horizon": horizon, "segment": index + 1,
            "segment_loss": float(loss.detach()), "parameter_gradient_norm": float(torch.linalg.vector_norm(flat.double())),
            "parameter_gradient_max_abs": float(flat.abs().max()), "finite": int(bool(torch.isfinite(flat).all())),
        })
    total_grads = torch.autograd.grad(total_loss, parameters, retain_graph=True, allow_unused=True)
    flat_total = _flat_grad(total_grads, parameters).detach()
    parameter_rows: list[dict[str, Any]] = []
    for group in (*PARAMETER_GROUPS, "other"):
        selected = [(name, parameter, grad) for (name, parameter), grad in zip(policy.named_parameters(), total_grads) if _parameter_group(name) == group]
        if not selected:
            continue
        group_flat = torch.cat([(torch.zeros_like(parameter) if grad is None else grad).reshape(-1) for _, parameter, grad in selected])
        parameter_rows.append({
            "mode": mode, "horizon": horizon, "parameter_group": group,
            "gradient_norm": float(torch.linalg.vector_norm(group_flat.double())),
            "max_absolute_gradient": float(group_flat.abs().max()),
            "finite": int(bool(torch.isfinite(group_flat).all())),
            "parameter_count": group_flat.numel(),
        })
    boundary_rows: list[dict[str, Any]] = []
    final_objective = segment_losses[-1] / segments + final
    for index, boundary in enumerate(boundaries[:-1]):
        future_grads = torch.autograd.grad(final_objective, boundary, retain_graph=True, allow_unused=True)
        boundary_rows.append({
            "mode": mode, "horizon": horizon, "boundary_after_segment": index + 1,
            "physical_state_adjoint_norm": _norm(list(future_grads[:6])),
            "hidden_adjoint_norm": _norm([future_grads[6]]),
            "integral_adjoint_norm": _norm([future_grads[7]]),
            "future_loss_has_causal_adjoint": int(any(value is not None and bool(torch.any(value != 0)) for value in future_grads)),
        })
    early_action_grad = torch.autograd.grad(final_objective, first_actions[0], retain_graph=True, allow_unused=True)[0]
    elapsed = time.perf_counter() - started
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
        peak_memory = torch.cuda.max_memory_allocated(torch.device(device))
    else:
        peak_memory = 0
    signature = torch.cat([value.detach().reshape(-1).cpu() for value in values])
    summary = {
        "mode": mode, "horizon": horizon, "batch_size": initial_state.position.shape[0],
        "dtype": str(dtype), "device": str(device), "loss": float(total_loss.detach()),
        "gradient_norm": float(torch.linalg.vector_norm(flat_total.double())),
        "max_absolute_gradient": float(flat_total.abs().max()),
        "gradient_finite": int(bool(torch.isfinite(flat_total).all())),
        "early_segment_action_adjoint_from_final_loss": _norm([early_action_grad]),
        "runtime_s": elapsed, "peak_gpu_memory_bytes": peak_memory,
    }
    return ModeAudit(mode, summary, parameter_rows, boundary_rows, segment_rows, flat_total.cpu(), signature)


def run_gradient_audit(
    checkpoint_path: str | Path,
    initial_state: L2FState,
    *,
    horizon: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    audits = [
        audit_mode(checkpoint_path, initial_state, mode=mode, horizon=horizon, device=device, dtype=dtype)
        for mode in ("legacy_detach", "full_bptt", "no_detach_decay", "checkpoint_recompute")
    ]
    reference = next(item for item in audits if item.mode == "full_bptt")
    summaries: list[dict[str, Any]] = []
    for item in audits:
        forward_error = float((item.final_signature - reference.final_signature).abs().max())
        denominator = torch.linalg.vector_norm(item.gradient.double()) * torch.linalg.vector_norm(reference.gradient.double())
        cosine = float(torch.dot(item.gradient.double(), reference.gradient.double()) / denominator.clamp_min(1.0e-30))
        row = dict(item.summary)
        row["forward_max_abs_error_vs_full_bptt"] = forward_error
        row["gradient_cosine_vs_full_bptt"] = cosine
        row["gradient_relative_norm_vs_full_bptt"] = float(
            torch.linalg.vector_norm(item.gradient.double()) / torch.linalg.vector_norm(reference.gradient.double()).clamp_min(1.0e-30)
        )
        summaries.append(row)
        for parameter_row in item.parameter_rows:
            parameter_row["gradient_cosine_vs_full_bptt"] = cosine
        for segment_row in item.segment_rows:
            segment_row["gradient_cosine_vs_full_bptt"] = cosine
    parameter_rows = [row for item in audits for row in (*item.parameter_rows, *item.segment_rows)]
    boundary_rows = [row for item in audits for row in item.boundary_rows]
    return summaries, parameter_rows, boundary_rows

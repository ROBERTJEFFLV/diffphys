from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train  # noqa: E402
from diagnostics.formal_rollout import clone_state, load_q2_policy  # noqa: E402
from env_l2f import L2FLossConfig, L2FParams, L2FSimulator, L2FState  # noqa: E402
from multiple_shooting import (  # noqa: E402
    CONTINUITY_FIELDS,
    RecurrentSystemState,
    SegmentResult,
    ShootingBoundary,
    augmented_lagrangian_terms,
    clone_recurrent,
    continuity_max_abs,
    continuity_residuals,
    continuity_rms,
    detach_recurrent,
    q2_h1000_task_loss,
    recurrent_tensors,
    rollout_q2_segment,
    update_duals_,
    zero_duals,
)
from policy_observation import initial_observation_state  # noqa: E402
from retain_bank import apply_retain_bank_samples, load_retain_bank  # noqa: E402
from temporal_decay import resolve_step_gradient_decay  # noqa: E402


MODE_TBPTT = "tbptt"
MODE_MULTIPLE_SHOOTING = "multiple-shooting"


@dataclass(frozen=True)
class TrainingSchedule:
    state: L2FState
    retain_mask: torch.Tensor
    retain_indices: torch.Tensor
    batch_indices: torch.Tensor
    update_batch_ids: torch.Tensor


class _ArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, line: str) -> list[str]:
        return shlex.split(line, comments=True, posix=True)


def parse_args() -> argparse.Namespace:
    parser = _ArgumentParser(
        description="Minimal paired H1000 TBPTT/multiple-shooting screen.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--mode", choices=(MODE_TBPTT, MODE_MULTIPLE_SHOOTING), required=True)
    parser.add_argument(
        "--base-config", type=Path, default=ROOT / "configs/time_horizon_T2.args"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--training-schedule",
        type=Path,
        default=None,
        help="Optional pre-generated fixed L2F state/retain batch shared by paired arms.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--segment-steps", type=int, default=250)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--shooting-lr", type=float, default=2.0e-3)
    parser.add_argument("--penalty-rho", type=float, default=10.0)
    parser.add_argument("--dual-update-every", type=int, default=5)
    parser.add_argument("--future-credit-probe-steps", type=int, default=8)
    parser.add_argument("--continuity-tolerance", type=float, default=5.0e-2)
    parser.add_argument("--abort-continuity-rms", type=float, default=5.0)
    parser.add_argument("--abort-policy-grad-norm", type=float, default=100.0)
    parser.add_argument("--abort-shooting-grad-norm", type=float, default=1000.0)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def _load_train_args(path: Path) -> argparse.Namespace:
    candidate = path if path.is_absolute() else ROOT / path
    previous = sys.argv
    try:
        sys.argv = ["train.py", f"@{candidate}"]
        args = train.parse_args()
        train.apply_direct_h500_training_defaults(args)
        return args
    finally:
        sys.argv = previous


def _validate(args: argparse.Namespace, train_args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.updates <= 0:
        raise ValueError("batch size and updates must be positive")
    if args.segment_steps != 250 or args.segments != 4:
        raise ValueError("the minimal validation is fixed at H1000=4xH250")
    if args.shooting_lr <= 0.0 or args.penalty_rho <= 0.0:
        raise ValueError("shooting learning rate and penalty rho must be positive")
    if args.dual_update_every < 0 or args.future_credit_probe_steps < 0:
        raise ValueError("dual update cadence must be non-negative; zero selects pure penalty")
    if train_args.horizon != 250:
        raise ValueError("base Q2 config must retain H250 differentiable segments")
    if train_args.sim_backend not in ("cuda", "torch"):
        raise ValueError("multiple shooting needs the step-wise cuda or torch backend")
    if train_args.lambda_capability_aux != 0.0 or train_args.lambda_response_aux != 0.0:
        raise ValueError("the minimal implementation is registered for Q2 auxiliary losses")
    if train_args.w_retain != 0.0:
        raise ValueError("the Q2 multiple-shooting screen expects retain loss weight zero")
    if train_args.tail_selection_mode != "independent":
        raise ValueError("the Q2 multiple-shooting screen expects independent CVaR")
    if train_args.action_noise_max != 0.0 or train_args.observation_noise_max != 0.0:
        raise ValueError("the fixed paired screen expects the current zero-noise Q2 config")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_signature(
    state: L2FState,
    retain_mask: torch.Tensor,
    retain_indices: torch.Tensor,
    batch_indices: torch.Tensor | None = None,
    update_batch_ids: torch.Tensor | None = None,
) -> str:
    digest = hashlib.sha256()
    for field in fields(L2FState):
        value = getattr(state, field.name).detach().contiguous().cpu()
        digest.update(field.name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    digest.update(retain_mask.detach().contiguous().cpu().numpy().tobytes())
    digest.update(retain_indices.detach().contiguous().cpu().numpy().tobytes())
    if batch_indices is not None:
        digest.update(batch_indices.detach().contiguous().cpu().numpy().tobytes())
    if update_batch_ids is not None:
        digest.update(update_batch_ids.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _select_state(state: L2FState, indices: torch.Tensor) -> L2FState:
    return L2FState(
        **{
            field.name: getattr(state, field.name).index_select(0, indices)
            for field in fields(L2FState)
        }
    )


def _load_training_schedule(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TrainingSchedule:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("training schedule must be a mapping")
    state_payload = payload.get("state")
    expected = set(L2FState.__dataclass_fields__)
    if not isinstance(state_payload, dict) or set(state_payload) != expected:
        raise ValueError("training schedule state schema does not match L2FState")
    state = L2FState(
        **{
            name: torch.as_tensor(value, device=device, dtype=dtype)
            for name, value in state_payload.items()
        }
    )
    retain_mask = torch.as_tensor(
        payload.get("retain_mask"), device=device, dtype=torch.bool
    )
    retain_indices = torch.as_tensor(
        payload.get("retain_indices"), device=device, dtype=torch.long
    )
    scenario_count = state.position.shape[0]
    if retain_mask.shape != (scenario_count,) or retain_indices.shape != (scenario_count,):
        raise ValueError("training schedule retain metadata has invalid shape")
    raw_batch_indices = payload.get("batch_indices")
    raw_update_batch_ids = payload.get("update_batch_ids")
    if raw_batch_indices is None and raw_update_batch_ids is None:
        batch_indices = torch.arange(
            scenario_count, device=device, dtype=torch.long
        ).unsqueeze(0)
        update_batch_ids = torch.zeros(1, device=device, dtype=torch.long)
    elif raw_batch_indices is None or raw_update_batch_ids is None:
        raise ValueError(
            "training schedule must provide both batch_indices and update_batch_ids"
        )
    else:
        batch_indices = torch.as_tensor(
            raw_batch_indices, device=device, dtype=torch.long
        )
        update_batch_ids = torch.as_tensor(
            raw_update_batch_ids, device=device, dtype=torch.long
        )
    if batch_indices.ndim != 2 or update_batch_ids.ndim != 1:
        raise ValueError("training schedule batch routing tensors have invalid rank")
    flattened = batch_indices.flatten()
    expected_indices = torch.arange(scenario_count, device=device, dtype=torch.long)
    if flattened.numel() != scenario_count or not bool(
        torch.equal(flattened.sort().values, expected_indices)
    ):
        raise ValueError(
            "training schedule batches must partition every scenario exactly once"
        )
    if update_batch_ids.numel() == 0 or bool(
        ((update_batch_ids < 0) | (update_batch_ids >= batch_indices.shape[0])).any().item()
    ):
        raise ValueError("training schedule update_batch_ids contains an invalid batch id")
    return TrainingSchedule(
        state=state,
        retain_mask=retain_mask,
        retain_indices=retain_indices,
        batch_indices=batch_indices,
        update_batch_ids=update_batch_ids,
    )


def _flat_grad_norm(
    gradients: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None],
) -> float:
    square = torch.zeros((), dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            square = square + gradient.detach().double().square().sum().cpu()
    return float(torch.sqrt(square).item())


def _parameter_grad_norm(parameters: tuple[torch.nn.Parameter, ...]) -> float:
    return _flat_grad_norm([parameter.grad for parameter in parameters])


def _all_parameter_grads_finite(parameters: tuple[torch.nn.Parameter, ...]) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in parameters
    )


def _tensor_list_norm(values: tuple[torch.Tensor | None, ...]) -> float:
    return _flat_grad_norm(values)


def _motor_aux_weight(
    train_args: argparse.Namespace,
    *,
    physical_steps_after_segment: int,
    accepted_updates: int,
) -> float:
    if train_args.aux_weight_ramp_physical_steps > 0:
        return train._ramped_auxiliary_weight_by_physical_steps(
            train_args.lambda_motor_aux,
            physical_steps_after_segment,
            train_args.aux_weight_ramp_physical_steps,
        )
    return train._ramped_auxiliary_weight(
        train_args.lambda_motor_aux,
        accepted_updates,
        train_args.aux_weight_ramp_updates,
    )


def _rollout_segments(
    *,
    policy: torch.nn.Module,
    sim: L2FSimulator,
    initial: RecurrentSystemState,
    shooting_boundaries: torch.nn.ModuleList | None,
    train_args: argparse.Namespace,
    loss_config: L2FLossConfig,
    state_step_decay: float,
    hidden_step_decay: float,
    retain_mask: torch.Tensor,
    mode: str,
    update_index: int,
    segment_steps: int,
    segment_count: int,
    batch_size: int,
    backend: str,
) -> tuple[list[SegmentResult], list[RecurrentSystemState]]:
    shooting_states = (
        [boundary.materialize(initial.state) for boundary in shooting_boundaries]
        if shooting_boundaries is not None
        else []
    )
    results: list[SegmentResult] = []
    current = initial
    for segment_index in range(segment_count):
        if segment_index > 0:
            current = (
                shooting_states[segment_index - 1]
                if mode == MODE_MULTIPLE_SHOOTING
                else detach_recurrent(results[-1].end)
            )
        physical_steps_after_segment = (
            (update_index - 1) * batch_size * segment_steps * segment_count
            + (segment_index + 1) * batch_size * segment_steps
        )
        result = rollout_q2_segment(
            policy,
            sim,
            current,
            train_args=train_args,
            loss_config=loss_config,
            state_step_decay=state_step_decay,
            hidden_step_decay=hidden_step_decay,
            retain_mask=retain_mask,
            segment_index=segment_index,
            segment_steps=segment_steps,
            segment_count=segment_count,
            motor_aux_weight=_motor_aux_weight(
                train_args,
                physical_steps_after_segment=physical_steps_after_segment,
                accepted_updates=update_index - 1,
            ),
            backend=backend,
        )
        results.append(result)
    return results, shooting_states


@torch.no_grad()
def _initialize_shooting_boundaries(
    *,
    policy: torch.nn.Module,
    sim: L2FSimulator,
    initial: RecurrentSystemState,
    train_args: argparse.Namespace,
    loss_config: L2FLossConfig,
    state_step_decay: float,
    hidden_step_decay: float,
    retain_mask: torch.Tensor,
    segment_steps: int,
    segment_count: int,
    backend: str,
) -> torch.nn.ModuleList:
    current = clone_recurrent(initial)
    boundaries = torch.nn.ModuleList()
    for segment_index in range(segment_count):
        result = rollout_q2_segment(
            policy,
            sim,
            current,
            train_args=train_args,
            loss_config=loss_config,
            state_step_decay=state_step_decay,
            hidden_step_decay=hidden_step_decay,
            retain_mask=retain_mask,
            segment_index=segment_index,
            segment_steps=segment_steps,
            segment_count=segment_count,
            motor_aux_weight=0.0,
            backend=backend,
        )
        current = clone_recurrent(result.end)
        if segment_index + 1 < segment_count:
            boundaries.append(ShootingBoundary(current))
    return boundaries


def _future_only_credit_probe(
    *,
    policy: torch.nn.Module,
    sim: L2FSimulator,
    initial: RecurrentSystemState,
    train_args: argparse.Namespace,
    loss_config: L2FLossConfig,
    state_step_decay: float,
    hidden_step_decay: float,
    retain_mask: torch.Tensor,
    segment_steps: int,
    segment_count: int,
    batch_size: int,
    backend: str,
    shooting_lr: float,
    penalty_rho: float,
    steps: int,
) -> list[dict[str, Any]]:
    """Trace final-segment-only task credit through all three constraints.

    Policy parameters are not stepped.  Only shooting variables are optimized,
    so a non-zero constraint adjoint at segment 1 can only have arrived from the
    final-segment task through the intervening shooting states and penalties.
    """

    if steps == 0:
        return []
    boundaries = _initialize_shooting_boundaries(
        policy=policy,
        sim=sim,
        initial=initial,
        train_args=train_args,
        loss_config=loss_config,
        state_step_decay=state_step_decay,
        hidden_step_decay=hidden_step_decay,
        retain_mask=retain_mask,
        segment_steps=segment_steps,
        segment_count=segment_count,
        backend=backend,
    ).to(initial.state.position.device)
    optimizer = torch.optim.AdamW(
        boundaries.parameters(), lr=shooting_lr, weight_decay=0.0
    )
    duals: list[dict[str, torch.Tensor]] | None = None
    rows: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        policy.zero_grad(set_to_none=True)
        results, shooting_states = _rollout_segments(
            policy=policy,
            sim=sim,
            initial=initial,
            shooting_boundaries=boundaries,
            train_args=train_args,
            loss_config=loss_config,
            state_step_decay=state_step_decay,
            hidden_step_decay=hidden_step_decay,
            retain_mask=retain_mask,
            mode=MODE_MULTIPLE_SHOOTING,
            update_index=1,
            segment_steps=segment_steps,
            segment_count=segment_count,
            batch_size=batch_size,
            backend=backend,
        )
        _, _, final_tail = q2_h1000_task_loss(results, train_args)
        final_only_task = (
            results[-1].task_loss / float(segment_count)
            + float(train_args.final_tail_weight) * final_tail
        )
        residuals = [
            continuity_residuals(results[index].end, shooting_states[index])
            for index in range(segment_count - 1)
        ]
        if duals is None:
            duals = zero_duals(residuals)
        continuity_loss, boundary_terms = augmented_lagrangian_terms(
            residuals, duals, rho=penalty_rho
        )
        early_adjoint = torch.autograd.grad(
            continuity_loss,
            results[0].first_action,
            retain_graph=True,
            allow_unused=True,
        )[0]
        boundary_policy_norms: list[float] = []
        policy_parameters = tuple(policy.parameters())
        for term in boundary_terms:
            gradient = torch.autograd.grad(
                term,
                policy_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            boundary_policy_norms.append(_flat_grad_norm(gradient))
        objective = final_only_task + continuity_loss
        objective.backward()
        shooting_gradient_norm = _parameter_grad_norm(tuple(boundaries.parameters()))
        optimizer.step()
        metrics = _continuity_metrics(residuals)
        rows.append(
            {
                "probe_step": step,
                "final_only_task": float(final_only_task.detach().item()),
                "continuity_objective": float(continuity_loss.detach().item()),
                **metrics,
                "boundary_1_policy_gradient_norm": boundary_policy_norms[0],
                "boundary_2_policy_gradient_norm": boundary_policy_norms[1],
                "boundary_3_policy_gradient_norm": boundary_policy_norms[2],
                "continuity_to_segment_1_action": _tensor_list_norm((early_adjoint,)),
                "shooting_gradient_norm": shooting_gradient_norm,
            }
        )
    policy.zero_grad(set_to_none=True)
    return rows


def _continuity_metrics(
    residuals: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    if not residuals:
        return {
            "continuity_rms": 0.0,
            "continuity_max_abs": 0.0,
            **{
                f"continuity_{name}_rms": 0.0
                for name in CONTINUITY_FIELDS
            },
        }
    return {
        "continuity_rms": float(
            torch.stack(tuple(continuity_rms(item) for item in residuals)).mean().detach().item()
        ),
        "continuity_max_abs": float(
            torch.stack(tuple(continuity_max_abs(item) for item in residuals)).max().detach().item()
        ),
        **{
            f"continuity_{name}_rms": float(
                torch.sqrt(
                    torch.stack(tuple(item[name].square().mean() for item in residuals)).mean()
                ).detach().item()
            )
            for name in CONTINUITY_FIELDS
        },
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty multiple-shooting log")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    train_args = _load_train_args(args.base_config)
    _validate(args, train_args)
    train_args.batch_size = args.batch_size
    train_args.device = args.device
    train_args.sim_backend = "cuda" if args.device == "cuda" else "torch"
    device = train.resolve_device(args.device)
    backend = train.resolve_sim_backend(train_args.sim_backend, device)
    torch.manual_seed(train_args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_args.seed)

    sim = L2FSimulator(
        L2FParams(
            dt=train_args.dt,
            max_initial_position=train_args.max_initial_position,
            max_initial_velocity=train_args.max_initial_velocity,
            max_initial_angle=train_args.max_initial_angle,
            max_initial_omega=train_args.max_initial_omega,
            disturbance_force_max=train_args.disturbance_force_max,
            external_force_ratio=train_args.external_force_ratio,
        )
    )
    loss_config = L2FLossConfig(
        p_scale=train_args.p_scale,
        v_scale=train_args.v_scale,
        omega_scale=train_args.omega_scale,
        huber_beta=train_args.huber_beta,
        w_p=train_args.w_p,
        w_v=train_args.w_v,
        w_omega=train_args.w_omega,
    )
    checkpoint_path = Path(train_args.init_checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    policy, _ = load_q2_policy(checkpoint_path, device=device, dtype=torch.float32)
    policy.train()
    state_step_decay = resolve_step_gradient_decay(
        mode=train_args.gradient_decay_mode,
        dt=train_args.dt,
        current_base=train_args.state_grad_decay,
        alpha=train_args.state_grad_alpha,
    )
    hidden_step_decay = resolve_step_gradient_decay(
        mode=train_args.gradient_decay_mode,
        dt=train_args.dt,
        current_base=train_args.hidden_grad_decay,
        alpha=train_args.hidden_grad_alpha,
    )

    # Match train.py: reset the paired sampler stream after policy construction.
    torch.manual_seed(train_args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_args.seed)
    retain_bank_path = Path(train_args.retain_bank_path)
    if not retain_bank_path.is_absolute():
        retain_bank_path = ROOT / retain_bank_path
    schedule_path = None
    if args.training_schedule is not None:
        schedule_path = (
            args.training_schedule
            if args.training_schedule.is_absolute()
            else ROOT / args.training_schedule
        )
        schedule = _load_training_schedule(
            schedule_path, device=device, dtype=torch.float32
        )
        if schedule.batch_indices.shape[1] != args.batch_size:
            raise ValueError(
                "training schedule batch size does not match --batch-size: "
                f"{schedule.batch_indices.shape[1]} != {args.batch_size}"
            )
        if schedule.update_batch_ids.numel() == 1:
            update_batch_ids = schedule.update_batch_ids.expand(args.updates)
        elif schedule.update_batch_ids.numel() < args.updates:
            raise ValueError(
                "training schedule has fewer routed updates than --updates: "
                f"{schedule.update_batch_ids.numel()} < {args.updates}"
            )
        else:
            update_batch_ids = schedule.update_batch_ids[: args.updates]
    else:
        initial_state = sim.reset(
            args.batch_size,
            device=device,
            sample_dynamics=train_args.sample_dynamics,
            sampled_dynamics_level=train_args.sampled_dynamics_level,
            broad_sampler=train_args.broad_sampler,
            balanced_dynamics_sampling=train_args.balanced_dynamics_sampling,
            sample_external_force=not train_args.disable_sampled_external_force,
        )
        retain_bank = load_retain_bank(retain_bank_path)
        reset_mask = torch.ones(args.batch_size, device=device, dtype=torch.bool)
        retain_mask, retain_indices = apply_retain_bank_samples(
            initial_state,
            retain_bank,
            reset_mask,
            fraction=train_args.retain_fraction,
        )
        schedule = TrainingSchedule(
            state=initial_state,
            retain_mask=retain_mask,
            retain_indices=retain_indices,
            batch_indices=torch.arange(
                args.batch_size, device=device, dtype=torch.long
            ).unsqueeze(0),
            update_batch_ids=torch.zeros(1, device=device, dtype=torch.long),
        )
        update_batch_ids = schedule.update_batch_ids.expand(args.updates)

    initial_batches: list[RecurrentSystemState] = []
    retain_mask_batches: list[torch.Tensor] = []
    scenario_indices_by_batch: list[torch.Tensor] = []
    for indices in schedule.batch_indices:
        batch_state = _select_state(schedule.state, indices)
        initial_batches.append(
            RecurrentSystemState(
                state=clone_state(batch_state),
                hidden=policy.initial_hidden(
                    args.batch_size, device=device, dtype=batch_state.position.dtype
                ),
                integral=initial_observation_state(
                    args.batch_size, device=device, dtype=batch_state.position.dtype
                ).integral_position,
            )
        )
        retain_mask_batches.append(schedule.retain_mask.index_select(0, indices))
        scenario_indices_by_batch.append(indices)
    initial_signature = _state_signature(
        schedule.state,
        schedule.retain_mask,
        schedule.retain_indices,
        schedule.batch_indices,
        update_batch_ids,
    )

    shooting_boundary_banks: torch.nn.ModuleList | None = None
    if args.mode == MODE_MULTIPLE_SHOOTING:
        shooting_boundary_banks = torch.nn.ModuleList(
            [
                _initialize_shooting_boundaries(
                    policy=policy,
                    sim=sim,
                    initial=initial,
                    train_args=train_args,
                    loss_config=loss_config,
                    state_step_decay=state_step_decay,
                    hidden_step_decay=hidden_step_decay,
                    retain_mask=batch_retain_mask,
                    segment_steps=args.segment_steps,
                    segment_count=args.segments,
                    backend=backend,
                )
                for initial, batch_retain_mask in zip(
                    initial_batches, retain_mask_batches
                )
            ]
        ).to(device)

    if args.future_credit_probe_steps > 0 and len(initial_batches) != 1:
        raise ValueError(
            "future credit probe is only defined for a single fixed training batch"
        )

    future_credit_rows = (
        _future_only_credit_probe(
            policy=policy,
            sim=sim,
            initial=initial_batches[0],
            train_args=train_args,
            loss_config=loss_config,
            state_step_decay=state_step_decay,
            hidden_step_decay=hidden_step_decay,
            retain_mask=retain_mask_batches[0],
            segment_steps=args.segment_steps,
            segment_count=args.segments,
            batch_size=args.batch_size,
            backend=backend,
            shooting_lr=args.shooting_lr,
            penalty_rho=args.penalty_rho,
            steps=args.future_credit_probe_steps,
        )
        if shooting_boundary_banks is not None
        else []
    )

    policy_parameters = tuple(policy.parameters())
    optimizer_groups: list[dict[str, Any]] = [
        {
            "params": policy_parameters,
            "lr": train_args.lr,
            "weight_decay": train_args.weight_decay,
        }
    ]
    shooting_parameters: tuple[torch.nn.Parameter, ...] = ()
    if shooting_boundary_banks is not None:
        shooting_parameters = tuple(shooting_boundary_banks.parameters())
        optimizer_groups.append(
            {
                "params": shooting_parameters,
                "lr": args.shooting_lr,
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=train_args.lr,
        weight_decay=train_args.weight_decay,
    )
    dual_banks: list[list[dict[str, torch.Tensor]] | None] = [
        None for _ in initial_batches
    ]
    rows: list[dict[str, Any]] = []
    stop_reason = "completed"
    batch_visit_counts = [0 for _ in initial_batches]

    for update in range(1, args.updates + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        active_batch_id = int(update_batch_ids[update - 1].item())
        batch_visit_counts[active_batch_id] += 1
        initial = initial_batches[active_batch_id]
        retain_mask = retain_mask_batches[active_batch_id]
        shooting_boundaries = (
            shooting_boundary_banks[active_batch_id]
            if shooting_boundary_banks is not None
            else None
        )
        duals = dual_banks[active_batch_id]
        results, shooting_states = _rollout_segments(
            policy=policy,
            sim=sim,
            initial=initial,
            shooting_boundaries=shooting_boundaries,
            train_args=train_args,
            loss_config=loss_config,
            state_step_decay=state_step_decay,
            hidden_step_decay=hidden_step_decay,
            retain_mask=retain_mask,
            mode=args.mode,
            update_index=update,
            segment_steps=args.segment_steps,
            segment_count=args.segments,
            batch_size=args.batch_size,
            backend=backend,
        )
        task_loss, early_tail, final_tail = q2_h1000_task_loss(results, train_args)
        residuals = (
            [
                continuity_residuals(results[index].end, shooting_states[index])
                for index in range(args.segments - 1)
            ]
            if args.mode == MODE_MULTIPLE_SHOOTING
            else []
        )
        if residuals and duals is None:
            duals = zero_duals(residuals)
            dual_banks[active_batch_id] = duals
        if residuals:
            assert duals is not None
            continuity_loss, boundary_terms = augmented_lagrangian_terms(
                residuals, duals, rho=args.penalty_rho
            )
        else:
            continuity_loss = task_loss * 0.0
            boundary_terms = []
        objective = task_loss + continuity_loss

        segment_gradient_norms: list[float] = []
        for result in results:
            segment_gradients = torch.autograd.grad(
                result.task_loss / float(args.segments),
                policy_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            segment_gradient_norms.append(_flat_grad_norm(segment_gradients))
        continuity_policy_gradient_norms: list[float] = []
        for boundary_term in boundary_terms:
            gradients = torch.autograd.grad(
                boundary_term,
                policy_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            continuity_policy_gradient_norms.append(_flat_grad_norm(gradients))

        final_objective = (
            results[-1].task_loss / float(args.segments)
            + float(train_args.final_tail_weight) * final_tail
        )
        direct_future_adjoint = torch.autograd.grad(
            final_objective,
            results[0].first_action,
            retain_graph=True,
            allow_unused=True,
        )[0]
        continuity_adjoint = (
            torch.autograd.grad(
                continuity_loss,
                results[0].first_action,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if residuals
            else None
        )
        final_task_boundary_grad = 0.0
        if shooting_states:
            boundary_gradients = torch.autograd.grad(
                final_objective,
                recurrent_tensors(shooting_states[-1]),
                retain_graph=True,
                allow_unused=True,
            )
            final_task_boundary_grad = _tensor_list_norm(boundary_gradients)

        objective.backward()
        policy_grad_norm = _parameter_grad_norm(policy_parameters)
        shooting_grad_norm = _parameter_grad_norm(shooting_parameters)
        gradients_finite = _all_parameter_grads_finite(
            policy_parameters + shooting_parameters
        )
        continuity_values = _continuity_metrics(residuals)
        finite = (
            bool(torch.isfinite(objective).item())
            and gradients_finite
            and all(math.isfinite(value) for value in continuity_values.values())
        )
        skipped = False
        skip_reason = ""
        clip_scale = 1.0
        policy_grad_after_clip = policy_grad_norm
        if not finite:
            skipped = True
            skip_reason = "nonfinite"
        elif policy_grad_norm > args.abort_policy_grad_norm:
            skipped = True
            skip_reason = "policy_gradient_explosion"
        elif shooting_grad_norm > args.abort_shooting_grad_norm:
            skipped = True
            skip_reason = "shooting_gradient_explosion"
        elif continuity_values["continuity_rms"] > args.abort_continuity_rms:
            skipped = True
            skip_reason = "continuity_divergence"
        else:
            _, policy_grad_after_clip, clip_scale = train._apply_fp64_global_grad_clip(
                policy, train_args.grad_clip
            )
            optimizer.step()
            if (
                residuals
                and args.dual_update_every > 0
                and update % args.dual_update_every == 0
            ):
                assert duals is not None
                update_duals_(duals, residuals, rho=args.penalty_rho)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_seconds = time.perf_counter() - update_started

        row: dict[str, Any] = {
            "phase": "train",
            "mode": args.mode,
            "update": update,
            "schedule_batch_id": active_batch_id,
            "schedule_batch_visit": batch_visit_counts[active_batch_id],
            "scenario_indices": ";".join(
                str(int(value))
                for value in scenario_indices_by_batch[active_batch_id].tolist()
            ),
            "physical_steps": update
            * args.batch_size
            * args.segment_steps
            * args.segments,
            "task_loss": float(task_loss.detach().item()),
            "early_tail_loss": float(early_tail.detach().item()),
            "final_tail_loss": float(final_tail.detach().item()),
            "continuity_objective": float(continuity_loss.detach().item()),
            "total_objective": float(objective.detach().item()),
            **continuity_values,
            "segment_1_gradient_norm": segment_gradient_norms[0],
            "segment_2_gradient_norm": segment_gradient_norms[1],
            "segment_3_gradient_norm": segment_gradient_norms[2],
            "segment_4_gradient_norm": segment_gradient_norms[3],
            "boundary_1_policy_gradient_norm": continuity_policy_gradient_norms[0]
            if continuity_policy_gradient_norms
            else 0.0,
            "boundary_2_policy_gradient_norm": continuity_policy_gradient_norms[1]
            if continuity_policy_gradient_norms
            else 0.0,
            "boundary_3_policy_gradient_norm": continuity_policy_gradient_norms[2]
            if continuity_policy_gradient_norms
            else 0.0,
            "global_policy_gradient_norm": policy_grad_norm,
            "policy_gradient_after_clip": policy_grad_after_clip,
            "shooting_gradient_norm": shooting_grad_norm,
            "clip_scale": clip_scale,
            "clipped": int(clip_scale < 0.999999),
            "skipped": int(skipped),
            "skip_reason": skip_reason,
            "all_finite": int(finite),
            "direct_final_task_to_segment_1_action": _tensor_list_norm(
                (direct_future_adjoint,)
            ),
            "continuity_to_segment_1_action": _tensor_list_norm((continuity_adjoint,)),
            "final_task_to_boundary_3_state": final_task_boundary_grad,
            "dual_1_rms": float(
                torch.sqrt(
                    torch.stack(
                        tuple(duals[0][name].square().mean() for name in CONTINUITY_FIELDS)
                    ).mean()
                ).item()
            )
            if duals
            else 0.0,
            "dual_2_rms": float(
                torch.sqrt(
                    torch.stack(
                        tuple(duals[1][name].square().mean() for name in CONTINUITY_FIELDS)
                    ).mean()
                ).item()
            )
            if duals
            else 0.0,
            "dual_3_rms": float(
                torch.sqrt(
                    torch.stack(
                        tuple(duals[2][name].square().mean() for name in CONTINUITY_FIELDS)
                    ).mean()
                ).item()
            )
            if duals
            else 0.0,
            "penalty_rho": args.penalty_rho,
            "update_seconds": update_seconds,
        }
        rows.append(row)
        if update == 1 or update % args.log_every == 0 or skipped:
            print(
                f"mode={args.mode} update={update}/{args.updates} "
                f"task={row['task_loss']:.6f} continuity={row['continuity_rms']:.3e} "
                f"grad={policy_grad_norm:.3e} shooting_grad={shooting_grad_norm:.3e} "
                f"cross={row['continuity_to_segment_1_action']:.3e} "
                f"clip={row['clipped']} skip={skip_reason}",
                flush=True,
            )
        if skipped:
            stop_reason = skip_reason
            break

    # Measure every persistent boundary bank after the final policy update.
    with torch.no_grad():
        final_tasks: list[torch.Tensor] = []
        final_residuals: list[dict[str, torch.Tensor]] = []
        for batch_id, (initial, retain_mask) in enumerate(
            zip(initial_batches, retain_mask_batches)
        ):
            shooting_boundaries = (
                shooting_boundary_banks[batch_id]
                if shooting_boundary_banks is not None
                else None
            )
            final_results, final_shooting_states = _rollout_segments(
                policy=policy,
                sim=sim,
                initial=initial,
                shooting_boundaries=shooting_boundaries,
                train_args=train_args,
                loss_config=loss_config,
                state_step_decay=state_step_decay,
                hidden_step_decay=hidden_step_decay,
                retain_mask=retain_mask,
                mode=args.mode,
                update_index=len(rows) + 1,
                segment_steps=args.segment_steps,
                segment_count=args.segments,
                batch_size=args.batch_size,
                backend=backend,
            )
            final_task_for_batch, _, _ = q2_h1000_task_loss(
                final_results, train_args
            )
            final_tasks.append(final_task_for_batch)
            if final_shooting_states:
                final_residuals.extend(
                    continuity_residuals(
                        final_results[index].end, final_shooting_states[index]
                    )
                    for index in range(args.segments - 1)
                )
        final_task = torch.stack(tuple(final_tasks)).mean()
        final_continuity = _continuity_metrics(final_residuals)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training.csv"
    _write_rows(log_path, rows)
    if future_credit_rows:
        _write_rows(args.output_dir / "future_credit_probe.csv", future_credit_rows)
    train_rows = [row for row in rows if row["phase"] == "train"]
    nonzero_cross = [
        float(row["continuity_to_segment_1_action"])
        for row in train_rows
        if float(row["continuity_to_segment_1_action"]) > 0.0
    ]
    disturbed_continuity = [
        float(row["continuity_rms"]) for row in train_rows[1:]
    ]
    summary = {
        "mode": args.mode,
        "requested_updates": args.updates,
        "completed_updates": sum(1 - int(row["skipped"]) for row in train_rows),
        "stop_reason": stop_reason,
        "initial_task_loss": float(train_rows[0]["task_loss"]),
        "final_task_loss": float(final_task.item()),
        "initial_continuity_rms": float(train_rows[0]["continuity_rms"]),
        "peak_disturbed_continuity_rms": max(disturbed_continuity, default=0.0),
        "final_continuity_rms": final_continuity["continuity_rms"],
        "final_continuity_max_abs": final_continuity["continuity_max_abs"],
        "continuity_tolerance": args.continuity_tolerance,
        "continuity_converged": int(
            args.mode == MODE_TBPTT
            or final_continuity["continuity_rms"] <= args.continuity_tolerance
        ),
        "max_global_policy_gradient_norm": max(
            float(row["global_policy_gradient_norm"]) for row in train_rows
        ),
        "max_shooting_gradient_norm": max(
            float(row["shooting_gradient_norm"]) for row in train_rows
        ),
        "clip_count": sum(int(row["clipped"]) for row in train_rows),
        "skip_count": sum(int(row["skipped"]) for row in train_rows),
        "all_finite": int(all(int(row["all_finite"]) for row in train_rows)),
        "max_continuity_to_segment_1_action": max(nonzero_cross, default=0.0),
        "final_continuity_to_segment_1_action": float(
            train_rows[-1]["continuity_to_segment_1_action"]
        ),
        "cross_segment_credit_nonzero": int(bool(nonzero_cross)),
        "future_only_probe_steps": len(future_credit_rows),
        "future_only_first_nonzero_segment_1_step": next(
            (
                int(row["probe_step"])
                for row in future_credit_rows
                if float(row["continuity_to_segment_1_action"]) > 0.0
            ),
            0,
        ),
        "future_only_max_continuity_to_segment_1_action": max(
            (
                float(row["continuity_to_segment_1_action"])
                for row in future_credit_rows
            ),
            default=0.0,
        ),
        "future_only_cross_segment_credit_nonzero": int(
            any(
                float(row["continuity_to_segment_1_action"]) > 0.0
                for row in future_credit_rows
            )
        ),
        "initial_state_sha256": initial_signature,
        "retain_count": int(schedule.retain_mask.sum().item()),
        "batch_size": args.batch_size,
        "scenario_count": int(schedule.state.position.shape[0]),
        "schedule_batch_count": len(initial_batches),
        "schedule_batch_visit_min": min(batch_visit_counts),
        "schedule_batch_visit_max": max(batch_visit_counts),
        "horizon": args.segment_steps * args.segments,
        "segment_steps": args.segment_steps,
        "seed": train_args.seed,
        "state_step_decay": state_step_decay,
        "hidden_step_decay": hidden_step_decay,
        "policy_lr": train_args.lr,
        "shooting_lr": args.shooting_lr if shooting_boundary_banks is not None else 0.0,
        "penalty_rho": args.penalty_rho if shooting_boundary_banks is not None else 0.0,
        "dual_update_every": args.dual_update_every if shooting_boundary_banks is not None else 0,
        "mean_update_seconds": sum(float(row["update_seconds"]) for row in train_rows)
        / len(train_rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint = {
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "shooting_boundary_banks": shooting_boundary_banks.state_dict()
        if shooting_boundary_banks is not None
        else None,
        "dual_banks": dual_banks,
        "summary": summary,
    }
    torch.save(checkpoint, args.output_dir / "model.pt")
    metadata = {
        "base_config": str((args.base_config if args.base_config.is_absolute() else ROOT / args.base_config).resolve()),
        "base_config_sha256": _sha256(
            args.base_config if args.base_config.is_absolute() else ROOT / args.base_config
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "retain_bank": str(retain_bank_path.resolve()),
        "retain_bank_sha256": _sha256(retain_bank_path),
        "initial_state_sha256": initial_signature,
        "training_schedule": str(schedule_path.resolve()) if schedule_path else None,
        "training_schedule_sha256": _sha256(schedule_path) if schedule_path else None,
        "args": vars(args),
    }
    # pathlib values are converted explicitly for a portable manifest.
    metadata["args"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in metadata["args"].items()
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {log_path}")
    print(f"wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

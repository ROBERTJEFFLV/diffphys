from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import load_q2_policy  # noqa: E402
from env_l2f import L2FParams, L2FSimulator, L2FState  # noqa: E402
from equilibrium_control import (  # noqa: E402
    EquilibriumCenteredPolicy,
    analytic_equilibrium_target,
    equilibrium_invariance_loss,
)
from policy_observation import initial_observation_state  # noqa: E402
from checkpointed_exact_bptt import (  # noqa: E402
    RecurrentSystemState,
    StrictShootingConfig,
    strict_multiple_shooting_objective,
)


DEFAULT_CHECKPOINT = (
    ROOT
    / "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the independent equilibrium-centered strict-MS branch."
    )
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/equilibrium_strict")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("torch", "cuda"), default="torch")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--segment-steps", type=int, default=250)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--energy-interval", type=int, default=25)
    parser.add_argument("--state-decay-base", type=float, default=1.0)
    parser.add_argument("--hidden-decay-base", type=float, default=1.0)
    parser.add_argument("--contraction-rate", type=float, default=0.25)
    parser.add_argument("--contraction-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--contraction-weight", type=float, default=1.0e-9)
    parser.add_argument("--prediction-weight", type=float, default=1.0e-4)
    parser.add_argument("--invariance-weight", type=float, default=1.0)
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--warmup-segments", type=int, default=1)
    parser.add_argument("--warmup-contraction-weight", type=float, default=2.0e-6)
    parser.add_argument("--warmup-prediction-weight", type=float, default=1.0)
    parser.add_argument("--warmup-state-decay-base", type=float, default=0.5)
    parser.add_argument("--warmup-hidden-decay-base", type=float, default=0.7)
    parser.add_argument("--cvar-fraction", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=5.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--grad-skip-threshold", type=float, default=1.0e3)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--fixed-dynamics", action="store_true")
    parser.add_argument("--no-external-force", action="store_true")
    return parser.parse_args()


def _all_finite_state(state: L2FState) -> bool:
    return all(
        bool(torch.isfinite(getattr(state, name)).all().item())
        for name in (
            "position",
            "velocity",
            "rotation",
            "omega",
            "motor",
            "previous_action",
        )
    )


def _gradient_norm(policy: torch.nn.Module) -> torch.Tensor:
    total = torch.zeros((), device=next(policy.parameters()).device, dtype=torch.float64)
    for parameter in policy.parameters():
        if parameter.grad is not None:
            total = total + parameter.grad.detach().double().square().sum()
    return torch.sqrt(total)


def _clip_gradient(policy: torch.nn.Module, maximum: float) -> tuple[float, float]:
    before_tensor = _gradient_norm(policy)
    before = float(before_tensor.item())
    scale = 1.0
    if maximum > 0.0 and math.isfinite(before):
        scale = min(1.0, float(maximum) / (before + 1.0e-12))
        if scale < 1.0:
            for parameter in policy.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
    return before, scale


def _checkpoint_payload(
    policy: EquilibriumCenteredPolicy,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    optimizer_updates: int,
    *,
    attempted_update: int | None = None,
    physical_steps: int = 0,
) -> dict[str, Any]:
    attempted = optimizer_updates if attempted_update is None else attempted_update
    return {
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "architecture": policy.architecture_metadata(),
        "args": vars(args),
        "optimizer_updates": optimizer_updates,
        "attempted_update": attempted,
        "physical_steps": physical_steps,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "strict_continuity": "exact direct substitution with checkpoint recomputation",
    }


def main() -> None:
    args = _parse_args()
    if args.updates < 1 or args.batch_size < 1:
        raise ValueError("updates and batch-size must be positive")
    if not 0 <= args.warmup_updates <= args.updates:
        raise ValueError("warmup-updates must be in [0, updates]")
    if not 1 <= args.warmup_segments <= args.segments:
        raise ValueError("warmup-segments must be in [1, segments]")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.backend == "cuda" and device.type != "cuda":
        raise ValueError("the CUDA backend requires --device cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source_path = args.source_checkpoint.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    source, checkpoint_args = load_q2_policy(
        source_path,
        device=device,
        dtype=torch.float32,
    )
    policy = EquilibriumCenteredPolicy.from_motor_gru(source).train()
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    simulator = L2FSimulator(L2FParams(dt=float(checkpoint_args.get("dt", 0.01))))
    shooting_config = StrictShootingConfig(
        segment_steps=args.segment_steps,
        segment_count=args.segments,
        energy_interval_steps=args.energy_interval,
        state_step_decay=float(args.state_decay_base) ** simulator.params.dt,
        hidden_step_decay=float(args.hidden_decay_base) ** simulator.params.dt,
        backend=args.backend,
        use_checkpoint_recompute=True,
    )
    shooting_config.validate()
    fieldnames = (
        "update",
        "phase",
        "segments",
        "horizon",
        "physical_steps",
        "contraction_weight",
        "prediction_weight",
        "loss",
        "weighted_contraction",
        "raw_contraction",
        "raw_mean_violation",
        "raw_violation_cvar",
        "terminal_energy",
        "weighted_prediction",
        "raw_prediction",
        "weighted_invariance",
        "raw_invariance",
        "invariance_physics",
        "invariance_hidden",
        "feasible_fraction",
        "continuity_rms",
        "grad_norm_before_clip",
        "grad_scale",
        "update_applied",
        "skip_reason",
        "final_position",
        "final_velocity",
        "final_omega",
        "elapsed_seconds",
    )
    torch.save(
        _checkpoint_payload(policy, optimizer, args, 0, physical_steps=0),
        args.output_dir / "model_initial.pt",
    )
    log_path = args.output_dir / "train.csv"
    cumulative_physical_steps = 0
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for update in range(1, args.updates + 1):
            started = time.perf_counter()
            warmup = update <= args.warmup_updates
            active_config = replace(
                shooting_config,
                segment_count=(args.warmup_segments if warmup else args.segments),
                state_step_decay=(
                    float(args.warmup_state_decay_base) ** simulator.params.dt
                    if warmup
                    else shooting_config.state_step_decay
                ),
                hidden_step_decay=(
                    float(args.warmup_hidden_decay_base) ** simulator.params.dt
                    if warmup
                    else shooting_config.hidden_step_decay
                ),
            )
            contraction_weight = (
                args.warmup_contraction_weight if warmup else args.contraction_weight
            )
            prediction_weight = (
                args.warmup_prediction_weight if warmup else args.prediction_weight
            )
            cumulative_physical_steps += (
                args.batch_size * active_config.segment_steps * active_config.segment_count
            )
            state = simulator.reset(
                args.batch_size,
                device=device,
                dtype=torch.float32,
                sample_dynamics=not args.fixed_dynamics,
                sampled_dynamics_level="broad",
                broad_sampler="physical-fit",
                balanced_dynamics_sampling=False,
                sample_external_force=not args.no_external_force,
            )
            target = analytic_equilibrium_target(
                state,
                gravity=simulator.params.gravity,
            )
            if not bool(target.feasible.any().item()):
                raise RuntimeError("sampled batch contains no feasible equilibrium")
            initial = RecurrentSystemState(
                state=state,
                hidden=policy.initial_hidden(
                    args.batch_size,
                    device=device,
                    dtype=torch.float32,
                ),
                integral=initial_observation_state(
                    args.batch_size,
                    device=device,
                    dtype=torch.float32,
                ).integral_position,
            )
            optimizer.zero_grad(set_to_none=True)
            strict = strict_multiple_shooting_objective(
                policy,
                simulator,
                initial,
                target,
                shooting_config=active_config,
                contraction_rate=args.contraction_rate,
                contraction_epsilon=args.contraction_epsilon,
                cvar_fraction=args.cvar_fraction,
                contraction_weight=contraction_weight,
                prediction_weight=prediction_weight,
            )
            invariance = equilibrium_invariance_loss(
                policy,
                simulator,
                state,
                target,
                integral_input_frame="body",
            )
            total = strict.loss + float(args.invariance_weight) * invariance.loss
            skip_reason = ""
            applied = False
            grad_before = float("nan")
            grad_scale = float("nan")
            if not bool(torch.isfinite(total).item()) or not _all_finite_state(strict.rollout.end.state):
                skip_reason = "loss_or_state_nonfinite"
            else:
                total.backward()
                grad_before, grad_scale = _clip_gradient(policy, args.grad_clip)
                if not math.isfinite(grad_before):
                    skip_reason = "gradient_nonfinite"
                elif args.grad_skip_threshold > 0.0 and grad_before > args.grad_skip_threshold:
                    skip_reason = "gradient_above_abort_threshold"
                else:
                    optimizer.step()
                    applied = True
            end = strict.rollout.end.state
            row = {
                "update": update,
                "phase": (
                    f"H{active_config.horizon}-local-warmup"
                    if warmup
                    else f"H{active_config.horizon}-strict"
                ),
                "segments": active_config.segment_count,
                "horizon": active_config.horizon,
                "physical_steps": cumulative_physical_steps,
                "contraction_weight": contraction_weight,
                "prediction_weight": prediction_weight,
                "loss": float(total.detach().item()),
                "weighted_contraction": float(
                    (contraction_weight * strict.contraction.loss).detach().item()
                ),
                "raw_contraction": float(strict.contraction.loss.detach().item()),
                "raw_mean_violation": float(strict.contraction.mean_violation.detach().item()),
                "raw_violation_cvar": float(strict.contraction.violation_cvar.detach().item()),
                "terminal_energy": float(strict.contraction.terminal_energy.detach().item()),
                "weighted_prediction": float(
                    (prediction_weight * strict.rollout.prediction_loss).detach().item()
                ),
                "raw_prediction": float(strict.rollout.prediction_loss.detach().item()),
                "weighted_invariance": float(
                    (args.invariance_weight * invariance.loss).detach().item()
                ),
                "raw_invariance": float(invariance.loss.detach().item()),
                "invariance_physics": float(invariance.physics_loss.detach().item()),
                "invariance_hidden": float(invariance.hidden_loss.detach().item()),
                "feasible_fraction": float(target.feasible.float().mean().item()),
                "continuity_rms": 0.0,
                "grad_norm_before_clip": grad_before,
                "grad_scale": grad_scale,
                "update_applied": int(applied),
                "skip_reason": skip_reason,
                "final_position": float(torch.linalg.vector_norm(end.position, dim=-1).mean().item()),
                "final_velocity": float(torch.linalg.vector_norm(end.velocity, dim=-1).mean().item()),
                "final_omega": float(torch.linalg.vector_norm(end.omega, dim=-1).mean().item()),
                "elapsed_seconds": time.perf_counter() - started,
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row, sort_keys=True), flush=True)
            if skip_reason:
                torch.save(
                    _checkpoint_payload(
                        policy,
                        optimizer,
                        args,
                        update - 1,
                        attempted_update=update,
                        physical_steps=cumulative_physical_steps,
                    ),
                    args.output_dir / "model_aborted.pt",
                )
                raise RuntimeError(f"training stopped safely: {skip_reason}")
            if args.checkpoint_every > 0 and update % args.checkpoint_every == 0:
                torch.save(
                    _checkpoint_payload(
                        policy,
                        optimizer,
                        args,
                        update,
                        physical_steps=cumulative_physical_steps,
                    ),
                    args.output_dir / f"model_update_{update}.pt",
                )
    torch.save(
        _checkpoint_payload(
            policy,
            optimizer,
            args,
            args.updates,
            physical_steps=cumulative_physical_steps,
        ),
        args.output_dir / "model.pt",
    )
    manifest = {
        "architecture": policy.architecture_metadata(),
        "source_checkpoint": str(source_path),
        "strict_continuity": True,
        "shooting_variables_in_optimizer": False,
        "shooting_lr": None,
        "updates": args.updates,
        "physical_steps": cumulative_physical_steps,
        "warmup_updates": args.warmup_updates,
        "warmup_horizon": args.warmup_segments * args.segment_steps,
        "device": str(device),
        "backend": args.backend,
    }
    (args.output_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

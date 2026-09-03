from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import load_q2_policy  # noqa: E402
from env_l2f import L2FParams, L2FSimulator  # noqa: E402
from equilibrium_control import (  # noqa: E402
    EquilibriumCenteredPolicy,
    analytic_equilibrium_target,
    equilibrium_invariance_loss,
)
from policy_observation import initial_observation_state  # noqa: E402
from checkpointed_exact_bptt import (  # noqa: E402
    RecurrentSystemState,
    StrictShootingConfig,
    recurrent_tensors,
    strict_multiple_shooting_objective,
)


DEFAULT_CHECKPOINT = (
    ROOT
    / "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate exact H1000 segment recomputation against full BPTT."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/equilibrium_strict_h1000_validation.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--segment-steps", type=int, default=250)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--energy-interval", type=int, default=25)
    parser.add_argument("--state-decay-base", type=float, default=1.0)
    parser.add_argument("--hidden-decay-base", type=float, default=1.0)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0e4)
    parser.add_argument("--skip-direct", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flat_gradient(
    loss: torch.Tensor,
    policy: torch.nn.Module,
) -> torch.Tensor:
    parameters = tuple(policy.parameters())
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat(
        tuple(
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        )
    ).detach()


def _run(
    policy: EquilibriumCenteredPolicy,
    simulator: L2FSimulator,
    initial: RecurrentSystemState,
    target: Any,
    config: StrictShootingConfig,
    *,
    collect_boundary_credit: bool,
) -> tuple[dict[str, Any], torch.Tensor]:
    device = initial.state.position.device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    objective = strict_multiple_shooting_objective(
        policy,
        simulator,
        initial,
        target,
        shooting_config=config,
    )
    if collect_boundary_credit and len(objective.rollout.boundaries) > 1:
        first_boundary = recurrent_tensors(objective.rollout.boundaries[0])
        boundary_gradients = torch.autograd.grad(
            objective.rollout.energy[-1].sum(),
            first_boundary,
            retain_graph=True,
            allow_unused=True,
        )
        boundary_flat = torch.cat(
            tuple(
                (torch.zeros_like(value) if gradient is None else gradient).reshape(-1)
                for value, gradient in zip(first_boundary, boundary_gradients)
            )
        )
        boundary_norm = float(torch.linalg.vector_norm(boundary_flat.double()).item())
        boundary_finite = bool(torch.isfinite(boundary_flat).all().item())
    else:
        boundary_norm = None
        boundary_finite = None
    gradient = _flat_gradient(objective.loss, policy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    elapsed = time.perf_counter() - started
    end = objective.rollout.end.state
    row = {
        "loss": float(objective.loss.detach().item()),
        "prediction_loss": float(objective.rollout.prediction_loss.detach().item()),
        "contraction_mean_violation": float(objective.contraction.mean_violation.detach().item()),
        "contraction_cvar": float(objective.contraction.violation_cvar.detach().item()),
        "terminal_energy": float(objective.contraction.terminal_energy.detach().item()),
        "terminal_energy_cvar": float(objective.contraction.terminal_cvar.detach().item()),
        "gradient_norm": float(torch.linalg.vector_norm(gradient.double()).item()),
        "gradient_max_abs": float(gradient.abs().max().item()),
        "gradient_finite": bool(torch.isfinite(gradient).all().item()),
        "first_boundary_to_terminal_gradient_norm": boundary_norm,
        "first_boundary_to_terminal_gradient_finite": boundary_finite,
        "continuity_rms": 0.0,
        "continuity_enforcement": "exact direct substitution",
        "final_position_norm": float(torch.linalg.vector_norm(end.position, dim=-1).mean().item()),
        "final_velocity_norm": float(torch.linalg.vector_norm(end.velocity, dim=-1).mean().item()),
        "final_omega_norm": float(torch.linalg.vector_norm(end.omega, dim=-1).mean().item()),
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": peak_memory,
    }
    return row, gradient


def main() -> None:
    args = _parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    source, checkpoint_args = load_q2_policy(
        checkpoint_path,
        device=device,
        dtype=torch.float32,
    )
    converted = EquilibriumCenteredPolicy.from_motor_gru(source)
    simulator = L2FSimulator(L2FParams(dt=float(checkpoint_args.get("dt", 0.01))))
    state = simulator.reset(
        args.batch_size,
        device=device,
        dtype=torch.float32,
        sample_dynamics=True,
        sampled_dynamics_level="broad",
        broad_sampler="physical-fit",
        balanced_dynamics_sampling=False,
        sample_external_force=True,
    )
    target = analytic_equilibrium_target(state, gravity=simulator.params.gravity)
    if not bool(target.feasible.any().item()):
        raise RuntimeError("seeded validation batch contains no feasible equilibrium")
    initial = RecurrentSystemState(
        state=state,
        hidden=converted.initial_hidden(
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
    common_config = dict(
        segment_steps=args.segment_steps,
        segment_count=args.segments,
        energy_interval_steps=args.energy_interval,
        state_step_decay=float(args.state_decay_base) ** simulator.params.dt,
        hidden_step_decay=float(args.hidden_decay_base) ** simulator.params.dt,
        backend="torch",
    )
    payload: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "seed": args.seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "batch_size": args.batch_size,
        "horizon": args.segment_steps * args.segments,
        "segment_steps": args.segment_steps,
        "segments": args.segments,
        "energy_interval": args.energy_interval,
        "state_decay_base": args.state_decay_base,
        "hidden_decay_base": args.hidden_decay_base,
        "max_gradient_norm": args.max_gradient_norm,
        "feasible_fraction": float(target.feasible.float().mean().item()),
        "policy": converted.architecture_metadata(),
    }
    direct_gradient: torch.Tensor | None = None
    if not args.skip_direct:
        direct_policy = copy.deepcopy(converted)
        direct_row, direct_gradient = _run(
            direct_policy,
            simulator,
            initial,
            target,
            StrictShootingConfig(
                **common_config,
                use_checkpoint_recompute=False,
            ),
            collect_boundary_credit=False,
        )
        payload["full_bptt"] = direct_row
        del direct_policy
    recompute_policy = copy.deepcopy(converted)
    recompute_row, recompute_gradient = _run(
        recompute_policy,
        simulator,
        initial,
        target,
        StrictShootingConfig(
            **common_config,
            use_checkpoint_recompute=True,
        ),
        collect_boundary_credit=True,
    )
    payload["strict_segment_recompute"] = recompute_row
    if direct_gradient is not None:
        difference = recompute_gradient.double() - direct_gradient.double()
        denominator = torch.linalg.vector_norm(direct_gradient.double()).clamp_min(1.0e-30)
        cosine = torch.nn.functional.cosine_similarity(
            direct_gradient.double().unsqueeze(0),
            recompute_gradient.double().unsqueeze(0),
        ).item()
        payload["gradient_equivalence"] = {
            "cosine_similarity": float(cosine),
            "relative_l2_error": float(torch.linalg.vector_norm(difference).item() / denominator.item()),
            "max_abs_error": float(difference.abs().max().item()),
            "loss_abs_error": abs(
                payload["full_bptt"]["loss"] - recompute_row["loss"]
            ),
        }

    invariance_policy = copy.deepcopy(converted)
    invariance = equilibrium_invariance_loss(
        invariance_policy,
        simulator,
        state,
        target,
        integral_input_frame="body",
    )
    invariance_gradient = _flat_gradient(invariance.loss, invariance_policy)
    payload["equilibrium_invariance"] = {
        "loss": float(invariance.loss.detach().item()),
        "physics_loss": float(invariance.physics_loss.detach().item()),
        "hidden_loss": float(invariance.hidden_loss.detach().item()),
        "direction_loss": float(invariance.direction_loss.detach().item()),
        "trim_loss": float(invariance.trim_loss.detach().item()),
        "gradient_norm": float(torch.linalg.vector_norm(invariance_gradient.double()).item()),
        "gradient_finite": bool(torch.isfinite(invariance_gradient).all().item()),
    }
    all_finite = recompute_row["gradient_finite"] and math.isfinite(recompute_row["loss"])
    all_finite = all_finite and payload["equilibrium_invariance"]["gradient_finite"]
    payload["passed"] = bool(
        all_finite
        and recompute_row["gradient_norm"] <= args.max_gradient_norm
        and recompute_row["first_boundary_to_terminal_gradient_norm"] > 0.0
        and (
            direct_gradient is None
            or payload["gradient_equivalence"]["cosine_similarity"] > 0.99999
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

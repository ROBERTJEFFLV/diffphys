from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_l2f import L2FParams, L2FSimulator, L2FState  # noqa: E402
from l2f_cuda_backend import cuda_step, load_extension  # noqa: E402
from model import MotorGRUPolicy  # noqa: E402
from policy_observation import (  # noqa: E402
    build_policy_observation,
    initial_observation_state,
    mode_from_observation_dim,
    update_position_integral,
)
from retain_bank import (  # noqa: E402
    PHYSICAL_FIT_SAMPLER_SOURCE_METADATA_KEY,
    physical_fit_sampler_source_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen-baseline retain scenario bank.")
    parser.add_argument(
        "--checkpoint",
        default="reports/mainline_h500_finetune_safe_5000_lr5e6/model_step_4000.pt",
    )
    parser.add_argument("--output", default="reports/next_stage_retain_bank/retain_bank.pt")
    parser.add_argument("--csv", default="reports/next_stage_retain_bank/retain_bank.csv")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--sim-backend", default="cuda", choices=("cuda", "torch"))
    parser.add_argument("--seed", type=int, default=1007)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--steady-window-steps", type=int, default=100)
    parser.add_argument("--steady-required-fraction", type=float, default=0.95)
    parser.add_argument("--success-position", type=float, default=0.05)
    parser.add_argument("--success-velocity", type=float, default=0.10)
    parser.add_argument("--success-omega", type=float, default=0.20)
    parser.add_argument("--tail-fraction", type=float, default=0.20)
    parser.add_argument("--broad-sampler", default="physical", choices=("physical", "physical-fit"))
    return parser.parse_args()


def clone_state(state: L2FState) -> L2FState:
    return L2FState(**{name: getattr(state, name).detach().clone() for name in L2FState.__dataclass_fields__})


def finite_state(state: L2FState) -> torch.Tensor:
    result = torch.ones(state.position.shape[0], device=state.position.device, dtype=torch.bool)
    for name in ("position", "velocity", "rotation", "omega", "motor", "previous_action"):
        value = getattr(state, name).reshape(state.position.shape[0], -1)
        result &= torch.isfinite(value).all(dim=-1)
    return result


@torch.no_grad()
def evaluate_baseline(
    policy: MotorGRUPolicy,
    sim: L2FSimulator,
    state: L2FState,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = None
    observation_state = initial_observation_state(
        state.position.shape[0], device=state.position.device, dtype=state.position.dtype
    )
    mode = mode_from_observation_dim(policy.observation_dim)
    alive = torch.ones(state.position.shape[0], device=state.position.device, dtype=torch.bool)
    window: deque[torch.Tensor] = deque(maxlen=args.steady_window_steps)
    h500_success = torch.zeros_like(alive)
    for step in range(1, args.horizon + 1):
        observation, observed_position = build_policy_observation(
            state, observation_state, mode=mode
        )
        action, hidden = policy(observation, hidden)
        observation_state = update_position_integral(
            observation_state,
            observed_position,
            dt=sim.params.dt,
            integral_limit=0.5,
            integral_leak=0.0,
        )
        if args.sim_backend == "cuda":
            state = cuda_step(state, action, sim.params, grad_decay=1.0)
        else:
            state = sim.step(state, action, grad_decay=1.0)
        alive &= finite_state(state)
        step_success = (
            (state.position.norm(dim=-1) < args.success_position)
            & (state.velocity.norm(dim=-1) < args.success_velocity)
            & (state.omega.norm(dim=-1) < args.success_omega)
            & alive
        )
        window.append(step_success)
        if step == 500:
            fraction = torch.stack(tuple(window)).float().mean(dim=0)
            h500_success = (fraction >= args.steady_required_fraction) & alive
        if step % 1000 == 0 or step == args.horizon:
            print(f"baseline retain evaluation: H{step}/{args.horizon}", flush=True)
    final_fraction = torch.stack(tuple(window)).float().mean(dim=0)
    h10000_success = (final_fraction >= args.steady_required_fraction) & alive
    return h500_success, h10000_success


def main() -> None:
    args = parse_args()
    if args.horizon < 500 or args.steady_window_steps <= 0:
        raise ValueError("horizon must include H500 and steady window must be positive")
    if not 0.0 < args.tail_fraction <= 0.5:
        raise ValueError("tail fraction must be in (0, 0.5]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.sim_backend == "cuda":
        if device.type != "cuda":
            raise ValueError("compact CUDA requires --device cuda")
        load_extension()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    sim = L2FSimulator(
        L2FParams(
            dt=float(checkpoint_args.get("dt", 0.01)),
            max_initial_position=float(checkpoint_args.get("max_initial_position", 1.0)),
            max_initial_velocity=float(checkpoint_args.get("max_initial_velocity", 0.6)),
            max_initial_angle=float(checkpoint_args.get("max_initial_angle", 0.45)),
            max_initial_omega=float(checkpoint_args.get("max_initial_omega", 1.0)),
            disturbance_force_max=float(checkpoint_args.get("disturbance_force_max", 0.0)),
            external_force_ratio=float(checkpoint_args.get("external_force_ratio", 0.0)),
        )
    )
    policy = MotorGRUPolicy(
        encoder_dim=int(checkpoint_args.get("encoder_dim", 192)),
        hidden_dim=int(checkpoint_args.get("hidden_dim", 192)),
        encoder_depth=int(checkpoint_args.get("encoder_depth", 2)),
    ).to(device)
    policy.load_compatible_state_dict(checkpoint.get("model", checkpoint))
    policy.eval()

    initial_state = sim.reset(
        args.batch_size,
        device=device,
        sample_dynamics=True,
        sampled_dynamics_level="broad",
        broad_sampler=args.broad_sampler,
        balanced_dynamics_sampling=args.broad_sampler == "physical-fit",
        sample_external_force=True,
    )
    frozen_initial_state = clone_state(initial_state)
    h500_success, h10000_success = evaluate_baseline(policy, sim, initial_state, args)

    q = args.tail_fraction
    alpha_roll_cutoff = torch.quantile(frozen_initial_state.alpha_roll_max, q)
    alpha_yaw_cutoff = torch.quantile(frozen_initial_state.alpha_yaw_max, q)
    tau_fall_cutoff = torch.quantile(frozen_initial_state.motor_time_falling, 1.0 - q)
    low_roll = frozen_initial_state.alpha_roll_max <= alpha_roll_cutoff
    low_yaw = frozen_initial_state.alpha_yaw_max <= alpha_yaw_cutoff
    high_tau_fall = frozen_initial_state.motor_time_falling >= tau_fall_cutoff
    retain_mask = h10000_success & (low_roll | low_yaw | high_tau_fall)
    retain_indices = torch.nonzero(retain_mask, as_tuple=False).flatten()
    if retain_indices.numel() == 0:
        raise RuntimeError("fixed baseline produced an empty retain bank")

    state_payload = {
        name: getattr(frozen_initial_state, name)[retain_indices].detach().cpu()
        for name in L2FState.__dataclass_fields__
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state_payload,
        "scenario_id": retain_indices.detach().cpu(),
        "baseline_h500_success": h500_success[retain_indices].detach().cpu(),
        "baseline_h10000_success": h10000_success[retain_indices].detach().cpu(),
        "metadata": {
            "checkpoint": str(Path(args.checkpoint)),
            "seed": args.seed,
            "source_scenarios": args.batch_size,
            "horizon": args.horizon,
            "steady_window_steps": args.steady_window_steps,
            "steady_required_fraction": args.steady_required_fraction,
            "broad_sampler": args.broad_sampler,
            **(
                {
                    PHYSICAL_FIT_SAMPLER_SOURCE_METADATA_KEY:
                        physical_fit_sampler_source_sha256()
                }
                if args.broad_sampler == "physical-fit"
                else {}
            ),
            "tail_fraction": q,
            "alpha_roll_cutoff": float(alpha_roll_cutoff.item()),
            "alpha_yaw_cutoff": float(alpha_yaw_cutoff.item()),
            "tau_fall_cutoff": float(tau_fall_cutoff.item()),
        },
    }
    torch.save(payload, output)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        fieldnames = (
            "scenario_id",
            "baseline_h500_success",
            "baseline_h10000_success",
            "low_alpha_roll",
            "low_alpha_yaw",
            "large_tau_fall",
            "alpha_roll_max",
            "alpha_yaw_max",
            "tau_fall",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in retain_indices.tolist():
            writer.writerow(
                {
                    "scenario_id": index,
                    "baseline_h500_success": int(h500_success[index].item()),
                    "baseline_h10000_success": int(h10000_success[index].item()),
                    "low_alpha_roll": int(low_roll[index].item()),
                    "low_alpha_yaw": int(low_yaw[index].item()),
                    "large_tau_fall": int(high_tau_fall[index].item()),
                    "alpha_roll_max": float(frozen_initial_state.alpha_roll_max[index].item()),
                    "alpha_yaw_max": float(frozen_initial_state.alpha_yaw_max[index].item()),
                    "tau_fall": float(frozen_initial_state.motor_time_falling[index].item()),
                }
            )
    print(
        f"saved retain bank: {output} count={retain_indices.numel()} "
        f"source_H500={h500_success.float().mean().item():.6f} "
        f"source_H10000={h10000_success.float().mean().item():.6f}"
    )
    print(f"saved retain manifest: {csv_path}")


if __name__ == "__main__":
    main()

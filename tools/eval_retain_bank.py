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
from retain_bank import load_retain_bank  # noqa: E402
from policy_observation import (  # noqa: E402
    INTEGRAL_OBSERVATION_MODE,
    build_policy_observation,
    initial_observation_state,
    mode_from_observation_dim,
    observation_dim,
    update_position_integral,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline retention on the fixed retain bank.")
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument(
        "--baseline-checkpoint",
        default="reports/mainline_h500_finetune_safe_5000_lr5e6/model_step_4000.pt",
    )
    parser.add_argument("--retain-bank", default="reports/next_stage_retain_bank/retain_bank.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--sim-backend", default="cuda", choices=("cuda", "torch"))
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--steady-window-steps", type=int, default=100)
    parser.add_argument("--steady-required-fraction", type=float, default=0.95)
    return parser.parse_args()


def state_from_bank(path: str, device: torch.device) -> tuple[L2FState, torch.Tensor]:
    bank = load_retain_bank(path)
    state = L2FState(
        **{name: value.to(device) for name, value in bank.state.items()}
    )
    return state, bank.scenario_id


def clone_state(state: L2FState) -> L2FState:
    return L2FState(**{name: getattr(state, name).detach().clone() for name in L2FState.__dataclass_fields__})


def load_policy(path: str, device: torch.device) -> MotorGRUPolicy:
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    mode = args.get("observation_mode", None)
    if mode is None:
        state_dict = checkpoint.get("model", checkpoint)
        mode = mode_from_observation_dim(int(state_dict["encoder.0.weight"].shape[1]))
    policy = MotorGRUPolicy(
        observation_dim=observation_dim(mode),
        encoder_dim=int(args.get("encoder_dim", 192)),
        hidden_dim=int(args.get("hidden_dim", 192)),
        encoder_depth=int(args.get("encoder_depth", 2)),
        enable_integral_residual=bool(args.get("enable_integral_residual", False)),
        enable_damping_residual=bool(args.get("enable_rate_damping_residual", False)),
        integral_residual_hidden_dim=int(args.get("integral_residual_hidden_dim", 16)),
        damping_residual_hidden_dim=int(args.get("damping_residual_hidden_dim", 32)),
        integral_residual_scale=float(args.get("integral_residual_scale", 1.0)),
        damping_residual_scale=float(args.get("damping_residual_scale", 1.0)),
    ).to(device)
    policy.load_compatible_state_dict(checkpoint.get("model", checkpoint))
    policy.eval()
    policy.observation_mode = mode
    policy.integral_limit = float(args.get("integral_limit", 0.5))
    policy.integral_leak = float(args.get("integral_leak", 0.0))
    policy.integral_input_frame = str(args.get("integral_input_frame", "world"))
    return policy


@torch.no_grad()
def evaluate(
    policy: MotorGRUPolicy,
    sim: L2FSimulator,
    initial_state: L2FState,
    args: argparse.Namespace,
) -> dict[int, dict[str, torch.Tensor]]:
    state = clone_state(initial_state)
    hidden = None
    observation_state = initial_observation_state(
        state.position.shape[0], device=state.position.device, dtype=state.position.dtype
    )
    alive = torch.ones(state.position.shape[0], device=state.position.device, dtype=torch.bool)
    success_window: deque[torch.Tensor] = deque(maxlen=args.steady_window_steps)
    omega_window: deque[torch.Tensor] = deque(maxlen=args.steady_window_steps)
    results: dict[int, dict[str, torch.Tensor]] = {}
    for step in range(1, args.horizon + 1):
        observation, observed_position = build_policy_observation(
            state,
            observation_state,
            mode=policy.observation_mode,
            integral_input_frame=policy.integral_input_frame,
        )
        action, hidden = policy(observation, hidden)
        observation_state = update_position_integral(
            observation_state,
            observed_position,
            dt=sim.params.dt,
            integral_limit=policy.integral_limit,
            integral_leak=policy.integral_leak,
        )
        state = (
            cuda_step(state, action, sim.params, grad_decay=1.0)
            if args.sim_backend == "cuda"
            else sim.step(state, action, grad_decay=1.0)
        )
        for name in ("position", "velocity", "rotation", "omega", "motor", "previous_action"):
            alive &= torch.isfinite(getattr(state, name).reshape(state.position.shape[0], -1)).all(dim=-1)
        success_window.append(
            (state.position.norm(dim=-1) < 0.05)
            & (state.velocity.norm(dim=-1) < 0.10)
            & (state.omega.norm(dim=-1) < 0.20)
            & alive
        )
        omega_window.append(state.omega)
        if step in (500, args.horizon):
            fraction = torch.stack(tuple(success_window)).float().mean(dim=0)
            omega_rms = torch.stack(tuple(omega_window)).square().sum(dim=-1).mean(dim=0).sqrt()
            results[step] = {
                "steady": (fraction >= args.steady_required_fraction) & alive,
                "window_fraction": fraction,
                "omega_rms": omega_rms,
                "survival": alive.clone(),
            }
    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.sim_backend == "cuda":
        if device.type != "cuda" or not torch.cuda.is_available():
            raise SystemExit("compact CUDA retain evaluation requires CUDA")
        load_extension()
    state, scenario_ids = state_from_bank(args.retain_bank, device)
    sim = L2FSimulator(L2FParams())
    baseline = evaluate(load_policy(args.baseline_checkpoint, device), sim, state, args)
    candidate = evaluate(load_policy(args.candidate_checkpoint, device), sim, state, args)
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for index, scenario_id in enumerate(scenario_ids.tolist()):
        row: dict[str, object] = {"scenario_id": scenario_id}
        for horizon in (500, args.horizon):
            suffix = f"H{horizon}"
            baseline_success = bool(baseline[horizon]["steady"][index].item())
            candidate_success = bool(candidate[horizon]["steady"][index].item())
            row[f"baseline_steady_{suffix}"] = int(baseline_success)
            row[f"candidate_steady_{suffix}"] = int(candidate_success)
            row[f"baseline_window_fraction_{suffix}"] = float(baseline[horizon]["window_fraction"][index].item())
            row[f"candidate_window_fraction_{suffix}"] = float(candidate[horizon]["window_fraction"][index].item())
            row[f"baseline_omega_rms_{suffix}"] = float(baseline[horizon]["omega_rms"][index].item())
            row[f"candidate_omega_rms_{suffix}"] = float(candidate[horizon]["omega_rms"][index].item())
            row[f"baseline_success_to_candidate_failure_{suffix}"] = int(baseline_success and not candidate_success)
            row[f"baseline_failure_to_candidate_success_{suffix}"] = int(not baseline_success and candidate_success)
            row[f"candidate_window_failure_{suffix}"] = int(candidate[horizon]["window_fraction"][index] < args.steady_required_fraction)
            row[f"candidate_omega_failure_{suffix}"] = int(candidate[horizon]["omega_rms"][index] >= 0.20)
        rows.append(row)
    for horizon in (500, args.horizon):
        suffix = f"H{horizon}"
        baseline_success = baseline[horizon]["steady"]
        candidate_success = candidate[horizon]["steady"]
        summary_rows.append(
            {
                "horizon": horizon,
                "count": len(rows),
                "baseline_success_rate": float(baseline_success.float().mean().item()),
                "candidate_success_rate": float(candidate_success.float().mean().item()),
                "baseline_success_to_candidate_failure": int((baseline_success & ~candidate_success).sum().item()),
                "baseline_failure_to_candidate_success": int((~baseline_success & candidate_success).sum().item()),
                "candidate_window_failures": int((candidate[horizon]["window_fraction"] < args.steady_required_fraction).sum().item()),
                "candidate_omega_failures": int((candidate[horizon]["omega_rms"] >= 0.20).sum().item()),
                "candidate_survival_rate": float(candidate[horizon]["survival"].float().mean().item()),
            }
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in (("samples.csv", rows), ("summary.csv", summary_rows)):
        with (output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(values[0]))
            writer.writeheader()
            writer.writerows(values)
    print(f"saved retain evaluation: {output_dir}")


if __name__ == "__main__":
    main()

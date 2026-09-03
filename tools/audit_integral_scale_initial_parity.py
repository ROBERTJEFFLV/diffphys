from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import MotorGRUPolicy, compensate_integral_input_scale_  # noqa: E402


def _policy(checkpoint: dict[str, object]) -> MotorGRUPolicy:
    args = checkpoint["args"]
    policy = MotorGRUPolicy(
        observation_dim=25,
        encoder_dim=int(args["encoder_dim"]),
        hidden_dim=int(args["hidden_dim"]),
        encoder_depth=int(args["encoder_depth"]),
        enable_integral_residual=bool(args["enable_integral_residual"]),
        enable_damping_residual=bool(args["enable_rate_damping_residual"]),
        integral_residual_hidden_dim=int(args["integral_residual_hidden_dim"]),
        damping_residual_hidden_dim=int(args["damping_residual_hidden_dim"]),
        integral_residual_scale=float(args["integral_residual_scale"]),
        damping_residual_scale=float(args["damping_residual_scale"]),
    ).double().eval()
    policy.load_state_dict(checkpoint["model"])
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit compensated integral-scale parity.")
    parser.add_argument(
        "--checkpoint",
        default=(
            "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/"
            "checkpoints/model_update_2000.pt"
        ),
    )
    parser.add_argument("--multipliers", default="0.5,1,2")
    parser.add_argument(
        "--output",
        default="reports/integral_scale_h500_u1000_gpu2/initial_parity.csv",
    )
    args = parser.parse_args()
    checkpoint = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    base = _policy(checkpoint)
    torch.manual_seed(7103)
    observation = torch.randn(32, 25, dtype=torch.float64)
    observation[:, 18:21] = torch.empty(32, 3, dtype=torch.float64).uniform_(-0.5, 0.5)
    hidden = torch.randn(32, base.hidden_dim, dtype=torch.float64)
    with torch.no_grad():
        expected_action, expected_hidden = base(observation, hidden)
    rows: list[dict[str, object]] = []
    for multiplier in (float(value) for value in args.multipliers.split(",")):
        candidate = _policy(checkpoint)
        compensate_integral_input_scale_(candidate, multiplier)
        scaled_observation = observation.clone()
        scaled_observation[:, 18:21].mul_(multiplier)
        with torch.no_grad():
            action, next_hidden = candidate(scaled_observation, hidden)
        action_error = float((action - expected_action).abs().max())
        hidden_error = float((next_hidden - expected_hidden).abs().max())
        if action_error != 0.0 or hidden_error != 0.0:
            raise RuntimeError(
                f"multiplier {multiplier:g} changed initialization: "
                f"action={action_error} hidden={hidden_error}"
            )
        rows.append(
            {
                "integral_input_multiplier": multiplier,
                "action_max_abs_error": action_error,
                "hidden_max_abs_error": hidden_error,
            }
        )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()

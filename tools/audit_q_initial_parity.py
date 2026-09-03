from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import MotorGRUPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit zero-integral Q-branch parity against P4b.")
    parser.add_argument(
        "--checkpoint",
        default=(
            "reports/compact_cvar_ablation_h500_u2000_gpu2_fresh/seed_7/"
            "group_P4b/checkpoints/model_update_2000.pt"
        ),
    )
    parser.add_argument("--output", default="reports/q_residual_h500_u2000_gpu2/initial_parity.csv")
    parser.add_argument("--seed", type=int, default=317)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    checkpoint_args = checkpoint.get("args", {})
    common = {
        "observation_dim": 25,
        "encoder_dim": int(checkpoint_args.get("encoder_dim", 192)),
        "hidden_dim": int(checkpoint_args.get("hidden_dim", 192)),
        "encoder_depth": int(checkpoint_args.get("encoder_depth", 2)),
    }
    torch.manual_seed(args.seed)
    policies = {
        "Q0": MotorGRUPolicy(**common).eval(),
        "Q1": MotorGRUPolicy(**common, enable_integral_residual=True).eval(),
        "Q2": MotorGRUPolicy(
            **common,
            enable_integral_residual=True,
            enable_damping_residual=True,
        ).eval(),
    }
    for policy in policies.values():
        policy.load_compatible_state_dict(state_dict)
    observation = torch.randn(32, 25)
    observation[:, 18:21] = 0.0
    hidden = torch.randn(32, common["hidden_dim"])
    with torch.no_grad():
        reference_action, reference_hidden = policies["Q0"](observation, hidden)
        rows: list[dict[str, object]] = []
        for name, policy in policies.items():
            action, next_hidden, details = policy.forward_with_aux(observation, hidden)
            action_error = (action - reference_action).abs()
            hidden_error = (next_hidden - reference_hidden).abs()
            rows.append(
                {
                    "group": name,
                    "action_max_abs_error": float(action_error.max().item()),
                    "hidden_max_abs_error": float(hidden_error.max().item()),
                    "integral_contribution_max_abs": float(
                        details["integral_action_contribution"].abs().max().item()
                    ),
                    "damping_contribution_max_abs": float(
                        details["damping_action_contribution"].abs().max().item()
                    ),
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if any(float(row["action_max_abs_error"]) != 0.0 for row in rows):
        raise RuntimeError(f"zero-initialized residual action parity failed: {rows}")
    if any(float(row["hidden_max_abs_error"]) != 0.0 for row in rows):
        raise RuntimeError(f"zero-initialized hidden parity failed: {rows}")
    print(output)


if __name__ == "__main__":
    main()

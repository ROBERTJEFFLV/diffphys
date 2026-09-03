from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy.io
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import MotorGRUPolicy  # noqa: E402
from tools.export_motor_gru_to_mat import export_checkpoint_to_mat  # noqa: E402


def _matlab_quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate residual MotorGRU PyTorch/MATLAB parity.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=419)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    saved = checkpoint.get("args", {})
    input_dim = int(state_dict["encoder.0.weight"].shape[1])
    hidden_dim = int(state_dict["gru.weight_hh"].shape[1])
    policy = MotorGRUPolicy(
        observation_dim=input_dim,
        encoder_dim=int(saved.get("encoder_dim", state_dict["encoder.0.weight"].shape[0])),
        hidden_dim=int(saved.get("hidden_dim", hidden_dim)),
        encoder_depth=int(saved.get("encoder_depth", 2)),
        enable_integral_residual=bool(saved.get("enable_integral_residual", False)),
        enable_damping_residual=bool(saved.get("enable_rate_damping_residual", False)),
        integral_residual_hidden_dim=int(saved.get("integral_residual_hidden_dim", 16)),
        damping_residual_hidden_dim=int(saved.get("damping_residual_hidden_dim", 32)),
        integral_residual_scale=float(saved.get("integral_residual_scale", 1.0)),
        damping_residual_scale=float(saved.get("damping_residual_scale", 1.0)),
    ).double().eval()
    policy.load_compatible_state_dict({key: value.double() for key, value in state_dict.items()})

    generator = torch.Generator().manual_seed(args.seed)
    observation = torch.randn(args.batch_size, input_dim, generator=generator, dtype=torch.float64)
    hidden = torch.randn(args.batch_size, hidden_dim, generator=generator, dtype=torch.float64)
    with torch.no_grad():
        expected_action, expected_hidden = policy(observation, hidden)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "weights.mat"
    input_path = args.output_dir / "input.mat"
    output_path = args.output_dir / "matlab_output.mat"
    export_checkpoint_to_mat(args.checkpoint, weights_path)
    scipy.io.savemat(
        input_path,
        {"observation": observation.numpy(), "hidden": hidden.numpy()},
    )
    matlab_root = _matlab_quote(ROOT / "matlab_l2f")
    expression = (
        "restoredefaultpath; "
        f"addpath('{matlab_root}'); "
        f"w=l2f_prepare_motor_gru_weights(load('{_matlab_quote(weights_path)}')); "
        f"d=load('{_matlab_quote(input_path)}'); "
        "[action,next_hidden]=l2f_motor_gru_forward(w,d.observation,d.hidden); "
        f"save('{_matlab_quote(output_path)}','action','next_hidden','-v7');"
    )
    subprocess.run(["matlab", "-batch", expression], cwd=ROOT, check=True)
    actual = scipy.io.loadmat(output_path)
    action_error = np.abs(actual["action"] - expected_action.numpy())
    hidden_error = np.abs(actual["next_hidden"] - expected_hidden.numpy())
    rows = [
        {
            "checkpoint": str(args.checkpoint),
            "batch_size": args.batch_size,
            "action_max_abs_error": float(action_error.max()),
            "hidden_max_abs_error": float(hidden_error.max()),
        }
    ]
    report_path = args.output_dir / "parity.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if rows[0]["action_max_abs_error"] > 2.0e-10:
        raise RuntimeError(f"MATLAB action parity failed: {rows[0]}")
    if rows[0]["hidden_max_abs_error"] > 2.0e-10:
        raise RuntimeError(f"MATLAB hidden parity failed: {rows[0]}")
    print(report_path)


if __name__ == "__main__":
    main()

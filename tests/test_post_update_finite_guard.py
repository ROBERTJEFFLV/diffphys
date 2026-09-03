from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import torch

from model import MotorGRUPolicy


ROOT = Path(__file__).resolve().parents[1]


def test_finite_post_check_rolls_back_poisoning_update_with_cvar(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    log = tmp_path / "train.csv"
    command = (
        sys.executable,
        "train.py",
        "--device", "cpu",
        "--sim-backend", "torch",
        "--steps", "1",
        "--horizon", "2",
        "--batch-size", "2",
        "--encoder-dim", "16",
        "--hidden-dim", "12",
        "--encoder-depth", "2",
        "--tail-steps", "2",
        "--tail-window-steps", "2",
        "--steady-window-steps", "2",
        "--lr", "1e20",
        "--grad-clip", "1.0",
        "--grad-skip-threshold", "0",
        "--w-position-cvar", "0.001",
        "--post-update-check", "finite",
        "--log-every", "1",
        "--save-every", "0",
        "--training-diagnostics-every-updates", "1000",
        "--log-path", str(log),
        "--checkpoint-path", str(checkpoint),
    )
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

    with log.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["post_update_accepted"] == "0"
    assert row["update_applied"] == "0"
    assert row["skip_reason"] == "post_update_rejected"

    payload = torch.load(checkpoint, map_location="cpu")
    torch.manual_seed(7)
    initial = MotorGRUPolicy(encoder_dim=16, hidden_dim=12, encoder_depth=2)
    assert payload["optimizer_update"] == 0
    for name, expected in initial.state_dict().items():
        torch.testing.assert_close(payload["model"][name], expected, rtol=0.0, atol=0.0)

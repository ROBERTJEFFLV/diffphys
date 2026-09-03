from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (500, 1000, 2000, 5000, 10000)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize integral multiplier sensitivity.")
    parser.add_argument(
        "--input-root",
        default="reports/integral_scale_sensitivity_q2_matlab",
    )
    parser.add_argument("--multipliers", default="0,0.5,1,2,4")
    args = parser.parse_args()
    root = (ROOT / args.input_root).resolve()
    multipliers = tuple(float(value) for value in args.multipliers.split(","))
    output: list[dict[str, object]] = []

    for multiplier in multipliers:
        label = f"scale_{multiplier:g}".replace(".", "p")
        rows = _read(root / label / "samples.csv")
        force = np.sqrt(
            _num(rows, "external_force_x") ** 2
            + _num(rows, "external_force_y") ** 2
            + _num(rows, "external_force_z") ** 2
        )
        high_force = force >= np.quantile(force, 0.80)
        for horizon in HORIZONS:
            suffix = f"H{horizon}"
            success = _num(rows, f"position_hold_steady_{suffix}") > 0.5
            failed = ~success
            position = _num(rows, f"position_tail_rms_{suffix}")
            velocity = _num(rows, f"velocity_tail_rms_{suffix}")
            omega = _num(rows, f"omega_tail_rms_{suffix}")
            position_bad = position >= 0.05
            velocity_bad = velocity >= 0.10
            omega_bad = omega >= 0.20
            action_axes = np.stack(
                [_num(rows, f"action_{axis}_tail_rms_{suffix}") for axis in range(4)],
                axis=1,
            )
            delta_axes = np.stack(
                [_num(rows, f"action_{axis}_delta_tail_rms_{suffix}") for axis in range(4)],
                axis=1,
            )
            omega_xy = np.sqrt(
                _num(rows, f"omega_x_tail_rms_{suffix}") ** 2
                + _num(rows, f"omega_y_tail_rms_{suffix}") ** 2
            )
            output.append(
                {
                    "multiplier": multiplier,
                    "horizon": horizon,
                    "steady_success": float(np.mean(success)),
                    "position_only_failures": int(
                        np.sum(failed & position_bad & ~velocity_bad & ~omega_bad)
                    ),
                    "omega_related_failures": int(np.sum(failed & omega_bad)),
                    "highest_force_20_success": float(np.mean(success[high_force])),
                    "position_bias_5_6cm": int(
                        np.sum(failed & (position >= 0.05) & (position < 0.06))
                    ),
                    "position_bias_6_10cm": int(
                        np.sum(failed & (position >= 0.06) & (position < 0.10))
                    ),
                    "position_rms_ge_10cm": int(np.sum(failed & (position >= 0.10))),
                    "integral_norm_mean": float(np.mean(_num(rows, "integral_world_norm_mean"))),
                    "integral_clamp_ratio": float(np.mean(_num(rows, "integral_clamp_ratio"))),
                    "action_rms": float(np.mean(np.sqrt(np.mean(action_axes ** 2, axis=1)))),
                    "action_delta_rms": float(np.mean(np.sqrt(np.mean(delta_axes ** 2, axis=1)))),
                    "omega_xy_rms": float(np.mean(omega_xy)),
                }
            )
    _write(root / "sensitivity_summary.csv", output)
    print(root / "sensitivity_summary.csv")


if __name__ == "__main__":
    main()

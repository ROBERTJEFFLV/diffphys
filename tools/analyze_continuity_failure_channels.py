from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports/continuity_cadence_matlab_20260804"
DEFAULT_OUTPUT = ROOT / "reports/continuity_cadence_mechanism_audit_20260806"
SEEDS = (7, 17, 27)
ARMS = ("A", "B", "C")
HORIZONS = (500, 10000)
SCENARIO_FIELDS = (
    "sample_id",
    "mass_kg",
    "arm_length_m",
    "thrust_to_weight",
    "motor_time_rising_s",
    "motor_time_falling_s",
    "inertia_x",
    "inertia_y",
    "inertia_z",
    "external_force_x",
    "external_force_y",
    "external_force_z",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1024:
        raise ValueError(f"{path} has {len(rows)} samples, expected 1024")
    return rows


def _masks(rows: list[dict[str, str]], horizon: int) -> dict[str, list[bool]]:
    suffix = f"_H{horizon}"
    result = {
        "success": [int(row[f"position_hold_steady{suffix}"]) == 1 for row in rows],
        "position_fail": [int(row[f"position_pass_count{suffix}"]) < 95 for row in rows],
        "velocity_fail": [int(row[f"velocity_pass_count{suffix}"]) < 95 for row in rows],
        "omega_fail": [int(row[f"omega_pass_count{suffix}"]) < 95 for row in rows],
    }
    return result


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit which formal success channels drive flips.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data: dict[tuple[int, str], list[dict[str, str]]] = {}
    inputs: list[dict[str, str]] = []
    for seed in SEEDS:
        for arm in ARMS:
            path = input_dir / f"seed_{seed}/arm_{arm}/samples.csv"
            data[(seed, arm)] = _load(path)
            inputs.append(
                {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}
            )
    reference = data[(SEEDS[0], ARMS[0])]
    for key, rows in data.items():
        for index, (expected, actual) in enumerate(zip(reference, rows)):
            if any(expected[field] != actual[field] for field in SCENARIO_FIELDS):
                raise ValueError(
                    f"scenario order/parameters differ for seed={key[0]} arm={key[1]} "
                    f"at row {index}"
                )

    count_rows: list[dict[str, object]] = []
    flip_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        masks = {
            (seed, arm): _masks(data[(seed, arm)], horizon)
            for seed in SEEDS
            for arm in ARMS
        }
        for arm in ARMS:
            pooled = {
                key: [value for seed in SEEDS for value in masks[(seed, arm)][key]]
                for key in ("success", "position_fail", "velocity_fail", "omega_fail")
            }
            p = pooled["position_fail"]
            v = pooled["velocity_fail"]
            o = pooled["omega_fail"]
            count_rows.append(
                {
                    "arm": arm,
                    "horizon": horizon,
                    "labels": len(p),
                    "joint_success": sum(pooled["success"]),
                    "joint_failure": len(p) - sum(pooled["success"]),
                    "position_failure": sum(p),
                    "velocity_failure": sum(v),
                    "omega_failure": sum(o),
                    "position_only_failure": sum(pi and not vi and not oi for pi, vi, oi in zip(p, v, o)),
                    "velocity_only_failure": sum(vi and not pi and not oi for pi, vi, oi in zip(p, v, o)),
                    "omega_only_failure": sum(oi and not pi and not vi for pi, vi, oi in zip(p, v, o)),
                    "multi_channel_failure": sum((int(pi) + int(vi) + int(oi)) >= 2 for pi, vi, oi in zip(p, v, o)),
                    "temporal_intersection_only_failure": sum(
                        (not success) and not pi and not vi and not oi
                        for success, pi, vi, oi in zip(pooled["success"], p, v, o)
                    ),
                }
            )

        for control in ("A", "B"):
            gains = losses = 0
            gain_channels = {"position": 0, "velocity": 0, "omega": 0}
            loss_channels = {"position": 0, "velocity": 0, "omega": 0}
            for seed in SEEDS:
                control_mask = masks[(seed, control)]
                candidate_mask = masks[(seed, "C")]
                for index in range(1024):
                    gain = candidate_mask["success"][index] and not control_mask["success"][index]
                    loss = control_mask["success"][index] and not candidate_mask["success"][index]
                    gains += int(gain)
                    losses += int(loss)
                    if gain:
                        for channel in gain_channels:
                            gain_channels[channel] += int(control_mask[f"{channel}_fail"][index])
                    if loss:
                        for channel in loss_channels:
                            loss_channels[channel] += int(candidate_mask[f"{channel}_fail"][index])
            flip_rows.append(
                {
                    "comparison": f"C-{control}",
                    "horizon": horizon,
                    "candidate_gain_labels": gains,
                    "candidate_loss_labels": losses,
                    "net_labels": gains - losses,
                    "gain_control_position_fail": gain_channels["position"],
                    "gain_control_velocity_fail": gain_channels["velocity"],
                    "gain_control_omega_fail": gain_channels["omega"],
                    "loss_candidate_position_fail": loss_channels["position"],
                    "loss_candidate_velocity_fail": loss_channels["velocity"],
                    "loss_candidate_omega_fail": loss_channels["omega"],
                }
            )

    _write(output_dir / "FAILURE_CHANNEL_COUNTS.csv", count_rows)
    _write(output_dir / "FLIP_CHANNEL_BREAKDOWN.csv", flip_rows)
    provenance = {
        "formal_label_source": "MATLAB samples.csv",
        "success_rule": (
            "at least 95/100 final-window steps simultaneously satisfy position, "
            "velocity, and omega thresholds"
        ),
        "channel_failure_rule": "the corresponding channel alone passes fewer than 95/100 steps",
        "scenario_alignment_fields": SCENARIO_FIELDS,
        "input_files": inputs,
        "tool": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "tool_sha256": _sha256(Path(__file__).resolve()),
    }
    (output_dir / "FAILURE_CHANNEL_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "rows": len(count_rows) + len(flip_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import load_q2_policy, rollout_q2  # noqa: E402
from diagnostics.scenarios import formal_scenario_uid, load_matlab_scenarios  # noqa: E402
from env_l2f import L2FState  # noqa: E402
from l2f_cuda_backend import load_extension  # noqa: E402


DEFAULT_CHECKPOINTS = (
    "A_current=reports/temporal_decay_minimal/short_train/A_current/model.pt,"
    "C_alpha1=reports/temporal_decay_minimal/short_train/C_alpha1/model.pt,"
    "D_alpha0.25=reports/temporal_decay_minimal/short_train/D_alpha0p25/model.pt"
)
DEFAULT_LOGS = (
    "A_current=reports/temporal_decay_minimal/short_train/A_current/train.csv,"
    "C_alpha1=reports/temporal_decay_minimal/short_train/C_alpha1/train.csv,"
    "D_alpha0.25=reports/temporal_decay_minimal/short_train/D_alpha0p25/train.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the minimal temporal-decay screen on one fixed scenario set."
    )
    parser.add_argument("--checkpoints", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--train-logs", default=DEFAULT_LOGS)
    parser.add_argument(
        "--scenario-csv",
        type=Path,
        default=ROOT / "diagnostic_inputs/h10000_paired_96m_20260804/manifests/scenario_reset_exact.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/temporal_decay_minimal/evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--eval-seed", type=int, default=1007)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _parse_mapping(value: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in value.split(","):
        label, separator, path = item.strip().partition("=")
        if not separator or not label or not path:
            raise ValueError(f"expected label=path entry, got {item!r}")
        if label in out:
            raise ValueError(f"duplicate label: {label}")
        candidate = Path(path)
        out[label] = candidate if candidate.is_absolute() else ROOT / candidate
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_scenario_subset(
    path: Path,
    *,
    count: int,
    device: torch.device,
) -> tuple[list[int], list[int], L2FState]:
    scenario_ids, full_state = load_matlab_scenarios(path, device=device, dtype=torch.float32)
    if count <= 0 or count > len(scenario_ids):
        raise ValueError("batch size must be in [1, scenario count]")
    indices = torch.linspace(
        0,
        len(scenario_ids) - 1,
        steps=count,
        device=device,
        dtype=torch.float64,
    ).round().to(dtype=torch.long)
    selected = L2FState(
        **{
            field.name: getattr(full_state, field.name).index_select(0, indices)
            for field in fields(L2FState)
        }
    )
    cpu_indices = [int(value) for value in indices.cpu().tolist()]
    return [scenario_ids[index] for index in cpu_indices], cpu_indices, selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return statistics.fmean(values)


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def _summarize_evaluation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = sorted({str(row["checkpoint"]) for row in rows})
    horizons = sorted({int(row["horizon"]) for row in rows})
    summary: list[dict[str, Any]] = []
    for label in labels:
        for horizon in horizons:
            selected = [
                row
                for row in rows
                if row["checkpoint"] == label and int(row["horizon"]) == horizon
            ]
            summary.append(
                {
                    "variant": label,
                    "horizon": horizon,
                    "scenario_count": len(selected),
                    "steady_success_rate": _mean(selected, "steady_success"),
                    "survival_rate": _mean(selected, "survival"),
                    "position_tail_mean": _mean(selected, "position_tail_mean"),
                    "velocity_tail_mean": _mean(selected, "velocity_tail_mean"),
                    "omega_tail_mean": _mean(selected, "omega_tail_mean"),
                    "position_final_mean": _mean(selected, "position_final"),
                    "velocity_final_mean": _mean(selected, "velocity_final"),
                    "omega_final_mean": _mean(selected, "omega_final"),
                    "position_final_median": _median(selected, "position_final"),
                    "velocity_final_median": _median(selected, "velocity_final"),
                    "omega_final_median": _median(selected, "omega_final"),
                    "position_recovery_ratio_mean": _mean(
                        selected, "position_recovery_ratio"
                    ),
                    "velocity_recovery_ratio_mean": _mean(
                        selected, "velocity_recovery_ratio"
                    ),
                    "omega_recovery_ratio_mean": _mean(
                        selected, "omega_recovery_ratio"
                    ),
                }
            )
    return summary


def _training_summary(label: str, path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    boundaries = [
        row for row in rows if str(row.get("optimization_block_boundary", "0")) == "1"
    ]
    gradients = [
        float(row["grad_norm_fp64_before_clip"])
        for row in boundaries
        if row.get("grad_norm_fp64_before_clip", "") not in ("", "nan")
    ]
    finite_gradients = [value for value in gradients if math.isfinite(value)]
    applied = sum(int(float(row.get("update_applied", "0") or 0)) for row in boundaries)
    skipped = len(boundaries) - applied
    clip_count = sum(
        float(row.get("grad_scale", "1") or 1.0) < 0.999999 for row in boundaries
    )
    nonfinite = sum(
        "nonfinite" in str(row.get("skip_reason", "")).lower() for row in boundaries
    )
    return {
        "variant": label,
        "log_path": str(path.resolve()),
        "optimizer_boundaries": len(boundaries),
        "accepted_updates": applied,
        "skipped_updates": skipped,
        "nonfinite_failures": nonfinite,
        "clip_fraction": clip_count / max(len(boundaries), 1),
        "grad_norm_mean": statistics.fmean(finite_gradients)
        if finite_gradients
        else float("nan"),
        "grad_norm_p95": _quantile(finite_gradients, 0.95),
        "grad_norm_max": max(finite_gradients) if finite_gradients else float("nan"),
        "all_grad_norms_finite": int(len(finite_gradients) == len(gradients)),
        "final_optimizer_update": max(
            (int(float(row.get("optimizer_update", "0") or 0)) for row in rows),
            default=0,
        ),
        "final_physical_steps": max(
            (int(float(row.get("physical_steps", "0") or 0)) for row in rows),
            default=0,
        ),
    }


def _paired_rows(
    rows: list[dict[str, Any]], *, baseline: str = "A_current"
) -> list[dict[str, Any]]:
    metrics = (
        "steady_success",
        "position_tail_mean",
        "velocity_tail_mean",
        "omega_tail_mean",
        "position_recovery_ratio",
        "velocity_recovery_ratio",
        "omega_recovery_ratio",
    )
    labels = sorted({str(row["checkpoint"]) for row in rows})
    horizons = sorted({int(row["horizon"]) for row in rows})
    if baseline not in labels:
        raise ValueError(f"paired-evaluation baseline is missing: {baseline}")
    indexed = {
        (str(row["checkpoint"]), int(row["horizon"]), str(row["scenario_uid"])): row
        for row in rows
    }
    out: list[dict[str, Any]] = []
    for label in labels:
        if label == baseline:
            continue
        for horizon in horizons:
            baseline_uids = sorted(
                key[2]
                for key in indexed
                if key[0] == baseline and key[1] == horizon
            )
            variant_uids = sorted(
                key[2]
                for key in indexed
                if key[0] == label and key[1] == horizon
            )
            if baseline_uids != variant_uids:
                raise ValueError(
                    f"paired scenario mismatch for {label} at horizon {horizon}"
                )
            for metric in metrics:
                baseline_values = [
                    float(indexed[(baseline, horizon, uid)][metric])
                    for uid in baseline_uids
                ]
                variant_values = [
                    float(indexed[(label, horizon, uid)][metric])
                    for uid in baseline_uids
                ]
                deltas = [
                    variant_value - baseline_value
                    for baseline_value, variant_value in zip(
                        baseline_values, variant_values
                    )
                ]
                baseline_mean = statistics.fmean(baseline_values)
                variant_mean = statistics.fmean(variant_values)
                delta_mean = statistics.fmean(deltas)
                higher_is_better = metric == "steady_success"
                favorable = [
                    delta > 0.0 if higher_is_better else delta < 0.0
                    for delta in deltas
                ]
                out.append(
                    {
                        "baseline": baseline,
                        "variant": label,
                        "horizon": horizon,
                        "metric": metric,
                        "scenario_count": len(deltas),
                        "baseline_mean": baseline_mean,
                        "variant_mean": variant_mean,
                        "paired_delta_mean": delta_mean,
                        "paired_relative_percent": (
                            100.0 * delta_mean / baseline_mean
                            if baseline_mean != 0.0
                            else float("nan")
                        ),
                        "favorable_fraction": statistics.fmean(favorable),
                    }
                )
    return out


def _compact_rows(
    eval_summary: list[dict[str, Any]],
    train_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    train_by_label = {str(row["variant"]): row for row in train_summary}
    compact: list[dict[str, Any]] = []
    for label in sorted(train_by_label):
        row: dict[str, Any] = {
            "variant": label,
            "accepted_updates": train_by_label[label]["accepted_updates"],
            "skipped_updates": train_by_label[label]["skipped_updates"],
            "clip_fraction": train_by_label[label]["clip_fraction"],
            "grad_norm_p95": train_by_label[label]["grad_norm_p95"],
            "grad_norm_max": train_by_label[label]["grad_norm_max"],
        }
        for horizon in (500, 2000, 5000):
            match = next(
                item
                for item in eval_summary
                if item["variant"] == label and int(item["horizon"]) == horizon
            )
            for key in (
                "steady_success_rate",
                "position_tail_mean",
                "velocity_tail_mean",
                "omega_tail_mean",
            ):
                row[f"H{horizon}_{key}"] = match[key]
        compact.append(row)
    return compact


def main() -> None:
    args = parse_args()
    if args.horizon != 5000:
        raise ValueError("the preregistered fixed evaluation uses H5000")
    checkpoints = _parse_mapping(args.checkpoints)
    train_logs = _parse_mapping(args.train_logs)
    if checkpoints.keys() != train_logs.keys():
        raise ValueError("checkpoint and train-log labels must match")
    for path in (*checkpoints.values(), *train_logs.values(), args.scenario_csv):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the fixed compact-CUDA evaluation requires CUDA")
    load_extension()
    scenario_ids, scenario_indices, initial_state = _fixed_scenario_subset(
        args.scenario_csv, count=args.batch_size, device=device
    )
    uids = [formal_scenario_uid(args.eval_seed, scenario_id) for scenario_id in scenario_ids]
    initial_norms = {
        "position": torch.linalg.vector_norm(initial_state.position, dim=-1).cpu().tolist(),
        "velocity": torch.linalg.vector_norm(initial_state.velocity, dim=-1).cpu().tolist(),
        "omega": torch.linalg.vector_norm(initial_state.omega, dim=-1).cpu().tolist(),
    }

    evaluation_rows: list[dict[str, Any]] = []
    for label, checkpoint_path in checkpoints.items():
        policy, _ = load_q2_policy(
            checkpoint_path, device=device, dtype=torch.float32
        )
        result = rollout_q2(
            policy,
            initial_state,
            uids,
            checkpoint_label=label,
            seed=args.eval_seed,
            group=label,
            horizon=args.horizon,
            snapshot_horizons=(500, 2000, 5000),
            backend="cuda",
        )
        uid_to_index = {uid: index for index, uid in enumerate(uids)}
        for row in result.horizon_rows:
            index = uid_to_index[str(row["scenario_uid"])]
            for channel in ("position", "velocity", "omega"):
                initial = max(float(initial_norms[channel][index]), 1.0e-12)
                row[f"{channel}_recovery_ratio"] = float(row[f"{channel}_final"]) / initial
            evaluation_rows.append(row)
        del policy
        torch.cuda.empty_cache()

    eval_summary = _summarize_evaluation(evaluation_rows)
    paired_rows = _paired_rows(evaluation_rows)
    train_summary = [
        _training_summary(label, train_logs[label]) for label in checkpoints
    ]
    compact = _compact_rows(eval_summary, train_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "evaluation_samples.csv", evaluation_rows)
    _write_csv(args.output_dir / "evaluation_summary.csv", eval_summary)
    _write_csv(args.output_dir / "paired_deltas.csv", paired_rows)
    _write_csv(args.output_dir / "training_summary.csv", train_summary)
    _write_csv(args.output_dir / "compact_summary.csv", compact)
    metadata = {
        "scenario_csv": str(args.scenario_csv.resolve()),
        "scenario_csv_sha256": _sha256(args.scenario_csv),
        "scenario_indices": scenario_indices,
        "scenario_ids": scenario_ids,
        "eval_seed": args.eval_seed,
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "checkpoints": {
            label: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "train_log": str(train_logs[label].resolve()),
                "train_log_sha256": _sha256(train_logs[label]),
            }
            for label, path in checkpoints.items()
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output_dir / 'compact_summary.csv'}")


if __name__ == "__main__":
    main()

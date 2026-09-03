from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CHECKPOINT = REPO_ROOT / "reports/mainline_h500_finetune_safe_5000_lr5e6/model_step_4000.pt"
GROUP_CONFIGS = {
    "A": "configs/formal_belief_A_control.args",
    "B": "configs/formal_belief_B_motor.args",
    "C": "configs/formal_belief_C_motor_capability.args",
    "D": "configs/formal_belief_D_motor_response.args",
    "E": "configs/formal_belief_E_motor_capability_response.args",
}
REPORT_METRICS = (
    "success_rate",
    "tail_success_rate",
    "omega_failure_rate",
    "low_alpha_roll_success_rate",
    "low_alpha_roll_omega_failure_rate",
    "low_alpha_yaw_success_rate",
    "low_alpha_yaw_omega_failure_rate",
    "large_tau_fall_success_rate",
    "large_tau_fall_omega_failure_rate",
    "h500_success_rate",
    "h500_to_final_survival_rate",
    "h500_to_final_stay_success_rate",
)
TRAIN_SEGMENT_HORIZON = 250
TRAIN_EPISODE_STEPS = 500
TRAIN_OUTER_STEPS = 500


def _int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _group_list(value: str) -> list[str]:
    groups = [part.strip().upper() for part in value.split(",") if part.strip()]
    invalid = sorted(set(groups) - set(GROUP_CONFIGS))
    if invalid:
        raise ValueError(f"unknown groups: {','.join(invalid)}")
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the strict paired A-E belief auxiliary ablation.")
    parser.add_argument("--phase", choices=("train", "eval", "audit", "summarize", "all"), default="all")
    parser.add_argument("--groups", default="A,B,C,D,E")
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--eval-seeds", default="1007,1017,1027,1037,1047")
    parser.add_argument("--checkpoint-steps", default="100,200,300,400,500")
    parser.add_argument("--horizons", default="500,10000")
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--output-root", default="reports/formal_belief_ablation_h500_s500_gpu2")
    parser.add_argument("--hard-samples-path", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.groups = _group_list(args.groups)
    args.seeds = _int_list(args.seeds)
    args.eval_seed_values = _int_list(args.eval_seeds)
    args.checkpoint_step_values = _int_list(args.checkpoint_steps)
    args.horizon_values = _int_list(args.horizons)
    if len(args.seeds) < 5 and args.phase in {"all", "train"}:
        print("warning: formal decision requires at least five training seeds", file=sys.stderr)
    return args


def _run(command: list[str], *, dry_run: bool) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def _group_root(output_root: Path, seed: int, group: str) -> Path:
    return output_root / f"seed_{seed}" / f"group_{group}"


def _checkpoint_path(group_root: Path, checkpoint_step: int) -> Path:
    return group_root / "checkpoints" / f"model_step_{checkpoint_step}.pt"


def run_training(args: argparse.Namespace, output_root: Path) -> None:
    commands: list[list[str]] = []
    for seed in args.seeds:
        for group in args.groups:
            root = _group_root(output_root, seed, group)
            final_checkpoint = root / "checkpoints/model.pt"
            if final_checkpoint.exists() and not args.force:
                print(f"skip completed training: seed={seed} group={group}")
                continue
            command = [
                sys.executable,
                "train.py",
                f"@{GROUP_CONFIGS[group]}",
                "--seed",
                str(seed),
                "--log-path",
                str(root / "train.csv"),
                "--checkpoint-path",
                str(final_checkpoint),
                "--sampler-audit-path",
                str(root / "reset_samples.csv"),
            ]
            commands.append(command)
    if args.dry_run or args.max_parallel <= 1:
        for command in commands:
            _run(command, dry_run=args.dry_run)
        return
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = [executor.submit(_run, command, dry_run=False) for command in commands]
        for future in futures:
            future.result()


def _eval_command(
    *,
    config: str,
    checkpoint: Path,
    log_path: Path,
    samples_path: Path,
    horizon: int,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        "train.py",
        f"@{config}",
        "--eval-only",
        "--checkpoint-path",
        str(checkpoint),
        "--log-path",
        str(log_path),
        "--eval-samples-path",
        str(samples_path),
        "--trajectory-count",
        "0",
        "--batch-size",
        str(args.eval_batch_size),
        "--horizon",
        str(horizon),
        "--training-episode-steps",
        str(max(2000, horizon + 1)),
        "--eval-seeds",
        ",".join(str(seed) for seed in args.eval_seed_values),
    ]


def run_evaluation(args: argparse.Namespace, output_root: Path) -> None:
    baseline_root = output_root / "baseline"
    baseline_config = GROUP_CONFIGS["A"]
    for horizon in args.horizon_values:
        log_path = baseline_root / "eval" / f"H{horizon}" / "step_0.csv"
        samples_path = baseline_root / "eval" / f"H{horizon}" / "samples_step_0.csv"
        if not log_path.exists() or args.force:
            command = _eval_command(
                config=baseline_config,
                checkpoint=BASELINE_CHECKPOINT,
                log_path=log_path,
                samples_path=samples_path,
                horizon=horizon,
                args=args,
            )
            _run(command, dry_run=args.dry_run)

    for seed in args.seeds:
        for group in args.groups:
            root = _group_root(output_root, seed, group)
            for checkpoint_step in args.checkpoint_step_values:
                checkpoint = _checkpoint_path(root, checkpoint_step)
                if not checkpoint.exists() and not args.dry_run:
                    raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
                for horizon in args.horizon_values:
                    eval_root = root / "eval" / f"H{horizon}"
                    log_path = eval_root / f"step_{checkpoint_step}.csv"
                    samples_path = eval_root / f"samples_step_{checkpoint_step}.csv"
                    if log_path.exists() and samples_path.exists() and not args.force:
                        print(
                            f"skip completed eval: seed={seed} group={group} "
                            f"step={checkpoint_step} H={horizon}"
                        )
                        continue
                    command = _eval_command(
                        config=GROUP_CONFIGS[group],
                        checkpoint=checkpoint,
                        log_path=log_path,
                        samples_path=samples_path,
                        horizon=horizon,
                        args=args,
                    )
                    _run(command, dry_run=args.dry_run)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_train_log(path: Path, expected_updates: int) -> dict[str, int | bool]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    boundaries = [row for row in rows if row.get("episode_boundary") == "1"]
    bad = [
        row
        for row in boundaries
        if row.get("update_applied") != "1"
        or bool(row.get("skip_reason"))
        or row.get("rollout_valid") != "1"
        or row.get("episode_valid") != "1"
    ]
    return {
        "boundary_rows": len(boundaries),
        "expected_updates": expected_updates,
        "bad_boundary_rows": len(bad),
        "valid": len(boundaries) == expected_updates and not bad,
    }


def audit_pairing(args: argparse.Namespace, output_root: Path) -> None:
    segments_per_update = TRAIN_EPISODE_STEPS // TRAIN_SEGMENT_HORIZON
    expected_updates = TRAIN_OUTER_STEPS // segments_per_update
    report: dict[str, object] = {"valid": True, "seeds": {}}
    for seed in args.seeds:
        seed_report: dict[str, object] = {"groups": {}}
        hashes: dict[str, str] = {}
        for group in args.groups:
            root = _group_root(output_root, seed, group)
            reset_path = root / "reset_samples.csv"
            train_path = root / "train.csv"
            if not reset_path.exists() or not train_path.exists():
                raise FileNotFoundError(f"missing audit input for seed={seed} group={group}")
            digest = _sha256(reset_path)
            hashes[group] = digest
            train_audit = _audit_train_log(train_path, expected_updates)
            seed_report["groups"][group] = {"reset_sha256": digest, **train_audit}
            report["valid"] = bool(report["valid"]) and bool(train_audit["valid"])
        pairing_valid = len(set(hashes.values())) == 1
        seed_report["paired_reset_samples"] = pairing_valid
        report["valid"] = bool(report["valid"]) and pairing_valid
        report["seeds"][str(seed)] = seed_report
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "pairing_audit.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"pairing audit valid={report['valid']}: {path}")
    if not report["valid"]:
        raise RuntimeError("paired experiment audit failed; do not use these runs for a formal comparison")


def _aggregate_eval_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["eval_seed"] == "aggregate_mean":
            return row
    raise ValueError(f"aggregate_mean row missing: {path}")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_samples(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(int(row["eval_seed"]), int(row["sample_index"])): row for row in rows}


def _mean_column(rows: list[dict[str, str]], column: str) -> float:
    return statistics.fmean(float(row[column]) for row in rows) if rows else float("nan")


def summarize(args: argparse.Namespace, output_root: Path) -> None:
    long_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for group in args.groups:
            root = _group_root(output_root, seed, group)
            for checkpoint_step in args.checkpoint_step_values:
                for horizon in args.horizon_values:
                    path = root / "eval" / f"H{horizon}" / f"step_{checkpoint_step}.csv"
                    row = _aggregate_eval_row(path)
                    for metric in REPORT_METRICS:
                        long_rows.append(
                            {
                                "training_seed": seed,
                                "group": group,
                                "checkpoint_step": checkpoint_step,
                                "horizon": horizon,
                                "metric": metric,
                                "value": float(row[metric]),
                            }
                        )
    _write_csv(output_root / "summary_long.csv", long_rows)

    aggregate_rows: list[dict[str, object]] = []
    keys = sorted({(row["group"], row["checkpoint_step"], row["horizon"], row["metric"]) for row in long_rows})
    for group, checkpoint_step, horizon, metric in keys:
        values = [
            float(row["value"])
            for row in long_rows
            if (row["group"], row["checkpoint_step"], row["horizon"], row["metric"])
            == (group, checkpoint_step, horizon, metric)
        ]
        aggregate_rows.append(
            {
                "group": group,
                "checkpoint_step": checkpoint_step,
                "horizon": horizon,
                "metric": metric,
                "seed_count": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }
        )
    _write_csv(output_root / "summary_aggregate.csv", aggregate_rows)

    paired_rows: list[dict[str, object]] = []
    for group in args.groups:
        if group == "A":
            continue
        for checkpoint_step in args.checkpoint_step_values:
            for horizon in args.horizon_values:
                for metric in REPORT_METRICS:
                    deltas: list[float] = []
                    for seed in args.seeds:
                        values = {
                            str(row["group"]): float(row["value"])
                            for row in long_rows
                            if row["training_seed"] == seed
                            and row["checkpoint_step"] == checkpoint_step
                            and row["horizon"] == horizon
                            and row["metric"] == metric
                            and row["group"] in {"A", group}
                        }
                        deltas.append(values[group] - values["A"])
                    lower_is_better = "failure" in metric
                    consistent = [delta < 0.0 if lower_is_better else delta > 0.0 for delta in deltas]
                    paired_rows.append(
                        {
                            "group": group,
                            "checkpoint_step": checkpoint_step,
                            "horizon": horizon,
                            "metric": metric,
                            "paired_seed_count": len(deltas),
                            "mean_delta_vs_A": statistics.fmean(deltas),
                            "std_delta_vs_A": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
                            "improved_seed_fraction": statistics.fmean(float(value) for value in consistent),
                        }
                    )
    _write_csv(output_root / "paired_vs_control.csv", paired_rows)

    prediction_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for group in args.groups:
            train_path = _group_root(output_root, seed, group) / "train.csv"
            with train_path.open(newline="") as handle:
                rows = [row for row in csv.DictReader(handle) if row["update_applied"] == "1"]
            window = rows[-25:]
            prediction_rows.append(
                {
                    "training_seed": seed,
                    "group": group,
                    "update_count": len(rows),
                    "window_updates": len(window),
                    "motor_aux_loss_last25_mean": _mean_column(window, "motor_aux_loss"),
                    "capability_aux_loss_last25_mean": _mean_column(window, "capability_aux_loss"),
                    "response_aux_loss_last25_mean": _mean_column(window, "response_aux_loss"),
                }
            )
    _write_csv(output_root / "prediction_losses_by_seed.csv", prediction_rows)

    baseline_samples = _load_samples(output_root / "baseline/eval/H500/samples_step_0.csv")
    scenario_columns = (
        "mass",
        "thrust_to_weight",
        "alpha_roll_max",
        "alpha_yaw_max",
        "tau_rise",
        "tau_fall",
        "f_ext_x",
        "f_ext_y",
        "f_ext_z",
        "initial_position_x",
        "initial_position_y",
        "initial_position_z",
        "initial_velocity_x",
        "initial_velocity_y",
        "initial_velocity_z",
        "initial_rotation_00",
        "initial_rotation_01",
        "initial_rotation_02",
        "initial_rotation_10",
        "initial_rotation_11",
        "initial_rotation_12",
        "initial_rotation_20",
        "initial_rotation_21",
        "initial_rotation_22",
        "initial_omega_x",
        "initial_omega_y",
        "initial_omega_z",
    )
    baseline_success_keys = {key for key, row in baseline_samples.items() if row["success"] == "1"}
    retain_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for group in args.groups:
            for checkpoint_step in args.checkpoint_step_values:
                path = (
                    _group_root(output_root, seed, group)
                    / "eval/H500"
                    / f"samples_step_{checkpoint_step}.csv"
                )
                samples = _load_samples(path)
                if samples.keys() != baseline_samples.keys():
                    raise RuntimeError(f"evaluation sample keys differ from baseline: {path}")
                for key in samples:
                    if any(samples[key][column] != baseline_samples[key][column] for column in scenario_columns):
                        raise RuntimeError(f"evaluation scenario mismatch at {key}: {path}")
                retained = sum(samples[key]["success"] == "1" for key in baseline_success_keys)
                denominator = max(len(baseline_success_keys), 1)
                retain_rows.append(
                    {
                        "training_seed": seed,
                        "group": group,
                        "checkpoint_step": checkpoint_step,
                        "baseline_success_count": len(baseline_success_keys),
                        "baseline_success_retained": retained,
                        "baseline_success_retain_rate": retained / denominator,
                        "baseline_success_lost": len(baseline_success_keys) - retained,
                    }
                )
    _write_csv(output_root / "baseline_retain_by_seed.csv", retain_rows)

    if args.hard_samples_path:
        hard_path = Path(args.hard_samples_path)
        with hard_path.open(newline="") as handle:
            hard_rows = list(csv.DictReader(handle))
        if len(hard_rows) != 43:
            raise ValueError(f"hard sample manifest must contain exactly 43 rows, got {len(hard_rows)}")
        required = {"eval_seed", "sample_index"}
        if not hard_rows or not required.issubset(hard_rows[0]):
            raise ValueError("hard sample manifest requires eval_seed,sample_index columns")
        hard_keys = {(int(row["eval_seed"]), int(row["sample_index"])) for row in hard_rows}
        hard_metric_rows: list[dict[str, object]] = []
        for seed in args.seeds:
            for group in args.groups:
                for checkpoint_step in args.checkpoint_step_values:
                    for horizon in args.horizon_values:
                        path = (
                            _group_root(output_root, seed, group)
                            / "eval"
                            / f"H{horizon}"
                            / f"samples_step_{checkpoint_step}.csv"
                        )
                        samples = _load_samples(path)
                        missing = hard_keys - samples.keys()
                        if missing:
                            raise ValueError(f"{len(missing)} hard sample IDs missing from {path}")
                        selected = [samples[key] for key in sorted(hard_keys)]
                        hard_metric_rows.append(
                            {
                                "training_seed": seed,
                                "group": group,
                                "checkpoint_step": checkpoint_step,
                                "horizon": horizon,
                                "hard_sample_count": len(selected),
                                "success_rate": _mean_column(selected, "success"),
                                "omega_failure_rate": _mean_column(selected, "omega_failure"),
                                "h500_to_final_survival_rate": _mean_column(
                                    [row for row in selected if row["h500_success"] == "1"],
                                    "h500_to_final_survived",
                                ),
                                "h500_to_final_stay_success_rate": _mean_column(
                                    [row for row in selected if row["h500_success"] == "1"],
                                    "h500_to_final_stayed_success",
                                ),
                            }
                        )
        _write_csv(output_root / "hard43_by_seed.csv", hard_metric_rows)
    else:
        print(
            "43-sample report omitted: pass --hard-samples-path with eval_seed,sample_index columns",
            file=sys.stderr,
        )
    print(f"saved formal summaries under: {output_root}")


def main() -> None:
    args = parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve()
    if not BASELINE_CHECKPOINT.exists() and not args.dry_run:
        raise FileNotFoundError(f"baseline checkpoint missing: {BASELINE_CHECKPOINT}")
    if args.phase in {"train", "all"}:
        run_training(args, output_root)
    if args.phase in {"audit", "all"} and not args.dry_run:
        audit_pairing(args, output_root)
    if args.phase in {"eval", "all"}:
        run_evaluation(args, output_root)
    if args.phase in {"summarize", "all"} and not args.dry_run:
        summarize(args, output_root)


if __name__ == "__main__":
    main()

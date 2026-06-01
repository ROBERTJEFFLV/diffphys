#!/usr/bin/env python3
import argparse
import csv
import math
import re
import statistics
from pathlib import Path


EVAL_RE = re.compile(r"seed(?P<seed>\d+)_evalseed(?P<eval_seed>\d+)_(?P<mode>fixed|sampled)_(?P<model>euler|l2f)\.csv$")


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value, default=math.nan):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def mean(values):
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else math.nan


def pstdev(values):
    values = [v for v in values if math.isfinite(v)]
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else math.nan


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return value
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def metadata_value(metadata, key, default=""):
    return metadata.get(key, default)


def read_metadata(run_root):
    path = run_root / "reports" / "run_metadata.csv"
    if not path.exists():
        return {}
    rows = read_csv(path)
    return {row["key"]: row["value"] for row in rows}


def last_train_row(run_root, seed):
    path = run_root / "logs" / f"seed{seed}_train.csv"
    if not path.exists():
        return None
    rows = read_csv(path)
    return rows[-1] if rows else None


def collect_eval(run_root):
    by_seed = {}
    for path in (run_root / "logs").glob("seed*_evalseed*_*.csv"):
        match = EVAL_RE.match(path.name)
        if not match:
            continue
        seed = int(match.group("seed"))
        mode = match.group("mode")
        model = match.group("model")
        rows = read_csv(path)
        if not rows:
            continue
        by_seed.setdefault(seed, {}).setdefault(mode, {}).setdefault(model, []).append(rows[0])
    return by_seed


def eval_stats(rows):
    success = [to_float(row.get("success_rate")) for row in rows]
    invalid = [to_float(row.get("invalid_or_nan_rate")) for row in rows]
    p = [to_float(row.get("mean_final_position_norm")) for row in rows]
    v = [to_float(row.get("mean_final_velocity_norm")) for row in rows]
    w = [to_float(row.get("mean_final_angular_velocity_norm")) for row in rows]
    return {
        "success_mean": mean(success),
        "success_std": pstdev(success),
        "invalid_mean": mean(invalid),
        "p_mean": mean(p),
        "v_mean": mean(v),
        "w_mean": mean(w),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    report_dir = run_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata(run_root)
    eval_by_seed = collect_eval(run_root)
    seeds = sorted(eval_by_seed)
    init_path = metadata_value(metadata, "init_actor_path", "none")
    zero_shot = metadata_value(metadata, "zero_shot", "0") == "1"
    init_type = "fixed-pretrained" if init_path and init_path != "none" else "scratch"
    if zero_shot:
        init_type = "fixed-pretrained-zero-shot"
    level = metadata_value(metadata, "sampled_dynamics_level", "unknown")

    summary_rows = []
    for seed in seeds:
        train = last_train_row(run_root, seed)
        row = {
            "train_seed": seed,
            "init_type": init_type,
            "sampled_dynamics_level": level,
            "zero_shot": str(zero_shot).lower(),
            "train_final_loss": "nan",
            "train_final_p": "nan",
            "train_final_v": "nan",
            "train_final_w": "nan",
            "train_invalid_rollouts": "nan",
            "train_skipped_steps": "nan",
            "train_nan_or_inf": "nan",
            "dynamics_rejection_rate": "nan",
            "mean_mass": "nan",
            "mean_thrust_to_weight_ratio": "nan",
            "mean_torque_to_inertia_ratio": "nan",
            "mean_motor_tau": "nan",
        }
        if train:
            row.update({
                "train_final_loss": train.get("total_loss_mean", train.get("loss_total", "nan")),
                "train_final_p": train.get("mean_final_position_norm", "nan"),
                "train_final_v": train.get("mean_final_velocity_norm", "nan"),
                "train_final_w": train.get("mean_final_angular_velocity_norm", "nan"),
                "train_invalid_rollouts": train.get("invalid_rollout_count", "nan"),
                "train_skipped_steps": train.get("num_skipped_steps", "nan"),
                "train_nan_or_inf": train.get("nan_or_inf_flag", "nan"),
                "dynamics_rejection_rate": train.get("dynamics_rejection_rate", "nan"),
                "mean_mass": train.get("mass", "nan"),
                "mean_thrust_to_weight_ratio": train.get("thrust_to_weight_ratio", "nan"),
                "mean_torque_to_inertia_ratio": train.get("torque_to_inertia_ratio", "nan"),
                "mean_motor_tau": train.get("motor_tau_mean", "nan"),
            })
        for mode in ("sampled", "fixed"):
            for model in ("euler", "l2f"):
                stats = eval_stats(eval_by_seed.get(seed, {}).get(mode, {}).get(model, []))
                prefix = f"{mode}_{model}"
                row[f"{prefix}_success_mean"] = stats["success_mean"]
                row[f"{prefix}_success_std"] = stats["success_std"]
                row[f"{prefix}_invalid_rate"] = stats["invalid_mean"]
                row[f"{prefix}_final_p"] = stats["p_mean"]
                row[f"{prefix}_final_v"] = stats["v_mean"]
                row[f"{prefix}_final_w"] = stats["w_mean"]
        row["sampled_beats_old_fixed_baselines"] = (
            to_float(row.get("sampled_euler_success_mean")) > 0.11
            and to_float(row.get("sampled_l2f_success_mean")) > 0.09
        )
        row["sampled_nonzero_both"] = (
            to_float(row.get("sampled_euler_success_mean")) > 0
            and to_float(row.get("sampled_l2f_success_mean")) > 0
        )
        summary_rows.append(row)

    fieldnames = [
        "train_seed", "init_type", "sampled_dynamics_level", "zero_shot",
        "train_final_loss", "train_final_p", "train_final_v", "train_final_w",
        "train_invalid_rollouts", "train_skipped_steps", "train_nan_or_inf",
        "dynamics_rejection_rate", "mean_mass", "mean_thrust_to_weight_ratio",
        "mean_torque_to_inertia_ratio", "mean_motor_tau",
    ]
    for mode in ("sampled", "fixed"):
        for model in ("euler", "l2f"):
            prefix = f"{mode}_{model}"
            fieldnames.extend([
                f"{prefix}_success_mean", f"{prefix}_success_std", f"{prefix}_invalid_rate",
                f"{prefix}_final_p", f"{prefix}_final_v", f"{prefix}_final_w",
            ])
    fieldnames.extend(["sampled_beats_old_fixed_baselines", "sampled_nonzero_both"])

    summary_csv = report_dir / "sampled_dynamics_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    sampled_euler = [to_float(row.get("sampled_euler_success_mean")) for row in summary_rows]
    sampled_l2f = [to_float(row.get("sampled_l2f_success_mean")) for row in summary_rows]
    fixed_euler = [to_float(row.get("fixed_euler_success_mean")) for row in summary_rows]
    fixed_l2f = [to_float(row.get("fixed_l2f_success_mean")) for row in summary_rows]
    invalid = [to_float(row.get("sampled_euler_invalid_rate")) for row in summary_rows] + [to_float(row.get("sampled_l2f_invalid_rate")) for row in summary_rows]
    rejection = [to_float(row.get("dynamics_rejection_rate")) for row in summary_rows]
    nonzero = sum(1 for row in summary_rows if row["sampled_nonzero_both"])
    beat = sum(1 for row in summary_rows if row["sampled_beats_old_fixed_baselines"])

    report_md = report_dir / "sampled_dynamics_report.md"
    with report_md.open("w") as f:
        f.write("# Sampled-Dynamics Audit Summary\n\n")
        f.write(f"Run root: `{run_root}`\n\n")
        f.write("This report only evaluates sampled-dynamics training/evaluation for the differentiable pretraining executable. It does not prove teacher-cost reduction, RAPTOR replacement, sampled dynamics solved in general, Sim2Real transfer, or full L2F dynamics equivalence.\n\n")
        f.write("## Metadata\n\n")
        f.write("| item | value |\n| --- | --- |\n")
        for key in ("run_name", "train_seeds", "steps", "sampled_dynamics_level", "init_actor_path", "zero_shot", "h128_schedule"):
            f.write(f"| {key} | {metadata_value(metadata, key, 'n/a')} |\n")
        f.write("\n## Aggregate\n\n")
        f.write("| metric | value |\n| --- | ---: |\n")
        f.write(f"| Sampled Euler success mean/std | {fmt(mean(sampled_euler))} / {fmt(pstdev(sampled_euler))} |\n")
        f.write(f"| Sampled L2F success mean/std | {fmt(mean(sampled_l2f))} / {fmt(pstdev(sampled_l2f))} |\n")
        f.write(f"| Fixed Euler success mean/std after run | {fmt(mean(fixed_euler))} / {fmt(pstdev(fixed_euler))} |\n")
        f.write(f"| Fixed L2F success mean/std after run | {fmt(mean(fixed_l2f))} / {fmt(pstdev(fixed_l2f))} |\n")
        f.write(f"| Seeds with nonzero sampled success on both eval models | {nonzero} / {len(summary_rows)} |\n")
        f.write(f"| Seeds beating old fixed baselines under sampled eval | {beat} / {len(summary_rows)} |\n")
        f.write(f"| Mean sampled invalid_or_nan_rate | {fmt(mean(invalid))} |\n")
        f.write(f"| Mean dynamics rejection rate | {fmt(mean(rejection))} |\n")
        f.write("\n## Per Seed\n\n")
        f.write("| seed | init | sampled Euler/L2F success | fixed Euler/L2F success | sampled p/v/w Euler | sampled p/v/w L2F | rejection | invalid | sampled nonzero both |\n")
        f.write("| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |\n")
        for row in summary_rows:
            f.write(
                f"| {row['train_seed']} | {row['init_type']} | "
                f"{fmt(row['sampled_euler_success_mean'])} / {fmt(row['sampled_l2f_success_mean'])} | "
                f"{fmt(row['fixed_euler_success_mean'])} / {fmt(row['fixed_l2f_success_mean'])} | "
                f"{fmt(row['sampled_euler_final_p'])} / {fmt(row['sampled_euler_final_v'])} / {fmt(row['sampled_euler_final_w'])} | "
                f"{fmt(row['sampled_l2f_final_p'])} / {fmt(row['sampled_l2f_final_v'])} / {fmt(row['sampled_l2f_final_w'])} | "
                f"{fmt(to_float(row['dynamics_rejection_rate']))} | "
                f"{fmt(max(to_float(row['sampled_euler_invalid_rate']), to_float(row['sampled_l2f_invalid_rate'])))} | "
                f"{str(row['sampled_nonzero_both']).lower()} |\n"
            )
    print(f"Wrote {summary_csv}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()

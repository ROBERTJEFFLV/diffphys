#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median


FAILED_SEEDS = {"6", "9"}
SUCCESS_CONTROL_SEEDS = {"2", "3", "7"}
PHASE_ORDER = ["H16_easy", "H32_easy", "H64_easy", "H128_easy", "H128_medium", "H128_full"]
EULER_BASELINE_SUCCESS = 0.11
L2F_BASELINE_SUCCESS = 0.09

TRAIN_RE = re.compile(r"fixed_vstrong_seed(?P<seed>\d+)_train\.csv$")
EVAL_RE = re.compile(r"fixed_vstrong_seed(?P<seed>\d+)_evalseed(?P<eval_seed>\d+)_(?P<model>euler|l2f)\.csv$")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value, default=math.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except ValueError:
        return default


def to_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def numbers(values):
    return [value for value in values if not math.isnan(value)]


def safe_mean(values):
    values = numbers(values)
    return sum(values) / len(values) if values else math.nan


def safe_median(values):
    values = numbers(values)
    return median(values) if values else math.nan


def first_float(row, names):
    for name in names:
        if name in row:
            value = to_float(row.get(name))
            if not math.isnan(value):
                return value
    return math.nan


def phase_name(row):
    phase = row.get("curriculum_phase_name") or row.get("h128_phase") or ""
    if phase:
        return phase
    horizon = row.get("current_horizon", "")
    difficulty = first_float(row, ("current_state_difficulty",))
    if horizon == "16":
        return "H16_easy"
    if horizon == "32":
        return "H32_easy"
    if horizon == "64":
        return "H64_easy"
    if horizon == "128" and not math.isnan(difficulty):
        if difficulty <= 0.3:
            return "H128_easy"
        if difficulty <= 0.75:
            return "H128_medium"
        return "H128_full"
    return "unknown"


def first_last_slope(rows, names):
    values = [first_float(row, names) for row in rows]
    values = numbers(values)
    if not values:
        return math.nan, math.nan, math.nan
    if len(values) == 1:
        return values[0], values[-1], 0.0
    return values[0], values[-1], (values[-1] - values[0]) / (len(values) - 1)


def bool_fraction(rows, names):
    if not rows:
        return math.nan
    count = 0
    active = 0
    for row in rows:
        for name in names:
            if name in row:
                count += 1
                active += 1 if to_bool(row.get(name)) else 0
                break
    return active / count if count else math.nan


def median_ratio(rows, numerator_names, denominator_names):
    ratios = []
    for row in rows:
        numerator = first_float(row, numerator_names)
        denominator = first_float(row, denominator_names)
        if math.isnan(numerator) or math.isnan(denominator) or abs(denominator) < 1e-12:
            continue
        ratios.append(numerator / denominator)
    return safe_median(ratios)


def parse_train(run_root):
    train = {}
    for path in sorted((run_root / "logs").glob("*_train.csv")):
        match = TRAIN_RE.match(path.name)
        if not match:
            continue
        rows = read_csv(path)
        if rows:
            train[match.group("seed")] = {"path": path, "rows": rows}
    return train


def parse_eval(run_root):
    evals = defaultdict(lambda: defaultdict(list))
    for path in sorted((run_root / "logs").glob("*_eval*.csv")):
        match = EVAL_RE.match(path.name)
        if not match:
            continue
        rows = read_csv(path)
        if not rows:
            continue
        row = rows[0]
        evals[match.group("seed")][match.group("model")].append({
            "success": first_float(row, ("success_rate",)),
            "p": first_float(row, ("mean_final_position_norm",)),
            "v": first_float(row, ("mean_final_velocity_norm",)),
            "w": first_float(row, ("mean_final_angular_velocity_norm",)),
            "invalid": first_float(row, ("invalid_or_nan_rate",)),
        })
    return evals


def aggregate_eval(records):
    if not records:
        return {"success": math.nan, "p": math.nan, "v": math.nan, "w": math.nan, "invalid": math.nan}
    return {
        "success": safe_mean([record["success"] for record in records]),
        "p": safe_mean([record["p"] for record in records]),
        "v": safe_mean([record["v"] for record in records]),
        "w": safe_mean([record["w"] for record in records]),
        "invalid": safe_mean([record["invalid"] for record in records]),
    }


def summarize_phase(seed, phase, rows, eval_summary):
    first_loss, final_loss, slope = first_last_slope(rows, ("total_loss_mean", "loss_total"))
    total_names = ("total_loss_mean", "loss_total")
    actor_pre_names = ("actor_grad_norm_pre_clip", "actor_grad_norm_before_clip")
    actor_scale_names = ("actor_grad_scale",)
    action_pre_names = ("action_grad_norm_pre_clip", "action_grad_norm_before_clip")
    action_scale_names = ("action_grad_clip_scale", "action_grad_scale")
    return {
        "seed": seed,
        "group": "failed" if seed in FAILED_SEEDS else "success_control" if seed in SUCCESS_CONTROL_SEEDS else "other",
        "phase": phase,
        "rows": len(rows),
        "first_step": int(to_float(rows[0].get("step"), 0)) if rows else 0,
        "last_step": int(to_float(rows[-1].get("step"), 0)) if rows else 0,
        "first_loss": first_loss,
        "median_loss": safe_median([first_float(row, total_names) for row in rows]),
        "final_loss": final_loss,
        "loss_slope": slope,
        "median_final_p": safe_median([first_float(row, ("mean_final_position_norm",)) for row in rows]),
        "median_final_v": safe_median([first_float(row, ("mean_final_velocity_norm",)) for row in rows]),
        "median_final_w": safe_median([first_float(row, ("mean_final_angular_velocity_norm",)) for row in rows]),
        "final_p": first_float(rows[-1], ("mean_final_position_norm",)),
        "final_v": first_float(rows[-1], ("mean_final_velocity_norm",)),
        "final_w": first_float(rows[-1], ("mean_final_angular_velocity_norm",)),
        "median_actor_grad_norm_pre_clip": safe_median([first_float(row, actor_pre_names) for row in rows]),
        "actor_grad_clip_fraction": bool_fraction(rows, ("actor_grad_clipped_flag",)),
        "median_actor_grad_clip_scale": safe_median([first_float(row, actor_scale_names) for row in rows]),
        "median_action_grad_norm_pre_clip": safe_median([first_float(row, action_pre_names) for row in rows]),
        "action_grad_clip_fraction": bool_fraction(rows, ("action_grad_clip_active_flag", "action_grad_clipped_flag")),
        "median_action_grad_clip_scale": safe_median([first_float(row, action_scale_names) for row in rows]),
        "action_mean": safe_median([first_float(row, ("action_mean",)) for row in rows]),
        "action_std": safe_median([first_float(row, ("action_std",)) for row in rows]),
        "action_abs_mean": safe_median([first_float(row, ("action_abs_mean",)) for row in rows]),
        "action_saturation_ratio": safe_median([first_float(row, ("action_saturation_ratio", "action_clamp_rate")) for row in rows]),
        "action_delta_mean": safe_median([first_float(row, ("action_delta_mean",)) for row in rows]),
        "action_delta_max": safe_median([first_float(row, ("action_delta_max",)) for row in rows]),
        "loss_position_fraction": median_ratio(rows, ("loss_position", "position_loss_mean"), total_names),
        "loss_velocity_fraction": median_ratio(rows, ("loss_velocity", "velocity_loss_mean"), total_names),
        "loss_angular_velocity_fraction": median_ratio(rows, ("loss_angular_velocity", "angular_velocity_loss_mean"), total_names),
        "loss_terminal_position_fraction": median_ratio(rows, ("loss_terminal_position", "terminal_position_loss_mean"), total_names),
        "loss_terminal_velocity_fraction": median_ratio(rows, ("loss_terminal_velocity", "terminal_velocity_loss_mean"), total_names),
        "loss_terminal_angular_velocity_fraction": median_ratio(rows, ("loss_terminal_angular_velocity", "terminal_angular_velocity_loss_mean"), total_names),
        "training_success_rate": safe_median([first_float(row, ("training_success_rate",)) for row in rows]),
        "eval_euler_success": eval_summary["euler"]["success"],
        "eval_l2f_success": eval_summary["l2f"]["success"],
    }


def compare_group(phase_rows, field, group):
    return safe_median([row[field] for row in phase_rows if row["group"] == group])


def generate_answers(phase_summaries):
    lines = []
    failed = [row for row in phase_summaries if row["group"] == "failed"]
    controls = [row for row in phase_summaries if row["group"] == "success_control"]
    failed_h128 = [row for row in failed if row["phase"].startswith("H128")]
    control_h128 = [row for row in controls if row["phase"].startswith("H128")]
    failed_pre_h128 = [row for row in failed if row["phase"] in ("H16_easy", "H32_easy", "H64_easy")]

    pre_loss = safe_median([row["final_loss"] for row in failed_pre_h128])
    h128_loss = safe_median([row["final_loss"] for row in failed_h128])
    lines.append(f"1. Seeds 6/9 before H128 median final phase loss is {fmt(pre_loss)}; H128 median final phase loss is {fmt(h128_loss)}. This indicates whether collapse is concentrated after H128.")

    failed_actor = safe_median([row["median_actor_grad_norm_pre_clip"] for row in failed_h128])
    control_actor = safe_median([row["median_actor_grad_norm_pre_clip"] for row in control_h128])
    lines.append(f"2. H128 median actor grad pre-clip: failed seeds {fmt(failed_actor)}, controls {fmt(control_actor)}.")

    failed_actor_clip = safe_median([row["actor_grad_clip_fraction"] for row in failed_h128])
    control_actor_clip = safe_median([row["actor_grad_clip_fraction"] for row in control_h128])
    lines.append(f"3. H128 actor-gradient clip fraction: failed seeds {fmt(failed_actor_clip)}, controls {fmt(control_actor_clip)}.")

    failed_action_clip = safe_median([row["action_grad_clip_fraction"] for row in failed_h128])
    failed_action_scale = safe_median([row["median_action_grad_clip_scale"] for row in failed_h128])
    lines.append(f"4. H128 action-gradient clip fraction is {fmt(failed_action_clip)} with median scale {fmt(failed_action_scale)} for failed seeds.")

    failed_abs = safe_median([row["action_abs_mean"] for row in failed_h128])
    control_abs = safe_median([row["action_abs_mean"] for row in control_h128])
    failed_sat = safe_median([row["action_saturation_ratio"] for row in failed_h128])
    control_sat = safe_median([row["action_saturation_ratio"] for row in control_h128])
    lines.append(f"5. H128 action abs/saturation medians: failed {fmt(failed_abs)} / {fmt(failed_sat)}, controls {fmt(control_abs)} / {fmt(control_sat)}.")

    failed_tv = safe_median([row["loss_terminal_velocity_fraction"] for row in failed_h128])
    failed_tp = safe_median([row["loss_terminal_position_fraction"] for row in failed_h128])
    control_tv = safe_median([row["loss_terminal_velocity_fraction"] for row in control_h128])
    control_tp = safe_median([row["loss_terminal_position_fraction"] for row in control_h128])
    lines.append(f"6. H128 terminal velocity/position fractions: failed {fmt(failed_tv)} / {fmt(failed_tp)}, controls {fmt(control_tv)} / {fmt(control_tp)}.")

    full_failed = [row for row in failed if row["phase"] == "H128_full"]
    failed_full_slope = safe_median([row["loss_slope"] for row in full_failed])
    failed_full_eval = safe_median([row["eval_l2f_success"] for row in full_failed])
    lines.append(f"7. H128_full failed-seed median loss slope is {fmt(failed_full_slope)} and final L2F eval success median is {fmt(failed_full_eval)}.")

    full_rows = [row for row in phase_summaries if row["phase"] == "H128_full"]
    failed_full_rows = safe_median([row["rows"] for row in full_rows if row["group"] == "failed"])
    control_full_rows = safe_median([row["rows"] for row in full_rows if row["group"] == "success_control"])
    lines.append(f"8. H128_full row counts: failed seeds median {fmt(failed_full_rows)}, controls median {fmt(control_full_rows)}. Compare this with slope and eval success to judge whether H128_full is long enough.")
    return lines


def write_outputs(run_root, phase_summaries, answers):
    report_dir = run_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "h128_failure_analysis.csv"
    md_path = report_dir / "h128_failure_analysis.md"
    fieldnames = [
        "seed", "group", "phase", "rows", "first_step", "last_step",
        "first_loss", "median_loss", "final_loss", "loss_slope",
        "median_final_p", "median_final_v", "median_final_w", "final_p", "final_v", "final_w",
        "median_actor_grad_norm_pre_clip", "actor_grad_clip_fraction", "median_actor_grad_clip_scale",
        "median_action_grad_norm_pre_clip", "action_grad_clip_fraction", "median_action_grad_clip_scale",
        "action_mean", "action_std", "action_abs_mean", "action_saturation_ratio", "action_delta_mean", "action_delta_max",
        "loss_position_fraction", "loss_velocity_fraction", "loss_angular_velocity_fraction",
        "loss_terminal_position_fraction", "loss_terminal_velocity_fraction", "loss_terminal_angular_velocity_fraction",
        "training_success_rate", "eval_euler_success", "eval_l2f_success",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in phase_summaries:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# H128 Failure Analysis\n\n")
        f.write(f"Run root: `{run_root}`\n\n")
        f.write("This analysis is fixed-dynamics only and compares failed seeds 6/9 against successful controls 2/3/7.\n\n")
        f.write("## Answers\n\n")
        for line in answers:
            f.write(f"{line}\n\n")
        f.write("## Phase Summary\n\n")
        f.write("| seed | group | phase | final loss | slope | final p/v/w | actor grad med | actor clip frac | action grad med | action clip frac | action abs | action sat | terminal v frac | terminal w frac | train success | eval Euler/L2F |\n")
        f.write("| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in phase_summaries:
            f.write(
                f"| {row['seed']} | {row['group']} | {row['phase']} | {fmt(row['final_loss'])} | {fmt(row['loss_slope'])} | "
                f"{fmt(row['final_p'])} / {fmt(row['final_v'])} / {fmt(row['final_w'])} | "
                f"{fmt(row['median_actor_grad_norm_pre_clip'])} | {fmt(row['actor_grad_clip_fraction'])} | "
                f"{fmt(row['median_action_grad_norm_pre_clip'])} | {fmt(row['action_grad_clip_fraction'])} | "
                f"{fmt(row['action_abs_mean'])} | {fmt(row['action_saturation_ratio'])} | "
                f"{fmt(row['loss_terminal_velocity_fraction'])} | {fmt(row['loss_terminal_angular_velocity_fraction'])} | "
                f"{fmt(row['training_success_rate'])} | {fmt(row['eval_euler_success'])} / {fmt(row['eval_l2f_success'])} |\n"
            )
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    run_root = args.run_root
    train = parse_train(run_root)
    evals = parse_eval(run_root)
    phase_summaries = []
    for seed in sorted(train, key=lambda s: int(s)):
        if seed not in FAILED_SEEDS and seed not in SUCCESS_CONTROL_SEEDS:
            continue
        rows_by_phase = defaultdict(list)
        for row in train[seed]["rows"]:
            rows_by_phase[phase_name(row)].append(row)
        eval_summary = {
            "euler": aggregate_eval(evals[seed]["euler"]),
            "l2f": aggregate_eval(evals[seed]["l2f"]),
        }
        for phase in PHASE_ORDER:
            rows = rows_by_phase.get(phase, [])
            if rows:
                phase_summaries.append(summarize_phase(seed, phase, rows, eval_summary))
    answers = generate_answers(phase_summaries)
    csv_path, md_path = write_outputs(run_root, phase_summaries, answers)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

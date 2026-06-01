#!/usr/bin/env python3
import csv
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev


SCRIPT_DIR = Path(__file__).resolve().parent
RL_TOOLS_DIR = SCRIPT_DIR.parents[3]
RUN_NAME = os.environ.get("RUN_NAME", "fixed_vstrong_10seed_audit")
RUN_ROOT = RL_TOOLS_DIR / "runs" / "diff_pre_training" / RUN_NAME
LOG_DIR = RUN_ROOT / "logs"
CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
REPORT_DIR = RUN_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SEEDS = [seed for seed in os.environ.get("TRAIN_SEEDS", "0 1 2 3 4 5 6 7 8 9").split() if seed]
EVAL_SEEDS = [seed for seed in os.environ.get("EVAL_SEEDS", "10000 10001 10002").split() if seed]
EVAL_MODELS = [model for model in os.environ.get("EVAL_MODELS", "euler l2f").split() if model]

EULER_BASELINE_SUCCESS = 0.11
L2F_BASELINE_SUCCESS = 0.09
EULER_BASELINE_PVW = (2.22573, 2.26618, 0.939211)
L2F_BASELINE_PVW = (2.44163, 2.65331, 1.02248)
NEAR_ZERO_SUCCESS = 0.02

TRAIN_RE = re.compile(r"fixed_vstrong_seed(?P<seed>\d+)_train\.csv$")
EVAL_RE = re.compile(r"fixed_vstrong_seed(?P<train_seed>\d+)_evalseed(?P<eval_seed>\d+)_(?P<model>euler|l2f)\.csv$")


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


def to_int(value, default=0):
    number = to_float(value)
    if math.isnan(number):
        return default
    return int(number)


def bool_text(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def fmt(value):
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "not available"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def valid_numbers(values):
    return [value for value in values if not math.isnan(value)]


def safe_mean(values):
    values = valid_numbers(values)
    return mean(values) if values else math.nan


def safe_std(values):
    values = valid_numbers(values)
    return pstdev(values) if len(values) > 1 else 0.0 if values else math.nan


def safe_median(values):
    values = valid_numbers(values)
    return median(values) if values else math.nan


def first_float(row, names, default=math.nan):
    for name in names:
        if name in row:
            value = to_float(row.get(name), math.nan)
            if not math.isnan(value):
                return value
    return default


def parse_train_logs():
    data = {}
    for path in sorted(LOG_DIR.glob("*_train.csv")):
        match = TRAIN_RE.match(path.name)
        if not match:
            continue
        rows = read_csv(path)
        if not rows:
            continue
        seed = match.group("seed")
        first = rows[0]
        last = rows[-1]
        first_loss = first_float(first, ("total_loss_mean", "loss_total"))
        final_loss = first_float(last, ("total_loss_mean", "loss_total"))
        invalid_rollouts = sum(to_int(row.get("invalid_rollout_count"), 0) for row in rows)
        nan_or_inf = any(bool_text(row.get("nan_or_inf_flag", "")) for row in rows)
        data[seed] = {
            "seed": seed,
            "path": path,
            "rows": rows,
            "checkpoint": CHECKPOINT_DIR / f"fixed_vstrong_seed{seed}_policy.bin",
            "first_loss": first_loss,
            "final_loss": final_loss,
            "loss_reduction": first_loss - final_loss,
            "loss_increased": final_loss > first_loss if not math.isnan(first_loss) and not math.isnan(final_loss) else False,
            "final_p": first_float(last, ("mean_final_position_norm",)),
            "final_v": first_float(last, ("mean_final_velocity_norm",)),
            "final_w": first_float(last, ("mean_final_angular_velocity_norm",)),
            "applied_steps": to_int(last.get("num_applied_steps"), 0),
            "skipped_steps": to_int(last.get("num_skipped_steps"), 0),
            "invalid_rollouts": invalid_rollouts,
            "nan_or_inf": nan_or_inf,
            "action_saturation_median": safe_median([first_float(row, ("action_saturation_ratio", "action_clamp_rate")) for row in rows]),
            "action_saturation_high_fraction": high_fraction(rows, ("action_saturation_ratio", "action_clamp_rate"), 0.5),
            "actor_grad_clip_fraction": high_fraction(rows, ("actor_grad_clipped_flag",), 0.5, boolean=True),
            "actor_grad_scale_median": safe_median([first_float(row, ("actor_grad_scale",)) for row in rows]),
            "diagnostic_unavailable_fields": last.get("diagnostic_unavailable_fields", ""),
        }
    return data


def high_fraction(rows, names, threshold, boolean=False):
    count = 0
    high = 0
    for row in rows:
        if boolean:
            value = any(bool_text(row.get(name, "")) for name in names)
            count += 1
            high += 1 if value else 0
        else:
            value = first_float(row, names)
            if math.isnan(value):
                continue
            count += 1
            high += 1 if value > threshold else 0
    return high / count if count else math.nan


def parse_eval_logs():
    data = defaultdict(lambda: defaultdict(list))
    for path in sorted(LOG_DIR.glob("*_eval*.csv")):
        match = EVAL_RE.match(path.name)
        if not match:
            continue
        rows = read_csv(path)
        if not rows:
            continue
        row = rows[0]
        train_seed = match.group("train_seed")
        model = match.group("model")
        data[train_seed][model].append({
            "train_seed": train_seed,
            "eval_seed": match.group("eval_seed"),
            "model": model,
            "path": path,
            "success": first_float(row, ("success_rate",)),
            "mean_p": first_float(row, ("mean_final_position_norm",)),
            "mean_v": first_float(row, ("mean_final_velocity_norm",)),
            "mean_w": first_float(row, ("mean_final_angular_velocity_norm",)),
            "invalid": first_float(row, ("invalid_or_nan_rate",)),
        })
    return data


def aggregate_eval(records):
    if not records:
        return None
    successes = [record["success"] for record in records]
    return {
        "count": len(records),
        "mean_success": safe_mean(successes),
        "std_success": safe_std(successes),
        "min_success": min(valid_numbers(successes)) if valid_numbers(successes) else math.nan,
        "max_success": max(valid_numbers(successes)) if valid_numbers(successes) else math.nan,
        "mean_p": safe_mean([record["mean_p"] for record in records]),
        "mean_v": safe_mean([record["mean_v"] for record in records]),
        "mean_w": safe_mean([record["mean_w"] for record in records]),
        "mean_invalid": safe_mean([record["invalid"] for record in records]),
    }


def loss_values(rows):
    return [first_float(row, ("total_loss_mean", "loss_total")) for row in rows]


def early_divergence(train):
    rows = train.get("rows", [])
    if not rows:
        return False
    first_loss = train["first_loss"]
    final_loss = train["final_loss"]
    if math.isnan(first_loss) or math.isnan(final_loss) or final_loss <= first_loss:
        return False
    early = []
    for row in rows:
        step = to_int(row.get("step"), 0)
        if step <= 1000:
            early.append(first_float(row, ("total_loss_mean", "loss_total")))
    early = valid_numbers(early)
    return bool(early and max(early) > first_loss * 1.25)


def curriculum_collapse(train):
    rows = train.get("rows", [])
    losses = loss_values(rows)
    if len(rows) < 30:
        return False
    for index in range(1, len(rows)):
        current_state = first_float(rows[index], ("current_state_difficulty",))
        previous_state = first_float(rows[index - 1], ("current_state_difficulty",))
        state_jump = (
            not math.isnan(current_state)
            and not math.isnan(previous_state)
            and abs(current_state - previous_state) > 0.05
        )
        changed = (
            rows[index].get("current_horizon") != rows[index - 1].get("current_horizon")
            or state_jump
        )
        if not changed:
            continue
        before = valid_numbers(losses[max(0, index - 20):index])
        after = valid_numbers(losses[index:min(len(rows), index + 50)])
        if not before or not after:
            continue
        before_med = median(before)
        after_med = median(after)
        if before_med > 0 and (after_med > before_med * 2 or max(after) > before_med * 5):
            return True
    return False


def classify_seed(row, medians):
    if not row["status_complete"]:
        return "INCOMPLETE", "missing train, checkpoint, or required eval artifacts"
    euler_success = row["euler_success_mean"]
    l2f_success = row["l2f_success_mean"]
    beats_euler = row["beats_euler_baseline"]
    beats_l2f = row["beats_l2f_baseline"]
    beats_both = beats_euler and beats_l2f
    if beats_both and l2f_success >= 0.25 and row["loss_reduction"] > 0:
        return "PASS_STRONG", "beats both baselines, L2F success >= 0.25, and loss decreased"
    if beats_both:
        return "PASS_WEAK", "beats both baselines but misses strong-pass threshold"
    if beats_euler and not beats_l2f:
        return "TYPE_F_EULER_TO_L2F_TRANSFER_MISMATCH", "Euler beats baseline but L2F does not"

    near_zero = max(euler_success, l2f_success) <= NEAR_ZERO_SUCCESS
    if near_zero and row["loss_increased_flag"] and row["early_divergence_flag"]:
        return "TYPE_A_EARLY_DIVERGENCE", "loss increased early and both eval models have near-zero success"
    if row["curriculum_collapse_flag"]:
        return "TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE", "loss spiked after horizon or state curriculum changed"
    if near_zero and row["action_saturation_high_fraction"] > 0.3:
        return "TYPE_C_ACTION_SATURATION_COLLAPSE", "action saturation ratio is high for a large fraction of training"

    pass_train_w_median = medians.get("pass_train_w", math.nan)
    pass_eval_w_median = medians.get("pass_eval_w", math.nan)
    angular_high = (
        (not math.isnan(pass_train_w_median) and row["final_train_w"] > pass_train_w_median * 2)
        or (not math.isnan(pass_eval_w_median) and max(row["euler_final_w_mean"], row["l2f_final_w_mean"]) > pass_eval_w_median * 2)
    )
    if angular_high and (row["euler_final_p_mean"] < row["euler_final_w_mean"] or row["l2f_final_p_mean"] < row["l2f_final_w_mean"]):
        return "TYPE_D_ANGULAR_INSTABILITY", "angular velocity is abnormally high relative to passing seeds"

    hidden_mean = row.get("gru_hidden_norm_mean", math.nan)
    hidden_max = row.get("gru_hidden_norm_max", math.nan)
    if not math.isnan(hidden_mean) or not math.isnan(hidden_max):
        return "TYPE_E_GRU_HIDDEN_INSTABILITY", "GRU hidden norm diagnostic is abnormal"

    if row["loss_reduction"] > max(abs(row["first_loss"]) * 0.2, 1e-6) and not beats_euler and not beats_l2f:
        return "TYPE_G_OBJECTIVE_MISMATCH", "loss decreased but both evaluation success rates remain below baseline"
    if row["loss_reduction"] > 0:
        return "TYPE_H_SLOW_OR_WEAK_CONVERGENCE", "loss decreased but success remains weak or near baseline"
    return "TYPE_UNKNOWN_FAILURE", "failed seed did not match the simple diagnostic rules"


train_data = parse_train_logs()
eval_data = parse_eval_logs()

missing = []
rows = []
for seed in TRAIN_SEEDS:
    train = train_data.get(seed)
    checkpoint = CHECKPOINT_DIR / f"fixed_vstrong_seed{seed}_policy.bin"
    if train is None:
        missing.append(f"train seed {seed} train CSV")
    if not checkpoint.exists():
        missing.append(f"train seed {seed} checkpoint")
    aggregates = {model: aggregate_eval(eval_data.get(seed, {}).get(model, [])) for model in EVAL_MODELS}
    for model in EVAL_MODELS:
        present_eval_seeds = {record["eval_seed"] for record in eval_data.get(seed, {}).get(model, [])}
        for eval_seed in EVAL_SEEDS:
            if eval_seed not in present_eval_seeds:
                missing.append(f"train seed {seed} eval seed {eval_seed} model {model}")

    euler = aggregates.get("euler")
    l2f = aggregates.get("l2f")
    status_complete = train is not None and checkpoint.exists() and all(
        aggregate is not None and aggregate["count"] == len(EVAL_SEEDS)
        for aggregate in aggregates.values()
    )
    row = {
        "train_seed": seed,
        "status_complete": status_complete,
        "first_loss": train["first_loss"] if train else math.nan,
        "final_loss": train["final_loss"] if train else math.nan,
        "loss_reduction": train["loss_reduction"] if train else math.nan,
        "loss_increased_flag": train["loss_increased"] if train else False,
        "final_train_p": train["final_p"] if train else math.nan,
        "final_train_v": train["final_v"] if train else math.nan,
        "final_train_w": train["final_w"] if train else math.nan,
        "total_applied_steps": train["applied_steps"] if train else 0,
        "total_skipped_steps": train["skipped_steps"] if train else 0,
        "total_invalid_rollouts": train["invalid_rollouts"] if train else 0,
        "any_nan_or_inf": train["nan_or_inf"] if train else True,
        "checkpoint": str(checkpoint),
        "early_divergence_flag": early_divergence(train) if train else False,
        "curriculum_collapse_flag": curriculum_collapse(train) if train else False,
        "action_saturation_median": train["action_saturation_median"] if train else math.nan,
        "action_saturation_high_fraction": train["action_saturation_high_fraction"] if train else math.nan,
        "actor_grad_clip_fraction": train["actor_grad_clip_fraction"] if train else math.nan,
        "actor_grad_scale_median": train["actor_grad_scale_median"] if train else math.nan,
        "diagnostic_unavailable_fields": train["diagnostic_unavailable_fields"] if train else "",
        "euler_success_mean": euler["mean_success"] if euler else math.nan,
        "euler_success_std": euler["std_success"] if euler else math.nan,
        "euler_success_min": euler["min_success"] if euler else math.nan,
        "euler_success_max": euler["max_success"] if euler else math.nan,
        "euler_final_p_mean": euler["mean_p"] if euler else math.nan,
        "euler_final_v_mean": euler["mean_v"] if euler else math.nan,
        "euler_final_w_mean": euler["mean_w"] if euler else math.nan,
        "euler_invalid_or_nan_rate_mean": euler["mean_invalid"] if euler else math.nan,
        "l2f_success_mean": l2f["mean_success"] if l2f else math.nan,
        "l2f_success_std": l2f["std_success"] if l2f else math.nan,
        "l2f_success_min": l2f["min_success"] if l2f else math.nan,
        "l2f_success_max": l2f["max_success"] if l2f else math.nan,
        "l2f_final_p_mean": l2f["mean_p"] if l2f else math.nan,
        "l2f_final_v_mean": l2f["mean_v"] if l2f else math.nan,
        "l2f_final_w_mean": l2f["mean_w"] if l2f else math.nan,
        "l2f_invalid_or_nan_rate_mean": l2f["mean_invalid"] if l2f else math.nan,
    }
    row["beats_euler_baseline"] = row["euler_success_mean"] > EULER_BASELINE_SUCCESS
    row["beats_l2f_baseline"] = row["l2f_success_mean"] > L2F_BASELINE_SUCCESS
    row["beats_both_baselines"] = row["beats_euler_baseline"] and row["beats_l2f_baseline"]
    rows.append(row)

pass_rows = [row for row in rows if row["beats_both_baselines"]]
medians = {
    "pass_train_w": safe_median([row["final_train_w"] for row in pass_rows]),
    "pass_eval_w": safe_median([max(row["euler_final_w_mean"], row["l2f_final_w_mean"]) for row in pass_rows]),
}

for row in rows:
    failure_type, reason = classify_seed(row, medians)
    row["failure_type"] = failure_type
    row["failure_reason_short"] = reason

summary_csv = REPORT_DIR / "fixed_vstrong_10seed_audit_summary.csv"
fieldnames = [
    "train_seed", "status_complete", "first_loss", "final_loss", "loss_reduction", "loss_increased_flag",
    "final_train_p", "final_train_v", "final_train_w", "total_applied_steps", "total_skipped_steps",
    "total_invalid_rollouts", "any_nan_or_inf",
    "euler_success_mean", "euler_success_std", "euler_success_min", "euler_success_max",
    "euler_final_p_mean", "euler_final_v_mean", "euler_final_w_mean", "euler_invalid_or_nan_rate_mean",
    "l2f_success_mean", "l2f_success_std", "l2f_success_min", "l2f_success_max",
    "l2f_final_p_mean", "l2f_final_v_mean", "l2f_final_w_mean", "l2f_invalid_or_nan_rate_mean",
    "beats_euler_baseline", "beats_l2f_baseline", "beats_both_baselines",
    "action_saturation_median", "action_saturation_high_fraction",
    "actor_grad_clip_fraction", "actor_grad_scale_median",
    "early_divergence_flag", "curriculum_collapse_flag",
    "failure_type", "failure_reason_short",
]
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})

completed_train = sum(1 for seed in TRAIN_SEEDS if seed in train_data)
expected_eval_files = len(TRAIN_SEEDS) * len(EVAL_SEEDS) * len(EVAL_MODELS)
completed_eval_files = sum(
    len(eval_data.get(seed, {}).get(model, []))
    for seed in TRAIN_SEEDS
    for model in EVAL_MODELS
)
euler_successes = [row["euler_success_mean"] for row in rows if row["status_complete"]]
l2f_successes = [row["l2f_success_mean"] for row in rows if row["status_complete"]]
euler_mean = safe_mean(euler_successes)
euler_std = safe_std(euler_successes)
l2f_mean = safe_mean(l2f_successes)
l2f_std = safe_std(l2f_successes)
euler_beating = sum(1 for row in rows if row["beats_euler_baseline"])
l2f_beating = sum(1 for row in rows if row["beats_l2f_baseline"])
both_beating = sum(1 for row in rows if row["beats_both_baselines"])
catastrophic = sum(
    1 for row in rows
    if row["status_complete"] and max(row["euler_success_mean"], row["l2f_success_mean"]) <= NEAR_ZERO_SUCCESS
)
total_skipped = sum(row["total_skipped_steps"] for row in rows)
any_nan_or_inf = any(row["any_nan_or_inf"] for row in rows)
mean_invalid_rate = safe_mean(
    [row["euler_invalid_or_nan_rate_mean"] for row in rows if row["status_complete"]]
    + [row["l2f_invalid_or_nan_rate_mean"] for row in rows if row["status_complete"]]
)
failure_counts = Counter(row["failure_type"] for row in rows)
failed_rows = [row for row in rows if not row["failure_type"].startswith("PASS")]
failed_seeds = [row["train_seed"] for row in failed_rows]
catastrophic_failure_types = Counter(row["failure_type"] for row in failed_rows if max(row["euler_success_mean"], row["l2f_success_mean"]) <= NEAR_ZERO_SUCCESS)
repeated_catastrophic = any(count >= 2 for count in catastrophic_failure_types.values())

if missing:
    decision = "INCOMPLETE"
elif both_beating >= 9 and catastrophic == 0 and l2f_mean >= 0.25 and mean_invalid_rate <= 1e-9:
    decision = "STRONG_STABILITY"
elif both_beating >= 7 and not repeated_catastrophic:
    decision = "ACCEPTABLE_BUT_UNSTABLE"
else:
    decision = "UNSTABLE_NEEDS_FIX"

def recommendation_for_counts(counts):
    for failure_type, _count in counts.most_common():
        if failure_type.startswith("PASS") or failure_type == "INCOMPLETE":
            continue
        if failure_type == "TYPE_A_EARLY_DIVERGENCE":
            return "try lower actor learning rate and lower actor-grad-clip."
        if failure_type == "TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE":
            return "slow down or freeze horizon/state curriculum."
        if failure_type == "TYPE_C_ACTION_SATURATION_COLLAPSE":
            return "reduce gradient step size, add or adjust action regularization, or reduce terminal weights."
        if failure_type == "TYPE_D_ANGULAR_INSTABILITY":
            return "reduce w-terminal-w and possibly w-w."
        if failure_type == "TYPE_E_GRU_HIDDEN_INSTABILITY":
            return "inspect GRU BPTT, hidden-state norm, reset-hidden ablation, and gradient clipping through the recurrent path."
        if failure_type == "TYPE_F_EULER_TO_L2F_TRANSFER_MISMATCH":
            return "investigate Euler-to-L2F model mismatch."
        if failure_type == "TYPE_G_OBJECTIVE_MISMATCH":
            return "revise objective/success alignment."
        if failure_type == "TYPE_H_SLOW_OR_WEAK_CONVERGENCE":
            return "try longer training before changing objective."
    return "complete missing runs before choosing a targeted ablation." if missing else "inspect failed seed logs manually before changing training."


failure_summary_lines = [
    f"| {failure_type} | {count} |"
    for failure_type, count in sorted(failure_counts.items())
]
train_table = [
    f"| {row['train_seed']} | {fmt(row['first_loss'])} | {fmt(row['final_loss'])} | {fmt(row['loss_reduction'])} | "
    f"{fmt(row['final_train_p'])} / {fmt(row['final_train_v'])} / {fmt(row['final_train_w'])} | "
    f"{row['total_applied_steps']} / {row['total_skipped_steps']} | {row['total_invalid_rollouts']} | "
    f"{str(row['any_nan_or_inf']).lower()} | `{row['checkpoint']}` |"
    for row in rows
]
eval_table = [
    f"| {row['train_seed']} | {fmt(row['euler_success_mean'])} / {fmt(row['euler_success_std'])} | "
    f"{fmt(row['l2f_success_mean'])} / {fmt(row['l2f_success_std'])} | "
    f"{fmt(row['euler_final_p_mean'])} / {fmt(row['euler_final_v_mean'])} / {fmt(row['euler_final_w_mean'])} | "
    f"{fmt(row['l2f_final_p_mean'])} / {fmt(row['l2f_final_v_mean'])} / {fmt(row['l2f_final_w_mean'])} | "
    f"{str(row['beats_both_baselines']).lower()} | {row['failure_type']} |"
    for row in rows
]
missing_text = "\n".join(f"- {item}" for item in missing) if missing else "None."
most_common_failure = next((item for item in failure_counts.most_common() if not item[0].startswith("PASS")), None)
most_likely = (
    f"{most_common_failure[0]} is the most common non-pass label."
    if most_common_failure
    else "No non-pass failure type was observed."
)
unavailable_fields = sorted({
    field
    for row in rows
    for field in str(row.get("diagnostic_unavailable_fields", "")).split(";")
    if field
})
unavailable_text = ", ".join(unavailable_fields) if unavailable_fields else "None reported."

report = f"""# Fixed SO(3) 10-Seed Stability Audit

Run root: `{RUN_ROOT}`

This audit only evaluates fixed-dynamics repeatability of SO(3) differentiable pretraining. It does not prove teacher-cost reduction, replacement of RAPTOR teachers, sampled-dynamics adaptation, Sim2Real transfer, or full L2F dynamics equivalence.

## Completeness

| item | value |
| --- | ---: |
| expected train seeds | {len(TRAIN_SEEDS)} |
| completed train seeds | {completed_train} |
| expected eval files | {expected_eval_files} |
| completed eval files | {completed_eval_files} |

Missing artifacts:

{missing_text}

Diagnostic fields unavailable without broader model/optimizer instrumentation: {unavailable_text}

## Training Seed Table

| seed | first loss | final loss | loss reduction | final p/v/w | applied/skipped steps | invalid rollouts | NaN/Inf | checkpoint |
| ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
{chr(10).join(train_table)}

## Evaluation Table

| seed | Euler success mean/std | L2F success mean/std | Euler p/v/w | L2F p/v/w | beats both baselines | failure type |
| ---: | ---: | ---: | --- | --- | --- | --- |
{chr(10).join(eval_table)}

## Failure-Mode Summary

| failure type | count |
| --- | ---: |
{chr(10).join(failure_summary_lines)}

Failed seeds: {", ".join(failed_seeds) if failed_seeds else "None"}.

Most likely failure explanation: {most_likely}

## Aggregate Metrics

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across train seeds | {fmt(euler_mean)} / {fmt(euler_std)} |
| L2F fixed mean/std success across train seeds | {fmt(l2f_mean)} / {fmt(l2f_std)} |
| seeds beating Euler baseline > {EULER_BASELINE_SUCCESS} | {euler_beating} / {len(TRAIN_SEEDS)} |
| seeds beating L2F baseline > {L2F_BASELINE_SUCCESS} | {l2f_beating} / {len(TRAIN_SEEDS)} |
| seeds beating both baselines | {both_beating} / {len(TRAIN_SEEDS)} |
| catastrophic zero-success seeds | {catastrophic} |
| total skipped actor steps | {total_skipped} |
| any NaN/Inf | {str(any_nan_or_inf).lower()} |
| mean invalid_or_nan_rate | {fmt(mean_invalid_rate)} |

Old first-order fixed v-strong baselines:

| Eval model | success | mean final p/v/w |
| --- | ---: | --- |
| Euler fixed | {EULER_BASELINE_SUCCESS} | {EULER_BASELINE_PVW[0]} / {EULER_BASELINE_PVW[1]} / {EULER_BASELINE_PVW[2]} |
| L2F fixed | {L2F_BASELINE_SUCCESS} | {L2F_BASELINE_PVW[0]} / {L2F_BASELINE_PVW[1]} / {L2F_BASELINE_PVW[2]} |

## Decision

{decision}

## Recommended Next Action

Do not automatically modify training from this audit alone; run a targeted ablation next: {recommendation_for_counts(failure_counts)}
"""

report_path = REPORT_DIR / "fixed_vstrong_10seed_audit_report.md"
report_path.write_text(report, encoding="utf-8")

print(f"Wrote {summary_csv}")
print(f"Wrote {report_path}")
print(f"Decision: {decision}")

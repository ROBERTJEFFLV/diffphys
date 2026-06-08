#!/usr/bin/env python3
import csv
import math
import os
import re
from pathlib import Path
from statistics import mean, pstdev


SCRIPT_DIR = Path(__file__).resolve().parent
RL_TOOLS_DIR = SCRIPT_DIR.parents[3]
RUN_ROOT = RL_TOOLS_DIR / "runs" / "diff_pre_training" / "fixed_vstrong_multiseed"
LOG_DIR = RUN_ROOT / "logs"
CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
REPORT_DIR = RUN_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SEEDS = [seed for seed in os.environ.get("TRAIN_SEEDS", "0 1 2").split() if seed]
EVAL_SEEDS = [seed for seed in os.environ.get("EVAL_SEEDS", "10000 10001 10002").split() if seed]

EULER_BASELINE_SUCCESS = 0.11
EULER_BASELINE_PVW = (2.22573, 2.26618, 0.939211)
L2F_BASELINE_SUCCESS = 0.09
L2F_BASELINE_PVW = (2.44163, 2.65331, 1.02248)

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


def fmt(value):
    if value is None:
        return "not available"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "not available"
    return f"{value:.6g}"


def safe_mean(values):
    values = [value for value in values if not math.isnan(value)]
    return mean(values) if values else math.nan


def safe_std(values):
    values = [value for value in values if not math.isnan(value)]
    return pstdev(values) if len(values) > 1 else 0.0 if values else math.nan


def parse_train_logs():
    data = {}
    for path in sorted(LOG_DIR.glob("*_train.csv")):
        match = TRAIN_RE.match(path.name)
        if not match:
            continue
        seed = match.group("seed")
        rows = read_csv(path)
        if not rows:
            continue
        first = rows[0]
        last = rows[-1]
        invalid_rollouts = sum(int(to_float(row.get("invalid_rollout_count"), 0)) for row in rows)
        nan_or_inf = any(str(row.get("nan_or_inf_flag", "")).lower() == "true" for row in rows)
        checkpoint = CHECKPOINT_DIR / f"fixed_vstrong_seed{seed}_policy.bin"
        first_loss = to_float(first.get("total_loss_mean"))
        final_loss = to_float(last.get("total_loss_mean"))
        data[seed] = {
            "seed": seed,
            "path": path,
            "checkpoint": checkpoint,
            "first_loss": first_loss,
            "final_loss": final_loss,
            "loss_reduction": first_loss - final_loss,
            "final_p": to_float(last.get("mean_final_position_norm")),
            "final_v": to_float(last.get("mean_final_velocity_norm")),
            "final_w": to_float(last.get("mean_final_angular_velocity_norm")),
            "applied_steps": int(to_float(last.get("num_applied_steps"), 0)),
            "skipped_steps": int(to_float(last.get("num_skipped_steps"), 0)),
            "invalid_rollouts": invalid_rollouts,
            "nan_or_inf": nan_or_inf,
        }
    return data


def parse_eval_logs():
    data = {}
    for path in sorted(LOG_DIR.glob("*_eval*.csv")):
        match = EVAL_RE.match(path.name)
        if not match:
            continue
        train_seed = match.group("train_seed")
        eval_seed = match.group("eval_seed")
        model = match.group("model")
        rows = read_csv(path)
        if not rows:
            continue
        row = rows[0]
        data.setdefault(train_seed, {}).setdefault(model, []).append({
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "model": model,
            "path": path,
            "success": to_float(row.get("success_rate")),
            "near_p": to_float(row.get("near_success_rate_p")),
            "near_pv": to_float(row.get("near_success_rate_pv")),
            "mean_p": to_float(row.get("mean_final_position_norm")),
            "mean_v": to_float(row.get("mean_final_velocity_norm")),
            "mean_w": to_float(row.get("mean_final_angular_velocity_norm")),
            "invalid": to_float(row.get("invalid_or_nan_rate")),
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
        "min_success": min(successes),
        "max_success": max(successes),
        "mean_p": safe_mean([record["mean_p"] for record in records]),
        "mean_v": safe_mean([record["mean_v"] for record in records]),
        "mean_w": safe_mean([record["mean_w"] for record in records]),
        "mean_invalid": safe_mean([record["invalid"] for record in records]),
    }


train_data = parse_train_logs()
eval_data = parse_eval_logs()

missing = []
for seed in TRAIN_SEEDS:
    if seed not in train_data:
        missing.append(f"train seed {seed}")
    for model in ("euler", "l2f"):
        records = eval_data.get(seed, {}).get(model, [])
        present_eval_seeds = {record["eval_seed"] for record in records}
        for eval_seed in EVAL_SEEDS:
            if eval_seed not in present_eval_seeds:
                missing.append(f"train seed {seed} eval seed {eval_seed} model {model}")

seed_model_rows = []
for seed in sorted(set(TRAIN_SEEDS) | set(train_data.keys()), key=lambda s: int(s)):
    train = train_data.get(seed)
    for model in ("euler", "l2f"):
        aggregate = aggregate_eval(eval_data.get(seed, {}).get(model, []))
        baseline = EULER_BASELINE_SUCCESS if model == "euler" else L2F_BASELINE_SUCCESS
        row = {
            "train_seed": seed,
            "eval_model": model,
            "train_first_loss": fmt(train["first_loss"]) if train else "not available",
            "train_final_loss": fmt(train["final_loss"]) if train else "not available",
            "train_loss_reduction": fmt(train["loss_reduction"]) if train else "not available",
            "train_final_p": fmt(train["final_p"]) if train else "not available",
            "train_final_v": fmt(train["final_v"]) if train else "not available",
            "train_final_w": fmt(train["final_w"]) if train else "not available",
            "applied_steps": train["applied_steps"] if train else "not available",
            "skipped_steps": train["skipped_steps"] if train else "not available",
            "invalid_rollouts": train["invalid_rollouts"] if train else "not available",
            "nan_or_inf": str(train["nan_or_inf"]).lower() if train else "not available",
            "eval_count": aggregate["count"] if aggregate else 0,
            "mean_success": fmt(aggregate["mean_success"]) if aggregate else "not available",
            "std_success": fmt(aggregate["std_success"]) if aggregate else "not available",
            "min_success": fmt(aggregate["min_success"]) if aggregate else "not available",
            "max_success": fmt(aggregate["max_success"]) if aggregate else "not available",
            "mean_final_p": fmt(aggregate["mean_p"]) if aggregate else "not available",
            "mean_final_v": fmt(aggregate["mean_v"]) if aggregate else "not available",
            "mean_final_w": fmt(aggregate["mean_w"]) if aggregate else "not available",
            "mean_invalid_or_nan_rate": fmt(aggregate["mean_invalid"]) if aggregate else "not available",
            "beats_success_baseline": str(bool(aggregate and aggregate["mean_success"] > baseline)).lower(),
            "checkpoint_path": str(train["checkpoint"]) if train else "not available",
        }
        seed_model_rows.append(row)

summary_csv = REPORT_DIR / "fixed_vstrong_multiseed_summary.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    fieldnames = list(seed_model_rows[0].keys()) if seed_model_rows else ["status"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(seed_model_rows)

per_seed = {}
for seed in TRAIN_SEEDS:
    euler = aggregate_eval(eval_data.get(seed, {}).get("euler", []))
    l2f = aggregate_eval(eval_data.get(seed, {}).get("l2f", []))
    per_seed[seed] = {"euler": euler, "l2f": l2f}

euler_seed_success = [per_seed[seed]["euler"]["mean_success"] for seed in TRAIN_SEEDS if per_seed[seed]["euler"]]
l2f_seed_success = [per_seed[seed]["l2f"]["mean_success"] for seed in TRAIN_SEEDS if per_seed[seed]["l2f"]]
euler_mean = safe_mean(euler_seed_success)
euler_std = safe_std(euler_seed_success)
l2f_mean = safe_mean(l2f_seed_success)
l2f_std = safe_std(l2f_seed_success)

euler_beating = sum(1 for value in euler_seed_success if value > EULER_BASELINE_SUCCESS)
l2f_beating = sum(1 for value in l2f_seed_success if value > L2F_BASELINE_SUCCESS)
both_beating = 0
for seed in TRAIN_SEEDS:
    euler = per_seed[seed]["euler"]
    l2f = per_seed[seed]["l2f"]
    if euler and l2f and euler["mean_success"] > EULER_BASELINE_SUCCESS and l2f["mean_success"] > L2F_BASELINE_SUCCESS:
        both_beating += 1

all_invalid = []
for seed in TRAIN_SEEDS:
    for model in ("euler", "l2f"):
        aggregate = per_seed[seed][model]
        if aggregate:
            all_invalid.append(aggregate["mean_invalid"])
any_train_nan = any(train_data.get(seed, {}).get("nan_or_inf", True) for seed in TRAIN_SEEDS)
total_skipped = sum(train_data.get(seed, {}).get("skipped_steps", 0) for seed in TRAIN_SEEDS)

if missing:
    verdict = "INCOMPLETE"
    verdict_reason = "not all required training/evaluation artifacts are available"
elif both_beating >= 2:
    strong = (
        both_beating == len(TRAIN_SEEDS)
        and not math.isnan(l2f_mean)
        and l2f_mean >= 0.25
        and all(abs(value) <= 1e-9 for value in all_invalid)
        and not any_train_nan
        and total_skipped == 0
    )
    verdict = "PASS"
    verdict_reason = "fixed SO(3) differentiable pretraining appears repeatable"
    if strong:
        verdict_reason += " and meets strong-pass criteria"
else:
    if not math.isnan(euler_mean) and not math.isnan(l2f_mean) and (euler_mean > EULER_BASELINE_SUCCESS or l2f_mean > L2F_BASELINE_SUCCESS):
        verdict = "PARTIAL"
        verdict_reason = "mean performance improved but seed repeatability is insufficient"
    else:
        verdict = "FAIL"
        verdict_reason = "result is not repeatable; do not proceed to teacher-cost claims"

train_table = []
for seed in sorted(train_data, key=lambda s: int(s)):
    row = train_data[seed]
    train_table.append(
        f"| {seed} | {fmt(row['first_loss'])} | {fmt(row['final_loss'])} | {fmt(row['loss_reduction'])} | "
        f"{fmt(row['final_p'])} / {fmt(row['final_v'])} / {fmt(row['final_w'])} | "
        f"{row['applied_steps']} | {row['skipped_steps']} | {row['invalid_rollouts']} | {str(row['nan_or_inf']).lower()} | `{row['checkpoint']}` |"
    )

eval_table = []
for seed in sorted(per_seed, key=lambda s: int(s)):
    for model in ("euler", "l2f"):
        aggregate = per_seed[seed][model]
        if aggregate:
            eval_table.append(
                f"| {seed} | {model} | {fmt(aggregate['mean_success'])} | {fmt(aggregate['std_success'])} | "
                f"{fmt(aggregate['min_success'])} / {fmt(aggregate['max_success'])} | "
                f"{fmt(aggregate['mean_p'])} / {fmt(aggregate['mean_v'])} / {fmt(aggregate['mean_w'])} | "
                f"{fmt(aggregate['mean_invalid'])} |"
            )
        else:
            eval_table.append(f"| {seed} | {model} | not available | not available | not available | not available | not available |")

missing_text = "\n".join(f"- {item}" for item in missing) if missing else "None."

report = f"""# Fixed SO(3) Multi-Seed Repeatability

Run root: `{RUN_ROOT}`

## Verdict

{verdict}: {verdict_reason}.

This experiment only tests fixed-dynamics repeatability. It does not prove teacher-cost reduction, RAPTOR replacement, sampled-dynamics success, Sim2Real transfer, or full L2F dynamics equivalence.

## Required Artifact Completeness

Missing artifacts:

{missing_text}

## Training Seed Table

| train seed | first loss | final loss | loss reduction | final mean p/v/w | applied steps | skipped steps | invalid rollout count | NaN/Inf flag | checkpoint |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
{chr(10).join(train_table) if train_table else "| not available | not available | not available | not available | not available | not available | not available | not available | not available | not available |"}

## Evaluation By Training Seed

| train seed | eval model | mean success | std success | min/max success | mean final p/v/w | mean invalid_or_nan_rate |
| --- | --- | ---: | ---: | --- | --- | ---: |
{chr(10).join(eval_table) if eval_table else "| not available | not available | not available | not available | not available | not available | not available |"}

## Overall Aggregate

| metric | value |
| --- | ---: |
| Euler fixed mean/std success across training seeds | {fmt(euler_mean)} / {fmt(euler_std)} |
| L2F fixed mean/std success across training seeds | {fmt(l2f_mean)} / {fmt(l2f_std)} |
| Train seeds beating old Euler success baseline > {EULER_BASELINE_SUCCESS} | {euler_beating} / {len(TRAIN_SEEDS)} |
| Train seeds beating old L2F success baseline > {L2F_BASELINE_SUCCESS} | {l2f_beating} / {len(TRAIN_SEEDS)} |
| Train seeds beating both success baselines | {both_beating} / {len(TRAIN_SEEDS)} |
| Total skipped actor steps | {total_skipped} |
| Any training NaN/Inf | {str(any_train_nan).lower()} |

Old first-order fixed v-strong baselines:

| Eval model | success | mean final p/v/w |
| --- | ---: | --- |
| Euler fixed | {EULER_BASELINE_SUCCESS} | {EULER_BASELINE_PVW[0]} / {EULER_BASELINE_PVW[1]} / {EULER_BASELINE_PVW[2]} |
| L2F fixed | {L2F_BASELINE_SUCCESS} | {L2F_BASELINE_PVW[0]} / {L2F_BASELINE_PVW[1]} / {L2F_BASELINE_PVW[2]} |
"""

report_path = REPORT_DIR / "fixed_vstrong_multiseed_report.md"
report_path.write_text(report, encoding="utf-8")

print(f"Wrote {summary_csv}")
print(f"Wrote {report_path}")
print(f"Verdict: {verdict}")

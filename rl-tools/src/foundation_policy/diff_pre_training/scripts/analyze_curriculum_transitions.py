#!/usr/bin/env python3
"""Analyze curriculum transitions from training CSV logs."""

import argparse
import csv
import os
import sys
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description="Analyze curriculum transitions")
    p.add_argument("--run-root", required=True, help="Run directory")
    p.add_argument("--window-before", type=int, default=100)
    p.add_argument("--window-after", type=int, default=300)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def safe_float(v, default=None):
    try:
        f = float(v)
        if f != f or abs(f) > 1e20:
            return default
        return f
    except (ValueError, TypeError):
        return default


def median(values):
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def analyze_run(run_root, window_before, window_after, output_dir):
    log_dir = os.path.join(run_root, "logs")
    if not os.path.isdir(log_dir):
        print(f"No logs directory: {log_dir}")
        return

    train_files = sorted(f for f in os.listdir(log_dir) if f.endswith("_train.csv"))
    if not train_files:
        print(f"No training CSVs found in {log_dir}")
        return

    out_dir = output_dir or os.path.join(run_root, "reports")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    seed_transitions = defaultdict(list)
    simultaneous_count = 0
    total_transitions = 0

    for fname in train_files:
        seed = fname.replace("_train.csv", "")
        path = os.path.join(log_dir, fname)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        if not data:
            continue

        cols = data[0].keys()
        has_horizon = "current_horizon" in cols
        has_state = "state_curriculum_changed" in cols or "current_state_difficulty" in cols

        prev_horizon = None
        for i, row in enumerate(data):
            step = int(row.get("step", i))
            ch = int(row.get("current_horizon", 0))

            horizon_changed = prev_horizon is not None and ch != prev_horizon
            prev_horizon = ch

            # Detect state curriculum change via flag or difficulty change
            state_changed_flag = row.get("state_curriculum_changed", "false").lower() == "true"
            state_diff = safe_float(row.get("current_state_difficulty", "1"), 1.0)

            # Check if horizon_changed_flag exists
            hc_flag = row.get("horizon_changed_flag", "").lower() == "true"
            sc_flag = row.get("state_curriculum_changed_flag", "").lower() == "true"

            is_simultaneous = (horizon_changed or hc_flag) and (state_changed_flag or sc_flag)
            is_transition = (horizon_changed or hc_flag) or (state_changed_flag or sc_flag)

            if is_transition:
                total_transitions += 1
                if is_simultaneous:
                    simultaneous_count += 1

                before_start = max(0, i - window_before)
                after_end = min(len(data), i + window_after + 1)

                def window_avg(key, rng):
                    vals = [safe_float(data[j].get(key, "")) for j in range(*rng)]
                    vals = [v for v in vals if v is not None]
                    return sum(vals) / len(vals) if vals else float("nan")

                total_loss_before = window_avg("total_loss_mean", (before_start, i))
                total_loss_after = window_avg("total_loss_mean", (i + 1, after_end))

                rows.append({
                    "seed": seed,
                    "transition_step": step,
                    "old_horizon": step > 0 and int(data[i - 1].get("current_horizon", 0)) or 0,
                    "new_horizon": ch,
                    "horizon_changed": "true" if (horizon_changed or hc_flag) else "false",
                    "state_curriculum_changed": "true" if (state_changed_flag or sc_flag) else "false",
                    "steps_since_horizon_change": row.get("steps_since_horizon_change", ""),
                    "steps_since_state_change": row.get("steps_since_state_change", ""),
                    "total_loss_before": total_loss_before,
                    "total_loss_after": total_loss_after,
                    "ratio_total_loss": total_loss_after / total_loss_before if total_loss_before > 0 else float("nan"),
                })
                seed_transitions[seed].append(step)

    # Write CSV
    csv_path = os.path.join(out_dir, "curriculum_transition_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    # Write MD report
    md_path = os.path.join(out_dir, "curriculum_transition_analysis.md")
    with open(md_path, "w") as f:
        f.write("# Curriculum Transition Analysis\n\n")
        f.write(f"Run root: `{run_root}`\n\n")
        f.write(f"Training CSV files analyzed: {len(train_files)}\n\n")
        f.write(f"Total transitions detected: {total_transitions}\n\n")

        if has_horizon and has_state:
            f.write(f"Simultaneous horizon+state transitions: {simultaneous_count} / {total_transitions}\n\n")
            f.write("## Did horizon and state curriculum often change simultaneously?\n\n")
            if total_transitions > 0:
                ratio = simultaneous_count / total_transitions
                if ratio > 0.3:
                    f.write(f"**YES** — {simultaneous_count}/{total_transitions} ({ratio:.1%}) transitions were simultaneous.\n\n")
                elif ratio > 0:
                    f.write(f"Some simultaneous transitions found: {simultaneous_count}/{total_transitions} ({ratio:.1%}).\n\n")
                else:
                    f.write("No simultaneous transitions found.\n\n")
        else:
            f.write("CSV does not contain horizon/state change columns. Cannot detect simultaneous transitions.\n\n")

        f.write("## Per-seed transition counts\n\n")
        for seed in sorted(seed_transitions.keys()):
            f.write(f"- seed {seed}: {len(seed_transitions[seed])} transitions at steps {seed_transitions[seed]}\n")

        if rows:
            ratios = [r["ratio_total_loss"] for r in rows if not (r["ratio_total_loss"] != r["ratio_total_loss"])]
            if ratios:
                spike_count = sum(1 for r in ratios if r > 2.0)
                f.write(f"\nLoss spike (ratio > 2.0) after transitions: {spike_count}/{len(ratios)}\n")

    print(f"Analysis written to {csv_path} and {md_path}")
    print(f"Transitions: {total_transitions}, Simultaneous: {simultaneous_count}")


if __name__ == "__main__":
    args = parse_args()
    analyze_run(args.run_root, args.window_before, args.window_after, args.output_dir)

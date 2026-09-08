"""One minimal DEV performance gate; never reads the FINAL banks."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from response_training import atomic_json, file_hash, source_hash

# Register these before seeing results. They are development continuation
# criteria, not deployment thresholds and not historical V4/V5 gate changes.
MINIMUM_UPDATES = 100
RECENT_DEVELOPMENT_CHECKS = 3
RECENT_TRAINING_UPDATES = 25
MAX_LOSS_RATIO = 10.0
MAX_PERSISTENT_SATURATION = 0.95


def assess(progress):
    history, development = progress.get("history", []), progress.get("development", [])
    if not history or len(development) < 2:
        return {"passed": False, "checks": {"enough_evidence": False}, "actual_updates": progress.get("updates", 0)}
    recent = development[-min(RECENT_DEVELOPMENT_CHECKS, len(development) - 1):]
    base = development[0]
    mean = lambda rows, name: sum(row[name] for row in rows) / len(rows)
    first_train = history[:min(RECENT_TRAINING_UPDATES, len(history))]
    last_train = history[-RECENT_TRAINING_UPDATES:]
    finite = all(row.get("numerics_finite", False) and all(
        math.isfinite(row.get(name, float("nan"))) for name in (
            "task_loss", "gradient_norm", "position_rms", "velocity_rms", "omega_rms",
            "steady_success_rate", "motor_saturation_fraction",
        )
    ) for row in history) and all(row.get("finite", False) for row in development)
    recent_metrics = {key: mean(recent, key) for key in (
        "score", "position_rms", "velocity_rms", "omega_rms",
        "steady_success_rate", "motor_saturation_fraction",
    )}
    checks = {
        "at_least_100_updates": progress.get("updates", 0) >= MINIMUM_UPDATES,
        "numerics_finite": finite and progress.get("status") != "failed",
        "task_loss_not_exploding": (
            recent_metrics["score"] <= MAX_LOSS_RATIO * max(base["score"], 1e-12)
            and mean(last_train, "task_loss") <= MAX_LOSS_RATIO * max(mean(first_train, "task_loss"), 1e-12)
        ),
        "position_improving_on_fixed_dev": recent_metrics["position_rms"] < base["position_rms"],
        "omega_improving_on_fixed_dev": recent_metrics["omega_rms"] < base["omega_rms"],
        "motors_not_persistently_pegged": mean(last_train, "motor_saturation_fraction") < MAX_PERSISTENT_SATURATION,
    }
    return {
        "passed": all(checks.values()), "checks": checks, "actual_updates": progress.get("updates", 0),
        "baseline_development": base, "recent_development": recent_metrics,
        "thresholds": {"minimum_updates": MINIMUM_UPDATES, "max_loss_ratio": MAX_LOSS_RATIO,
                       "max_persistent_saturation": MAX_PERSISTENT_SATURATION,
                       "recent_development_checks": RECENT_DEVELOPMENT_CHECKS},
        "meaning": "permission to continue the same simulation-development run, not proof of adaptation or deployment safety",
        "final_seeds_consumed": [], "deployment_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/response_pooled_v1/seed7/latest.training.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/response_pooled_v1/preflight.json"))
    args = parser.parse_args()
    value = torch.load(args.checkpoint, map_location="cpu")
    if value.get("binding", {}).get("source_sha256") != source_hash():
        raise ValueError("preflight checkpoint belongs to changed source")
    if value["binding"]["optimizer"] != "adam":
        raise ValueError("this preflight gate is for self-rollout training, not MS")
    report = assess(value["progress"])
    report.update(checkpoint=str(args.checkpoint), checkpoint_sha256=file_hash(args.checkpoint), source_sha256=source_hash())
    atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

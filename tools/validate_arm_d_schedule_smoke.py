from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/arm_d_cpu_schedule_smoke_20260806"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the full compressed Arm-D CPU schedule smoke.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    log_path = output_dir / "train.csv"
    checkpoint_path = output_dir / "model.pt"
    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    observed = {
        "segments": len(rows),
        "physical_steps": int(rows[-1]["physical_steps"]),
        "h500_reset_episodes": sum(
            int(row["reset_episode_boundary"])
            and int(row["episode_target_steps"]) == 500
            for row in rows
        ),
        "h1000_reset_episodes": sum(
            int(row["reset_episode_boundary"])
            and int(row["episode_target_steps"]) == 1000
            for row in rows
        ),
        "optimizer_boundaries": sum(int(row["optimization_block_boundary"]) for row in rows),
        "updates_applied": sum(int(row["update_applied"]) for row in rows),
        "tail_supervision_starts": sum(int(row["first_tail_supervision_segment"]) for row in rows),
        "tail_supervision_boundaries": sum(int(row["tail_supervision_block_boundary"]) for row in rows),
        "mid_h1000_optimizer_boundaries": sum(
            int(row["optimization_block_boundary"])
            and int(row["episode_target_steps"]) == 1000
            and not int(row["reset_episode_boundary"])
            for row in rows
        ),
        "invalid_rollouts": sum(not int(row["rollout_valid"]) for row in rows),
        "rejected_or_skipped_boundaries": sum(
            int(row["optimization_block_boundary"]) and not int(row["update_applied"])
            for row in rows
        ),
    }
    expected = {
        "segments": 150,
        "physical_steps": 37500,
        "h500_reset_episodes": 59,
        "h1000_reset_episodes": 8,
        "optimizer_boundaries": 67,
        "updates_applied": 67,
        "tail_supervision_starts": 75,
        "tail_supervision_boundaries": 75,
        "mid_h1000_optimizer_boundaries": 0,
        "invalid_rollouts": 0,
        "rejected_or_skipped_boundaries": 0,
    }
    all_expected = observed == expected
    result = {
        "scope": "CPU batch-1 schedule/numerics smoke; no performance claim",
        "observed": observed,
        "expected": expected,
        "all_expected": all_expected,
        "source_hashes": {
            "train.py": _sha256(ROOT / "train.py"),
            "configs/continuity_cadence_D.args": _sha256(
                ROOT / "configs/continuity_cadence_D.args"
            ),
            "train.csv": _sha256(log_path),
            "model.pt": _sha256(checkpoint_path),
            Path(__file__).resolve().relative_to(ROOT).as_posix(): _sha256(Path(__file__).resolve()),
        },
    }
    (output_dir / "SCHEDULE_SUMMARY.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not all_expected:
        raise RuntimeError("Arm-D schedule smoke does not match the compressed design")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

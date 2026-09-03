from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.phase1_fast_common import atomic_write_dataframe, first_existing_column, read_table
from diagnostics.snapshot_selection import SnapshotSelectionConfig, select_decisive_snapshots


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "scenario_uid": ("scenario_uid", "scenario_id", "uid"),
        "step": ("step", "time_step", "t"),
        "failure_group": ("failure_group", "stage1a_stratum", "outcome_group", "group"),
        "position_norm": ("position_norm", "p_norm", "position_rms", "pos_norm"),
        "omega_norm": ("omega_norm", "angular_rate_norm", "omega_rms", "rate_norm"),
    }
    output = pd.DataFrame({target: frame[first_existing_column(frame, options)] for target, options in aliases.items()})
    clamp = first_existing_column(
        frame, ("integral_clamp_fraction", "clamp_fraction", "integral_clamped", "clamped"), required=False
    )
    output["integral_clamp_fraction"] = frame[clamp].astype(float) if clamp else 0.0
    for optional in (
        "checkpoint", "seed", "policy_group", "dynamic_hard", "high_force", "low_roll", "low_yaw"
    ):
        if optional in frame.columns:
            output[optional] = frame[optional]
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the deterministic Phase 1 JVP/Arnoldi snapshot plan.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-group-scenarios", type=int, default=8)
    parser.add_argument("--snapshots-per-scenario", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1007)
    args = parser.parse_args()
    plan = select_decisive_snapshots(
        normalize(read_table(args.input)),
        config=SnapshotSelectionConfig(
            per_group_scenarios=args.per_group_scenarios,
            snapshots_per_scenario=args.snapshots_per_scenario,
            seed=args.seed,
        ),
    )
    if plan.empty:
        raise RuntimeError("snapshot plan is empty; no output written")
    atomic_write_dataframe(plan, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

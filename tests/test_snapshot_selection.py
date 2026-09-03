from __future__ import annotations

import pandas as pd

from diagnostics.snapshot_selection import SnapshotSelectionConfig, select_decisive_snapshots


def test_snapshot_selection_is_bounded_deterministic_and_checkpoint_safe() -> None:
    records = []
    for checkpoint in ("q2-t0", "q2-t2"):
        for group in ("success", "position-only", "omega-related", "dynamic-hard", "ignored"):
            for scenario in range(6):
                for step in range(1, 101):
                    records.append({
                        "checkpoint": checkpoint,
                        "seed": 7,
                        "scenario_uid": f"shared-{group}-{scenario}",
                        "step": step,
                        "failure_group": group,
                        "position_norm": (scenario + 1) * step / 10000,
                        "omega_norm": (7 - scenario) * (101 - step) / 10000,
                        "integral_clamp_fraction": float(step > 75 and group == "position-only"),
                    })
    frame = pd.DataFrame(records)
    config = SnapshotSelectionConfig(per_group_scenarios=4, snapshots_per_scenario=3, seed=7)
    first = select_decisive_snapshots(frame, config=config)
    second = select_decisive_snapshots(frame, config=config)
    pd.testing.assert_frame_equal(first, second)
    assert "ignored" not in set(first["failure_group"])
    assert first.groupby(["checkpoint", "failure_group"])["scenario_uid"].nunique().max() <= 4
    assert first.groupby(["checkpoint", "scenario_uid"]).size().max() <= 3
    assert first.shape[0] <= 2 * 4 * 4 * 3

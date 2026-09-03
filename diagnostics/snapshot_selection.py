from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SnapshotSelectionConfig:
    per_group_scenarios: int = 8
    snapshots_per_scenario: int = 3
    late_window_steps: int = 500
    minimum_spacing_steps: int = 25
    seed: int = 1007
    allowed_groups: tuple[str, ...] = ("success", "position-only", "omega-related", "dynamic-hard")


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{value}:phase1-snapshot:{seed}".encode()).hexdigest()


def _spaced_order(scores: np.ndarray, steps: np.ndarray, minimum_spacing: int) -> list[int]:
    order = np.argsort(np.nan_to_num(scores, nan=-np.inf), kind="stable")[::-1]
    selected = []
    for index in order:
        if all(abs(int(steps[index]) - int(steps[previous])) >= minimum_spacing for previous in selected):
            selected.append(int(index))
    return selected


def select_decisive_snapshots(
    frame: pd.DataFrame,
    *,
    scenario_column: str = "scenario_uid",
    step_column: str = "step",
    failure_group_column: str = "failure_group",
    position_column: str = "position_norm",
    omega_column: str = "omega_norm",
    clamp_column: str = "integral_clamp_fraction",
    identity_columns: Sequence[str] = ("checkpoint", "seed"),
    config: SnapshotSelectionConfig = SnapshotSelectionConfig(),
) -> pd.DataFrame:
    required = {scenario_column, step_column, failure_group_column, position_column, omega_column}
    if missing := required.difference(frame.columns):
        raise KeyError(f"Missing snapshot selection columns: {sorted(missing)}")
    working = frame.copy()
    if clamp_column not in working.columns:
        working[clamp_column] = 0.0
    identities = [column for column in identity_columns if column in working.columns]
    unit_columns = [*identities, scenario_column]
    working = working[working[failure_group_column].isin(config.allowed_groups)]
    working = working.sort_values([*unit_columns, step_column])
    scenario_table = (
        working.groupby(unit_columns, observed=True, dropna=False)
        .agg(
            failure_group=(failure_group_column, "first"),
            position_score=(position_column, "mean"),
            omega_score=(omega_column, "mean"),
            clamp_score=(clamp_column, "mean"),
        )
        .reset_index()
    )
    selected_units: list[tuple[object, ...]] = []
    outer_groups = [*identities, "failure_group"]
    for _, candidates in scenario_table.groupby(outer_groups, observed=True, dropna=False, sort=True):
        severity = sum(
            candidates[column].rank(pct=True, method="average")
            for column in ("position_score", "omega_score", "clamp_score")
        )
        candidates = candidates.assign(
            selection_rank=severity,
            hash_rank=[_stable_hash(str(uid), config.seed) for uid in candidates[scenario_column]],
        )
        high_count = min(max(1, config.per_group_scenarios // 2), candidates.shape[0])
        high = candidates.nlargest(high_count, "selection_rank")
        remaining = candidates.drop(index=high.index).sort_values("hash_rank")
        chosen = pd.concat((high, remaining.head(config.per_group_scenarios - high_count)))
        selected_units.extend(tuple(row[column] for column in unit_columns) for _, row in chosen.iterrows())

    records: list[dict[str, object]] = []
    indexed = working.set_index(unit_columns, drop=False)
    for unit_key in selected_units:
        key = unit_key if len(unit_columns) > 1 else unit_key[0]
        scenario = indexed.loc[key]
        if isinstance(scenario, pd.Series):
            scenario = scenario.to_frame().T
        max_step = int(scenario[step_column].max())
        late = scenario[scenario[step_column].astype(int) >= max_step - config.late_window_steps + 1]
        steps = late[step_column].to_numpy(dtype=np.int64)
        signals = {
            "late_position_peak": late[position_column].to_numpy(dtype=np.float64),
            "late_omega_peak": late[omega_column].to_numpy(dtype=np.float64),
            "late_clamp_peak": late[clamp_column].to_numpy(dtype=np.float64),
        }
        used_steps: list[int] = []
        for reason, scores in signals.items():
            candidate = next(
                (
                    index for index in _spaced_order(scores, steps, config.minimum_spacing_steps)
                    if all(abs(int(steps[index]) - previous) >= config.minimum_spacing_steps for previous in used_steps)
                ),
                None,
            )
            if candidate is None:
                continue
            row = late.iloc[candidate]
            step = int(row[step_column])
            records.append({
                **{column: row[column] for column in identities},
                "scenario_uid": str(row[scenario_column]),
                "step": step,
                "selection_reason": reason,
                "failure_group": row[failure_group_column],
                "position_norm": float(row[position_column]),
                "omega_norm": float(row[omega_column]),
                "integral_clamp_fraction": float(row[clamp_column]),
                **{
                    column: row[column]
                    for column in ("policy_group", "dynamic_hard", "high_force", "low_roll", "low_yaw")
                    if column in row.index
                },
            })
            used_steps.append(step)
            if len(used_steps) >= config.snapshots_per_scenario:
                break
    result = pd.DataFrame(records)
    if result.empty:
        return result
    return result.sort_values([*identities, "failure_group", "scenario_uid", "step"]).reset_index(drop=True)

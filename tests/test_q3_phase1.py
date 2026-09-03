from __future__ import annotations

import csv
from dataclasses import fields

import numpy as np
import torch

from diagnostics.scenarios import formal_scenario_uid, load_matlab_scenarios, scenario_group_masks
from env_l2f import L2FState


def _row(index: int) -> dict[str, float]:
    row: dict[str, float] = {"scenario_id": index}
    for prefix, width in (("position", 3), ("velocity", 3), ("omega", 3),
                          ("motor", 4), ("previous_action", 4),
                          ("thrust_coeff_c0", 4), ("thrust_coeff_c1", 4),
                          ("thrust_coeff_c2", 4)):
        for axis in range(width):
            row[f"{prefix}_{axis}"] = 0.0
    for r in range(3):
        for c in range(3):
            row[f"rotation_{r}{c}"] = float(r == c)
    for axis in range(3):
        row[f"external_force_{axis}"] = float(index + axis)
    for name in ("mass", "thrust_to_weight", "torque_to_inertia", "rotor_distance_factor",
                 "inertia_factor", "motor_time_rising", "motor_time_falling",
                 "rotor_torque_constant", "cbrt_mass", "force_std", "arm_length",
                 "inertia_x", "inertia_y", "inertia_z", "alpha_roll_max",
                 "alpha_pitch_max", "alpha_yaw_max", "eta_yaw", "jz_over_jxy",
                 "dt_alpha_roll_max", "dt_alpha_yaw_max"):
        row[name] = float(index)
    return row


def test_phase1_scenario_loader_and_groups(tmp_path) -> None:
    rows = [_row(i) for i in range(1, 11)]
    path = tmp_path / "scenarios.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ids, state = load_matlab_scenarios(path)
    assert ids == list(range(1, 11))
    assert isinstance(state, L2FState)
    assert {item.name for item in fields(state)} == {item.name for item in fields(L2FState)}
    assert state.rotation.dtype == torch.float64
    masks = scenario_group_masks(rows)
    assert all(mask.shape == (10,) and mask.dtype == np.bool_ for mask in masks.values())
    assert masks["high_force"].sum() == 2
    assert formal_scenario_uid(1007, 1) == formal_scenario_uid(1007, 1)
    assert formal_scenario_uid(1007, 1) != formal_scenario_uid(1007, 2)

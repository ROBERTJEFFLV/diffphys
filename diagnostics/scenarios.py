from __future__ import annotations

import csv
import hashlib
from dataclasses import fields
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from env_l2f import L2FState


DYNAMIC_HARD_IDS = frozenset(
    (53, 69, 97, 147, 161, 237, 351, 355, 396, 467, 514, 569,
     608, 636, 650, 677, 682, 732, 768, 834, 864, 904)
)


def formal_scenario_uid(eval_seed: int, scenario_id: int) -> str:
    payload = f"matlab-physical-broad:{int(eval_seed)}:{int(scenario_id)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_scenario_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _columns(rows: list[dict[str, str]], names: Iterable[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)


def _column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def load_matlab_scenarios(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[list[int], L2FState]:
    """Load exact states exported by ``export_q3_phase1_scenarios``."""

    rows = read_scenario_rows(path)
    if not rows:
        raise ValueError("scenario CSV is empty")
    ids = [int(float(row["scenario_id"])) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario IDs must be unique")

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, device=device, dtype=dtype)

    rotation = _columns(rows, ("rotation_00", "rotation_01", "rotation_02",
                               "rotation_10", "rotation_11", "rotation_12",
                               "rotation_20", "rotation_21", "rotation_22")).reshape(-1, 3, 3)
    values: dict[str, torch.Tensor] = {
        "position": tensor(_columns(rows, ("position_0", "position_1", "position_2"))),
        "velocity": tensor(_columns(rows, ("velocity_0", "velocity_1", "velocity_2"))),
        "rotation": tensor(rotation),
        "omega": tensor(_columns(rows, ("omega_0", "omega_1", "omega_2"))),
        "motor": tensor(_columns(rows, tuple(f"motor_{i}" for i in range(4)))),
        "previous_action": tensor(_columns(rows, tuple(f"previous_action_{i}" for i in range(4)))),
        "external_force": tensor(_columns(rows, ("external_force_0", "external_force_1", "external_force_2"))),
        "thrust_coeff_c0": tensor(_columns(rows, tuple(f"thrust_coeff_c0_{i}" for i in range(4)))),
        "thrust_coeff_c1": tensor(_columns(rows, tuple(f"thrust_coeff_c1_{i}" for i in range(4)))),
        "thrust_coeff_c2": tensor(_columns(rows, tuple(f"thrust_coeff_c2_{i}" for i in range(4)))),
    }
    scalar_map = {
        "mass": "mass",
        "thrust_to_weight": "thrust_to_weight",
        "torque_to_inertia": "torque_to_inertia",
        "rotor_distance_factor": "rotor_distance_factor",
        "inertia_factor": "inertia_factor",
        "motor_time_rising": "motor_time_rising",
        "motor_time_falling": "motor_time_falling",
        "rotor_torque_constant": "rotor_torque_constant",
        "cbrt_mass": "cbrt_mass",
        "force_std": "force_std",
        "arm_length": "arm_length",
        "inertia_x": "inertia_x",
        "inertia_y": "inertia_y",
        "inertia_z": "inertia_z",
        "alpha_roll_max": "alpha_roll_max",
        "alpha_pitch_max": "alpha_pitch_max",
        "alpha_yaw_max": "alpha_yaw_max",
        "eta_yaw": "eta_yaw",
        "jz_over_jxy": "jz_over_jxy",
        "dt_alpha_roll_max": "dt_alpha_roll_max",
        "dt_alpha_yaw_max": "dt_alpha_yaw_max",
    }
    values.update({field: tensor(_column(rows, column)) for field, column in scalar_map.items()})
    expected = {field.name for field in fields(L2FState)}
    if values.keys() != expected:
        raise AssertionError(f"state schema mismatch: missing={expected-values.keys()}, extra={values.keys()-expected}")
    return ids, L2FState(**values)


def scenario_group_masks(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    """Reproduce the formal report's fixed 20% group definitions."""

    force = np.linalg.norm(_columns(rows, ("external_force_0", "external_force_1", "external_force_2")), axis=1)
    roll = _column(rows, "alpha_roll_max")
    yaw = _column(rows, "alpha_yaw_max")
    rise = _column(rows, "motor_time_rising")
    fall = _column(rows, "motor_time_falling")
    ids = np.asarray([int(float(row["scenario_id"])) for row in rows])
    return {
        "dynamic_hard": np.isin(ids, tuple(DYNAMIC_HARD_IDS)),
        "high_force": force >= np.quantile(force, 0.80),
        "low_roll": roll <= np.quantile(roll, 0.20),
        "low_yaw": yaw <= np.quantile(yaw, 0.20),
        "slow_motor_rise": rise >= np.quantile(rise, 0.80),
        "slow_motor_fall": fall >= np.quantile(fall, 0.80),
        "low_thrust_to_weight": _column(rows, "thrust_to_weight") <= np.quantile(_column(rows, "thrust_to_weight"), 0.20),
    }


def validate_against_matlab_samples(
    scenario_rows: list[dict[str, str]],
    sample_rows: list[dict[str, str]],
) -> dict[str, float]:
    """Compare reset parameters against the existing formal MATLAB output."""

    if len(scenario_rows) != len(sample_rows):
        raise ValueError("scenario/sample row counts differ")
    mappings = {
        "mass": "mass_kg",
        "cbrt_mass": "cbrt_mass",
        "arm_length": "arm_length_m",
        "thrust_to_weight": "thrust_to_weight",
        "torque_to_inertia": "torque_to_inertia",
        "motor_time_rising": "motor_time_rising_s",
        "motor_time_falling": "motor_time_falling_s",
        "force_std": "force_std",
        "inertia_x": "inertia_x",
        "inertia_y": "inertia_y",
        "inertia_z": "inertia_z",
        "alpha_roll_max": "alpha_roll_max",
        "alpha_yaw_max": "alpha_yaw_max",
        "eta_yaw": "eta_yaw",
        "jz_over_jxy": "jz_over_jxy",
        "external_force_0": "external_force_x",
        "external_force_1": "external_force_y",
        "external_force_2": "external_force_z",
    }
    errors: dict[str, float] = {}
    for source, target in mappings.items():
        lhs = np.asarray([float(row[source]) for row in scenario_rows])
        rhs = np.asarray([float(row[target]) for row in sample_rows])
        errors[source] = float(np.max(np.abs(lhs - rhs)))
    return errors

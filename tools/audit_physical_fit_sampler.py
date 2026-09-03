from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_l2f import (  # noqa: E402
    L2FParams,
    L2FRaptorEpisodeDynamics,
    _joint_stratified_units,
    sample_physical_fit_episode_dynamics,
)


DEFAULT_OUTPUT = ROOT / "reports/physical_fit_sampler_audit_20260806"
DEFAULT_SEEDS = (7, 17, 27, 1007)
GRID_BINS = 4
GRID_DIMENSIONS = 4
GRID_CELLS = GRID_BINS**GRID_DIMENSIONS

METRICS = (
    ("mass_kg", "kg"),
    ("arm_length_m", "m"),
    ("motor_span_m", "m"),
    ("thrust_to_weight", "ratio"),
    ("minimum_thrust_to_weight", "ratio"),
    ("maximum_thrust_to_weight_reconstructed", "ratio"),
    ("required_static_thrust_to_weight", "ratio"),
    ("thrust_coeff_c0_per_rotor_N", "N"),
    ("thrust_coeff_c1_per_rotor_N", "N"),
    ("thrust_coeff_c2_abs_max_N", "N"),
    ("inertia_x", "kg*m^2"),
    ("inertia_y", "kg*m^2"),
    ("inertia_z", "kg*m^2"),
    ("kxy_realized", "ratio"),
    ("jz_over_jxy", "ratio"),
    ("inertia_z_margin", "kg*m^2"),
    ("alpha_roll_max", "rad/s^2"),
    ("alpha_pitch_max", "rad/s^2"),
    ("alpha_yaw_max", "rad/s^2"),
    ("eta_yaw", "ratio"),
    ("rotor_torque_constant", "m"),
    ("motor_time_rising_s", "s"),
    ("motor_time_falling_s", "s"),
    ("force_std_N", "N"),
    ("disturbance_acc_std_mps2", "m/s^2"),
    ("external_acc_norm_mps2", "m/s^2"),
)

SAMPLE_FLAGS = (
    "finite_all_returned_fields",
    "positive_critical_fields",
    "nonnegative_force_std",
    "inertia_triangle_valid",
    "fall_time_not_faster_than_rise",
    "thrust_curve_reconstruction_valid",
    "derived_capability_reconstruction_valid",
    "static_translation_trim_feasible",
    "identical_per_motor_thrust_curves",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _close(value: torch.Tensor, target: float) -> torch.Tensor:
    tolerance = max(abs(target) * 1.0e-6, 1.0e-8)
    return torch.abs(value - target) <= tolerance


def _tensor_close(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.isclose(value, target, rtol=5.0e-5, atol=1.0e-7)


def _per_sample_all(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1).all(dim=-1)


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _reconstruction_flags(
    dynamics: L2FRaptorEpisodeDynamics,
    nominal: L2FParams,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    mass = dynamics.mass
    gravity = float(nominal.gravity)
    c0 = dynamics.thrust_coeff_c0
    c1 = dynamics.thrust_coeff_c1
    c2 = dynamics.thrust_coeff_c2
    minimum_thrust = (c0 - c1 + c2).clamp_min(0.0)
    maximum_thrust = (c0 + c1 + c2).clamp_min(0.0)
    thrust_delta = maximum_thrust[:, 0] - minimum_thrust[:, 0]

    expected_c0 = (mass * gravity / 4.0)[:, None].expand_as(c0)
    expected_c1 = ((dynamics.thrust_to_weight - 1.0) * mass * gravity / 4.0)[
        :, None
    ].expand_as(c1)
    reconstructed_tw = maximum_thrust.sum(dim=-1) / (mass * gravity)
    thrust_curve_valid = (
        _per_sample_all(_tensor_close(c0, expected_c0))
        & _per_sample_all(_tensor_close(c1, expected_c1))
        & _per_sample_all(_close(c2, 0.0))
        & _tensor_close(reconstructed_tw, dynamics.thrust_to_weight)
    )

    reconstructed_roll = dynamics.arm_length * thrust_delta / dynamics.inertia_x
    reconstructed_pitch = dynamics.arm_length * thrust_delta / dynamics.inertia_y
    reconstructed_yaw = (
        dynamics.rotor_torque_constant * (2.0 * thrust_delta) / dynamics.inertia_z
    )
    reconstructed_eta = reconstructed_yaw / reconstructed_roll
    reconstructed_jz_ratio = dynamics.inertia_z / dynamics.inertia_x
    derived_valid = (
        _tensor_close(dynamics.cbrt_mass, mass.pow(1.0 / 3.0))
        & _tensor_close(
            dynamics.rotor_distance_factor,
            dynamics.arm_length / float(nominal.arm_length),
        )
        & _tensor_close(
            dynamics.inertia_factor,
            torch.full_like(mass, float(nominal.inertia_x)) / dynamics.inertia_x,
        )
        & _tensor_close(dynamics.torque_to_inertia, dynamics.alpha_roll_max)
        & _tensor_close(reconstructed_roll, dynamics.alpha_roll_max)
        & _tensor_close(reconstructed_pitch, dynamics.alpha_pitch_max)
        & _tensor_close(reconstructed_yaw, dynamics.alpha_yaw_max)
        & _tensor_close(reconstructed_eta, dynamics.eta_yaw)
        & _tensor_close(reconstructed_jz_ratio, dynamics.jz_over_jxy)
        & _tensor_close(
            dynamics.dt_alpha_roll_max,
            dynamics.alpha_roll_max * float(nominal.dt),
        )
        & _tensor_close(
            dynamics.dt_alpha_yaw_max,
            dynamics.alpha_yaw_max * float(nominal.dt),
        )
    )
    reconstructed = {
        "minimum_thrust_to_weight": minimum_thrust.sum(dim=-1) / (mass * gravity),
        "maximum_thrust_to_weight_reconstructed": reconstructed_tw,
    }
    return thrust_curve_valid, derived_valid, reconstructed


def _sample_rows(
    *,
    seeds: tuple[int, ...],
    samples_per_seed: int,
) -> tuple[list[dict[str, object]], dict[str, torch.Tensor]]:
    nominal = L2FParams()
    rows: list[dict[str, object]] = []
    columns: dict[str, list[torch.Tensor]] = {name: [] for name, _ in METRICS}
    for name in SAMPLE_FLAGS:
        columns[name] = []

    for seed in seeds:
        torch.manual_seed(seed)
        dynamics = sample_physical_fit_episode_dynamics(
            nominal,
            samples_per_seed,
            device=torch.device("cpu"),
            dtype=torch.float32,
            balanced=True,
        )
        mass = dynamics.mass
        arm = dynamics.arm_length
        inertia_x = dynamics.inertia_x
        inertia_y = dynamics.inertia_y
        inertia_z = dynamics.inertia_z
        thrust_curve_valid, derived_valid, reconstructed = _reconstruction_flags(
            dynamics, nominal
        )

        finite_all = torch.ones(samples_per_seed, dtype=torch.bool)
        for field in fields(dynamics):
            value = getattr(dynamics, field.name)
            finite_all &= _per_sample_all(torch.isfinite(value))

        critical_positive_names = (
            "mass",
            "thrust_coeff_c0",
            "thrust_coeff_c1",
            "thrust_to_weight",
            "torque_to_inertia",
            "rotor_distance_factor",
            "inertia_factor",
            "motor_time_rising",
            "motor_time_falling",
            "rotor_torque_constant",
            "cbrt_mass",
            "arm_length",
            "inertia_x",
            "inertia_y",
            "inertia_z",
            "alpha_roll_max",
            "alpha_pitch_max",
            "alpha_yaw_max",
            "eta_yaw",
            "jz_over_jxy",
            "dt_alpha_roll_max",
            "dt_alpha_yaw_max",
        )
        positive_critical = torch.ones(samples_per_seed, dtype=torch.bool)
        for name in critical_positive_names:
            positive_critical &= _per_sample_all(getattr(dynamics, name) > 0.0)

        inertia_triangle_valid = (
            (inertia_x <= inertia_y + inertia_z)
            & (inertia_y <= inertia_x + inertia_z)
            & (inertia_z <= inertia_x + inertia_y)
        )
        required_thrust = torch.linalg.vector_norm(
            torch.tensor(
                (0.0, 0.0, float(nominal.gravity)),
                dtype=mass.dtype,
            )[None, :]
            - dynamics.external_force / mass[:, None],
            dim=-1,
        )
        required_ratio = required_thrust / float(nominal.gravity)
        trim_tolerance = 5.0e-6
        static_trim = (
            required_ratio + trim_tolerance >= reconstructed["minimum_thrust_to_weight"]
        ) & (
            required_ratio - trim_tolerance
            <= reconstructed["maximum_thrust_to_weight_reconstructed"]
        )

        metric_tensors = {
            "mass_kg": mass,
            "arm_length_m": arm,
            "motor_span_m": 2.0 * arm,
            "thrust_to_weight": dynamics.thrust_to_weight,
            **reconstructed,
            "required_static_thrust_to_weight": required_ratio,
            "thrust_coeff_c0_per_rotor_N": dynamics.thrust_coeff_c0[:, 0],
            "thrust_coeff_c1_per_rotor_N": dynamics.thrust_coeff_c1[:, 0],
            "thrust_coeff_c2_abs_max_N": dynamics.thrust_coeff_c2.abs().amax(dim=-1),
            "inertia_x": inertia_x,
            "inertia_y": inertia_y,
            "inertia_z": inertia_z,
            "kxy_realized": inertia_x / torch.clamp(mass * arm * arm, min=1.0e-12),
            "jz_over_jxy": dynamics.jz_over_jxy,
            "inertia_z_margin": inertia_x + inertia_y - inertia_z,
            "alpha_roll_max": dynamics.alpha_roll_max,
            "alpha_pitch_max": dynamics.alpha_pitch_max,
            "alpha_yaw_max": dynamics.alpha_yaw_max,
            "eta_yaw": dynamics.eta_yaw,
            "rotor_torque_constant": dynamics.rotor_torque_constant,
            "motor_time_rising_s": dynamics.motor_time_rising,
            "motor_time_falling_s": dynamics.motor_time_falling,
            "force_std_N": dynamics.force_std,
            "disturbance_acc_std_mps2": dynamics.force_std / mass,
            "external_acc_norm_mps2": torch.linalg.vector_norm(
                dynamics.external_force / mass[:, None], dim=-1
            ),
        }
        sample_flags = {
            "finite_all_returned_fields": finite_all,
            "positive_critical_fields": positive_critical,
            "nonnegative_force_std": dynamics.force_std >= 0.0,
            "inertia_triangle_valid": inertia_triangle_valid,
            "fall_time_not_faster_than_rise": (
                dynamics.motor_time_falling >= dynamics.motor_time_rising
            ),
            "thrust_curve_reconstruction_valid": thrust_curve_valid,
            "derived_capability_reconstruction_valid": derived_valid,
            "static_translation_trim_feasible": static_trim,
            "identical_per_motor_thrust_curves": (
                _per_sample_all(_tensor_close(dynamics.thrust_coeff_c0, dynamics.thrust_coeff_c0[:, :1]))
                & _per_sample_all(_tensor_close(dynamics.thrust_coeff_c1, dynamics.thrust_coeff_c1[:, :1]))
                & _per_sample_all(_tensor_close(dynamics.thrust_coeff_c2, dynamics.thrust_coeff_c2[:, :1]))
            ),
        }
        for name, tensor in {**metric_tensors, **sample_flags}.items():
            columns[name].append(tensor.detach().cpu())

        values = {
            name: tensor.detach().cpu().tolist()
            for name, tensor in {**metric_tensors, **sample_flags}.items()
        }
        for index in range(samples_per_seed):
            row: dict[str, object] = {"seed": seed, "sample_index": index}
            row.update({name: values[name][index] for name, _ in METRICS})
            row.update({name: int(values[name][index]) for name in SAMPLE_FLAGS})
            rows.append(row)

    merged = {name: torch.cat(parts) for name, parts in columns.items()}
    return rows, merged


def _summary_rows(columns: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    quantiles = torch.tensor(
        (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99), dtype=torch.float64
    )
    rows: list[dict[str, object]] = []
    for name, unit in METRICS:
        values = columns[name].to(dtype=torch.float64)
        q = torch.quantile(values, quantiles).tolist()
        rows.append(
            {
                "metric": name,
                "unit": unit,
                "count": values.numel(),
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=True).item()),
                "min": float(values.min().item()),
                "p01": q[0],
                "p05": q[1],
                "p25": q[2],
                "p50": q[3],
                "p75": q[4],
                "p95": q[5],
                "p99": q[6],
                "max": float(values.max().item()),
            }
        )
    return rows


def _balanced_grid_row(
    *, seeds: tuple[int, ...], samples_per_seed: int
) -> dict[str, object]:
    expected_per_cell = samples_per_seed // GRID_CELLS
    failing_seeds = 0
    for seed in seeds:
        torch.manual_seed(seed)
        units = _joint_stratified_units(
            samples_per_seed,
            torch.device("cpu"),
            torch.float32,
            bins=GRID_BINS,
            dimensions=GRID_DIMENSIONS,
        )
        cell_id = torch.zeros(samples_per_seed, dtype=torch.long)
        stride = 1
        for unit in units:
            bin_id = torch.floor(unit * GRID_BINS).to(dtype=torch.long).clamp(0, GRID_BINS - 1)
            cell_id += stride * bin_id
            stride *= GRID_BINS
        counts = torch.bincount(cell_id, minlength=GRID_CELLS)
        if not bool((counts == expected_per_cell).all().item()):
            failing_seeds += 1
    return {
        "check": "balanced_roots_cover_all_4pow4_joint_cells",
        "kind": "sampling_design_constraint",
        "requirement": (
            f"each of {GRID_CELLS} cells appears {expected_per_cell} times per seed"
        ),
        "sample_count": len(seeds),
        "violations": failing_seeds,
        "violation_fraction": failing_seeds / len(seeds),
        "passed": int(failing_seeds == 0),
    }


def _constraint_rows(
    columns: dict[str, torch.Tensor],
    *,
    seeds: tuple[int, ...],
    samples_per_seed: int,
) -> list[dict[str, object]]:
    count = int(columns["mass_kg"].numel())
    tolerance = 1.0e-6
    disturbance_upper = 0.35 * (columns["thrust_to_weight"] - 1.0)
    checks = (
        (
            "all_returned_dynamics_fields_finite",
            ~columns["finite_all_returned_fields"],
            "every tensor field of L2FRaptorEpisodeDynamics is finite",
        ),
        (
            "critical_physical_fields_strictly_positive",
            ~columns["positive_critical_fields"],
            "mass, curve gains, inertias, time constants and derived capabilities > 0",
        ),
        (
            "force_std_nonnegative",
            ~columns["nonnegative_force_std"],
            "force_std >= 0",
        ),
        (
            "principal_inertia_triangle",
            ~columns["inertia_triangle_valid"],
            "each principal inertia <= sum of the other two",
        ),
        (
            "mass_in_0p02_5p00",
            (columns["mass_kg"] < 0.02 - tolerance)
            | (columns["mass_kg"] > 5.00 + tolerance),
            "0.02 <= mass <= 5.00 kg",
        ),
        (
            "arm_length_in_0p028_0p50",
            (columns["arm_length_m"] < 0.028 - tolerance)
            | (columns["arm_length_m"] > 0.50 + tolerance),
            "0.028 <= arm length <= 0.50 m",
        ),
        (
            "thrust_to_weight_in_1p45_5p50",
            (columns["thrust_to_weight"] < 1.45 - tolerance)
            | (columns["thrust_to_weight"] > 5.50 + tolerance),
            "1.45 <= T/W <= 5.50",
        ),
        (
            "jz_over_jxy_in_1p45_1p95",
            (columns["jz_over_jxy"] < 1.45 - tolerance)
            | (columns["jz_over_jxy"] > 1.95 + tolerance),
            "1.45 <= Jz/Jxy <= 1.95",
        ),
        (
            "realized_inertia_coefficient_in_0p045_0p50",
            (columns["kxy_realized"] < 0.045 - tolerance)
            | (columns["kxy_realized"] > 0.50 + tolerance),
            "0.045 <= Jx/(mass*arm^2) <= 0.50 after authority rescaling",
        ),
        (
            "roll_authority_in_35_2200",
            (columns["alpha_roll_max"] < 35.0 - tolerance)
            | (columns["alpha_roll_max"] > 2200.0 + tolerance),
            "35 <= alpha_roll <= 2200 rad/s^2",
        ),
        (
            "pitch_authority_in_35_2200",
            (columns["alpha_pitch_max"] < 35.0 - tolerance)
            | (columns["alpha_pitch_max"] > 2200.0 + tolerance),
            "35 <= alpha_pitch <= 2200 rad/s^2",
        ),
        (
            "yaw_to_roll_authority_in_0p02_1p00",
            (columns["eta_yaw"] < 0.02 - tolerance)
            | (columns["eta_yaw"] > 1.00 + tolerance),
            "0.02 <= alpha_yaw/alpha_roll <= 1.00",
        ),
        (
            "realized_rotor_torque_constant_in_0p006_0p035",
            (columns["rotor_torque_constant"] < 0.006 - tolerance)
            | (columns["rotor_torque_constant"] > 0.035 + tolerance),
            "0.006 <= realized rotor torque constant <= 0.035 m",
        ),
        (
            "rise_time_in_0p025_0p18",
            (columns["motor_time_rising_s"] < 0.025 - tolerance)
            | (columns["motor_time_rising_s"] > 0.18 + tolerance),
            "0.025 <= tau_rise <= 0.18 s",
        ),
        (
            "fall_time_in_0p03_0p35",
            (columns["motor_time_falling_s"] < 0.03 - tolerance)
            | (columns["motor_time_falling_s"] > 0.35 + tolerance),
            "0.03 <= tau_fall <= 0.35 s",
        ),
        (
            "fall_time_not_faster_than_rise",
            ~columns["fall_time_not_faster_than_rise"],
            "tau_fall >= tau_rise",
        ),
        (
            "fall_to_rise_ratio_not_above_2p6",
            columns["motor_time_falling_s"]
            > 2.6 * columns["motor_time_rising_s"] + tolerance,
            "tau_fall <= 2.6*tau_rise after clipping",
        ),
        (
            "thrust_curve_matches_declared_thrust_to_weight",
            ~columns["thrust_curve_reconstruction_valid"],
            "c0, c1, c2 and reconstructed max thrust match declared T/W",
        ),
        (
            "derived_capabilities_reconstruct_from_primitive_fields",
            ~columns["derived_capability_reconstruction_valid"],
            "mass/arm/inertia/torque fields reconstruct all reported capabilities",
        ),
        (
            "disturbance_std_within_declared_surplus_scaled_bound",
            (columns["disturbance_acc_std_mps2"] < -tolerance)
            | (columns["disturbance_acc_std_mps2"] > disturbance_upper + tolerance),
            "0 <= force_std/m <= 0.35*(T/W-1) m/s^2",
        ),
        (
            "static_translation_thrust_magnitude_feasible",
            ~columns["static_translation_trim_feasible"],
            "required constant-force balance lies between min and max total thrust",
        ),
    )
    rows: list[dict[str, object]] = []
    for name, violation_mask, requirement in checks:
        violations = int(violation_mask.sum().item())
        rows.append(
            {
                "check": name,
                "kind": "checked_necessary_or_source_consistency_constraint",
                "requirement": requirement,
                "sample_count": count,
                "violations": violations,
                "violation_fraction": violations / count,
                "passed": int(violations == 0),
            }
        )
    rows.append(_balanced_grid_row(seeds=seeds, samples_per_seed=samples_per_seed))
    return rows


def _coverage_rows(columns: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    count = int(columns["mass_kg"].numel())
    diagnostics = (
        ("tau_rise_le_35ms", columns["motor_time_rising_s"] <= 0.035, "fast-motor coverage"),
        ("mass_at_0p02_lower_bound", _close(columns["mass_kg"], 0.02), "boundary pileup"),
        ("mass_at_5p00_upper_bound", _close(columns["mass_kg"], 5.00), "boundary pileup"),
        ("arm_at_0p028_lower_cap", _close(columns["arm_length_m"], 0.028), "cap pileup"),
        ("arm_at_0p50_upper_cap", _close(columns["arm_length_m"], 0.50), "cap pileup"),
        ("thrust_to_weight_at_1p45_lower_cap", _close(columns["thrust_to_weight"], 1.45), "cap pileup"),
        ("thrust_to_weight_at_5p50_upper_cap", _close(columns["thrust_to_weight"], 5.50), "cap pileup"),
        ("alpha_roll_at_35_lower_cap", _close(columns["alpha_roll_max"], 35.0), "cap pileup"),
        ("alpha_roll_at_2200_upper_cap", _close(columns["alpha_roll_max"], 2200.0), "cap pileup"),
        ("eta_yaw_at_0p02_lower_cap", _close(columns["eta_yaw"], 0.02), "cap pileup"),
        ("eta_yaw_at_1_upper_cap", _close(columns["eta_yaw"], 1.0), "cap pileup"),
        ("tau_rise_at_25ms_lower_cap", _close(columns["motor_time_rising_s"], 0.025), "cap pileup"),
        ("tau_rise_at_180ms_upper_cap", _close(columns["motor_time_rising_s"], 0.18), "cap pileup"),
        ("tau_fall_at_30ms_lower_cap", _close(columns["motor_time_falling_s"], 0.03), "cap pileup"),
        ("tau_fall_at_350ms_upper_cap", _close(columns["motor_time_falling_s"], 0.35), "cap pileup"),
        ("jz_ratio_at_1p45_lower_bound", _close(columns["jz_over_jxy"], 1.45), "boundary pileup"),
        ("jz_ratio_at_1p95_upper_bound", _close(columns["jz_over_jxy"], 1.95), "boundary pileup"),
        ("kxy_realized_at_0p045_lower_bound", _close(columns["kxy_realized"], 0.045), "boundary pileup"),
        ("kxy_realized_at_0p50_upper_bound", _close(columns["kxy_realized"], 0.50), "boundary pileup"),
        ("rotor_torque_constant_at_0p006_lower_bound", _close(columns["rotor_torque_constant"], 0.006), "boundary pileup"),
        ("rotor_torque_constant_at_0p035_upper_bound", _close(columns["rotor_torque_constant"], 0.035), "boundary pileup"),
        ("jx_equals_jy", _close(columns["inertia_x"] - columns["inertia_y"], 0.0), "model assumption"),
        ("quadratic_thrust_coefficient_is_zero", _close(columns["thrust_coeff_c2_abs_max_N"], 0.0), "model assumption"),
        ("all_four_motor_thrust_curves_identical", columns["identical_per_motor_thrust_curves"], "model assumption"),
        ("zero_motor_command_is_nominal_hover_without_external_force", _close(columns["thrust_coeff_c0_per_rotor_N"] * 4.0 / (columns["mass_kg"] * L2FParams().gravity), 1.0), "model assumption"),
    )
    return [
        {
            "diagnostic": name,
            "kind": kind,
            "sample_count": count,
            "matched_count": int(mask.sum().item()),
            "matched_fraction": float(mask.float().mean().item()),
        }
        for name, mask, kind in diagnostics
    ]


def _summary_markdown(
    *,
    total_samples: int,
    constraints: list[dict[str, object]],
    coverage: list[dict[str, object]],
    seeds: tuple[int, ...],
) -> str:
    failed = [str(row["check"]) for row in constraints if int(row["passed"]) != 1]
    status = "all passed" if not failed else "failed: " + ", ".join(failed)
    coverage_by_name = {str(row["diagnostic"]): row for row in coverage}

    def fraction(name: str) -> float:
        return float(coverage_by_name[name]["matched_fraction"])

    return f"""# Physical-fit sampler audit

This report is a source-consistency and necessary-constraint audit of the current
`physical-fit` sampler. It sampled {total_samples:,} CPU float32 cases across seeds
{', '.join(str(seed) for seed in seeds)}. The explicitly checked gates are **{status}**.

Passing these gates means the sampled tensors were finite, the listed source
relationships reconstructed, principal inertias obeyed the rigid-body triangle
inequalities, motor fall time was not shorter than rise time, the declared bounds
held, the 4^4 balanced root grid was covered, and a constant-force translational
thrust-magnitude balance existed. It does **not** establish that the empirical
parameter distribution represents real aircraft, that an attitude/torque trim is
reachable, or that trajectories remain controllable.

## Coverage diagnostics

- Fast motors (`tau_rise <= 35 ms`): {fraction('tau_rise_le_35ms'):.3%}.
- Arm lower/upper cap pileups: {fraction('arm_at_0p028_lower_cap'):.3%} / {fraction('arm_at_0p50_upper_cap'):.3%}.
- Roll-authority lower/upper cap pileups: {fraction('alpha_roll_at_35_lower_cap'):.3%} / {fraction('alpha_roll_at_2200_upper_cap'):.3%}.
- Yaw-ratio lower/upper cap pileups: {fraction('eta_yaw_at_0p02_lower_cap'):.3%} / {fraction('eta_yaw_at_1_upper_cap'):.3%}.
- Rise-time lower/upper cap pileups: {fraction('tau_rise_at_25ms_lower_cap'):.3%} / {fraction('tau_rise_at_180ms_upper_cap'):.3%}.
- Fall-time lower/upper cap pileups: {fraction('tau_fall_at_30ms_lower_cap'):.3%} / {fraction('tau_fall_at_350ms_upper_cap'):.3%}.

`Jx == Jy`, a zero quadratic thrust coefficient, identical per-motor thrust
curves, and zero-command nominal hover are construction assumptions here, not
validated facts about a cross-size fleet. See `COVERAGE_DIAGNOSTICS.csv` and
`DISTRIBUTION_SUMMARY.csv` for the measured rates and tails.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the current physical-fit sampler.")
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--samples-per-seed", type=int, default=16384)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-raw-samples", action="store_true")
    args = parser.parse_args()
    if args.samples_per_seed <= 0:
        raise ValueError("--samples-per-seed must be positive")
    if args.samples_per_seed % GRID_CELLS != 0:
        raise ValueError(
            f"--samples-per-seed must be divisible by {GRID_CELLS} so the joint balanced grid can be audited"
        )

    seeds = tuple(args.seeds)
    output_dir = args.output_dir.resolve()
    rows, columns = _sample_rows(seeds=seeds, samples_per_seed=args.samples_per_seed)
    summary = _summary_rows(columns)
    constraints = _constraint_rows(
        columns, seeds=seeds, samples_per_seed=args.samples_per_seed
    )
    coverage = _coverage_rows(columns)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_raw_samples:
        _write_csv(
            output_dir / "SAMPLES.csv",
            rows,
            ("seed", "sample_index", *(name for name, _ in METRICS), *SAMPLE_FLAGS),
        )
    _write_csv(output_dir / "DISTRIBUTION_SUMMARY.csv", summary, tuple(summary[0]))
    _write_csv(output_dir / "HARD_CONSTRAINTS.csv", constraints, tuple(constraints[0]))
    _write_csv(output_dir / "COVERAGE_DIAGNOSTICS.csv", coverage, tuple(coverage[0]))
    (output_dir / "SUMMARY.md").write_text(
        _summary_markdown(
            total_samples=len(rows), constraints=constraints, coverage=coverage, seeds=seeds
        ),
        encoding="utf-8",
    )

    source_files = (ROOT / "env_l2f.py", Path(__file__).resolve())
    hard_constraints_passed = all(int(row["passed"]) == 1 for row in constraints)
    provenance = {
        "sampler": "physical-fit",
        "audit_scope": "checked necessary constraints and source consistency; not empirical fleet validation",
        "balanced": True,
        "joint_root_grid": {
            "dimensions": GRID_DIMENSIONS,
            "bins_per_dimension": GRID_BINS,
            "cells": GRID_CELLS,
        },
        "device": "cpu",
        "dtype": "torch.float32",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "seeds": seeds,
        "samples_per_seed": args.samples_per_seed,
        "total_samples": len(rows),
        "hard_constraints_passed": hard_constraints_passed,
        "source_hashes": {
            path.relative_to(ROOT).as_posix(): _sha256(path) for path in source_files
        },
    }
    (output_dir / "RUN_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **provenance}, indent=2))
    if not hard_constraints_passed:
        failed = [str(row["check"]) for row in constraints if int(row["passed"]) != 1]
        raise RuntimeError("physical-fit sampler audit failed: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

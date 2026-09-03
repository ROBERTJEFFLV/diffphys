from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "diagnostic_inputs/h10000_paired_96m_20260804/matlab_eval"
DEFAULT_OUTPUT = ROOT / "reports/physical_failure_axes_audit_20260806"
SEEDS = (7, 17, 27)
AXES = (
    "mass_kg",
    "arm_length_m",
    "thrust_to_weight",
    "alpha_roll_max",
    "alpha_yaw_max",
    "eta_yaw",
    "jz_over_jxy",
    "motor_time_rising_s",
    "motor_time_falling_s",
    "force_std",
    "required_tilt_deg",
)
SCENARIO_FIELDS = ("sample_id",) + AXES + (
    "inertia_x",
    "inertia_y",
    "inertia_z",
    "external_force_x",
    "external_force_y",
    "external_force_z",
)
HORIZONS = (500, 10_000)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def equal_rank_groups(values: list[float], groups: int = 4) -> list[list[int]]:
    """Return stable, equal-count rank groups; ties use sample order."""
    if groups < 2 or len(values) == 0 or len(values) % groups:
        raise ValueError("values must be nonempty and divisible by groups")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    width = len(values) // groups
    return [order[group * width : (group + 1) * width] for group in range(groups)]


def assert_scenario_pairing(
    reference: list[dict[str, str]],
    candidate: list[dict[str, str]],
) -> None:
    if len(reference) != len(candidate):
        raise ValueError("scenario row count differs across seeds")
    for row_index, (left, right) in enumerate(zip(reference, candidate)):
        for field in SCENARIO_FIELDS:
            if left.get(field) != right.get(field):
                raise ValueError(
                    f"scenario mismatch row={row_index} field={field}: "
                    f"{left.get(field)!r} != {right.get(field)!r}"
                )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rate(
    rows_by_seed: list[list[dict[str, str]]],
    indices: list[int],
    horizon: int,
) -> tuple[int, int, float]:
    field = f"position_hold_steady_H{horizon}"
    values = [int(rows[index][field]) for rows in rows_by_seed for index in indices]
    successes = sum(values)
    total = len(values)
    return successes, total, 100.0 * successes / total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exploratory common-scenario quartile audit of historical T0 failure axes."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [input_root / f"seed_{seed}" / "T0" / "samples.csv" for seed in SEEDS]
    rows_by_seed = [_read_csv(path) for path in paths]
    if any(len(rows) != 1024 for rows in rows_by_seed):
        raise ValueError("expected exactly 1,024 formal scenarios per seed")
    for rows in rows_by_seed[1:]:
        assert_scenario_pairing(rows_by_seed[0], rows)

    reference = rows_by_seed[0]
    feasible_indices = [
        index
        for index, row in enumerate(reference)
        if (
            float(row["inertia_x"]) <= float(row["inertia_y"]) + float(row["inertia_z"])
            and float(row["inertia_y"]) <= float(row["inertia_x"]) + float(row["inertia_z"])
            and float(row["inertia_z"]) <= float(row["inertia_x"]) + float(row["inertia_y"])
        )
    ]
    if len(feasible_indices) != 400:
        raise ValueError(f"expected 400 inertia-feasible scenarios, got {len(feasible_indices)}")
    panels = {
        "all_1024": list(range(len(reference))),
        "inertia_feasible_400": feasible_indices,
    }
    quartile_rows: list[dict[str, object]] = []
    spread_rows: list[dict[str, object]] = []
    for panel, panel_indices in panels.items():
        for axis in AXES:
            panel_values = [float(reference[index][axis]) for index in panel_indices]
            local_groups = equal_rank_groups(panel_values)
            groups = [
                [panel_indices[local_index] for local_index in local_group]
                for local_group in local_groups
            ]
            rates: dict[int, list[float]] = {horizon: [] for horizon in HORIZONS}
            for quartile, indices in enumerate(groups, start=1):
                axis_values = [float(reference[index][axis]) for index in indices]
                record: dict[str, object] = {
                    "panel": panel,
                    "axis": axis,
                    "quartile": quartile,
                    "scenario_count": len(indices),
                    "seed_scenario_count": len(indices) * len(SEEDS),
                    "axis_min": min(axis_values),
                    "axis_max": max(axis_values),
                }
                for horizon in HORIZONS:
                    successes, total, rate = _rate(rows_by_seed, indices, horizon)
                    record[f"successes_H{horizon}"] = successes
                    record[f"total_H{horizon}"] = total
                    record[f"success_rate_pct_H{horizon}"] = rate
                    rates[horizon].append(rate)
                quartile_rows.append(record)
            spread_rows.append(
                {
                    "panel": panel,
                    "axis": axis,
                    "success_range_pp_H500": max(rates[500]) - min(rates[500]),
                    "lowest_quartile_H500": rates[500].index(min(rates[500])) + 1,
                    "success_range_pp_H10000": max(rates[10_000]) - min(rates[10_000]),
                    "lowest_quartile_H10000": rates[10_000].index(min(rates[10_000])) + 1,
                }
            )
    spread_rows.sort(
        key=lambda row: (str(row["panel"]), -float(row["success_range_pp_H500"]))
    )
    _write_csv(output_dir / "QUARTILE_SUCCESS.csv", quartile_rows)
    _write_csv(output_dir / "AXIS_SPREAD_SUMMARY.csv", spread_rows)

    by_axis_quartile = {
        (str(row["panel"]), str(row["axis"]), int(row["quartile"])): row
        for row in quartile_rows
    }
    selected_axes = ("mass_kg", "thrust_to_weight", "force_std", "required_tilt_deg")
    lines = [
        "# Historical T0 physical failure-axis audit",
        "",
        "This is an exploratory marginal stratification of the same 1,024 formal",
        "scenarios for seeds 7/17/27. Each axis is split into four equal-count",
        "rank groups using the common scenario order. It is not a causal or",
        "multivariate attribution: several sampled parameters are correlated.",
        "",
        "| Axis | Q1 H500 | Q2 H500 | Q3 H500 | Q4 H500 | Q1 H10000 | Q2 H10000 | Q3 H10000 | Q4 H10000 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for axis in selected_axes:
        h500 = [
            float(by_axis_quartile[("all_1024", axis, q)]["success_rate_pct_H500"])
            for q in range(1, 5)
        ]
        h10000 = [
            float(by_axis_quartile[("all_1024", axis, q)]["success_rate_pct_H10000"])
            for q in range(1, 5)
        ]
        lines.append(
            f"| `{axis}` | "
            + " | ".join(f"{value:.2f}%" for value in h500 + h10000)
            + " |"
        )
    top = [row for row in spread_rows if row["panel"] == "all_1024"][:4]
    feasible_q4_h500 = {
        axis: float(
            by_axis_quartile[("inertia_feasible_400", axis, 4)][
                "success_rate_pct_H500"
            ]
        )
        for axis in ("required_tilt_deg", "thrust_to_weight", "force_std")
    }
    lines.extend(
        [
            "",
            "Mass and arm length are nearly flat in this marginal audit. The largest",
            "H500 separations are associated with required tilt, thrust-to-weight,",
            "external-force scale, and low angular authority. In particular, the",
            "highest thrust-to-weight quartile performs worse, so the present action",
            "normalization/sensitivity deserves a controlled frozen-policy probe; high",
            "T/W should not automatically be treated as an easier vehicle.",
            "",
            "Largest H500 quartile ranges: "
            + ", ".join(
                f"`{row['axis']}` {float(row['success_range_pp_H500']):.2f} pp"
                for row in top
            )
            + ".",
            "",
            "The same adverse directions remain on the 400-scenario inertia-feasible",
            "panel: the Q4 H500 rates are "
            f"{feasible_q4_h500['required_tilt_deg']:.2f}% for required tilt, "
            f"{feasible_q4_h500['thrust_to_weight']:.2f}% for thrust-to-weight,",
            f"and {feasible_q4_h500['force_std']:.2f}% for force scale. They are therefore not",
            "created solely by the impossible-inertia scenarios.",
            "",
            "A new challenge bank should stratify these axes independently and retain",
            "the full joint parameter record. The historical retain bank is a success",
            "safeguard, not a challenge or coverage set.",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tool_path = Path(__file__).resolve()
    provenance = {
        "scope": "exploratory marginal quartile audit; no causal claim",
        "policy": "historical T0 at 96M",
        "seeds": list(SEEDS),
        "scenario_count": 1024,
        "inertia_feasible_scenario_count": len(feasible_indices),
        "pairing": "exact common sample_id and physical-field strings across seeds",
        "quartile_definition": "stable equal-count rank groups, 256 common scenarios each",
        "success_rule": "formal MATLAB simultaneous joint steady label at each horizon",
        "axes": list(AXES),
        "source_hashes": {
            tool_path.relative_to(ROOT).as_posix(): _sha256(tool_path),
            **{
                path.relative_to(ROOT).as_posix(): _sha256(path)
                for path in paths
            },
        },
    }
    (output_dir / "RUN_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "top_axes": top}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

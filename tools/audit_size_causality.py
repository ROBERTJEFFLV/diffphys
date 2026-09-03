from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_l2f import L2FParams, L2FSimulator, L2FState  # noqa: E402


DEFAULT_OUTPUT = ROOT / "reports/size_causality_audit_20260806"
DEFAULT_MASSES = (0.02, 0.08, 0.32, 1.28, 5.0)
TRACKED_DYNAMIC_FIELDS = (
    "position",
    "velocity",
    "rotation",
    "omega",
    "motor",
    "previous_action",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_masses(value: str) -> tuple[float, ...]:
    masses = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if len(masses) < 2 or any(mass <= 0.0 for mass in masses):
        raise argparse.ArgumentTypeError("provide at least two positive masses")
    return masses


def _repeat_groups(value: torch.Tensor, variants: int) -> torch.Tensor:
    return value.repeat_interleave(variants, dim=0)


def _uniform_log(count: int, low: float, high: float) -> torch.Tensor:
    unit = torch.rand(count, dtype=torch.float64)
    return torch.exp(math.log(low) + unit * (math.log(high) - math.log(low)))


def build_matched_state(
    *, simulator: L2FSimulator, groups: int, masses: tuple[float, ...]
) -> tuple[L2FState, list[dict[str, float]]]:
    """Construct scale variants with identical normalized control capabilities."""

    variants = len(masses)
    params = simulator.params
    base = simulator.reset(groups, device="cpu", dtype=torch.float64)
    mass_values = torch.tensor(masses, dtype=torch.float64)
    mass = mass_values.repeat(groups)
    arm = (0.21257 * mass.pow(0.5028)).clamp(0.028, 0.50)

    thrust_to_weight_group = torch.empty(groups, dtype=torch.float64).uniform_(1.50, 5.50)
    alpha_roll_group = _uniform_log(groups, 35.0, 2200.0)
    eta_yaw_group = _uniform_log(groups, 0.02, 1.0)
    jz_ratio_group = torch.empty(groups, dtype=torch.float64).uniform_(1.45, 1.95)
    tau_rise_group = torch.empty(groups, dtype=torch.float64).uniform_(0.04, 0.18)
    fall_factor_group = torch.empty(groups, dtype=torch.float64).uniform_(1.0, 2.6)
    tau_fall_group = (tau_rise_group * fall_factor_group).clamp(0.04, 0.35)
    disturbance_acc_std = 0.20
    disturbance_acc_group = (
        torch.randn(groups, 3, dtype=torch.float64) * disturbance_acc_std
    )

    thrust_to_weight = _repeat_groups(thrust_to_weight_group, variants)
    alpha_roll = _repeat_groups(alpha_roll_group, variants)
    eta_yaw = _repeat_groups(eta_yaw_group, variants)
    jz_ratio = _repeat_groups(jz_ratio_group, variants)
    tau_rise = _repeat_groups(tau_rise_group, variants)
    tau_fall = _repeat_groups(tau_fall_group, variants)
    disturbance_acc = _repeat_groups(disturbance_acc_group, variants)

    hover_thrust = mass * float(params.gravity) / 4.0
    max_thrust = thrust_to_weight * hover_thrust
    min_thrust = torch.clamp(2.0 * hover_thrust - max_thrust, min=0.0)
    thrust_delta = torch.clamp(max_thrust - min_thrust, min=1.0e-12)
    thrust_c0 = hover_thrust[:, None].expand(-1, 4).clone()
    thrust_c1 = ((thrust_to_weight - 1.0) * hover_thrust)[:, None].expand(-1, 4).clone()
    thrust_c2 = torch.zeros_like(thrust_c0)

    inertia_x = arm * thrust_delta / alpha_roll
    inertia_y = inertia_x.clone()
    inertia_z = jz_ratio * inertia_x
    alpha_yaw = eta_yaw * alpha_roll
    rotor_torque_constant = alpha_yaw * inertia_z / (2.0 * thrust_delta)
    external_force = mass[:, None] * disturbance_acc

    state = L2FState(
        position=_repeat_groups(base.position, variants),
        velocity=_repeat_groups(base.velocity, variants),
        rotation=_repeat_groups(base.rotation, variants),
        omega=_repeat_groups(base.omega, variants),
        motor=_repeat_groups(base.motor, variants),
        previous_action=_repeat_groups(base.previous_action, variants),
        external_force=external_force,
        mass=mass,
        thrust_coeff_c0=thrust_c0,
        thrust_coeff_c1=thrust_c1,
        thrust_coeff_c2=thrust_c2,
        thrust_to_weight=thrust_to_weight,
        torque_to_inertia=alpha_roll,
        rotor_distance_factor=arm / float(params.arm_length),
        inertia_factor=torch.full_like(inertia_x, float(params.inertia_x)) / inertia_x,
        motor_time_rising=tau_rise,
        motor_time_falling=tau_fall,
        rotor_torque_constant=rotor_torque_constant,
        cbrt_mass=mass.pow(1.0 / 3.0),
        force_std=mass * disturbance_acc_std,
        arm_length=arm,
        inertia_x=inertia_x,
        inertia_y=inertia_y,
        inertia_z=inertia_z,
        alpha_roll_max=alpha_roll,
        alpha_pitch_max=alpha_roll.clone(),
        alpha_yaw_max=alpha_yaw,
        eta_yaw=eta_yaw,
        jz_over_jxy=jz_ratio,
        dt_alpha_roll_max=alpha_roll * float(params.dt),
        dt_alpha_yaw_max=alpha_yaw * float(params.dt),
    )
    mass_rows = [
        {
            "variant": index,
            "mass_kg": float(mass_value),
            "arm_length_m": min(
                max(float(0.21257 * mass_value**0.5028), 0.028), 0.50
            ),
        }
        for index, mass_value in enumerate(masses)
    ]
    return state, mass_rows


def _excitation_for_step(step: int, groups: int) -> torch.Tensor:
    """Return held limit/differential commands that cover all control axes."""

    templates = torch.tensor(
        (
            (1.0, 1.0, 1.0, 1.0),
            (-1.0, -1.0, -1.0, -1.0),
            (0.0, 1.0, 0.0, -1.0),
            (0.0, -1.0, 0.0, 1.0),
            (-1.0, 0.0, 1.0, 0.0),
            (1.0, 0.0, -1.0, 0.0),
            (1.0, -1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0, 1.0),
        ),
        dtype=torch.float64,
    )
    block_steps = 12
    group_offset = torch.arange(groups, dtype=torch.long) % templates.shape[0]
    template_index = ((step // block_steps) + group_offset) % templates.shape[0]
    return templates[template_index]


def _coverage_rows(counters: dict[str, float]) -> list[dict[str, object]]:
    checks = (
        ("action_reaches_negative_limit", counters["action_min"], "value <= -1", counters["action_min"] <= -1.0),
        ("action_reaches_positive_limit", counters["action_max"], "value >= +1", counters["action_max"] >= 1.0),
        ("positive_limit_commands", counters["positive_limit_commands"], "count > 0", counters["positive_limit_commands"] > 0),
        ("negative_limit_commands", counters["negative_limit_commands"], "count > 0", counters["negative_limit_commands"] > 0),
        ("motor_rising_branch", counters["motor_rising_branch"], "count > 0", counters["motor_rising_branch"] > 0),
        ("motor_falling_branch", counters["motor_falling_branch"], "count > 0", counters["motor_falling_branch"] > 0),
        ("raw_thrust_clamp_branch", counters["raw_thrust_clamp_branch"], "count > 0", counters["raw_thrust_clamp_branch"] > 0),
        ("positive_raw_thrust_branch", counters["positive_raw_thrust_branch"], "count > 0", counters["positive_raw_thrust_branch"] > 0),
        ("roll_differential_commands", counters["roll_differential_commands"], "count > 0", counters["roll_differential_commands"] > 0),
        ("pitch_differential_commands", counters["pitch_differential_commands"], "count > 0", counters["pitch_differential_commands"] > 0),
        ("yaw_differential_commands", counters["yaw_differential_commands"], "count > 0", counters["yaw_differential_commands"] > 0),
    )
    return [
        {
            "diagnostic": name,
            "value": value,
            "requirement": requirement,
            "passed": int(passed),
        }
        for name, value, requirement, passed in checks
    ]


def run_invariance_experiment(
    *, groups: int, steps: int, masses: tuple[float, ...], seed: int
) -> tuple[
    list[dict[str, object]],
    list[dict[str, float]],
    list[dict[str, object]],
]:
    if groups <= 0 or steps <= 0:
        raise ValueError("groups and steps must be positive")
    torch.manual_seed(seed)
    simulator = L2FSimulator()
    state, mass_rows = build_matched_state(
        simulator=simulator, groups=groups, masses=masses
    )
    variants = len(masses)
    max_abs = {name: 0.0 for name in TRACKED_DYNAMIC_FIELDS}
    squared_sum = {name: 0.0 for name in TRACKED_DYNAMIC_FIELDS}
    value_count = {name: 0 for name in TRACKED_DYNAMIC_FIELDS}
    counters = {
        "action_min": math.inf,
        "action_max": -math.inf,
        "positive_limit_commands": 0.0,
        "negative_limit_commands": 0.0,
        "motor_rising_branch": 0.0,
        "motor_falling_branch": 0.0,
        "raw_thrust_clamp_branch": 0.0,
        "positive_raw_thrust_branch": 0.0,
        "roll_differential_commands": 0.0,
        "pitch_differential_commands": 0.0,
        "yaw_differential_commands": 0.0,
    }

    for step in range(steps):
        action_group = _excitation_for_step(step, groups)
        action = _repeat_groups(action_group, variants)

        motor_group = state.motor.reshape(groups, variants, 4)[:, 0]
        tau_rise_group = state.motor_time_rising.reshape(groups, variants)[:, 0, None]
        tau_fall_group = state.motor_time_falling.reshape(groups, variants)[:, 0, None]
        rising = action_group >= motor_group
        motor_tau = torch.where(rising, tau_rise_group, tau_fall_group)
        motor_next = motor_group + torch.clamp(
            float(simulator.params.dt) / motor_tau, 0.0, 1.0
        ) * (action_group - motor_group)
        c0 = state.thrust_coeff_c0.reshape(groups, variants, 4)[:, 0]
        c1 = state.thrust_coeff_c1.reshape(groups, variants, 4)[:, 0]
        c2 = state.thrust_coeff_c2.reshape(groups, variants, 4)[:, 0]
        raw_thrust = c0 + c1 * motor_next + c2 * motor_next * motor_next

        counters["action_min"] = min(counters["action_min"], float(action_group.min().item()))
        counters["action_max"] = max(counters["action_max"], float(action_group.max().item()))
        counters["positive_limit_commands"] += float((action_group == 1.0).sum().item())
        counters["negative_limit_commands"] += float((action_group == -1.0).sum().item())
        counters["motor_rising_branch"] += float(rising.sum().item())
        counters["motor_falling_branch"] += float((~rising).sum().item())
        counters["raw_thrust_clamp_branch"] += float((raw_thrust <= 0.0).sum().item())
        counters["positive_raw_thrust_branch"] += float((raw_thrust > 0.0).sum().item())
        counters["roll_differential_commands"] += float(
            ((action_group[:, 1] - action_group[:, 3]).abs() > 0.5).sum().item()
        )
        counters["pitch_differential_commands"] += float(
            ((action_group[:, 2] - action_group[:, 0]).abs() > 0.5).sum().item()
        )
        counters["yaw_differential_commands"] += float(
            (
                (
                    action_group[:, 0]
                    - action_group[:, 1]
                    + action_group[:, 2]
                    - action_group[:, 3]
                ).abs()
                > 0.5
            ).sum().item()
        )

        state = simulator.step(state, action)
        for name in TRACKED_DYNAMIC_FIELDS:
            value = getattr(state, name).reshape(groups, variants, -1)
            difference = value - value[:, :1]
            max_abs[name] = max(max_abs[name], float(difference.abs().max().item()))
            squared_sum[name] += float((difference * difference).sum().item())
            value_count[name] += difference.numel()

    # The construction is analytically scale-equivalent.  The absolute check
    # allows only accumulated float64 solver roundoff under the deliberately
    # discontinuous limit-command sequence.
    tolerance = 1.0e-9
    rows = [
        {
            "state_field": name,
            "groups": groups,
            "mass_variants": variants,
            "steps": steps,
            "max_abs_difference": max_abs[name],
            "rms_difference": math.sqrt(squared_sum[name] / value_count[name]),
            "tolerance": tolerance,
            "passed": int(max_abs[name] <= tolerance),
        }
        for name in TRACKED_DYNAMIC_FIELDS
    ]
    return rows, mass_rows, _coverage_rows(counters)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(
    *,
    rows: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
    masses: tuple[float, ...],
    groups: int,
    steps: int,
) -> str:
    maximum = max(float(row["max_abs_difference"]) for row in rows)
    invariant = all(int(row["passed"]) == 1 for row in rows)
    tolerance = float(rows[0]["tolerance"])
    coverage = all(int(row["passed"]) == 1 for row in coverage_rows)
    return f"""# Constructive scale-equivalent causality audit

This is a **constructed counterexample**, not a fleet-identification benchmark.
For each of {groups} independently drawn normalized-dynamics groups, the simulator
was instantiated at masses {', '.join(f'{mass:g}' for mass in masses)} kg. Arm
length and dimensional coefficients changed with mass, while thrust-to-weight,
angular authority, inertia ratio, motor time constants, initial dynamic state and
external acceleration were matched within each group.

Under a shared {steps}-step command sequence, all tracked dynamic fields
(`position`, `velocity`, `rotation`, `omega`, `motor`, `previous_action`) were
invariant within the declared {tolerance:.0e} float64 diagnostic tolerance: **{invariant}**.
The largest observed absolute difference was {maximum:.3e}. The excitation
diagnostics covered command limits, rising/falling motor branches, raw-thrust
clamping, positive thrust, and roll/pitch/yaw differentials: **{coverage}**.

This counterexample is sufficient to reject the universal claim that absolute
mass or arm length must always be identifiable from these tracked trajectories in
the present scale-equivalent model. It does **not** show that size is never
inferable from correlations in a real fleet, richer sensors, aerodynamic effects,
contact, actuator saturation/failure, or other dimensional phenomena. The
physical parameter fields intentionally differ and are not included among the
invariance claims. H{steps} open-loop equality also does not establish closed-loop
GRU performance or real-aircraft generalization.
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Construct a scale-equivalent counterexample for absolute-size observability."
    )
    parser.add_argument("--groups", type=int, default=128)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--masses", type=_parse_masses, default=DEFAULT_MASSES)
    parser.add_argument("--seed", type=int, default=1007)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows, mass_rows, coverage_rows = run_invariance_experiment(
        groups=args.groups,
        steps=args.steps,
        masses=tuple(args.masses),
        seed=args.seed,
    )
    output_dir = args.output_dir.resolve()
    _write_csv(output_dir / "TRAJECTORY_INVARIANCE.csv", rows)
    _write_csv(output_dir / "MASS_ARM_VARIANTS.csv", mass_rows)
    _write_csv(output_dir / "EXCITATION_COVERAGE.csv", coverage_rows)
    (output_dir / "SUMMARY.md").write_text(
        _summary_markdown(
            rows=rows,
            coverage_rows=coverage_rows,
            masses=tuple(args.masses),
            groups=args.groups,
            steps=args.steps,
        ),
        encoding="utf-8",
    )
    passed = all(int(row["passed"]) == 1 for row in rows)
    excitation_coverage_passed = all(
        int(row["passed"]) == 1 for row in coverage_rows
    )
    provenance = {
        "experiment": "constructive scale-equivalent counterexample",
        "claim_scope": "tracked dynamic trajectories only; physical parameter fields intentionally differ",
        "groups": args.groups,
        "steps": args.steps,
        "masses_kg": tuple(args.masses),
        "seed": args.seed,
        "device": "cpu",
        "dtype": "torch.float64",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mass_scaled_constant_external_acceleration": True,
        "tracked_dynamic_fields": TRACKED_DYNAMIC_FIELDS,
        "all_tracked_dynamic_fields_invariant": passed,
        "excitation_design": "12-step held collective and roll/pitch/yaw differential limit commands",
        "excitation_coverage_passed": excitation_coverage_passed,
        "source_hashes": {
            "env_l2f.py": _sha256(ROOT / "env_l2f.py"),
            Path(__file__).resolve().relative_to(ROOT).as_posix(): _sha256(Path(__file__).resolve()),
        },
    }
    (output_dir / "RUN_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **provenance}, indent=2))
    if not passed:
        raise RuntimeError("scale-equivalent tracked trajectories exceeded the tolerance")
    if not excitation_coverage_passed:
        raise RuntimeError("scale-equivalent audit did not cover all declared excitation branches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

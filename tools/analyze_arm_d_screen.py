from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.phase1_fast_common import (  # noqa: E402
    artifact_fingerprint,
    atomic_write_dataframe,
    atomic_write_json,
    sha256_file,
)
from tools.validate_continuity_cadence_screen import (  # noqa: E402
    ARM_CONFIGS as FROZEN_ARM_CONFIGS,
    EXPECTED_CODE_FILES,
    INITIAL_CHECKPOINT_RELATIVE,
    INITIAL_CHECKPOINT_SHA256,
    MANIFEST_SCHEMA_VERSION,
    _expected_config_closure,
    _validate_frozen_records,
)


SEEDS = (7, 17, 27)
CAUSAL_ARMS = ("B", "D")
PROMOTION_ARMS = ("A", "B", "D")
ARMS = PROMOTION_ARMS
HORIZONS = (500, 10_000)
CAUSAL_COMPARISONS = (("D", "B"),)
PROMOTION_COMPARISONS = (("D", "A"), ("D", "B"))
BOOTSTRAP_COUNT = 20_000
BOOTSTRAP_SEED = 20_260_806
EQUIVALENCE_MARGIN = 0.005  # probability scale: +/-0.5 percentage points
NONINFERIORITY_MARGIN = 0.005
RESET_TOLERANCE = 1.0e-12
FORMAL_SCENARIO_SHA256 = (
    "97ea0e439fbc87f5fefbe89155cb07cb176274f10904f8a556b1a97d1004954b"
)

RESET_MAPPINGS = {
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

REJECT_REASONS = {
    "adaptive_gate_suspicious",
    "adaptive_gate_hard_grad",
    "post_update_rejected",
}
GRADIENT_SKIP_REASONS = {
    "grad_skip_threshold",
    "grad_norm_nonfinite",
    "grad_tensor_nonfinite",
}
EXPECTED_TRAINING = {
    "A": {
        "updates": 75,
        "resets": 75,
        "h500_resets": 75,
        "h1000_resets": 0,
        "tail_blocks": 75,
    },
    "B": {
        "updates": 67,
        "resets": 67,
        "h500_resets": 59,
        "h1000_resets": 8,
        "tail_blocks": 67,
    },
    "D": {
        "updates": 67,
        "resets": 67,
        "h500_resets": 59,
        "h1000_resets": 8,
        "tail_blocks": 75,
    },
}


def _artifact_path(root: Path, *, seed: int, arm: str, relative: str) -> Path:
    """Resolve either a shared-root or an arm-specific artifact layout."""

    candidates = (
        root / f"seed_{seed}" / f"arm_{arm}" / relative,
        root / f"seed_{seed}" / relative,
        root / f"arm_{arm}" / f"seed_{seed}" / relative,
    )
    existing = tuple(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))
    if len(existing) != 1:
        rendered = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"expected exactly one seed={seed} arm={arm} artifact {relative!r}; "
            f"found {len(existing)} among: {rendered}"
        )
    return existing[0]


def _roots(
    *,
    shared: Path | None,
    overrides: dict[str, Path | None],
    label: str,
    arms: tuple[str, ...] = PROMOTION_ARMS,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for arm in arms:
        root = overrides[arm] if overrides[arm] is not None else shared
        if root is None:
            raise ValueError(
                f"missing {label} root for arm {arm}; provide --{label}-root or "
                f"--arm-{arm.lower()}-{label}-root"
            )
        resolved[arm] = root.resolve()
    return resolved


def _validate_training_manifest(
    manifest_path: Path,
    *,
    run_dir: Path,
    seed: int,
    arm: str,
) -> tuple[str, str, str]:
    """Validate the schema-v2 frozen launch closure without folding in outcomes."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = EXPECTED_TRAINING[arm]
    expected_fields = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "seed": seed,
        "arm": arm,
        "initial_checkpoint": INITIAL_CHECKPOINT_RELATIVE.as_posix(),
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
        "physical_step_budget": 9_600_000,
        "segment_horizon": 250,
        "batch_size": 256,
        "expected_segments": 150,
        "expected_reset_episodes": expected["resets"],
        "expected_optimizer_commits": expected["updates"],
        "expected_tail_supervision_blocks": expected["tail_blocks"],
    }
    errors = [
        f"manifest {key}={payload.get(key)!r}, expected {value!r}"
        for key, value in expected_fields.items()
        if payload.get(key) != value
    ]
    errors.extend(
        _validate_frozen_records(
            payload.get("code_files"),
            label="code_files",
            expected_paths=EXPECTED_CODE_FILES,
        )
    )
    errors.extend(
        _validate_frozen_records(
            payload.get("config_closure"),
            label="config_closure",
            expected_paths=_expected_config_closure(arm),
        )
    )
    command = payload.get("command")
    expected_config = f"@{FROZEN_ARM_CONFIGS[arm].as_posix()}"
    if not isinstance(command, list) or expected_config not in command:
        errors.append(f"manifest command does not select {expected_config}")
    command_path = run_dir / "command.txt"
    if not command_path.is_file() or not command_path.read_text(
        encoding="utf-8"
    ).strip():
        errors.append("run lacks a non-empty command.txt")

    initial_checkpoint = ROOT / INITIAL_CHECKPOINT_RELATIVE
    if not initial_checkpoint.is_file():
        errors.append(f"frozen initial checkpoint is missing: {initial_checkpoint}")
    elif sha256_file(initial_checkpoint) != INITIAL_CHECKPOINT_SHA256:
        errors.append("frozen initial checkpoint file hash changed")

    torch_version = payload.get("torch_version")
    cuda_build = payload.get("torch_cuda_build")
    device_name = payload.get("cuda_device_name")
    if payload.get("cuda_available") is not True:
        errors.append("formal run manifest does not declare CUDA available")
    if not isinstance(payload.get("cuda_device_count"), int) or int(
        payload.get("cuda_device_count", 0)
    ) < 1:
        errors.append("formal run manifest has no CUDA device")
    for label, value in (
        ("torch_version", torch_version),
        ("torch_cuda_build", cuda_build),
        ("cuda_device_name", device_name),
    ):
        if not isinstance(value, str) or not value:
            errors.append(f"formal run manifest lacks {label}")
    if errors:
        rendered = "; ".join(errors)
        raise RuntimeError(
            f"training provenance failed for seed={seed} arm={arm}: {rendered}"
        )
    return str(torch_version), str(cuda_build), str(device_name)


def _path_has_suffix(value: object, *parts: str) -> bool:
    normalized = str(value).replace("\\", "/").rstrip("/")
    return normalized.endswith("/".join(parts))


def _numeric_equals(value: object, expected: int) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and numeric == float(expected))


def _inertia_feasible(scenarios: pd.DataFrame) -> np.ndarray:
    x = scenarios["inertia_x"].to_numpy(dtype=np.float64)
    y = scenarios["inertia_y"].to_numpy(dtype=np.float64)
    z = scenarios["inertia_z"].to_numpy(dtype=np.float64)
    tolerance = 1.0e-12
    return (
        (x <= y + z + tolerance)
        & (y <= x + z + tolerance)
        & (z <= x + y + tolerance)
    )


def _validate_samples(
    samples: pd.DataFrame,
    scenarios: pd.DataFrame,
    *,
    source: Path,
) -> dict[str, float]:
    if len(samples) != len(scenarios):
        raise RuntimeError(
            f"formal sample count mismatch: {source} has {len(samples)}, "
            f"manifest has {len(scenarios)}"
        )
    sample_ids = pd.to_numeric(samples["sample_id"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    expected_ids = pd.to_numeric(
        scenarios["scenario_id"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    ids_are_integral = bool(
        np.isfinite(sample_ids).all()
        and np.isfinite(expected_ids).all()
        and np.equal(sample_ids, np.floor(sample_ids)).all()
        and np.equal(expected_ids, np.floor(expected_ids)).all()
    )
    if not ids_are_integral or not np.array_equal(sample_ids, expected_ids):
        raise RuntimeError(f"sample IDs/order do not match manifest: {source}")

    errors: dict[str, float] = {}
    for scenario_column, sample_column in RESET_MAPPINGS.items():
        if scenario_column not in scenarios or sample_column not in samples:
            raise RuntimeError(
                f"missing paired scenario field {scenario_column!r}/{sample_column!r}: "
                f"{source}"
            )
        scenario_values = pd.to_numeric(
            scenarios[scenario_column], errors="coerce"
        ).to_numpy(dtype=np.float64)
        sample_values = pd.to_numeric(samples[sample_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(scenario_values).all() or not np.isfinite(sample_values).all():
            raise RuntimeError(
                f"non-finite paired scenario field {scenario_column!r}/{sample_column!r}: "
                f"{source}"
            )
        error = float(np.max(np.abs(scenario_values - sample_values)))
        errors[scenario_column] = error
        if error > RESET_TOLERANCE:
            raise RuntimeError(
                f"scenario parameter mismatch for {scenario_column}: {error:.3e} in {source}"
            )
    return errors


def _load_formal_metrics(
    *,
    roots: dict[str, Path],
    scenarios: pd.DataFrame,
    arms: tuple[str, ...] = PROMOTION_ARMS,
) -> tuple[pd.DataFrame, list[Path], dict[str, float]]:
    expected_uids = scenarios["scenario_uid"].astype(str).tolist()
    rows: list[pd.DataFrame] = []
    inputs: list[Path] = []
    maximum_errors = {column: 0.0 for column in RESET_MAPPINGS}
    for seed in SEEDS:
        for arm in arms:
            path = _artifact_path(
                roots[arm], seed=seed, arm=arm, relative="samples.csv"
            )
            samples = pd.read_csv(path)
            errors = _validate_samples(samples, scenarios, source=path)
            for column, error in errors.items():
                maximum_errors[column] = max(maximum_errors[column], error)
            for horizon in HORIZONS:
                success_column = f"position_hold_steady_H{horizon}"
                if success_column not in samples:
                    raise RuntimeError(f"missing formal label {success_column}: {path}")
                raw_success = pd.to_numeric(
                    samples[success_column], errors="coerce"
                ).to_numpy(dtype=np.float64)
                if not np.isfinite(raw_success).all() or not np.isin(
                    raw_success, (0.0, 1.0)
                ).all():
                    raise RuntimeError(f"formal labels are not binary: {path}, H{horizon}")
                success = raw_success.astype(np.int64)
                rows.append(
                    pd.DataFrame(
                        {
                            "seed": seed,
                            "arm": arm,
                            "horizon": horizon,
                            "scenario_id": samples["sample_id"].to_numpy(dtype=np.int64),
                            "scenario_uid": expected_uids,
                            "success": success,
                        }
                    )
                )
            inputs.append(path)
    metrics = pd.concat(rows, ignore_index=True)
    expected_rows = len(SEEDS) * len(arms) * len(HORIZONS) * len(scenarios)
    if len(metrics) != expected_rows:
        raise RuntimeError(f"formal metric rows={len(metrics)}, expected={expected_rows}")
    keys = ["seed", "arm", "horizon", "scenario_uid"]
    if metrics.duplicated(keys).any():
        raise RuntimeError("formal metrics do not have one-to-one scenario keys")
    return metrics, inputs, maximum_errors


def _bootstrap_intervals(
    difference: np.ndarray,
    *,
    n_bootstrap: int = BOOTSTRAP_COUNT,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return paired scenario-block and crossed seed x scenario intervals."""

    if difference.ndim != 2 or difference.shape[0] != len(SEEDS):
        raise ValueError("difference must have shape [3 training seeds, scenarios]")
    if difference.shape[1] == 0 or n_bootstrap <= 0:
        raise ValueError("bootstrap requires scenarios and a positive replication count")
    seed_count, scenario_count = difference.shape
    rng = np.random.default_rng(seed)
    scenario_estimates = np.empty(n_bootstrap, dtype=np.float64)
    crossed_estimates = np.empty(n_bootstrap, dtype=np.float64)
    chunk = 250
    for start in range(0, n_bootstrap, chunk):
        stop = min(start + chunk, n_bootstrap)
        count = stop - start
        scenario_indices = rng.integers(
            0, scenario_count, size=(count, scenario_count)
        )
        scenario_estimates[start:stop] = difference[:, scenario_indices].mean(
            axis=(0, 2)
        )
        seed_indices = rng.integers(0, seed_count, size=(count, seed_count))
        crossed = difference[
            seed_indices[:, :, None],
            scenario_indices[:, None, :],
        ]
        crossed_estimates[start:stop] = crossed.mean(axis=(1, 2))
    quantiles = (0.025, 0.975)
    scenario_ci = tuple(
        float(value) for value in np.quantile(scenario_estimates, quantiles)
    )
    crossed_ci = tuple(
        float(value) for value in np.quantile(crossed_estimates, quantiles)
    )
    return scenario_ci, crossed_ci


def _effect_tables(
    metrics: pd.DataFrame,
    *,
    scenario_uids: list[str],
    feasible_mask: np.ndarray,
    arms: tuple[str, ...] = PROMOTION_ARMS,
    comparisons: tuple[tuple[str, str], ...] = PROMOTION_COMPARISONS,
    n_bootstrap: int = BOOTSTRAP_COUNT,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    flip_rows: list[dict[str, object]] = []
    strata = (
        ("all", np.ones(len(scenario_uids), dtype=bool)),
        ("inertia_feasible", feasible_mask),
    )
    for horizon in HORIZONS:
        selected_horizon = metrics[metrics["horizon"].eq(horizon)]
        pivot = selected_horizon.pivot(
            index="scenario_uid", columns=["seed", "arm"], values="success"
        ).reindex(scenario_uids)
        expected_columns = pd.MultiIndex.from_product((SEEDS, arms))
        if not expected_columns.isin(pivot.columns).all() or pivot.isna().any().any():
            raise RuntimeError(f"formal scenario pairing is incomplete at H{horizon}")

        for left, right in comparisons:
            difference = np.stack(
                [
                    pivot[(seed, left)].to_numpy(dtype=np.float64)
                    - pivot[(seed, right)].to_numpy(dtype=np.float64)
                    for seed in SEEDS
                ]
            )
            for stratum, mask in strata:
                selected = difference[:, mask]
                scenario_ci, crossed_ci = _bootstrap_intervals(
                    selected,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed,
                )
                seed_means = selected.mean(axis=1)
                pair_rows.append(
                    {
                        "left_arm": left,
                        "right_arm": right,
                        "horizon": horizon,
                        "stratum": stratum,
                        "scenario_count": int(mask.sum()),
                        "point_difference": float(selected.mean()),
                        "point_difference_pp": float(100.0 * selected.mean()),
                        "scenario_ci_low": scenario_ci[0],
                        "scenario_ci_high": scenario_ci[1],
                        "scenario_ci_low_pp": 100.0 * scenario_ci[0],
                        "scenario_ci_high_pp": 100.0 * scenario_ci[1],
                        "crossed_ci_low": crossed_ci[0],
                        "crossed_ci_high": crossed_ci[1],
                        "crossed_ci_low_pp": 100.0 * crossed_ci[0],
                        "crossed_ci_high_pp": 100.0 * crossed_ci[1],
                        "positive_seed_count": int((seed_means > 0.0).sum()),
                        "minimum_seed_difference": float(seed_means.min()),
                        "crossed_equivalent_pm0p5pp": bool(
                            crossed_ci[0] >= -EQUIVALENCE_MARGIN
                            and crossed_ci[1] <= EQUIVALENCE_MARGIN
                        ),
                    }
                )
                for seed_index, seed in enumerate(SEEDS):
                    seed_rows.append(
                        {
                            "seed": seed,
                            "left_arm": left,
                            "right_arm": right,
                            "horizon": horizon,
                            "stratum": stratum,
                            "scenario_count": int(mask.sum()),
                            "difference": float(seed_means[seed_index]),
                            "difference_pp": float(100.0 * seed_means[seed_index]),
                        }
                    )

                left_values = np.stack(
                    [pivot[(seed, left)].to_numpy(dtype=bool)[mask] for seed in SEEDS]
                )
                right_values = np.stack(
                    [pivot[(seed, right)].to_numpy(dtype=bool)[mask] for seed in SEEDS]
                )
                for seed_index, seed_label in (
                    *((index, str(seed)) for index, seed in enumerate(SEEDS)),
                    (None, "pooled"),
                ):
                    if seed_index is None:
                        left_selected = left_values.reshape(-1)
                        right_selected = right_values.reshape(-1)
                    else:
                        left_selected = left_values[seed_index]
                        right_selected = right_values[seed_index]
                    gains = int((left_selected & ~right_selected).sum())
                    losses = int((~left_selected & right_selected).sum())
                    flip_rows.append(
                        {
                            "left_arm": left,
                            "right_arm": right,
                            "horizon": horizon,
                            "stratum": stratum,
                            "seed": seed_label,
                            "labels": len(left_selected),
                            "left_success_right_failure": gains,
                            "left_failure_right_success": losses,
                            "net_labels": gains - losses,
                        }
                    )
    return pd.DataFrame(pair_rows), pd.DataFrame(seed_rows), pd.DataFrame(flip_rows)


def _load_training_safety(
    roots: dict[str, Path],
    *,
    arms: tuple[str, ...] = PROMOTION_ARMS,
) -> tuple[pd.DataFrame, list[Path], bool]:
    rows: list[dict[str, object]] = []
    inputs: list[Path] = []
    reset_hashes: dict[tuple[int, str], str] = {}
    runtime_signatures: dict[tuple[int, str], tuple[str, str, str]] = {}
    for seed in SEEDS:
        for arm in arms:
            root = roots[arm]
            log_path = _artifact_path(root, seed=seed, arm=arm, relative="train.csv")
            reset_path = _artifact_path(
                root, seed=seed, arm=arm, relative="reset_samples.csv"
            )
            checkpoint_path = _artifact_path(
                root, seed=seed, arm=arm, relative="checkpoints/model.pt"
            )
            manifest_path = _artifact_path(
                root, seed=seed, arm=arm, relative="RUN_MANIFEST.json"
            )
            run_dir = log_path.parent
            artifact_run_dirs = {
                run_dir,
                reset_path.parent,
                checkpoint_path.parent.parent,
                manifest_path.parent,
            }
            if len(artifact_run_dirs) != 1:
                raise RuntimeError(
                    f"training artifacts span multiple run directories: seed={seed} arm={arm}"
                )
            runtime_signature = _validate_training_manifest(
                manifest_path,
                run_dir=run_dir,
                seed=seed,
                arm=arm,
            )
            runtime_signatures[(seed, arm)] = runtime_signature
            command_path = run_dir / "command.txt"
            training = pd.read_csv(log_path)
            if training.empty:
                raise RuntimeError(f"empty training log: {log_path}")
            expected = EXPECTED_TRAINING[arm]
            skip_reason = training["skip_reason"].fillna("").astype(str)
            invalid_rollouts = int((training["rollout_valid"].astype(int) == 0).sum())
            rejected_updates = int(skip_reason.isin(REJECT_REASONS).sum())
            gradient_skips = int(skip_reason.isin(GRADIENT_SKIP_REASONS).sum())
            reset_boundaries = int(training["reset_episode_boundary"].astype(int).sum())
            optimizer_boundaries = int(
                training["optimization_block_boundary"].astype(int).sum()
            )
            updates_applied = int(training["update_applied"].astype(int).sum())
            h500_resets = int(
                (
                    training["reset_episode_boundary"].astype(bool)
                    & training["episode_target_steps"].eq(500)
                ).sum()
            )
            h1000_resets = int(
                (
                    training["reset_episode_boundary"].astype(bool)
                    & training["episode_target_steps"].eq(1000)
                ).sum()
            )
            mid_h1000_optimizer_boundaries = int(
                (
                    training["optimization_block_boundary"].astype(bool)
                    & training["episode_target_steps"].eq(1000)
                    & ~training["reset_episode_boundary"].astype(bool)
                ).sum()
            )
            tail_columns = {
                "first_tail_supervision_segment",
                "tail_supervision_block_boundary",
                "tail_supervision_block_horizon",
            }
            has_tail_columns = tail_columns.issubset(training.columns)
            if has_tail_columns:
                tail_starts = int(
                    training["first_tail_supervision_segment"].astype(int).sum()
                )
                tail_boundaries = int(
                    training["tail_supervision_block_boundary"].astype(int).sum()
                )
            else:
                # Frozen A/B logs predate the decoupled cadence columns and used
                # optimizer-coupled independent CVaR events.
                tail_starts = optimizer_boundaries
                tail_boundaries = optimizer_boundaries
            d_tail_horizon_valid = bool(
                arm != "D"
                or (
                    has_tail_columns
                    and training["tail_supervision_block_horizon"].eq(500).all()
                )
            )
            schedule_valid = bool(
                len(training) == 150
                and int(training.iloc[-1]["physical_steps"]) == 9_600_000
                and int(training.iloc[-1]["optimizer_update"]) == expected["updates"]
                and optimizer_boundaries == expected["updates"]
                and updates_applied == expected["updates"]
                and reset_boundaries == expected["resets"]
                and h500_resets == expected["h500_resets"]
                and h1000_resets == expected["h1000_resets"]
                and tail_starts == expected["tail_blocks"]
                and tail_boundaries == expected["tail_blocks"]
                and d_tail_horizon_valid
                and (arm != "D" or mid_h1000_optimizer_boundaries == 0)
            )
            digest = sha256_file(reset_path)
            reset_hashes[(seed, arm)] = digest
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "rows": len(training),
                    "physical_steps": int(training.iloc[-1]["physical_steps"]),
                    "optimizer_boundaries": optimizer_boundaries,
                    "updates_applied": updates_applied,
                    "reset_boundaries": reset_boundaries,
                    "h500_reset_episodes": h500_resets,
                    "h1000_reset_episodes": h1000_resets,
                    "mid_h1000_optimizer_boundaries": mid_h1000_optimizer_boundaries,
                    "tail_supervision_starts": tail_starts,
                    "tail_supervision_boundaries": tail_boundaries,
                    "invalid_rollouts": invalid_rollouts,
                    "rejected_updates": rejected_updates,
                    "gradient_skips": gradient_skips,
                    "schedule_valid": schedule_valid,
                    "reset_samples_sha256": digest,
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "checkpoint_bytes": checkpoint_path.stat().st_size,
                    "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                    "torch_version": runtime_signature[0],
                    "torch_cuda_build": runtime_signature[1],
                    "cuda_device_name": runtime_signature[2],
                }
            )
            inputs.extend(
                (log_path, reset_path, checkpoint_path, manifest_path, command_path)
            )

    if not {"B", "D"}.issubset(arms):
        raise ValueError("Arm-D analysis requires both B and D training roots")
    reset_pairing_valid = all(
        reset_hashes[(seed, "B")] == reset_hashes[(seed, "D")] for seed in SEEDS
    )
    unique_runtime_signatures = set(runtime_signatures.values())
    if len(unique_runtime_signatures) != 1:
        raise RuntimeError(
            "active-stage runs do not share one Torch/CUDA runtime and device class: "
            f"{sorted(unique_runtime_signatures)}"
        )
    return pd.DataFrame(rows), inputs, reset_pairing_valid


def _validate_matlab_linkage(
    roots: dict[str, Path],
    training_safety: pd.DataFrame,
    *,
    arms: tuple[str, ...] = PROMOTION_ARMS,
) -> tuple[pd.DataFrame, list[Path]]:
    """Bind every formal label file to its indexed final training checkpoint."""

    grouped_roots: dict[Path, list[str]] = {}
    for arm in arms:
        grouped_roots.setdefault(roots[arm].resolve(), []).append(arm)

    inputs: list[Path] = []
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    safety_index = training_safety.set_index(["seed", "arm"])
    for root, root_arms in grouped_roots.items():
        checkpoints_path = root / "CHECKPOINTS.csv"
        eval_manifest_path = root / "eval_manifest.csv"
        if not checkpoints_path.is_file() or not eval_manifest_path.is_file():
            missing = [
                str(path)
                for path in (checkpoints_path, eval_manifest_path)
                if not path.is_file()
            ]
            raise FileNotFoundError(
                "MATLAB root lacks required linkage indexes: " + ", ".join(missing)
            )
        checkpoints = pd.read_csv(checkpoints_path)
        eval_manifest = pd.read_csv(eval_manifest_path)
        inputs.extend((checkpoints_path, eval_manifest_path))

        expected_keys = {(seed, arm) for seed in SEEDS for arm in root_arms}
        checkpoint_keys = list(
            zip(
                pd.to_numeric(checkpoints["seed"], errors="coerce"),
                checkpoints["arm"].astype(str).str.upper(),
            )
        )
        if (
            len(checkpoint_keys) != len(expected_keys)
            or len(set(checkpoint_keys)) != len(checkpoint_keys)
            or set(checkpoint_keys) != expected_keys
        ):
            raise RuntimeError(
                f"CHECKPOINTS.csv keys are not exactly active arms {sorted(expected_keys)}: "
                f"{checkpoints_path}"
            )
        expected_labels = {
            f"seed_{seed}_arm_{arm}_physical_steps_9600000"
            for seed, arm in expected_keys
        }
        manifest_labels = eval_manifest["label"].astype(str).tolist()
        if (
            len(manifest_labels) != len(expected_labels)
            or len(set(manifest_labels)) != len(manifest_labels)
            or set(manifest_labels) != expected_labels
        ):
            raise RuntimeError(
                "eval_manifest.csv labels are not exactly the active seed/arm set: "
                f"{eval_manifest_path}"
            )

        for seed, arm in sorted(expected_keys):
            checkpoint_row = checkpoints[
                pd.to_numeric(checkpoints["seed"], errors="coerce").eq(seed)
                & checkpoints["arm"].astype(str).str.upper().eq(arm)
            ].iloc[0]
            label = f"seed_{seed}_arm_{arm}_physical_steps_9600000"
            eval_row = eval_manifest[eval_manifest["label"].astype(str).eq(label)].iloc[0]
            expected_suffix = (f"seed_{seed}", f"arm_{arm}")
            sample_path = _artifact_path(
                root, seed=seed, arm=arm, relative="samples.csv"
            )
            model_mat_path = _artifact_path(
                root, seed=seed, arm=arm, relative="model.mat"
            )
            training_row = safety_index.loc[(seed, arm)]
            checks = {
                "checkpoint_label": str(checkpoint_row["label"]) == label,
                "physical_steps": _numeric_equals(
                    checkpoint_row["physical_steps"], 9_600_000
                ),
                "checkpoint_path": _path_has_suffix(
                    checkpoint_row["checkpoint_path"],
                    *expected_suffix,
                    "checkpoints",
                    "model.pt",
                ),
                "checkpoint_sha256": str(checkpoint_row["checkpoint_sha256"])
                == str(training_row["checkpoint_sha256"]),
                "checkpoint_bytes": _numeric_equals(
                    checkpoint_row["checkpoint_bytes"],
                    int(training_row["checkpoint_bytes"]),
                ),
                "model_mat_path": _path_has_suffix(
                    checkpoint_row["model_mat_path"], *expected_suffix, "model.mat"
                ),
                "model_mat_sha256": str(checkpoint_row["model_mat_sha256"])
                == sha256_file(model_mat_path),
                "sample_output_path": _path_has_suffix(
                    checkpoint_row["sample_output_path"],
                    *expected_suffix,
                    "samples.csv",
                ),
                "eval_seed": _numeric_equals(checkpoint_row["eval_seed"], 1007),
                "horizon": _numeric_equals(checkpoint_row["horizon"], 10_000),
                "eval_checkpoint_path": _path_has_suffix(
                    eval_row["checkpoint_path"],
                    *expected_suffix,
                    "checkpoints",
                    "model.pt",
                ),
                "eval_weights_path": _path_has_suffix(
                    eval_row["weights_path"], *expected_suffix, "model.mat"
                ),
                "eval_sample_path": _path_has_suffix(
                    eval_row["sample_output_path"], *expected_suffix, "samples.csv"
                ),
                "eval_seed_manifest": _numeric_equals(eval_row["eval_seed"], 1007),
                "eval_horizon_manifest": _numeric_equals(
                    eval_row["horizon"], 10_000
                ),
                "eval_batch_size": _numeric_equals(eval_row["batch_size"], 1024),
            }
            failed = sorted(name for name, passed in checks.items() if not passed)
            if failed:
                errors.append(f"seed={seed} arm={arm}: {failed}")
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "checkpoint_sha256": str(training_row["checkpoint_sha256"]),
                    "samples_sha256": sha256_file(sample_path),
                    "model_mat_sha256": sha256_file(model_mat_path),
                    "eval_seed": eval_row["eval_seed"],
                    "horizon": eval_row["horizon"],
                    "batch_size": eval_row["batch_size"],
                    "linkage_valid": not failed,
                    "failed_checks": ";".join(failed),
                }
            )
            inputs.extend((sample_path, model_mat_path))
    if errors:
        raise RuntimeError("formal MATLAB linkage failed: " + "; ".join(errors))
    return pd.DataFrame(rows), inputs


def _performance_summary(
    metrics: pd.DataFrame,
    *,
    feasible_uids: set[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for stratum, selected in (
        ("all", metrics),
        ("inertia_feasible", metrics[metrics["scenario_uid"].isin(feasible_uids)]),
    ):
        frame = (
            selected.groupby(["seed", "arm", "horizon"], as_index=False)
            .agg(success_rate=("success", "mean"), scenario_count=("success", "size"))
            .assign(stratum=stratum)
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[
        ["seed", "arm", "horizon", "stratum", "scenario_count", "success_rate"]
    ]


def _effect(
    effects: pd.DataFrame,
    left: str,
    right: str,
    horizon: int,
    stratum: str = "all",
) -> pd.Series:
    selected = effects[
        effects["left_arm"].eq(left)
        & effects["right_arm"].eq(right)
        & effects["horizon"].eq(horizon)
        & effects["stratum"].eq(stratum)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"effect lookup is not unique: {left}-{right}, H{horizon}, {stratum}"
        )
    return selected.iloc[0]


def _gate(
    category: str,
    name: str,
    *,
    value: object,
    threshold: object,
    comparison: str,
    passed: bool,
    value_high: object = np.nan,
    threshold_high: object = np.nan,
) -> dict[str, object]:
    return {
        "category": category,
        "gate": name,
        "value": value,
        "value_high": value_high,
        "threshold": threshold,
        "threshold_high": threshold_high,
        "comparison": comparison,
        "passed": bool(passed),
    }


def _decision_gates(
    effects: pd.DataFrame,
    safety: pd.DataFrame,
    *,
    reset_pairing_valid: bool,
    stage: str = "promotion",
) -> tuple[pd.DataFrame, dict[str, object]]:
    if stage not in {"causal", "promotion"}:
        raise ValueError("stage must be 'causal' or 'promotion'")
    active_arms = CAUSAL_ARMS if stage == "causal" else PROMOTION_ARMS
    observed_arms = set(safety["arm"].astype(str))
    if observed_arms != set(active_arms):
        raise RuntimeError(
            f"safety arms {sorted(observed_arms)} do not match stage {stage}: "
            f"{list(active_arms)}"
        )

    db500 = _effect(effects, "D", "B", 500)
    db10000 = _effect(effects, "D", "B", 10_000)
    schedule_safe = bool(safety["schedule_valid"].all())
    nonfinite_count = int(safety["invalid_rollouts"].sum())
    rejected_count = int(safety["rejected_updates"].sum())
    gradient_skip_count = int(safety["gradient_skips"].sum())
    nonfinite_safe = nonfinite_count == 0
    rejected_safe = rejected_count == 0
    gradient_safe = gradient_skip_count == 0
    training_safety = bool(
        schedule_safe
        and nonfinite_safe
        and rejected_safe
        and gradient_safe
        and reset_pairing_valid
    )

    causal_h500 = bool(
        db500["point_difference"] > 0.0 and db500["crossed_ci_low"] > 0.0
    )
    causal_h500_point = bool(db500["point_difference"] > 0.0)
    causal_h500_crossed = bool(db500["crossed_ci_low"] > 0.0)
    causal_h10000 = bool(
        db10000["crossed_ci_low"] > -NONINFERIORITY_MARGIN
    )
    causal_performance_passed = causal_h500 and causal_h10000
    causal_package_supported = causal_performance_passed and training_safety

    rows = [
        _gate(
            "safety",
            "all_stage_arms_schedule_counts_valid",
            value=int(schedule_safe),
            threshold=1,
            comparison="==",
            passed=schedule_safe,
        ),
        _gate(
            "safety",
            "all_stage_arms_zero_nonfinite_rollouts",
            value=nonfinite_count,
            threshold=0,
            comparison="==",
            passed=nonfinite_safe,
        ),
        _gate(
            "safety",
            "all_stage_arms_zero_rejected_updates",
            value=rejected_count,
            threshold=0,
            comparison="==",
            passed=rejected_safe,
        ),
        _gate(
            "safety",
            "all_stage_arms_zero_gradient_skips",
            value=gradient_skip_count,
            threshold=0,
            comparison="==",
            passed=gradient_safe,
        ),
        _gate(
            "safety",
            "b_d_reset_samples_identical",
            value=int(reset_pairing_valid),
            threshold=1,
            comparison="==",
            passed=reset_pairing_valid,
        ),
        _gate(
            "safety",
            "all_training_safety_gates_passed",
            value=int(training_safety),
            threshold=1,
            comparison="==",
            passed=training_safety,
        ),
        _gate(
            "causal",
            "d_b_h500_point_positive",
            value=float(db500["point_difference"]),
            threshold=0.0,
            comparison=">",
            passed=causal_h500_point,
        ),
        _gate(
            "causal",
            "d_b_h500_crossed_superiority",
            value=float(db500["crossed_ci_low"]),
            threshold=0.0,
            comparison=">",
            passed=causal_h500_crossed,
        ),
        _gate(
            "causal",
            "d_b_h10000_crossed_noninferiority",
            value=float(db10000["crossed_ci_low"]),
            threshold=-NONINFERIORITY_MARGIN,
            comparison=">",
            passed=causal_h10000,
        ),
    ]

    active_comparisons = (
        CAUSAL_COMPARISONS if stage == "causal" else PROMOTION_COMPARISONS
    )
    for left, right in active_comparisons:
        for horizon in HORIZONS:
            effect = _effect(effects, left, right, horizon)
            equivalent = bool(
                effect["crossed_ci_low"] >= -EQUIVALENCE_MARGIN
                and effect["crossed_ci_high"] <= EQUIVALENCE_MARGIN
            )
            rows.append(
                _gate(
                    "equivalence",
                    f"{left.lower()}_{right.lower()}_h{horizon}_crossed_equivalence_pm0p5pp",
                    value=float(effect["crossed_ci_low"]),
                    value_high=float(effect["crossed_ci_high"]),
                    threshold=-EQUIVALENCE_MARGIN,
                    threshold_high=EQUIVALENCE_MARGIN,
                    comparison="95% CI within [low, high]",
                    passed=equivalent,
                )
            )

    rows.extend(
        (
            _gate(
                "causal",
                "cvar_event_package_performance_gates_passed",
                value=int(causal_performance_passed),
                threshold=1,
                comparison="==",
                passed=causal_performance_passed,
            ),
            _gate(
                "causal",
                "cvar_event_package_supported",
                value=int(causal_package_supported),
                threshold=1,
                comparison="==",
                passed=causal_package_supported,
            ),
            _gate(
                "causal",
                "cvar_event_package_screen_passed_with_safety",
                value=int(causal_package_supported),
                threshold=1,
                comparison="==",
                passed=causal_package_supported,
            ),
        )
    )

    both_equivalent = bool(
        db500["crossed_ci_low"] >= -EQUIVALENCE_MARGIN
        and db500["crossed_ci_high"] <= EQUIVALENCE_MARGIN
        and db10000["crossed_ci_low"] >= -EQUIVALENCE_MARGIN
        and db10000["crossed_ci_high"] <= EQUIVALENCE_MARGIN
    )
    rows.append(
        _gate(
            "equivalence",
            "d_b_both_horizons_crossed_equivalence_pm0p5pp",
            value=int(both_equivalent),
            threshold=1,
            comparison="==",
            passed=both_equivalent,
        )
    )
    if not training_safety:
        interpretation = "hard_safety_or_provenance_gate_failed"
    elif causal_package_supported:
        interpretation = "cvar_event_package_supported"
    elif both_equivalent:
        interpretation = "no_material_cvar_event_package_effect_within_pm0p5pp"
    elif (
        db500["crossed_ci_high"] < 0.0
        or db10000["crossed_ci_high"] < -NONINFERIORITY_MARGIN
    ):
        interpretation = "cvar_event_package_harmful"
    elif (
        db500["point_difference"] > 0.0
        and db10000["point_difference"] < -NONINFERIORITY_MARGIN
    ):
        interpretation = "cvar_event_package_early_long_horizon_tradeoff"
    else:
        interpretation = "inconclusive"

    promotion_passed: bool | None = None
    if stage == "promotion":
        da500 = _effect(effects, "D", "A", 500)
        da10000 = _effect(effects, "D", "A", 10_000)
        promotion_components = {
            "d_a_h500_noninferiority": bool(
                da500["crossed_ci_low"] > -NONINFERIORITY_MARGIN
            ),
            "d_a_h500_seed_floor": bool(
                da500["minimum_seed_difference"] >= -0.01
            ),
            "d_b_h500_superiority": causal_h500,
            "d_b_h10000_noninferiority": causal_h10000,
            "d_a_h10000_materiality": bool(
                da10000["point_difference"] >= 0.005
            ),
            "d_a_h10000_seed_consistency": bool(
                da10000["positive_seed_count"] >= 2
            ),
        }
        promotion_passed = training_safety and all(promotion_components.values())
        rows.extend(
            (
                _gate(
                    "promotion",
                    "training_safety_for_promotion",
                    value=int(training_safety),
                    threshold=1,
                    comparison="==",
                    passed=training_safety,
                ),
                _gate(
                    "promotion",
                    "d_a_h500_crossed_noninferiority",
                    value=float(da500["crossed_ci_low"]),
                    threshold=-NONINFERIORITY_MARGIN,
                    comparison=">",
                    passed=promotion_components["d_a_h500_noninferiority"],
                ),
                _gate(
                    "promotion",
                    "d_a_h500_no_seed_below_minus_1pp",
                    value=float(da500["minimum_seed_difference"]),
                    threshold=-0.01,
                    comparison=">=",
                    passed=promotion_components["d_a_h500_seed_floor"],
                ),
                _gate(
                    "promotion",
                    "d_b_h500_point_positive_for_promotion",
                    value=float(db500["point_difference"]),
                    threshold=0.0,
                    comparison=">",
                    passed=causal_h500_point,
                ),
                _gate(
                    "promotion",
                    "d_b_h500_crossed_superiority_for_promotion",
                    value=float(db500["crossed_ci_low"]),
                    threshold=0.0,
                    comparison=">",
                    passed=causal_h500_crossed,
                ),
                _gate(
                    "promotion",
                    "d_b_h10000_crossed_noninferiority_for_promotion",
                    value=float(db10000["crossed_ci_low"]),
                    threshold=-NONINFERIORITY_MARGIN,
                    comparison=">",
                    passed=causal_h10000,
                ),
                _gate(
                    "promotion",
                    "d_a_h10000_material_gain",
                    value=float(da10000["point_difference"]),
                    threshold=0.005,
                    comparison=">=",
                    passed=promotion_components["d_a_h10000_materiality"],
                ),
                _gate(
                    "promotion",
                    "d_a_h10000_positive_in_two_seeds",
                    value=int(da10000["positive_seed_count"]),
                    threshold=2,
                    comparison=">=",
                    passed=promotion_components["d_a_h10000_seed_consistency"],
                ),
                _gate(
                    "promotion",
                    "all_promotion_gates_passed",
                    value=int(promotion_passed),
                    threshold=1,
                    comparison="==",
                    passed=promotion_passed,
                ),
            )
        )
    decision = {
        "stage": stage,
        "cvar_event_package_interpretation": interpretation,
        "training_safety_passed": training_safety,
        "cvar_event_package_performance_gates_passed": causal_performance_passed,
        "cvar_event_package_supported": causal_package_supported,
        "causal_screen_passed_with_safety": causal_package_supported,
        "d_b_both_horizons_equivalent_pm0p5pp": both_equivalent,
        "promotion_evaluated": stage == "promotion",
        "promotion_passed": promotion_passed,
        "equivalence_definition": (
            "the full crossed 95% CI lies within [-0.005, +0.005] "
            "on the success-probability scale"
        ),
    }
    return pd.DataFrame(rows), decision


def _root_arguments(parser: argparse.ArgumentParser, label: str) -> None:
    parser.add_argument(
        f"--{label}-root",
        type=Path,
        help=f"Shared A/B/D {label} root (arm-specific options override it).",
    )
    for arm in ARMS:
        parser.add_argument(
            f"--arm-{arm.lower()}-{label}-root",
            type=Path,
            help=f"Arm-specific {label} root for {arm}.",
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the staged preregistered Arm-D screen from paired MATLAB labels."
        )
    )
    _root_arguments(parser, "matlab")
    _root_arguments(parser, "training")
    parser.add_argument(
        "--scenario-csv",
        type=Path,
        default=(
            ROOT
            / "diagnostic_inputs"
            / "h10000_paired_96m_20260804"
            / "manifests"
            / "SCENARIO_MANIFEST.csv"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("causal", "promotion"),
        default="promotion",
        help=(
            "causal requires fresh B/D and evaluates D-B only; promotion requires "
            "A/B/D and also evaluates D-A promotion gates (default: promotion)."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arms = CAUSAL_ARMS if args.stage == "causal" else PROMOTION_ARMS
    comparisons = (
        CAUSAL_COMPARISONS if args.stage == "causal" else PROMOTION_COMPARISONS
    )

    matlab_roots = _roots(
        shared=args.matlab_root,
        overrides={
            "A": args.arm_a_matlab_root,
            "B": args.arm_b_matlab_root,
            "D": args.arm_d_matlab_root,
        },
        label="matlab",
        arms=arms,
    )
    training_roots = _roots(
        shared=args.training_root,
        overrides={
            "A": args.arm_a_training_root,
            "B": args.arm_b_training_root,
            "D": args.arm_d_training_root,
        },
        label="training",
        arms=arms,
    )

    scenario_path = args.scenario_csv.resolve()
    if sha256_file(scenario_path) != FORMAL_SCENARIO_SHA256:
        raise RuntimeError("scenario manifest is not the frozen 1,024-scenario panel")
    scenarios = pd.read_csv(scenario_path)
    if (
        len(scenarios) != 1024
        or scenarios["scenario_id"].nunique() != 1024
        or scenarios["scenario_uid"].nunique() != 1024
    ):
        raise RuntimeError("scenario manifest must contain 1,024 unique IDs and UIDs")
    scenario_eval_seeds = pd.to_numeric(
        scenarios["eval_seed"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.equal(scenario_eval_seeds, 1007.0).all():
        raise RuntimeError("scenario manifest is not the frozen eval seed 1007 panel")
    feasible_mask = _inertia_feasible(scenarios)
    if int(feasible_mask.sum()) != 400:
        raise RuntimeError(
            f"historical inertia-feasible scenario count is {int(feasible_mask.sum())}, "
            "expected 400"
        )

    metrics, formal_inputs, reset_errors = _load_formal_metrics(
        roots=matlab_roots,
        scenarios=scenarios,
        arms=arms,
    )
    scenario_uids = scenarios["scenario_uid"].astype(str).tolist()
    pair_effects, seed_effects, flips = _effect_tables(
        metrics,
        scenario_uids=scenario_uids,
        feasible_mask=feasible_mask,
        arms=arms,
        comparisons=comparisons,
    )
    training_safety, training_inputs, reset_pairing_valid = _load_training_safety(
        training_roots,
        arms=arms,
    )
    evaluation_linkage, evaluation_inputs = _validate_matlab_linkage(
        matlab_roots,
        training_safety,
        arms=arms,
    )
    feasible_uids = set(scenarios.loc[feasible_mask, "scenario_uid"].astype(str))
    performance = _performance_summary(metrics, feasible_uids=feasible_uids)
    gates, decision = _decision_gates(
        pair_effects,
        training_safety,
        reset_pairing_valid=reset_pairing_valid,
        stage=args.stage,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_dataframe(performance, output / "PERFORMANCE_SUMMARY.csv")
    atomic_write_dataframe(pair_effects, output / "PAIRWISE_EFFECTS.csv")
    atomic_write_dataframe(seed_effects, output / "SEED_EFFECTS.csv")
    atomic_write_dataframe(flips, output / "SUCCESS_FLIPS.csv")
    atomic_write_dataframe(training_safety, output / "TRAINING_SAFETY.csv")
    atomic_write_dataframe(evaluation_linkage, output / "EVALUATION_LINKAGE.csv")
    atomic_write_dataframe(gates, output / "DECISION_GATES.csv")
    atomic_write_json(decision, output / "DECISION.json")

    inputs = tuple(
        dict.fromkeys(
            (scenario_path, *formal_inputs, *training_inputs, *evaluation_inputs)
        )
    )
    provenance = artifact_fingerprint(
        inputs=inputs,
        parameters={
            "seeds": SEEDS,
            "stage": args.stage,
            "arms": arms,
            "comparisons": comparisons,
            "horizons": HORIZONS,
            "bootstrap_count": BOOTSTRAP_COUNT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "equivalence_margin": EQUIVALENCE_MARGIN,
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "inertia_feasible_count": int(feasible_mask.sum()),
            "reset_tolerance": RESET_TOLERANCE,
            "maximum_reset_errors": reset_errors,
            **decision,
        },
        code_paths=(
            Path(__file__),
            ROOT / "tools" / "validate_continuity_cadence_screen.py",
        ),
    )
    provenance["outputs"] = {
        path.name: sha256_file(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "RUN_PROVENANCE.json"
    }
    atomic_write_json(provenance, output / "RUN_PROVENANCE.json")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

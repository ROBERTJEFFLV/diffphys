from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "reports" / "continuity_cadence_9p6m_20260804"
MANIFEST_SCHEMA_VERSION = 2
INITIAL_CHECKPOINT_RELATIVE = Path(
    "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/"
    "checkpoints/model_update_2000.pt"
)
INITIAL_CHECKPOINT_SHA256 = (
    "b401dc6f02beadf51d1b55b24b9056f0b00f17a7a8a5d94d3675554fdb15370d"
)
ARM_CONFIGS = {
    arm: Path(f"configs/continuity_cadence_{arm}.args") for arm in "ABCD"
}
EXPECTED_CODE_FILES = (
    Path("tools/run_continuity_cadence_screen.py"),
    Path("train.py"),
    Path("training_objectives.py"),
    Path("training_schedule.py"),
    Path("model.py"),
    Path("env_l2f.py"),
    Path("policy_observation.py"),
    Path("retain_bank.py"),
    Path("l2f_cuda_backend.py"),
    Path("l2f_full_cuda_backend.py"),
    Path("cuda_ext/binding.cpp"),
    Path("cuda_ext/l2f_step_kernel.cu"),
    Path("cuda_ext/full_rollout_kernel.cu"),
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED = {
    "A": {"updates": 75, "resets": 75, "mid_h1000_updates": 0, "tail_blocks": 75},
    "B": {"updates": 67, "resets": 67, "mid_h1000_updates": 0, "tail_blocks": 67},
    "C": {"updates": 75, "resets": 67, "mid_h1000_updates": 8, "tail_blocks": 75},
    "D": {"updates": 67, "resets": 67, "mid_h1000_updates": 0, "tail_blocks": 75},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _expected_config_closure(arm: str) -> tuple[Path, ...]:
    records: list[Path] = []
    seen: set[Path] = set()

    def visit(relative: Path) -> None:
        normalized = Path(relative.as_posix())
        if normalized in seen:
            return
        source = ROOT / normalized
        if not source.is_file():
            raise FileNotFoundError(f"missing expected config source: {source}")
        seen.add(normalized)
        records.append(normalized)
        for raw_line in source.read_text(encoding="utf-8-sig").splitlines():
            for token in shlex.split(raw_line, comments=True, posix=True):
                if token.startswith("@"):
                    visit(Path(token[1:]))

    visit(ARM_CONFIGS[arm])
    return tuple(records)


def _validate_frozen_records(
    value: object,
    *,
    label: str,
    expected_paths: tuple[Path, ...],
) -> list[str]:
    errors: list[str] = []
    expected = tuple(path.as_posix() for path in expected_paths)
    if not isinstance(value, list) or not value:
        return [f"schema-v2 manifest {label} must be a non-empty list"]

    paths: list[str] = []
    hashes: dict[str, str] = {}
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            errors.append(f"schema-v2 manifest {label}[{index}] is not an object")
            continue
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not path:
            errors.append(f"schema-v2 manifest {label}[{index}] has no path")
            continue
        paths.append(path)
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            errors.append(
                f"schema-v2 manifest {label} has invalid SHA-256 for {path!r}"
            )
            continue
        hashes[path] = digest

    duplicate_paths = sorted({path for path in paths if paths.count(path) > 1})
    if duplicate_paths:
        errors.append(
            f"schema-v2 manifest {label} has duplicate paths: {duplicate_paths}"
        )
    missing = sorted(set(expected) - set(paths))
    unexpected = sorted(set(paths) - set(expected))
    if missing:
        errors.append(f"schema-v2 manifest {label} is missing: {missing}")
    if unexpected:
        errors.append(f"schema-v2 manifest {label} has unexpected paths: {unexpected}")
    if not missing and not unexpected and not duplicate_paths and tuple(paths) != expected:
        errors.append(f"schema-v2 manifest {label} is not in canonical order")

    for relative in expected:
        source = ROOT / relative
        if not source.is_file():
            errors.append(f"schema-v2 frozen {label} source is missing: {source}")
            continue
        if hashes.get(relative) != _sha256(source):
            errors.append(f"schema-v2 frozen {label} hash changed: {relative}")
    return errors


def _validate_run(run_dir: Path, arm: str) -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    log_path = run_dir / "train.csv"
    checkpoint_path = run_dir / "checkpoints" / "model.pt"
    reset_path = run_dir / "reset_samples.csv"
    for path in (log_path, checkpoint_path, reset_path, run_dir / "RUN_MANIFEST.json"):
        if not path.is_file():
            errors.append(f"missing {path}")
    if errors:
        return {"run_dir": str(run_dir), "complete": False}, errors

    rows = _read_rows(log_path)
    expected = EXPECTED[arm]
    manifest_path = run_dir / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("manifest_schema_version")
    manifest_expected = {
        "expected_segments": 150,
        "expected_reset_episodes": expected["resets"],
        "expected_optimizer_commits": expected["updates"],
        "expected_tail_supervision_blocks": expected["tail_blocks"],
        "physical_step_budget": 9_600_000,
        "segment_horizon": 250,
        "batch_size": 256,
    }
    for key, value in manifest_expected.items():
        if key == "expected_tail_supervision_blocks" and arm != "D" and key not in manifest:
            continue
        if manifest.get(key) != value:
            errors.append(f"manifest {key}={manifest.get(key)!r}, expected {value!r}")
    seed_name = run_dir.parent.name
    seed_value = seed_name[len("seed_"):] if seed_name.startswith("seed_") else seed_name
    if manifest.get("seed") != int(seed_value):
        errors.append("manifest seed does not match run directory")
    if manifest.get("arm") != arm:
        errors.append("manifest arm does not match run directory")
    expected_checkpoint = INITIAL_CHECKPOINT_RELATIVE.as_posix()
    if manifest.get("initial_checkpoint") != expected_checkpoint:
        errors.append(
            "manifest initial_checkpoint="
            f"{manifest.get('initial_checkpoint')!r}, expected {expected_checkpoint!r}"
        )
    if manifest.get("initial_checkpoint_sha256") != INITIAL_CHECKPOINT_SHA256:
        errors.append("manifest initial checkpoint SHA-256 is not the frozen value")
    initial_checkpoint = ROOT / INITIAL_CHECKPOINT_RELATIVE
    if not initial_checkpoint.is_file():
        errors.append(f"frozen initial checkpoint is missing: {initial_checkpoint}")
    elif _sha256(initial_checkpoint) != INITIAL_CHECKPOINT_SHA256:
        errors.append("frozen initial checkpoint file hash changed")

    if arm == "D" and schema_version != MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"arm D manifest schema is {schema_version!r}, expected "
            f"{MANIFEST_SCHEMA_VERSION}"
        )
    if schema_version == MANIFEST_SCHEMA_VERSION:
        errors.extend(
            _validate_frozen_records(
                manifest.get("code_files"),
                label="code_files",
                expected_paths=EXPECTED_CODE_FILES,
            )
        )
        errors.extend(
            _validate_frozen_records(
                manifest.get("config_closure"),
                label="config_closure",
                expected_paths=_expected_config_closure(arm),
            )
        )
        command = manifest.get("command")
        expected_config_token = f"@{ARM_CONFIGS[arm].as_posix()}"
        if not isinstance(command, list) or expected_config_token not in command:
            errors.append(
                "schema-v2 manifest command does not select "
                f"{expected_config_token}"
            )
        command_path = run_dir / "command.txt"
        if not command_path.is_file() or not command_path.read_text(
            encoding="utf-8"
        ).strip():
            errors.append("schema-v2 run lacks a non-empty command.txt")
    if len(rows) != 150:
        errors.append(f"train rows={len(rows)}, expected 150")
    final_physical_steps = _as_int(rows[-1], "physical_steps") if rows else -1
    final_updates = _as_int(rows[-1], "optimizer_update") if rows else -1
    update_rows = sum(_as_int(row, "update_applied") for row in rows)
    reset_boundaries = sum(_as_int(row, "reset_episode_boundary") for row in rows)
    optimization_boundaries = sum(
        _as_int(row, "optimization_block_boundary") for row in rows
    )
    has_tail_cadence_columns = bool(rows) and {
        "tail_supervision_block_horizon",
        "tail_supervision_block_boundary",
        "first_tail_supervision_segment",
    }.issubset(rows[0])
    if has_tail_cadence_columns:
        tail_supervision_boundaries = sum(
            _as_int(row, "tail_supervision_block_boundary") for row in rows
        )
        tail_supervision_starts = sum(
            _as_int(row, "first_tail_supervision_segment") for row in rows
        )
    else:
        # Historical A/B/C logs predate the explicit cadence columns and used
        # optimizer-coupled CVaR blocks by construction.
        tail_supervision_boundaries = optimization_boundaries
        tail_supervision_starts = optimization_boundaries
    mid_h1000 = [
        row
        for row in rows
        if _as_int(row, "episode_target_steps") == 1000
        and _as_int(row, "optimization_block_boundary") == 1
        and _as_int(row, "reset_episode_boundary") == 0
    ]
    continuation_rows = [
        rows[index + 1]
        for index, row in enumerate(rows[:-1])
        if _as_int(row, "episode_target_steps") == 1000
        and _as_int(row, "optimization_block_boundary") == 1
        and _as_int(row, "reset_episode_boundary") == 0
    ]
    invalid_rollouts = sum(1 for row in rows if _as_int(row, "rollout_valid") == 0)
    rejected_updates = sum(
        1
        for row in rows
        if row["skip_reason"]
        in {
            "adaptive_gate_suspicious",
            "adaptive_gate_hard_grad",
            "grad_skip_threshold",
            "post_update_rejected",
            "grad_norm_nonfinite",
            "grad_tensor_nonfinite",
        }
    )

    if final_physical_steps != 9_600_000:
        errors.append(f"physical_steps={final_physical_steps}, expected 9600000")
    if final_updates != expected["updates"] or update_rows != expected["updates"]:
        errors.append(
            f"updates final/rows={final_updates}/{update_rows}, expected {expected['updates']}"
        )
    if reset_boundaries != expected["resets"]:
        errors.append(f"reset boundaries={reset_boundaries}, expected {expected['resets']}")
    if optimization_boundaries != expected["updates"]:
        errors.append(
            f"optimization boundaries={optimization_boundaries}, expected {expected['updates']}"
        )
    if (
        tail_supervision_boundaries != expected["tail_blocks"]
        or tail_supervision_starts != expected["tail_blocks"]
    ):
        errors.append(
            "tail supervision starts/boundaries="
            f"{tail_supervision_starts}/{tail_supervision_boundaries}, "
            f"expected {expected['tail_blocks']}"
        )
    if arm == "D":
        if not has_tail_cadence_columns:
            errors.append("arm D log lacks explicit tail-supervision cadence columns")
        elif any(_as_int(row, "tail_supervision_block_horizon") != 500 for row in rows):
            errors.append("arm D tail-supervision block horizon is not always H500")
    if len(mid_h1000) != expected["mid_h1000_updates"]:
        errors.append(
            f"mid-H1000 boundaries={len(mid_h1000)}, expected {expected['mid_h1000_updates']}"
        )
    if arm == "C" and any(_as_int(row, "force_reset_next") for row in mid_h1000):
        errors.append("arm C forced a reset after a mid-H1000 optimizer commit")
    if arm == "C":
        if len(continuation_rows) != expected["mid_h1000_updates"]:
            errors.append(
                f"mid-H1000 continuation rows={len(continuation_rows)}, "
                f"expected {expected['mid_h1000_updates']}"
            )
        for row in continuation_rows:
            if float(row["reset_mask"]) != 0.0:
                errors.append("arm C reset at the segment after a mid-H1000 update")
                break
            if float(row["hidden_state_norm_initial"]) <= 0.0:
                errors.append("arm C did not carry hidden state after a mid-H1000 update")
                break
            if float(row["integral_state_norm_initial"]) <= 0.0:
                errors.append("arm C did not carry integral state after a mid-H1000 update")
                break
    if invalid_rollouts:
        errors.append(f"non-finite/invalid rollout rows={invalid_rollouts}")
    if rejected_updates:
        errors.append(f"rejected/gradient-skip rows={rejected_updates}")

    return (
        {
            "run_dir": str(run_dir),
            "complete": not errors,
            "rows": len(rows),
            "physical_steps": final_physical_steps,
            "updates": final_updates,
            "reset_boundaries": reset_boundaries,
            "optimization_boundaries": optimization_boundaries,
            "tail_supervision_starts": tail_supervision_starts,
            "tail_supervision_boundaries": tail_supervision_boundaries,
            "mid_h1000_updates": len(mid_h1000),
            "invalid_rollouts": invalid_rollouts,
            "rejected_updates": rejected_updates,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "reset_samples_sha256": _sha256(reset_path),
        },
        errors,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--arms", default="A,B,C")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    arms = tuple(part.strip().upper() for part in args.arms.split(",") if part.strip())
    unknown = tuple(arm for arm in arms if arm not in EXPECTED)
    if not arms or unknown:
        raise ValueError(f"arms must be drawn from A,B,C,D; unknown={unknown}")
    summaries: list[dict[str, object]] = []
    failures: list[str] = []
    for seed in seeds:
        reset_hashes: dict[str, str] = {}
        for arm in arms:
            run_dir = args.root.resolve() / f"seed_{seed}" / f"arm_{arm}"
            summary, errors = _validate_run(run_dir, arm)
            summary.update({"seed": seed, "arm": arm})
            summaries.append(summary)
            if "reset_samples_sha256" in summary:
                reset_hashes[arm] = str(summary["reset_samples_sha256"])
            if errors and not (args.allow_partial and not summary.get("complete")):
                failures.extend(f"seed={seed} arm={arm}: {error}" for error in errors)
        if reset_hashes.get("B") and reset_hashes.get("C"):
            if reset_hashes["B"] != reset_hashes["C"]:
                failures.append(f"seed={seed}: B/C reset sample hashes differ")
        if reset_hashes.get("B") and reset_hashes.get("D"):
            if reset_hashes["B"] != reset_hashes["D"]:
                failures.append(f"seed={seed}: B/D reset sample hashes differ")

    print(json.dumps(summaries, indent=2, sort_keys=True))
    for failure in failures:
        print(f"ERROR: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "reports/continuity_cadence_9p6m_20260804"
DEFAULT_OUTPUT = ROOT / "reports/continuity_cadence_mechanism_audit_20260806"

FIRST_METRICS = (
    "early_position_cvar",
    "early_omega_cvar",
    "position",
    "velocity",
    "omega",
    "omega_decay_active_fraction",
)
BOUNDARY_METRICS = (
    "final_position_cvar",
    "final_omega_cvar",
    "grad_norm_fp64_before_clip",
    "max_abs_param_delta",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(row: dict[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name} at train step {row.get('step')}")
    return value


def extract_optimizer_blocks(rows: list[dict[str, str]], *, seed: int) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    current: list[dict[str, str]] = []
    for row in rows:
        if int(row["first_optimization_segment"]) == 1:
            if current:
                raise ValueError("new optimizer block started before the previous boundary")
            current = [row]
        else:
            if not current:
                raise ValueError("optimizer block has no first segment")
            current.append(row)
        if int(row["optimization_block_boundary"]) != 1:
            continue
        first = current[0]
        boundary = current[-1]
        block_type = "fresh_reset" if float(first["reset_mask"]) > 0.5 else "continuation"
        record: dict[str, object] = {
            "seed": seed,
            "block_index": len(blocks),
            "block_type": block_type,
            "episode_target_steps": int(first["episode_target_steps"]),
            "segment_count": len(current),
            "first_train_step": int(first["step"]),
            "boundary_train_step": int(boundary["step"]),
            "update_applied": int(boundary["update_applied"]),
        }
        record.update({f"first_{name}": _number(first, name) for name in FIRST_METRICS})
        record.update({f"boundary_{name}": _number(boundary, name) for name in BOUNDARY_METRICS})
        blocks.append(record)
        current = []
    if current:
        raise ValueError("training log ends with an incomplete optimizer block")
    return blocks


def _read_blocks(path: Path, *, seed: int) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return extract_optimizer_blocks(rows, seed=seed)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare effective Arm-C signals in fresh-reset and continuation H500 blocks."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    logs = sorted(run_root.glob("seed_*/arm_C/train.csv"))
    if len(logs) != 3:
        raise ValueError(f"expected exactly three Arm-C logs, found {len(logs)}")

    blocks: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    for path in logs:
        seed_name = path.parents[1].name
        seed_value = seed_name[len("seed_"):] if seed_name.startswith("seed_") else seed_name
        seed = int(seed_value)
        seed_blocks = _read_blocks(path, seed=seed)
        if len(seed_blocks) != 75:
            raise ValueError(f"seed {seed} has {len(seed_blocks)} optimizer blocks, expected 75")
        blocks.extend(seed_blocks)
        inputs.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
                "blocks": len(seed_blocks),
            }
        )

    counts = {
        kind: sum(record["block_type"] == kind for record in blocks)
        for kind in ("fresh_reset", "continuation")
    }
    if counts != {"fresh_reset": 201, "continuation": 24}:
        raise ValueError(f"unexpected block classification: {counts}")
    if any(int(record["update_applied"]) != 1 for record in blocks):
        raise ValueError("an optimizer boundary did not apply its update")

    numeric_metrics = tuple(
        name
        for name in blocks[0]
        if name.startswith("first_") or name.startswith("boundary_")
        if name not in {"first_train_step", "boundary_train_step"}
    )
    summary_rows: list[dict[str, object]] = []
    for metric in numeric_metrics:
        fresh = [float(record[metric]) for record in blocks if record["block_type"] == "fresh_reset"]
        continuation = [
            float(record[metric]) for record in blocks if record["block_type"] == "continuation"
        ]
        fresh_mean = fmean(fresh)
        continuation_mean = fmean(continuation)
        summary_rows.append(
            {
                "metric": metric,
                "fresh_reset_count": len(fresh),
                "fresh_reset_mean": fresh_mean,
                "continuation_count": len(continuation),
                "continuation_mean": continuation_mean,
                "continuation_over_fresh": (
                    continuation_mean / fresh_mean if fresh_mean != 0.0 else float("nan")
                ),
            }
        )

    h1000_pairs: list[dict[str, object]] = []
    for seed in (7, 17, 27):
        seed_blocks = sorted(
            (record for record in blocks if int(record["seed"]) == seed),
            key=lambda record: int(record["block_index"]),
        )
        pair_index = 0
        for index, fresh in enumerate(seed_blocks[:-1]):
            if (
                fresh["block_type"] != "fresh_reset"
                or int(fresh["episode_target_steps"]) != 1000
            ):
                continue
            continuation = seed_blocks[index + 1]
            if (
                continuation["block_type"] != "continuation"
                or int(continuation["episode_target_steps"]) != 1000
            ):
                raise ValueError(
                    f"seed {seed} H1000 fresh block is not followed by its continuation"
                )
            h1000_pairs.append(
                {
                    "seed": seed,
                    "pair_index": pair_index,
                    "h250_early_position_cvar": fresh["first_early_position_cvar"],
                    "h250_early_omega_cvar": fresh["first_early_omega_cvar"],
                    "h250_dense_position": fresh["first_position"],
                    "h250_dense_velocity": fresh["first_velocity"],
                    "h250_dense_omega": fresh["first_omega"],
                    "h500_final_position_cvar": fresh["boundary_final_position_cvar"],
                    "h500_final_omega_cvar": fresh["boundary_final_omega_cvar"],
                    "h500_commit_grad_norm": fresh["boundary_grad_norm_fp64_before_clip"],
                    "h500_commit_max_param_delta": fresh["boundary_max_abs_param_delta"],
                    "h750_early_position_cvar": continuation["first_early_position_cvar"],
                    "h750_early_omega_cvar": continuation["first_early_omega_cvar"],
                    "h750_dense_position": continuation["first_position"],
                    "h750_dense_velocity": continuation["first_velocity"],
                    "h750_dense_omega": continuation["first_omega"],
                    "h1000_final_position_cvar": continuation["boundary_final_position_cvar"],
                    "h1000_final_omega_cvar": continuation["boundary_final_omega_cvar"],
                    "h1000_commit_grad_norm": continuation["boundary_grad_norm_fp64_before_clip"],
                    "h1000_commit_max_param_delta": continuation["boundary_max_abs_param_delta"],
                }
            )
            pair_index += 1
    if len(h1000_pairs) != 24:
        raise ValueError(f"expected 24 paired H1000 episodes, found {len(h1000_pairs)}")

    h1000_summary = [
        {
            "metric": metric,
            "episode_count": len(h1000_pairs),
            "mean": fmean(float(record[metric]) for record in h1000_pairs),
        }
        for metric in h1000_pairs[0]
        if metric not in {"seed", "pair_index"}
    ]

    _write_csv(output_dir / "OPTIMIZER_BLOCKS.csv", blocks)
    _write_csv(output_dir / "EFFECTIVE_SIGNAL_SUMMARY.csv", summary_rows)
    _write_csv(output_dir / "H1000_EVENT_PAIRS.csv", h1000_pairs)
    _write_csv(output_dir / "H1000_EVENT_SUMMARY.csv", h1000_summary)
    provenance = {
        "analysis": "Arm C fresh-reset versus mid-H1000 continuation blocks",
        "input_root": run_root.relative_to(ROOT).as_posix(),
        "input_logs": inputs,
        "block_counts": counts,
        "tool": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "tool_sha256": _sha256(Path(__file__)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "RUN_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), **counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

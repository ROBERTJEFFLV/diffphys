from __future__ import annotations

import argparse
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_matlab_position_hold_eval import _complete, _evaluate, _export, _write_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_ROOT = ROOT / "reports" / "continuity_cadence_9p6m_20260804"
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "continuity_cadence_matlab_20260804"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _csv_arms(value: str) -> tuple[str, ...]:
    arms = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if not arms or any(arm not in {"A", "B", "C", "D"} for arm in arms):
        raise argparse.ArgumentTypeError("arms must be drawn from A,B,C,D")
    return arms


def _job(
    *,
    seed: int,
    arm: str,
    checkpoint: Path,
    output: Path,
    batch_size: int,
    eval_seed: int,
    horizon: int,
) -> dict[str, object]:
    label = f"seed_{seed}_arm_{arm}_physical_steps_9600000"
    return {
        "label": label,
        "checkpoint_path": str(checkpoint.resolve()),
        "weights_path": str((output / "model.mat").resolve()),
        "output_path": str((output / "summary.csv").resolve()),
        "sample_output_path": str((output / "samples.csv").resolve()),
        "mat_output_path": str((output / "metrics.mat").resolve()),
        "batch_size": batch_size,
        "eval_seed": eval_seed,
        "horizon": horizon,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export and formally evaluate the 9-run continuity/cadence screen."
    )
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", type=_csv_ints, default=(7, 17, 27))
    parser.add_argument("--arms", type=_csv_arms, default=("A", "B", "C"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-seed", type=int, default=1007)
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--phase", choices=("export", "eval", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.horizon < 500 or args.workers <= 0:
        raise ValueError("batch size/workers must be positive and horizon at least 500")

    training_root = args.training_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for arm in args.arms:
            checkpoint = training_root / f"seed_{seed}" / f"arm_{arm}" / "checkpoints" / "model.pt"
            if not checkpoint.is_file() and not args.dry_run:
                raise FileNotFoundError(checkpoint)
            rows.append(
                _job(
                    seed=seed,
                    arm=arm,
                    checkpoint=checkpoint,
                    output=output_root / f"seed_{seed}" / f"arm_{arm}",
                    batch_size=args.batch_size,
                    eval_seed=args.eval_seed,
                    horizon=args.horizon,
                )
            )

    if args.phase in {"export", "all"}:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(_export, row, args.force, args.dry_run) for row in rows
            ]
            for future in futures:
                future.result()
    _write_manifest(output_root / "eval_manifest.csv", rows)

    if args.phase in {"eval", "all"}:
        pending = [row for row in rows if args.force or not _complete(row)]
        manifests: list[Path] = []
        for row in pending:
            manifest = output_root / "job_manifests" / f"{row['label']}.csv"
            _write_manifest(manifest, [row])
            manifests.append(manifest)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(_evaluate, manifest, args.force, args.dry_run)
                for manifest in manifests
            ]
            for future in futures:
                future.result()

    if not args.dry_run:
        checkpoint_rows: list[dict[str, object]] = []
        for row in rows:
            checkpoint = Path(str(row["checkpoint_path"]))
            model_mat = Path(str(row["weights_path"]))
            checkpoint_rows.append(
                {
                    "label": row["label"],
                    "seed": int(str(row["label"]).split("_")[1]),
                    "arm": str(row["label"]).split("_")[3],
                    "physical_steps": 9_600_000,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "model_mat_path": str(model_mat),
                    "model_mat_sha256": _sha256(model_mat),
                    "sample_output_path": row["sample_output_path"],
                    "eval_seed": args.eval_seed,
                    "horizon": args.horizon,
                }
            )
        with (output_root / "CHECKPOINTS.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(checkpoint_rows[0]))
            writer.writeheader()
            writer.writerows(checkpoint_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

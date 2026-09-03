from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _label(multiplier: float) -> str:
    return f"scale_{multiplier:g}".replace(".", "p")


def _write_manifest(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)


def _export(checkpoint: Path, output: Path, multiplier: float, force: bool) -> None:
    if output.exists() and not force:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "tools/export_motor_gru_to_mat.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--integral-input-multiplier",
            str(multiplier),
        ],
        cwd=ROOT,
        check=True,
    )


def _evaluate(manifest: Path, force: bool) -> None:
    matlab_root = (ROOT / "matlab_l2f").as_posix().replace("'", "''")
    manifest_value = manifest.as_posix().replace("'", "''")
    expression = (
        "restoredefaultpath; rehash toolboxcache; "
        f"addpath('{matlab_root}'); "
        f"run_motor_gru_eval_manifest('{manifest_value}', 'force', {str(force).lower()});"
    )
    subprocess.run(["matlab", "-batch", expression], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no-training integral multiplier sensitivity.")
    parser.add_argument(
        "--checkpoint",
        default=(
            "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/"
            "checkpoints/model_update_2000.pt"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="reports/integral_scale_sensitivity_q2_matlab",
    )
    parser.add_argument("--multipliers", default="0,0.5,1,2,4")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-seed", type=int, default=1007)
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoint = (ROOT / args.checkpoint).resolve()
    output_root = (ROOT / args.output_root).resolve()
    multipliers = tuple(float(value) for value in args.multipliers.split(","))
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if any(value < 0.0 for value in multipliers):
        raise ValueError("multipliers must be non-negative")
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    jobs: list[tuple[Path, dict[str, object]]] = []
    for multiplier in multipliers:
        label = _label(multiplier)
        destination = output_root / label
        model_path = destination / "model.mat"
        _export(checkpoint, model_path, multiplier, args.force)
        row = {
            "label": label,
            "checkpoint_path": str(checkpoint),
            "weights_path": str(model_path),
            "output_path": str(destination / "summary.csv"),
            "sample_output_path": str(destination / "samples.csv"),
            "mat_output_path": str(destination / "metrics.mat"),
            "batch_size": args.batch_size,
            "eval_seed": args.eval_seed,
            "horizon": args.horizon,
        }
        manifest = output_root / "job_manifests" / f"{label}.csv"
        _write_manifest(manifest, row)
        complete = all(
            Path(str(row[key])).exists()
            for key in ("output_path", "sample_output_path", "mat_output_path")
        )
        if args.force or not complete:
            jobs.append((manifest, row))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_evaluate, manifest, args.force) for manifest, _ in jobs]
        for future in futures:
            future.result()

    marker = output_root / "SENSITIVITY_COMPLETE.txt"
    marker.write_text("complete\n", encoding="utf-8")
    print(marker)


if __name__ == "__main__":
    main()

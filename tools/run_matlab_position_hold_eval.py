from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CHECKPOINT = REPO_ROOT / "reports/mainline_h500_finetune_safe_5000_lr5e6/model_step_4000.pt"


def _items(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and run paired MATLAB position-hold evaluation.")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--checkpoint-step", type=int, default=500)
    parser.add_argument(
        "--checkpoint-update",
        type=int,
        default=0,
        help="Use model_update_N.pt and update_N output folders; zero keeps legacy step checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-physical-step",
        type=int,
        default=0,
        help="Use model_physical_steps_N.pt and physical_steps_N output folders.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-seed", type=int, default=1007)
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument(
        "--baseline-checkpoint",
        default=str(BASELINE_CHECKPOINT.relative_to(REPO_ROOT)),
        help="Reference checkpoint used by --include-baseline.",
    )
    parser.add_argument("--phase", choices=("export", "eval", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.groups = _items(args.groups)
    args.seeds = [int(value) for value in _items(args.seeds)]
    if args.horizon < 500:
        raise ValueError("formal position-hold evaluation requires horizon >= 500")
    if args.workers <= 0 or args.batch_size <= 0:
        raise ValueError("--workers and --batch-size must be positive")
    if args.checkpoint_update > 0 and args.checkpoint_physical_step > 0:
        raise ValueError("choose either update or physical-step checkpoint addressing")
    return args


def _job(label: str, checkpoint: Path, root: Path, args: argparse.Namespace) -> dict[str, object]:
    return {
        "label": label,
        "checkpoint_path": str(checkpoint.resolve()),
        "weights_path": str((root / "model.mat").resolve()),
        "output_path": str((root / "summary.csv").resolve()),
        "sample_output_path": str((root / "samples.csv").resolve()),
        "mat_output_path": str((root / "metrics.mat").resolve()),
        "batch_size": args.batch_size,
        "eval_seed": args.eval_seed,
        "horizon": args.horizon,
    }


def jobs(args: argparse.Namespace, experiment_root: Path, output_root: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    if args.include_baseline:
        baseline_checkpoint = Path(args.baseline_checkpoint)
        if not baseline_checkpoint.is_absolute():
            baseline_checkpoint = REPO_ROOT / baseline_checkpoint
        result.append(_job("baseline", baseline_checkpoint, output_root / "baseline", args))
    for seed in args.seeds:
        for group in args.groups:
            if args.checkpoint_physical_step > 0:
                checkpoint_name = f"model_physical_steps_{args.checkpoint_physical_step}.pt"
                checkpoint_label = f"physical_steps_{args.checkpoint_physical_step}"
            elif args.checkpoint_update > 0:
                checkpoint_name = f"model_update_{args.checkpoint_update}.pt"
                checkpoint_label = f"update_{args.checkpoint_update}"
            else:
                checkpoint_name = f"model_step_{args.checkpoint_step}.pt"
                checkpoint_label = f"step_{args.checkpoint_step}"
            checkpoint = (
                experiment_root / f"seed_{seed}" / f"group_{group}" / "checkpoints"
                / checkpoint_name
            )
            result.append(
                _job(
                    f"seed_{seed}_group_{group}_{checkpoint_label}",
                    checkpoint,
                    output_root / f"seed_{seed}" / f"group_{group}" / checkpoint_label,
                    args,
                )
            )
    return result


def _complete(job: dict[str, object]) -> bool:
    return all(Path(str(job[key])).exists() for key in ("output_path", "sample_output_path", "mat_output_path"))


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _export(job: dict[str, object], force: bool, dry_run: bool) -> None:
    checkpoint = Path(str(job["checkpoint_path"]))
    output = Path(str(job["weights_path"]))
    if not checkpoint.exists() and not dry_run:
        raise FileNotFoundError(checkpoint)
    if output.exists() and not force:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "tools/export_motor_gru_to_mat.py", "--checkpoint", str(checkpoint), "--output", str(output)]
    print(subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def _evaluate(manifest: Path, force: bool, dry_run: bool) -> None:
    matlab_root = (REPO_ROOT / "matlab_l2f").as_posix().replace("'", "''")
    manifest_value = manifest.as_posix().replace("'", "''")
    expression = (
        "restoredefaultpath; rehash toolboxcache; "
        f"addpath('{matlab_root}'); "
        f"run_motor_gru_eval_manifest('{manifest_value}', 'force', {str(force).lower()});"
    )
    command = ["matlab", "-batch", expression]
    print(subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    experiment_root = (REPO_ROOT / args.experiment_root).resolve()
    output_root = (REPO_ROOT / args.output_root).resolve()
    rows = jobs(args, experiment_root, output_root)
    if args.phase in {"export", "all"}:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_export, row, args.force, args.dry_run) for row in rows]
            for future in futures:
                future.result()
    if args.phase in {"eval", "all"}:
        pending = [row for row in rows if args.force or not _complete(row)]
        manifests: list[Path] = []
        for row in pending:
            manifest = output_root / "job_manifests" / f"{row['label']}.csv"
            _write_manifest(manifest, [row])
            manifests.append(manifest)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_evaluate, manifest, args.force, args.dry_run) for manifest in manifests]
            for future in futures:
                future.result()
    _write_manifest(output_root / "eval_manifest.csv", rows)


if __name__ == "__main__":
    main()

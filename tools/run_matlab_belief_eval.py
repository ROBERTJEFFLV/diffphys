from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "reports/formal_belief_ablation_h500_s500_gpu2"
BASELINE_CHECKPOINT = (
    REPO_ROOT / "reports/mainline_h500_finetune_safe_5000_lr5e6/model_step_4000.pt"
)
GROUPS = ("A", "B", "C", "D", "E")
TRAINING_SEEDS = (7, 17, 27, 37, 47)
CHECKPOINT_STEPS = (100, 200, 300, 400, 500)


def _int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _str_list(value: str) -> list[str]:
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and evaluate the paired A-E checkpoints with MATLAB streaming."
    )
    parser.add_argument("--phase", choices=("export", "eval", "all"), default="all")
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--seeds", default=",".join(str(value) for value in TRAINING_SEEDS))
    parser.add_argument(
        "--checkpoint-steps", default=",".join(str(value) for value in CHECKPOINT_STEPS)
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-seed", type=int, default=1007)
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fresh-process-per-job", action="store_true")
    parser.add_argument(
        "--output-root",
        default=str(EXPERIMENT_ROOT / "matlab_eval_streaming"),
    )
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.groups = _str_list(args.groups)
    args.seeds = _int_list(args.seeds)
    args.checkpoint_steps = _int_list(args.checkpoint_steps)
    invalid_groups = sorted(set(args.groups) - set(GROUPS))
    if invalid_groups:
        raise ValueError(f"unknown groups: {','.join(invalid_groups)}")
    if args.batch_size <= 0 or args.horizon < 500 or args.workers <= 0:
        raise ValueError("batch size/workers must be positive and horizon must be at least 500")
    return args


def _jobs(args: argparse.Namespace, output_root: Path) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    if not args.no_baseline:
        root = output_root / "baseline" / "step_0"
        jobs.append(
            _job_row(
                label="baseline_step_0",
                checkpoint=BASELINE_CHECKPOINT,
                root=root,
                batch_size=args.batch_size,
                eval_seed=args.eval_seed,
                horizon=args.horizon,
            )
        )
    for seed in args.seeds:
        for group in args.groups:
            for step in args.checkpoint_steps:
                checkpoint = (
                    EXPERIMENT_ROOT
                    / f"seed_{seed}"
                    / f"group_{group}"
                    / "checkpoints"
                    / f"model_step_{step}.pt"
                )
                root = output_root / f"seed_{seed}" / f"group_{group}" / f"step_{step}"
                jobs.append(
                    _job_row(
                        label=f"seed_{seed}_group_{group}_step_{step}",
                        checkpoint=checkpoint,
                        root=root,
                        batch_size=args.batch_size,
                        eval_seed=args.eval_seed,
                        horizon=args.horizon,
                    )
                )
    return jobs


def _job_row(
    *,
    label: str,
    checkpoint: Path,
    root: Path,
    batch_size: int,
    eval_seed: int,
    horizon: int,
) -> dict[str, object]:
    return {
        "label": label,
        "checkpoint_path": str(checkpoint.resolve()),
        "weights_path": str((root / "model.mat").resolve()),
        "output_path": str((root / "summary.csv").resolve()),
        "sample_output_path": str((root / "samples.csv").resolve()),
        "mat_output_path": str((root / "metrics.mat").resolve()),
        "batch_size": batch_size,
        "eval_seed": eval_seed,
        "horizon": horizon,
    }


def export_weights(jobs: list[dict[str, object]], *, force: bool, dry_run: bool) -> None:
    for index, job in enumerate(jobs, start=1):
        checkpoint = Path(str(job["checkpoint_path"]))
        output = Path(str(job["weights_path"]))
        if not checkpoint.exists() and not dry_run:
            raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
        if output.exists() and not force:
            print(f"[{index}/{len(jobs)}] skip exported: {job['label']}")
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "tools/export_motor_gru_to_mat.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ]
        print(subprocess.list2cmdline(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


def write_manifest(jobs: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(jobs[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(jobs)


def _job_complete(job: dict[str, object]) -> bool:
    return all(
        Path(str(job[name])).exists()
        for name in ("output_path", "sample_output_path", "mat_output_path")
    )


def _matlab_command(args: argparse.Namespace, manifest: Path) -> list[str]:
    matlab_root = (REPO_ROOT / "matlab_l2f").as_posix()
    manifest_value = manifest.as_posix().replace("'", "''")
    expression = (
        "restoredefaultpath; rehash toolboxcache; "
        f"addpath('{matlab_root}'); "
        f"run_motor_gru_eval_manifest('{manifest_value}', 'force', {str(args.force).lower()});"
    )
    return ["matlab", "-batch", expression]


def run_matlab(args: argparse.Namespace, manifests: list[Path]) -> None:
    commands = [_matlab_command(args, manifest) for manifest in manifests]
    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return
    if len(commands) == 1:
        subprocess.run(commands[0], cwd=REPO_ROOT, check=True)
        return
    with ThreadPoolExecutor(max_workers=min(args.workers, len(commands))) as executor:
        futures = [
            executor.submit(subprocess.run, command, cwd=REPO_ROOT, check=True)
            for command in commands
        ]
        for future in futures:
            future.result()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    jobs = _jobs(args, output_root)
    manifest = output_root / "eval_manifest.csv"
    if args.phase in {"export", "all"}:
        export_weights(jobs, force=args.force, dry_run=args.dry_run)
    write_manifest(jobs, manifest)
    print(f"wrote {len(jobs)} jobs: {manifest}")
    if args.phase in {"eval", "all"}:
        pending_jobs = [job for job in jobs if args.force or not _job_complete(job)]
        print(
            f"evaluation jobs complete={len(jobs) - len(pending_jobs)} "
            f"pending={len(pending_jobs)}",
            flush=True,
        )
        if not pending_jobs:
            return
        if args.fresh_process_per_job:
            manifest_dir = output_root / "job_manifests"
            manifests = []
            for job in pending_jobs:
                job_manifest = manifest_dir / f"{job['label']}.csv"
                write_manifest([job], job_manifest)
                manifests.append(job_manifest)
        else:
            worker_count = min(args.workers, len(pending_jobs))
            if worker_count == 1:
                manifests = [manifest]
            else:
                partitions = [pending_jobs[index::worker_count] for index in range(worker_count)]
                manifests = []
                for index, partition in enumerate(partitions):
                    worker_manifest = output_root / f"eval_manifest_worker_{index}.csv"
                    write_manifest(partition, worker_manifest)
                    manifests.append(worker_manifest)
        run_matlab(args, manifests)


if __name__ == "__main__":
    main()

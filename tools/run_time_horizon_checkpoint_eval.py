from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("T0", "T1", "T2", "T3")
PHYSICAL_STEPS = (32, 64, 96, 128, 192, 256)
BASELINE_SOURCE = ROOT / "reports/integral_scale_h500_u1000_matlab/baseline"


def _copy_baseline(output_root: Path) -> None:
    destination = output_root / "baseline"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("model.mat", "summary.csv", "samples.csv", "metrics.mat"):
        source = BASELINE_SOURCE / name
        if not source.exists():
            raise FileNotFoundError(source)
        target = destination / name
        if not target.exists():
            shutil.copy2(source, target)


def _checkpoint(training_root: Path, group: str, physical_steps: int) -> Path:
    return (
        training_root
        / "seed_7"
        / f"group_{group}"
        / "checkpoints"
        / f"model_physical_steps_{physical_steps}.pt"
    )


def _complete(output_root: Path, group: str, physical_steps: int) -> bool:
    destination = (
        output_root
        / "seed_7"
        / f"group_{group}"
        / f"physical_steps_{physical_steps}"
    )
    return all((destination / name).exists() for name in ("summary.csv", "samples.csv", "metrics.mat"))


def _evaluate_one(
    training_root: Path,
    output_root: Path,
    group: str,
    physical_steps: int,
) -> None:
    command = [
        sys.executable,
        "tools/run_matlab_position_hold_eval.py",
        "--experiment-root",
        str(training_root.relative_to(ROOT)),
        "--output-root",
        str(output_root.relative_to(ROOT)),
        "--groups",
        group,
        "--seeds",
        "7",
        "--checkpoint-physical-step",
        str(physical_steps),
        "--batch-size",
        "1024",
        "--eval-seed",
        "1007",
        "--horizon",
        "10000",
        "--workers",
        "1",
    ]
    log_root = output_root / "job_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    label = f"{group}_physical_steps_{physical_steps}"
    with (log_root / f"{label}.stdout.log").open("w", encoding="utf-8") as stdout, (
        log_root / f"{label}.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=True)


def _summarize(training_root: Path, output_root: Path, physical_steps: int) -> None:
    label = f"physical_steps_{physical_steps}"
    subprocess.run(
        [
            sys.executable,
            "tools/summarize_q_residual_eval.py",
            "--eval-root",
            str(output_root.relative_to(ROOT)),
            "--training-root",
            str(training_root.relative_to(ROOT)),
            "--groups",
            ",".join(GROUPS),
            "--seed",
            "7",
            "--checkpoint-label",
            label,
            "--output-subdir",
            f"aggregate_{label}",
            "--baseline-label",
            "q2",
            "--secondary-baseline-samples",
            "reports/q_residual_h500_u2000_matlab/baseline/samples.csv",
            "--secondary-baseline-label",
            "p4b",
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate T0-T3 physical checkpoints as soon as training writes them."
    )
    parser.add_argument("--training-root", default="reports/time_horizon_q2_256m_gpu2")
    parser.add_argument("--output-root", default="reports/time_horizon_q2_256m_matlab")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    training_root = (ROOT / args.training_root).resolve()
    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_baseline(output_root)

    pending = [
        (group, millions * 1_000_000)
        for millions in PHYSICAL_STEPS
        for group in GROUPS
        if not _complete(output_root, group, millions * 1_000_000)
    ]
    active: dict[Future[None], tuple[str, int]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while pending or active:
            finished = [future for future in active if future.done()]
            for future in finished:
                group, physical_steps = active.pop(future)
                future.result()
                print(f"evaluated {group} at {physical_steps}", flush=True)
            ready_index = next(
                (
                    index
                    for index, (group, physical_steps) in enumerate(pending)
                    if _checkpoint(training_root, group, physical_steps).exists()
                ),
                None,
            )
            while ready_index is not None and len(active) < args.workers:
                group, physical_steps = pending.pop(ready_index)
                future = executor.submit(
                    _evaluate_one,
                    training_root,
                    output_root,
                    group,
                    physical_steps,
                )
                active[future] = (group, physical_steps)
                ready_index = next(
                    (
                        index
                        for index, (candidate_group, candidate_steps) in enumerate(pending)
                        if _checkpoint(training_root, candidate_group, candidate_steps).exists()
                    ),
                    None,
                )
            if pending or active:
                time.sleep(15)

    for millions in PHYSICAL_STEPS:
        _summarize(training_root, output_root, millions * 1_000_000)
    (output_root / "TIME_HORIZON_EVAL_COMPLETE.txt").write_text(
        "complete\n", encoding="utf-8"
    )
    print(output_root / "TIME_HORIZON_EVAL_COMPLETE.txt")


if __name__ == "__main__":
    main()

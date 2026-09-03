from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "reports/time_horizon_q2_256m_gpu2"
PILOT_EVAL_ROOT = ROOT / "reports/time_horizon_q2_256m_matlab"
FINAL_PHYSICAL_STEPS = 256_000_000


def _items(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _run_one(group: str, seed: int, output_root: Path) -> None:
    destination = output_root / f"seed_{seed}" / f"group_{group}"
    final_checkpoint = (
        destination
        / "checkpoints"
        / f"model_physical_steps_{FINAL_PHYSICAL_STEPS}.pt"
    )
    if final_checkpoint.exists():
        return
    if destination.exists():
        raise RuntimeError(f"partial multi-seed run exists: {destination}")
    (destination / "checkpoints").mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "train.py",
        f"@configs/time_horizon_{group}.args",
        "--seed",
        str(seed),
        "--log-path",
        str(destination / "train.csv"),
        "--checkpoint-path",
        str(destination / "checkpoints" / "model.pt"),
        "--sampler-audit-path",
        str(destination / "reset_samples.csv"),
        "--sampler-audit-hash-only",
        "--sampler-audit-resets-only",
    ]
    (destination / "command.txt").write_text(
        subprocess.list2cmdline(command) + "\n", encoding="utf-8"
    )
    with (destination / "stdout.log").open("w", encoding="utf-8") as stdout, (
        destination / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=True)


def _copy_seed7(groups: tuple[str, str], output_root: Path) -> None:
    for group in groups:
        source = PILOT_ROOT / "seed_7" / f"group_{group}"
        if not (
            source / "checkpoints" / f"model_physical_steps_{FINAL_PHYSICAL_STEPS}.pt"
        ).exists():
            raise FileNotFoundError(source)
        destination = output_root / "seed_7" / f"group_{group}"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)


def _copy_baseline(output_root: Path) -> None:
    source = PILOT_EVAL_ROOT / "baseline"
    destination = output_root / "baseline"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("model.mat", "summary.csv", "samples.csv", "metrics.mat"):
        if not (destination / name).exists():
            shutil.copy2(source / name, destination / name)


def _copy_seed7_evaluation(
    groups: tuple[str, str], output_root: Path, physical_step: int
) -> None:
    """Reuse the already paired pilot evaluation for the reused seed 7."""

    for group in groups:
        source = (
            PILOT_EVAL_ROOT
            / "seed_7"
            / f"group_{group}"
            / f"physical_steps_{physical_step}"
        )
        required = ("model.mat", "summary.csv", "samples.csv", "metrics.mat")
        if not all((source / name).exists() for name in required):
            raise FileNotFoundError(f"incomplete pilot evaluation: {source}")
        destination = (
            output_root
            / "seed_7"
            / f"group_{group}"
            / f"physical_steps_{physical_step}"
        )
        destination.mkdir(parents=True, exist_ok=True)
        for name in required:
            if not (destination / name).exists():
                shutil.copy2(source / name, destination / name)


def _audit_training(
    groups: tuple[str, str], seeds: list[int], output_root: Path
) -> None:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        reset_sequences: dict[str, list[str]] = {}
        for group in groups:
            destination = output_root / f"seed_{seed}" / f"group_{group}"
            path = destination / "train.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                training = list(csv.DictReader(handle))
            applied = [row for row in training if row["update_applied"] == "1"]
            if len(training) != 4000 or int(float(training[-1]["physical_steps"])) != FINAL_PHYSICAL_STEPS:
                raise RuntimeError(f"incomplete final training: seed={seed} group={group}")
            bad = [
                row
                for row in training
                if row["skip_reason"] not in {"", "defer_update_until_episode_boundary"}
            ]
            if bad:
                raise RuntimeError(f"rejected update: seed={seed} group={group}")
            reset_path = destination / "reset_samples.csv"
            with reset_path.open(newline="", encoding="utf-8") as handle:
                reset_sequences[group] = [
                    row["state_sha256"] for row in csv.DictReader(handle)
                ]
            rows.append(
                {
                    "seed": seed,
                    "group": group,
                    "segments": len(training),
                    "optimizer_updates": len(applied),
                    "physical_steps": FINAL_PHYSICAL_STEPS,
                    "reset_count": len(reset_sequences[group]),
                    "reset_file_sha256": hashlib.sha256(
                        reset_path.read_bytes()
                    ).hexdigest(),
                    "final_loss": applied[-1]["loss"],
                }
            )
        common = min(len(reset_sequences[group]) for group in groups)
        if reset_sequences[groups[0]][:common] != reset_sequences[groups[1]][:common]:
            raise RuntimeError(
                f"paired reset stream prefix differs: seed={seed}, groups={groups}"
            )
    with (output_root / "training_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final control/candidate 3-seed validation.")
    parser.add_argument("--control-group", required=True, choices=("T0", "T1", "T2", "T3"))
    parser.add_argument("--candidate-group", required=True, choices=("T0", "T1", "T2", "T3"))
    parser.add_argument("--new-seeds", default="17,27")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--eval-physical-step",
        type=int,
        default=FINAL_PHYSICAL_STEPS,
        help="Checkpoint physical-step label to evaluate after every run finishes.",
    )
    parser.add_argument("--output-root", default="reports/time_horizon_q2_multiseed_gpu2")
    parser.add_argument("--eval-root", default="reports/time_horizon_q2_multiseed_matlab")
    args = parser.parse_args()
    if args.eval_physical_step <= 0 or args.eval_physical_step > FINAL_PHYSICAL_STEPS:
        raise ValueError(
            f"--eval-physical-step must be in (0, {FINAL_PHYSICAL_STEPS}]"
        )
    groups = (args.control_group, args.candidate_group)
    if groups[0] == groups[1]:
        raise ValueError("control and candidate groups must differ")
    seeds = _items(args.new_seeds)
    if not seeds or 7 in seeds:
        raise ValueError("--new-seeds must be non-empty and exclude reused pilot seed 7")
    output_root = (ROOT / args.output_root).resolve()
    eval_root = (ROOT / args.eval_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    _copy_seed7(groups, output_root)
    # Keep each paired seed under the same concurrency/load conditions.  A
    # per-seed barrier also prevents the shorter mixed-horizon run from
    # starting the next seed while its paired fixed-H500 run is still active.
    for seed in seeds:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(_run_one, group, seed, output_root)
                for group in groups
            ]
            for future in futures:
                future.result()
    all_seeds = [7, *seeds]
    _audit_training(groups, all_seeds, output_root)
    _copy_baseline(eval_root)
    _copy_seed7_evaluation(groups, eval_root, args.eval_physical_step)
    subprocess.run(
        [
            sys.executable,
            "tools/run_matlab_position_hold_eval.py",
            "--experiment-root",
            str(output_root.relative_to(ROOT)),
            "--output-root",
            str(eval_root.relative_to(ROOT)),
            "--groups",
            ",".join(groups),
            "--seeds",
            ",".join(str(seed) for seed in all_seeds),
            "--checkpoint-physical-step",
            str(args.eval_physical_step),
            "--batch-size",
            "1024",
            "--eval-seed",
            "1007",
            "--horizon",
            "10000",
            "--workers",
            str(args.workers),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "tools/summarize_time_horizon_multiseed.py",
            "--eval-root",
            str(eval_root.relative_to(ROOT)),
            "--control-group",
            args.control_group,
            "--candidate-group",
            args.candidate_group,
            "--seeds",
            ",".join(str(seed) for seed in all_seeds),
            "--physical-step",
            str(args.eval_physical_step),
        ],
        cwd=ROOT,
        check=True,
    )
    (eval_root / "MULTISEED_COMPLETE.txt").write_text(
        f"complete\neval_physical_step={args.eval_physical_step}\n",
        encoding="utf-8",
    )
    print(eval_root / "MULTISEED_COMPLETE.txt")


if __name__ == "__main__":
    main()

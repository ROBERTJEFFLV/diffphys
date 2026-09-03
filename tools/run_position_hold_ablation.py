from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GROUP_CONFIGS = {
    "R0": "configs/retain_tail_R0.args",
    "R1": "configs/retain_tail_R1.args",
    "R2": "configs/retain_tail_R2.args",
    "R3": "configs/retain_tail_R3.args",
}


def _items(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paired H500 position-hold ablation.")
    parser.add_argument("--phase", choices=("train", "audit", "all"), default="all")
    parser.add_argument("--groups", default=",".join(GROUP_CONFIGS))
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--optimizer-updates", type=int, default=2000)
    parser.add_argument(
        "--checkpoint-updates",
        default="250,500,750,1000,1500,2000",
    )
    parser.add_argument(
        "--output-root",
        default="reports/retain_tail_ablation_h500_u2000_gpu2",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.groups = _items(args.groups)
    args.seeds = [int(value) for value in _items(args.seeds)]
    unknown = sorted(set(args.groups) - set(GROUP_CONFIGS))
    if unknown:
        raise ValueError(f"unknown groups: {','.join(unknown)}")
    if args.max_parallel <= 0:
        raise ValueError("--max-parallel must be positive")
    return args


def _group_root(root: Path, seed: int, group: str) -> Path:
    return root / f"seed_{seed}" / f"group_{group}"


def _run(command: list[str], *, dry_run: bool) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def train(args: argparse.Namespace, root: Path) -> None:
    commands: list[list[str]] = []
    for seed in args.seeds:
        for group in args.groups:
            group_root = _group_root(root, seed, group)
            final_checkpoint = group_root / "checkpoints" / "model.pt"
            target_checkpoint = (
                group_root / "checkpoints" / f"model_update_{args.optimizer_updates}.pt"
            )
            if final_checkpoint.exists() and target_checkpoint.exists() and not args.force:
                print(f"skip completed seed={seed} group={group}")
                continue
            commands.append(
                [
                    sys.executable,
                    "train.py",
                    f"@{GROUP_CONFIGS[group]}",
                    "--seed",
                    str(seed),
                    "--optimizer-updates",
                    str(args.optimizer_updates),
                    "--checkpoint-updates",
                    args.checkpoint_updates,
                    "--log-path",
                    str(group_root / "train.csv"),
                    "--checkpoint-path",
                    str(final_checkpoint),
                    "--sampler-audit-path",
                    str(group_root / "reset_samples.csv"),
                ]
            )
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = [executor.submit(_run, command, dry_run=args.dry_run) for command in commands]
        for future in futures:
            future.result()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(args: argparse.Namespace, root: Path) -> None:
    for seed in args.seeds:
        expected_outer_steps = 2 * args.optimizer_updates
        for group in args.groups:
            log_path = _group_root(root, seed, group) / "train.csv"
            with log_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise RuntimeError(f"empty training log: {log_path}")
            last = rows[-1]
            final_update = int(float(last["optimizer_update"]))
            final_step = int(float(last["step"]))
            skipped = [row for row in rows if row.get("skip_reason", "")]
            deferred = [
                row
                for row in skipped
                if row["skip_reason"] == "defer_update_until_episode_boundary"
            ]
            rejected = [row for row in skipped if row not in deferred]
            if final_update != args.optimizer_updates:
                raise RuntimeError(
                    f"seed={seed} group={group} ended at optimizer update "
                    f"{final_update}, expected {args.optimizer_updates}"
                )
            if final_step != expected_outer_steps:
                raise RuntimeError(
                    f"seed={seed} group={group} consumed {final_step} outer segments, "
                    f"expected exactly {expected_outer_steps}"
                )
            if rejected:
                reasons = sorted({row["skip_reason"] for row in rejected})
                raise RuntimeError(
                    f"seed={seed} group={group} rejected optimizer updates: {reasons}"
                )
        hashes = {
            group: _sha256(_group_root(root, seed, group) / "reset_samples.csv")
            for group in args.groups
        }
        if len(set(hashes.values())) != 1:
            details = ", ".join(f"{group}={value}" for group, value in hashes.items())
            raise RuntimeError(f"unpaired reset samples for seed {seed}: {details}")
        print(f"seed={seed} paired_reset_sha256={next(iter(hashes.values()))}")


def main() -> None:
    args = parse_args()
    root = (REPO_ROOT / args.output_root).resolve()
    if args.phase in {"train", "all"}:
        train(args, root)
    if args.phase in {"audit", "all"} and not args.dry_run:
        audit(args, root)


if __name__ == "__main__":
    main()

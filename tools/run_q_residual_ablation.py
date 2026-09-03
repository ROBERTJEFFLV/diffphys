from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GROUP_CONFIGS = {
    "Q0": "configs/q_residual_Q0.args",
    "Q1": "configs/q_residual_Q1.args",
    "Q2": "configs/q_residual_Q2.args",
}


def _items(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paired P4b residual-control ablation.")
    parser.add_argument("--phase", choices=("train", "audit", "all"), default="all")
    parser.add_argument("--groups", default="Q0,Q1,Q2")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--optimizer-updates", type=int, default=2000)
    parser.add_argument("--checkpoint-updates", default="250,500,750,1000,1500,2000")
    parser.add_argument("--output-root", default="reports/q_residual_h500_u2000_gpu2")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
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
            target = group_root / "checkpoints" / f"model_update_{args.optimizer_updates}.pt"
            if final_checkpoint.exists() and target.exists() and not args.force:
                print(f"skip completed seed={seed} group={group}")
                continue
            command = [
                sys.executable,
                "train.py",
                f"@{GROUP_CONFIGS[group]}",
                "--seed", str(seed),
                "--optimizer-updates", str(args.optimizer_updates),
                "--checkpoint-updates", args.checkpoint_updates,
                "--log-path", str(group_root / "train.csv"),
                "--checkpoint-path", str(final_checkpoint),
                "--sampler-audit-path", str(group_root / "reset_samples.csv"),
            ]
            if args.resume:
                checkpoints = sorted(
                    (group_root / "checkpoints").glob("model_update_*.pt"),
                    key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
                )
                if checkpoints:
                    latest = checkpoints[-1]
                    latest_update = int(latest.stem.rsplit("_", 1)[-1])
                    if latest_update < args.optimizer_updates:
                        command.extend(
                            ["--init-checkpoint-path", str(latest), "--resume-training-state"]
                        )
            commands.append(command)
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
    audit_rows: list[dict[str, object]] = []
    paired_arg_names = (
        "seed", "batch_size", "horizon", "training_episode_steps", "lr",
        "optimizer_updates", "sampled_dynamics_level", "broad_sampler",
        "retain_bank_path", "retain_fraction", "persistent_episode_training",
        "update_timing", "disturbance_force_max", "external_force_ratio",
        "lambda_motor_aux", "w_position_cvar", "w_omega_cvar",
        "init_checkpoint_path",
    )
    for seed in args.seeds:
        hashes: dict[str, str] = {}
        paired_args: dict[str, tuple[object, ...]] = {}
        for group in args.groups:
            group_root = _group_root(root, seed, group)
            with (group_root / "train.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise RuntimeError(f"empty training log: {group_root / 'train.csv'}")
            last = rows[-1]
            if int(float(last["optimizer_update"])) != args.optimizer_updates:
                raise RuntimeError(f"seed={seed} group={group} did not reach requested update")
            if int(float(last["step"])) != 2 * args.optimizer_updates:
                raise RuntimeError(f"seed={seed} group={group} did not use H250x2 per update")
            rejected = [
                row for row in rows
                if row.get("skip_reason", "") not in ("", "defer_update_until_episode_boundary")
            ]
            if rejected:
                raise RuntimeError(
                    f"seed={seed} group={group} rejected updates: "
                    f"{sorted({row['skip_reason'] for row in rejected})}"
                )
            reset_path = group_root / "reset_samples.csv"
            hashes[group] = _sha256(reset_path)
            required_updates = [250, 500, 750, 1000, 1500, 2000]
            missing_checkpoints = [
                update for update in required_updates if update <= args.optimizer_updates
                and not (group_root / "checkpoints" / f"model_update_{update}.pt").exists()
            ]
            if missing_checkpoints:
                raise RuntimeError(
                    f"seed={seed} group={group} missing checkpoints {missing_checkpoints}"
                )
            final_checkpoint = torch.load(
                group_root / "checkpoints" / f"model_update_{args.optimizer_updates}.pt",
                map_location="cpu",
                weights_only=False,
            )
            saved_args = final_checkpoint["args"]
            paired_args[group] = tuple(saved_args[name] for name in paired_arg_names)
            if float(saved_args["lambda_capability_aux"]) != 0.0:
                raise RuntimeError(f"seed={seed} group={group} enabled capability auxiliary")
            if float(saved_args["lambda_response_aux"]) != 0.0:
                raise RuntimeError(f"seed={seed} group={group} enabled response auxiliary")
            if float(saved_args["w_retain"]) != 0.0:
                raise RuntimeError(f"seed={seed} group={group} enabled action retain")
            audit_rows.append(
                {
                    "seed": seed,
                    "group": group,
                    "optimizer_updates": args.optimizer_updates,
                    "outer_segments": int(float(last["step"])),
                    "reset_stream_sha256": hashes[group],
                    "final_loss": float(last["loss"]),
                    "integral_residual_action_rms": float(last["integral_residual_action_rms"]),
                    "damping_residual_action_rms": float(last["damping_residual_action_rms"]),
                    "omega_decay_loss": float(last["omega_decay_loss"]),
                }
            )
        if len(set(hashes.values())) != 1:
            raise RuntimeError(f"seed={seed} reset streams differ: {hashes}")
        if len(set(paired_args.values())) != 1:
            raise RuntimeError(f"seed={seed} paired fixed arguments differ: {paired_args}")
        print(f"seed={seed} paired_reset_sha256={next(iter(hashes.values()))}")
    output_path = root / "paired_training_audit.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)


def main() -> None:
    args = parse_args()
    root = (REPO_ROOT / args.output_root).resolve()
    if args.phase in {"train", "all"}:
        train(args, root)
    if args.phase in {"audit", "all"} and not args.dry_run:
        audit(args, root)


if __name__ == "__main__":
    main()

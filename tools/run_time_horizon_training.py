from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("T0", "T1", "T2", "T3")
PHYSICAL_CHECKPOINTS = (32, 64, 96, 128, 192, 256)


def _last_physical_steps(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return 0 if not rows else int(float(rows[-1]["physical_steps"]))


def _run(group: str, output_root: Path, force: bool) -> None:
    destination = output_root / "seed_7" / f"group_{group}"
    final_checkpoint = (
        destination / "checkpoints" / "model_physical_steps_256000000.pt"
    )
    if final_checkpoint.exists() and not force:
        return
    if destination.exists() and _last_physical_steps(destination / "train.csv") > 0:
        raise RuntimeError(
            f"partial time-horizon run exists for {group}; use a fresh output root"
        )
    (destination / "checkpoints").mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "train.py",
        f"@configs/time_horizon_{group}.args",
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
        subprocess.list2cmdline(command) + "\n",
        encoding="utf-8",
    )
    with (destination / "stdout.log").open("w", encoding="utf-8") as stdout, (
        destination / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=True)


def _reset_hashes(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["state_sha256"] for row in csv.DictReader(handle)]


def _audit(output_root: Path) -> None:
    audit_rows: list[dict[str, object]] = []
    reset_sequences: dict[str, list[str]] = {}
    for group in GROUPS:
        destination = output_root / "seed_7" / f"group_{group}"
        with (destination / "train.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        applied = [row for row in rows if row["update_applied"] == "1"]
        if len(rows) != 4000:
            raise RuntimeError(f"{group}: expected exactly 4000 H250 segments")
        if int(float(rows[-1]["physical_steps"])) != 256_000_000:
            raise RuntimeError(f"{group}: physical-step budget was not exhausted exactly")
        if any(row["skip_reason"] not in {"", "defer_update_until_episode_boundary"} for row in rows):
            raise RuntimeError(f"{group}: formal run contains a rejected/invalid update")
        expected_updates = 2000 if group in {"T0", "T1"} else None
        if expected_updates is not None and len(applied) != expected_updates:
            raise RuntimeError(f"{group}: fixed H500 run did not apply 2000 updates")
        for millions in PHYSICAL_CHECKPOINTS:
            checkpoint = (
                destination
                / "checkpoints"
                / f"model_physical_steps_{millions * 1_000_000}.pt"
            )
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
        reset_path = destination / "reset_samples.csv"
        reset_sequences[group] = _reset_hashes(reset_path)
        audit_rows.append(
            {
                "group": group,
                "integral_input_multiplier": 1.0 if group in {"T0", "T2"} else 0.5,
                "episode_horizon_schedule": (
                    "fixed-h500" if group in {"T0", "T1"} else "mixed-h500-h1000-h2000"
                ),
                "outer_segments": len(rows),
                "optimizer_updates": len(applied),
                "physical_steps": int(float(rows[-1]["physical_steps"])),
                "reset_file_sha256": hashlib.sha256(reset_path.read_bytes()).hexdigest(),
                "reset_count": len(reset_sequences[group]),
                "final_loss": applied[-1]["loss"],
            }
        )
    if reset_sequences["T0"] != reset_sequences["T1"]:
        raise RuntimeError("T0/T1 reset streams differ")
    if reset_sequences["T2"] != reset_sequences["T3"]:
        raise RuntimeError("T2/T3 reset streams differ")
    common = min(len(reset_sequences["T0"]), len(reset_sequences["T2"]))
    if reset_sequences["T0"][:common] != reset_sequences["T2"][:common]:
        raise RuntimeError("fixed/mixed schedules do not share the same reset sequence prefix")
    with (output_root / "paired_training_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired T0-T3 physical-step training.")
    parser.add_argument(
        "--output-root",
        default="reports/time_horizon_q2_256m_gpu2",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_run, group, output_root, args.force)
            for group in GROUPS
        ]
        for future in futures:
            future.result()
    _audit(output_root)
    print(output_root / "paired_training_audit.csv")


if __name__ == "__main__":
    main()

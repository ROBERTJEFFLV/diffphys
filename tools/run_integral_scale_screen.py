from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _parse_groups(value: str) -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    for item in value.split(","):
        name, raw_multiplier = item.split(":", maxsplit=1)
        multiplier = float(raw_multiplier)
        if multiplier <= 0.0:
            raise ValueError("formal scale-screen multipliers must be positive")
        result.append((name.strip(), multiplier))
    if len(result) != 3:
        raise ValueError("the formal scale screen requires exactly three groups")
    return result


def _last_update(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return 0 if not rows else int(float(rows[-1]["optimizer_update"]))


def _run(group: str, multiplier: float, root: Path, force: bool) -> None:
    destination = root / "seed_7" / f"group_{group}"
    final_checkpoint = destination / "checkpoints" / "model_update_1000.pt"
    if final_checkpoint.exists() and not force:
        return
    if destination.exists() and _last_update(destination / "train.csv") > 0:
        raise RuntimeError(
            f"partial scale-screen run exists for {group}; remove it or use a fresh output root"
        )
    (destination / "checkpoints").mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "train.py",
        "@configs/integral_scale_q2_common.args",
        "--integral-input-multiplier",
        str(multiplier),
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


def _audit(groups: list[tuple[str, float]], root: Path) -> None:
    rows: list[dict[str, object]] = []
    hashes: set[str] = set()
    for group, multiplier in groups:
        destination = root / "seed_7" / f"group_{group}"
        with (destination / "train.csv").open(newline="", encoding="utf-8") as handle:
            training = list(csv.DictReader(handle))
        applied = [row for row in training if row["update_applied"] == "1"]
        if len(training) != 2000 or len(applied) != 1000:
            raise RuntimeError(f"{group}: expected 2000 segments/1000 updates")
        reset_path = destination / "reset_samples.csv"
        digest = hashlib.sha256(reset_path.read_bytes()).hexdigest()
        hashes.add(digest)
        for update in (250, 500, 750, 1000):
            if not (destination / "checkpoints" / f"model_update_{update}.pt").exists():
                raise FileNotFoundError(f"{group}: missing update {update} checkpoint")
        rows.append(
            {
                "group": group,
                "integral_input_multiplier": multiplier,
                "outer_segments": len(training),
                "optimizer_updates": len(applied),
                "reset_stream_sha256": digest,
                "final_loss": applied[-1]["loss"],
                "integral_residual_action_rms": applied[-1]["integral_residual_action_rms"],
                "damping_residual_action_rms": applied[-1]["damping_residual_action_rms"],
            }
        )
    if len(hashes) != 1:
        raise RuntimeError(f"scale-screen reset streams differ: {sorted(hashes)}")
    with (root / "paired_training_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paired H500 integral scale screen.")
    parser.add_argument("--groups", default="S0:0.5,S1:1.0,S2:2.0")
    parser.add_argument("--output-root", default="reports/integral_scale_h500_u1000_gpu2")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    groups = _parse_groups(args.groups)
    root = (ROOT / args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_run, group, multiplier, root, args.force)
            for group, multiplier in groups
        ]
        for future in futures:
            future.result()
    _audit(groups, root)
    print(root / "paired_training_audit.csv")


if __name__ == "__main__":
    main()

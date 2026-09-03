from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("Q0", "Q1", "Q2")
P4B_CHECKPOINT = (
    "reports/compact_cvar_ablation_h500_u2000_gpu2_fresh/seed_7/"
    "group_P4b/checkpoints/model_update_2000.pt"
)


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Finish Q evaluation after paired training.")
    parser.add_argument("--training-root", default="reports/q_residual_h500_u2000_gpu2")
    parser.add_argument("--eval-root", default="reports/q_residual_h500_u2000_matlab")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--update", type=int, default=2000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    args = parser.parse_args()
    training_root = ROOT / args.training_root
    eval_root = ROOT / args.eval_root
    required = [
        training_root / f"seed_{args.seed}" / f"group_{group}"
        / "checkpoints" / f"model_update_{args.update}.pt"
        for group in GROUPS
    ]
    deadline = time.time() + args.timeout_hours * 3600.0
    while not all(path.exists() for path in required):
        if time.time() >= deadline:
            missing = [str(path) for path in required if not path.exists()]
            raise TimeoutError(f"timed out waiting for final checkpoints: {missing}")
        complete = sum(path.exists() for path in required)
        print(f"waiting for Q training: {complete}/{len(required)} final checkpoints", flush=True)
        time.sleep(args.poll_seconds)

    run(
        [
            sys.executable, "tools/run_q_residual_ablation.py",
            "--phase", "audit", "--groups", ",".join(GROUPS),
            "--seeds", str(args.seed), "--optimizer-updates", str(args.update),
            "--output-root", args.training_root,
        ],
        training_root / "formal_audit.log",
    )
    run(
        [
            sys.executable, "tools/run_matlab_position_hold_eval.py",
            "--experiment-root", args.training_root,
            "--output-root", args.eval_root,
            "--groups", ",".join(GROUPS),
            "--seeds", str(args.seed),
            "--checkpoint-update", str(args.update),
            "--batch-size", "1024", "--eval-seed", "1007", "--horizon", "10000",
            "--workers", "2", "--include-baseline",
            "--baseline-checkpoint", P4B_CHECKPOINT,
        ],
        eval_root / "formal_eval.log",
    )
    run(
        [
            sys.executable, "tools/summarize_q_residual_eval.py",
            "--eval-root", args.eval_root,
            "--training-root", args.training_root,
            "--groups", ",".join(GROUPS),
            "--seed", str(args.seed),
            "--checkpoint-label", f"update_{args.update}",
            "--baseline-label", "p4b",
        ],
        eval_root / "formal_summary.log",
    )
    q2_checkpoint = required[2]
    run(
        [
            sys.executable, "tools/validate_matlab_residual_forward.py",
            "--checkpoint", str(q2_checkpoint),
            "--output-dir", str(eval_root / "q2_forward_parity"),
        ],
        eval_root / "q2_forward_parity.log",
    )
    marker = eval_root / "FORMAL_Q_EXPERIMENT_COMPLETE.txt"
    marker.write_text("complete\n", encoding="utf-8")
    print(marker, flush=True)


if __name__ == "__main__":
    main()

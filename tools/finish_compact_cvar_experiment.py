from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("P0", "P1", "P2", "P3", "P4a", "P4b")


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
    parser = argparse.ArgumentParser(description="Finish evaluation after paired P training.")
    parser.add_argument("--training-root", default="reports/compact_cvar_ablation_h500_u2000_gpu2")
    parser.add_argument("--eval-root", default="reports/compact_cvar_ablation_h500_u2000_matlab")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--update", type=int, default=2000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    args = parser.parse_args()
    training_root = ROOT / args.training_root
    eval_root = ROOT / args.eval_root
    required = [
        training_root / f"seed_{args.seed}" / f"group_{group}" / "checkpoints" / f"model_update_{args.update}.pt"
        for group in GROUPS
    ]
    deadline = time.time() + args.timeout_hours * 3600.0
    while not all(path.exists() for path in required):
        if time.time() >= deadline:
            missing = [str(path) for path in required if not path.exists()]
            raise TimeoutError(f"timed out waiting for final checkpoints: {missing}")
        complete = sum(path.exists() for path in required)
        print(f"waiting for paired training: {complete}/{len(required)} final checkpoints", flush=True)
        time.sleep(args.poll_seconds)

    audit = [
        sys.executable, "tools/run_compact_cvar_ablation.py",
        "--phase", "audit", "--groups", ",".join(GROUPS),
        "--seeds", str(args.seed), "--optimizer-updates", str(args.update),
        "--output-root", args.training_root,
    ]
    run(audit, training_root / "formal_audit.log")

    matlab_eval = [
        sys.executable, "tools/run_matlab_position_hold_eval.py",
        "--experiment-root", args.training_root,
        "--output-root", args.eval_root,
        "--groups", ",".join(GROUPS),
        "--seeds", str(args.seed),
        "--checkpoint-update", str(args.update),
        "--batch-size", "1024", "--eval-seed", "1007", "--horizon", "10000",
        "--workers", "2", "--include-baseline",
    ]
    oracle = [
        sys.executable, "tools/eval_privileged_oracle.py",
        "--updates", "2000", "--restarts", "3",
        "--shooting-updates", "2000", "--shooting-restarts", "3",
        "--output-dir", "reports/privileged_oracle_standalone_u2000_r3",
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(run, matlab_eval, eval_root / "formal_eval.log"),
            executor.submit(run, oracle, ROOT / "reports/privileged_oracle_standalone_u2000_r3/formal_oracle.log"),
        )
        for future in futures:
            future.result()

    summarize = [
        sys.executable, "tools/summarize_retain_tail_eval.py",
        "--eval-root", args.eval_root,
        "--training-root", args.training_root,
        "--groups", ",".join(GROUPS),
        "--seeds", str(args.seed),
        "--checkpoint-label", f"update_{args.update}",
        "--oracle-results", "reports/privileged_oracle_standalone_u2000_r3/scenario_results.csv",
    ]
    run(summarize, eval_root / "formal_summary.log")
    render = [
        sys.executable, "tools/render_compact_cvar_report.py",
        "--eval-root", args.eval_root,
        "--oracle-root", "reports/privileged_oracle_standalone_u2000_r3",
    ]
    run(render, eval_root / "formal_report.log")
    print(f"formal paired experiment complete: {eval_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()

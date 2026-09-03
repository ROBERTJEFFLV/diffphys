from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "reports/integral_scale_h500_u1000_gpu2"
EVAL_ROOT = ROOT / "reports/integral_scale_h500_u1000_matlab"
GROUPS = ("S0", "S1", "S2")


def main() -> None:
    targets = [
        TRAINING_ROOT / "seed_7" / f"group_{group}" / "checkpoints" / "model_update_1000.pt"
        for group in GROUPS
    ]
    audit = TRAINING_ROOT / "paired_training_audit.csv"
    while not all(path.exists() for path in targets) or not audit.exists():
        complete = sum(path.exists() for path in targets)
        print(f"waiting for scale training: {complete}/3", flush=True)
        time.sleep(30)

    subprocess.run(
        [
            sys.executable,
            "tools/run_matlab_position_hold_eval.py",
            "--experiment-root",
            str(TRAINING_ROOT.relative_to(ROOT)),
            "--output-root",
            str(EVAL_ROOT.relative_to(ROOT)),
            "--groups",
            ",".join(GROUPS),
            "--seeds",
            "7",
            "--checkpoint-update",
            "1000",
            "--batch-size",
            "1024",
            "--eval-seed",
            "1007",
            "--horizon",
            "10000",
            "--workers",
            "2",
            "--include-baseline",
            "--baseline-checkpoint",
            (
                "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/"
                "checkpoints/model_update_2000.pt"
            ),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "tools/summarize_q_residual_eval.py",
            "--eval-root",
            str(EVAL_ROOT.relative_to(ROOT)),
            "--training-root",
            str(TRAINING_ROOT.relative_to(ROOT)),
            "--groups",
            ",".join(GROUPS),
            "--seed",
            "7",
            "--checkpoint-label",
            "update_1000",
        ],
        cwd=ROOT,
        check=True,
    )
    marker = EVAL_ROOT / "SCALE_SCREEN_COMPLETE.txt"
    marker.write_text("complete\n", encoding="utf-8")
    print(marker, flush=True)


if __name__ == "__main__":
    main()

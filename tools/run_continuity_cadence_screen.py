from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "continuity_cadence_9p6m_20260804"
INITIAL_CHECKPOINT = (
    ROOT
    / "reports"
    / "q_residual_h500_u2000_gpu2"
    / "seed_7"
    / "group_Q2"
    / "checkpoints"
    / "model_update_2000.pt"
)
INITIAL_CHECKPOINT_SHA256 = (
    "b401dc6f02beadf51d1b55b24b9056f0b00f17a7a8a5d94d3675554fdb15370d"
)
ARM_CONFIGS = {
    "A": ROOT / "configs" / "continuity_cadence_A.args",
    "B": ROOT / "configs" / "continuity_cadence_B.args",
    "C": ROOT / "configs" / "continuity_cadence_C.args",
    "D": ROOT / "configs" / "continuity_cadence_D.args",
}
CODE_PATHS = (
    ROOT / "tools" / "run_continuity_cadence_screen.py",
    ROOT / "train.py",
    ROOT / "training_objectives.py",
    ROOT / "training_schedule.py",
    ROOT / "model.py",
    ROOT / "env_l2f.py",
    ROOT / "policy_observation.py",
    ROOT / "retain_bank.py",
    ROOT / "l2f_cuda_backend.py",
    ROOT / "l2f_full_cuda_backend.py",
    ROOT / "cuda_ext" / "binding.cpp",
    ROOT / "cuda_ext" / "l2f_step_kernel.cu",
    ROOT / "cuda_ext" / "full_rollout_kernel.cu",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_initial_checkpoint() -> None:
    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing common initialization: {INITIAL_CHECKPOINT}")
    actual = _sha256(INITIAL_CHECKPOINT)
    if actual != INITIAL_CHECKPOINT_SHA256:
        raise RuntimeError(
            "common initialization checkpoint hash mismatch: "
            f"{actual}, expected {INITIAL_CHECKPOINT_SHA256}"
        )


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return parsed


def _parse_arms(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    unknown = tuple(arm for arm in parsed if arm not in ARM_CONFIGS)
    if not parsed or unknown:
        raise argparse.ArgumentTypeError(f"arms must be drawn from A,B,C,D; unknown={unknown}")
    return parsed


def _config_closure(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[Path] = set()

    def visit(current: Path) -> None:
        resolved = current.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        records.append(
            {
                "path": resolved.relative_to(ROOT).as_posix(),
                "sha256": _sha256(resolved),
            }
        )
        for raw_line in resolved.read_text(encoding="utf-8-sig").splitlines():
            for token in shlex.split(raw_line, comments=True, posix=True):
                if token.startswith("@"):
                    visit(ROOT / token[1:])

    visit(path)
    return records


def _command(seed: int, arm: str, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "train.py"),
        f"@{ARM_CONFIGS[arm].relative_to(ROOT).as_posix()}",
        "--seed",
        str(seed),
        "--log-path",
        str(run_dir / "train.csv"),
        "--checkpoint-path",
        str(run_dir / "checkpoints" / "model.pt"),
        "--sampler-audit-path",
        str(run_dir / "reset_samples.csv"),
    ]


def _write_run_manifest(run_dir: Path, *, seed: int, arm: str, command: list[str]) -> None:
    _validate_initial_checkpoint()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(
        subprocess.list2cmdline(command) + "\n",
        encoding="utf-8",
    )
    payload = {
        "manifest_schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "arm": arm,
        "command": command,
        "config_closure": _config_closure(ARM_CONFIGS[arm]),
        "code_files": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}
            for path in CODE_PATHS
        ],
        "initial_checkpoint": INITIAL_CHECKPOINT.relative_to(ROOT).as_posix(),
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "physical_step_budget": 9_600_000,
        "segment_horizon": 250,
        "batch_size": 256,
        "expected_segments": 150,
        "expected_reset_episodes": 75 if arm == "A" else 67,
        "expected_optimizer_commits": 67 if arm in {"B", "D"} else 75,
        "expected_tail_supervision_blocks": 67 if arm == "B" else 75,
    }
    (run_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the continuity/cadence screen and default-off Arm-D follow-up."
    )
    parser.add_argument("--seeds", type=_parse_csv_ints, default=(7, 17, 27))
    parser.add_argument("--arms", type=_parse_arms, default=("A", "B", "C"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    _validate_initial_checkpoint()
    if args.execute and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start the screen")
    if args.skip_existing and "D" in args.arms:
        raise ValueError(
            "Arm D forbids --skip-existing; validate or remove an incomplete run "
            "explicitly before a new preregistered launch"
        )

    output_root = args.output_root.resolve()
    failures: list[str] = []
    for seed in args.seeds:
        for arm in args.arms:
            run_dir = output_root / f"seed_{seed}" / f"arm_{arm}"
            command = _command(seed, arm, run_dir)
            print(subprocess.list2cmdline(command), flush=True)
            final_checkpoint = run_dir / "checkpoints" / "model.pt"
            if args.skip_existing and final_checkpoint.is_file():
                print(f"skip existing {final_checkpoint}", flush=True)
                continue
            _write_run_manifest(run_dir, seed=seed, arm=arm, command=command)
            if not args.execute:
                continue
            with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout_handle, (
                run_dir / "stderr.log"
            ).open("w", encoding="utf-8") as stderr_handle:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    check=False,
                )
            if result.returncode != 0:
                label = f"seed={seed} arm={arm} exit={result.returncode}"
                failures.append(label)
                print(f"FAILED {label}", file=sys.stderr, flush=True)
                if not args.continue_on_error:
                    return result.returncode

    if failures:
        print("failed runs: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

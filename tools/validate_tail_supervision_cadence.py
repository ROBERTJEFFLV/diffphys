from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/arm_d_cpu_semantic_smoke_20260806"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_arm(output_dir: Path, name: str, tail_horizon: int) -> list[dict[str, str]]:
    arm_dir = output_dir / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    command = (
        sys.executable,
        "train.py",
        "--device", "cpu",
        "--sim-backend", "torch",
        "--seed", "7",
        "--optimizer-updates", "1",
        "--horizon", "2",
        "--training-episode-steps", "8",
        "--persistent-episode-training",
        "--update-timing", "episode-boundary",
        "--batch-size", "4",
        "--encoder-dim", "16",
        "--hidden-dim", "12",
        "--tail-window-steps", "2",
        "--steady-window-steps", "2",
        "--tail-selection-mode", "independent",
        "--w-position-cvar", "0.001",
        "--w-omega-cvar", "0.001",
        "--early-tail-weight", "0.25",
        "--final-tail-weight", "1.0",
        "--correct-episode-boundary-weighting",
        "--tail-supervision-block-horizon", str(tail_horizon),
        "--training-diagnostics-every-updates", "1000",
        "--log-every", "1000",
        "--save-every", "0",
        "--post-update-check", "off",
        "--log-path", str(arm_dir / "train.csv"),
        "--checkpoint-path", str(arm_dir / "model.pt"),
    )
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    (arm_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (arm_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    return _read_rows(arm_dir / "train.csv")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU semantic counterfactual for Arm D cadence.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    coupled = _run_arm(output_dir, "coupled_B_semantics", 0)
    decoupled = _run_arm(output_dir, "decoupled_D_semantics", 4)
    if len(coupled) != 4 or len(decoupled) != 4:
        raise RuntimeError("semantic smoke must contain four H2 segments")

    forward_fields = (
        "tracking",
        "position",
        "velocity",
        "omega",
        "hidden_state_norm_initial",
        "integral_state_norm_initial",
        "mass",
        "thrust_to_weight",
    )
    forward_exact = all(
        coupled[index][field] == decoupled[index][field]
        for index in range(4)
        for field in forward_fields
    )
    coupled_updates = sum(int(row["update_applied"]) for row in coupled)
    decoupled_updates = sum(int(row["update_applied"]) for row in decoupled)
    coupled_early = sum(float(row["early_position_cvar"]) > 0.0 for row in coupled)
    decoupled_early = sum(float(row["early_position_cvar"]) > 0.0 for row in decoupled)
    coupled_final = sum(float(row["final_position_cvar"]) > 0.0 for row in coupled)
    decoupled_final = sum(float(row["final_position_cvar"]) > 0.0 for row in decoupled)
    coupled_checkpoint = output_dir / "coupled_B_semantics/model.pt"
    decoupled_checkpoint = output_dir / "decoupled_D_semantics/model.pt"
    coupled_payload = torch.load(coupled_checkpoint, map_location="cpu")
    decoupled_payload = torch.load(decoupled_checkpoint, map_location="cpu")
    coupled_model = coupled_payload["model"]
    decoupled_model = decoupled_payload["model"]
    if coupled_model.keys() != decoupled_model.keys():
        raise RuntimeError("semantic-smoke model state keys differ")
    model_tensors_equal = all(
        torch.equal(coupled_model[name], decoupled_model[name])
        for name in coupled_model
    )
    max_model_parameter_difference = max(
        float((coupled_model[name] - decoupled_model[name]).abs().max().item())
        for name in coupled_model
    )

    checks = [
        {
            "check": "pre_commit_forward_metrics_exact",
            "passed": int(forward_exact),
            "detail": "same seed/reset/policy; cadence changes only objective events before the shared commit",
        },
        {
            "check": "one_optimizer_commit_each",
            "passed": int(coupled_updates == decoupled_updates == 1),
            "detail": f"coupled={coupled_updates}; decoupled={decoupled_updates}",
        },
        {
            "check": "independent_event_count_doubles",
            "passed": int((coupled_early, coupled_final, decoupled_early, decoupled_final) == (1, 1, 2, 2)),
            "detail": (
                f"coupled early/final={coupled_early}/{coupled_final}; "
                f"decoupled={decoupled_early}/{decoupled_final}"
            ),
        },
        {
            "check": "extra_events_change_committed_parameters",
            "passed": int(not model_tensors_equal and max_model_parameter_difference > 0.0),
            "detail": (
                "model-state tensors differ after the single shared-boundary commit; "
                f"max_abs={max_model_parameter_difference:.9g}"
            ),
        },
    ]
    _write_csv(output_dir / "SEMANTIC_CHECKS.csv", checks)
    provenance = {
        "scope": "CPU semantic smoke only; no controller performance claim",
        "coupled_tail_horizon": 0,
        "decoupled_tail_horizon": 4,
        "optimizer_horizon": 8,
        "max_model_parameter_abs_difference": max_model_parameter_difference,
        "all_checks_passed": all(int(row["passed"]) == 1 for row in checks),
        "source_hashes": {
            "train.py": _sha256(ROOT / "train.py"),
            Path(__file__).resolve().relative_to(ROOT).as_posix(): _sha256(Path(__file__).resolve()),
            "coupled_train.csv": _sha256(output_dir / "coupled_B_semantics/train.csv"),
            "decoupled_train.csv": _sha256(output_dir / "decoupled_D_semantics/train.csv"),
            "coupled_checkpoint": _sha256(coupled_checkpoint),
            "decoupled_checkpoint": _sha256(decoupled_checkpoint),
        },
    }
    (output_dir / "RUN_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), **provenance}, indent=2))
    if not provenance["all_checks_passed"]:
        raise RuntimeError("Arm-D semantic smoke failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

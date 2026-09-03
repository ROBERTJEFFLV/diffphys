from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import load_q2_policy, run_formal_rollout
from diagnostics.phase1_fast_common import (
    artifact_fingerprint,
    atomic_write_dataframe,
    atomic_write_json,
    sha256_file,
)
from diagnostics.scenarios import load_matlab_scenarios, read_scenario_rows


DEFAULT_TRAINING_ROOT = ROOT / "reports" / "continuity_cadence_9p6m_20260804"
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "continuity_cadence_internal_eval_20260804"
DEFAULT_SCENARIOS = (
    ROOT
    / "diagnostic_inputs"
    / "h10000_paired_96m_20260804"
    / "manifests"
    / "SCENARIO_MANIFEST.csv"
)


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _csv_arms(value: str) -> tuple[str, ...]:
    arms = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if not arms or any(arm not in {"A", "B", "C", "D"} for arm in arms):
        raise argparse.ArgumentTypeError("arms must be drawn from A,B,C,D")
    return arms


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Corrected internal frozen evaluation for the continuity/cadence screen."
    )
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scenario-csv", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--seeds", type=_csv_ints, default=(7, 17, 27))
    parser.add_argument("--arms", type=_csv_arms, default=("A", "B", "C"))
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--scenario-limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    scenario_rows = read_scenario_rows(args.scenario_csv)
    scenario_ids, initial_state = load_matlab_scenarios(
        args.scenario_csv,
        device=device,
        dtype=dtype,
    )
    if args.scenario_limit is not None:
        if not 0 < args.scenario_limit <= len(scenario_rows):
            raise ValueError("scenario-limit is outside the manifest")
        scenario_rows = scenario_rows[: args.scenario_limit]
        scenario_ids = scenario_ids[: args.scenario_limit]
        indices = torch.arange(args.scenario_limit, device=device)
        from diagnostics.formal_rollout import select_state

        initial_state = select_state(initial_state, indices)
    if scenario_ids != [int(float(row["scenario_id"])) for row in scenario_rows]:
        raise RuntimeError("scenario IDs do not preserve manifest order")
    scenario_uids = [row["scenario_uid"] for row in scenario_rows]

    code_paths = [
        Path(__file__),
        ROOT / "diagnostics" / "formal_rollout.py",
        ROOT / "diagnostics" / "scenarios.py",
        ROOT / "diagnostics" / "physics.py",
        ROOT / "model.py",
        ROOT / "env_l2f.py",
        ROOT / "policy_observation.py",
        ROOT / "l2f_cuda_backend.py",
    ]
    summaries: list[dict[str, object]] = []
    for seed in args.seeds:
        for arm in args.arms:
            label = f"seed_{seed}_arm_{arm}_physical_steps_9600000"
            checkpoint = (
                args.training_root.resolve()
                / f"seed_{seed}"
                / f"arm_{arm}"
                / "checkpoints"
                / "model.pt"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            output = args.output_root.resolve() / f"seed_{seed}" / f"arm_{arm}"
            output.mkdir(parents=True, exist_ok=True)
            manifest_path = output / "RUN_PROVENANCE.json"
            metrics_path = output / "scenario_metrics.csv"
            integral_path = output / "integral_diagnostics.csv"
            provenance = artifact_fingerprint(
                (checkpoint, args.scenario_csv),
                parameters={
                    "label": label,
                    "seed": seed,
                    "arm": arm,
                    "horizon": args.horizon,
                    "scenario_limit": args.scenario_limit,
                    "scenario_count": len(scenario_rows),
                    "device": str(device),
                    "dtype": str(dtype),
                    "backend": "cuda" if device.type == "cuda" else "torch",
                    "integral_clamp_mode": "box",
                    "label_status": "corrected_internal_not_yet_matlab_confirmed",
                },
                code_paths=code_paths,
            )
            if manifest_path.is_file() and not args.force:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                outputs = existing.get("outputs", {})
                if (
                    existing.get("pipeline_fingerprint") == provenance["pipeline_fingerprint"]
                    and outputs
                    and all(
                        (output / name).is_file()
                        and sha256_file(output / name) == digest
                        for name, digest in outputs.items()
                    )
                ):
                    print(f"validated existing {label}", flush=True)
                    summaries.append(existing)
                    continue
                raise RuntimeError(f"existing output differs for {label}; use --force")

            policy, checkpoint_args = load_q2_policy(
                checkpoint,
                device=device,
                dtype=dtype,
            )
            started = time.perf_counter()
            result = run_formal_rollout(
                policy,
                initial_state,
                scenario_uids,
                checkpoint_label=label,
                seed=seed,
                group=arm,
                horizon=args.horizon,
                snapshot_horizons=tuple(
                    value for value in (500, 10_000) if value <= args.horizon
                ),
                backend="cuda" if device.type == "cuda" else "torch",
                intervention="full",
                record_branches=False,
                record_phase=False,
                streaming_accumulator=None,
                integral_clamp_mode="box",
            )
            elapsed = time.perf_counter() - started
            metrics = pd.DataFrame(result.horizon_rows)
            metrics["label_status"] = "corrected_internal_not_yet_matlab_confirmed"
            integral = pd.DataFrame(result.integral_rows)
            atomic_write_dataframe(metrics, metrics_path)
            atomic_write_dataframe(integral, integral_path)
            outputs = {
                metrics_path.name: sha256_file(metrics_path),
                integral_path.name: sha256_file(integral_path),
            }
            provenance.update(
                {
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "checkpoint_physical_steps": checkpoint_args.get("physical_step_budget"),
                    "scenario_manifest_sha256": sha256_file(args.scenario_csv),
                    "scenario_count": len(scenario_rows),
                    "rollout_elapsed_s": elapsed,
                    "outputs": outputs,
                }
            )
            atomic_write_json(provenance, manifest_path)
            summaries.append(provenance)
            print(f"evaluated {label} in {elapsed:.1f}s", flush=True)

    atomic_write_json(summaries, args.output_root.resolve() / "EVAL_INDEX.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

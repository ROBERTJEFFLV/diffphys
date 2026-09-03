from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_l2f import L2FState  # noqa: E402
from retain_bank import load_retain_bank  # noqa: E402
from tools.run_timed_multiple_shooting_value_screen import (  # noqa: E402
    _evaluation_rows,
    _gate_decision,
    _load_arm_policy,
    _load_train_args,
    _make_sim,
    _run_arm,
    _set_seed,
    _sha256,
    _state_from_payload,
    _state_payload,
    _summarize_results,
    _write_csv,
    _write_markdown_table,
    native_l2f_rollout,
)


class _ArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, line: str) -> list[str]:
        return shlex.split(line, comments=True, posix=True)


def parse_args() -> argparse.Namespace:
    parser = _ArgumentParser(
        description="Paired 128-scene persistent-boundary MS generalization screen.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--base-config", type=Path, default=ROOT / "configs/time_horizon_T2.args"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/multiple_shooting_diverse_128x8_screen",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=ROOT / "reports/multiple_shooting_timed_value_screen/eval_set.pt",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--training-scenarios", type=int, default=128)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--updates", type=int, default=129)
    parser.add_argument("--extra-batch-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--schedule-seed", type=int, default=7011)
    parser.add_argument("--bootstrap-seed", type=int, default=7003)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--arm-timeout-seconds", type=float, default=1200.0)
    return parser.parse_args()


def balanced_update_batch_ids(
    batch_count: int,
    cycles: int,
    *,
    extra_batch_id: int,
    seed: int,
) -> torch.Tensor:
    if batch_count <= 0 or cycles <= 0:
        raise ValueError("batch count and cycles must be positive")
    if not 0 <= extra_batch_id < batch_count:
        raise ValueError("extra batch id is out of range")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cycle_orders = [
        torch.randperm(batch_count, generator=generator) for _ in range(cycles)
    ]
    return torch.cat(
        tuple(cycle_orders) + (torch.tensor([extra_batch_id], dtype=torch.long),)
    )


def _concat_states(states: list[L2FState]) -> L2FState:
    return L2FState(
        **{
            field.name: torch.cat(
                tuple(getattr(state, field.name) for state in states), dim=0
            )
            for field in fields(L2FState)
        }
    )


def _scenario_sha256(state: L2FState, index: int) -> str:
    digest = hashlib.sha256()
    for field in fields(L2FState):
        value = getattr(state, field.name)[index].detach().contiguous().cpu()
        digest.update(field.name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write_diverse_schedule(
    path: Path,
    *,
    train_args: argparse.Namespace,
    batch_size: int,
    scenario_count: int,
    cycles: int,
    updates: int,
    extra_batch_id: int,
    seed: int,
    schedule_seed: int,
    device: torch.device,
) -> tuple[L2FState, dict[str, Any]]:
    if scenario_count % batch_size != 0:
        raise ValueError("training scenario count must be divisible by batch size")
    batch_count = scenario_count // batch_size
    update_batch_ids = balanced_update_batch_ids(
        batch_count,
        cycles,
        extra_batch_id=extra_batch_id,
        seed=schedule_seed,
    )
    if update_batch_ids.numel() != updates:
        raise ValueError(
            "locked update count must equal batch_count * cycles + 1: "
            f"{updates} != {update_batch_ids.numel()}"
        )
    sim = _make_sim(train_args)
    retain_path = Path(train_args.retain_bank_path)
    if not retain_path.is_absolute():
        retain_path = ROOT / retain_path
    retain_bank = load_retain_bank(retain_path)
    retain_per_batch = int(round(float(train_args.retain_fraction) * batch_size))
    total_retain = retain_per_batch * batch_count
    if total_retain > len(retain_bank):
        raise ValueError("retain bank is too small for globally unique selection")
    retain_generator = torch.Generator(device="cpu")
    retain_generator.manual_seed(seed + 1_000_003)
    globally_unique_retain_ids = torch.randperm(
        len(retain_bank), generator=retain_generator
    )[:total_retain]
    _set_seed(seed, device)
    states: list[L2FState] = []
    retain_masks: list[torch.Tensor] = []
    retain_indices: list[torch.Tensor] = []
    for batch_id in range(batch_count):
        state = sim.reset(
            batch_size,
            device=device,
            dtype=torch.float32,
            sample_dynamics=train_args.sample_dynamics,
            sampled_dynamics_level=train_args.sampled_dynamics_level,
            broad_sampler=train_args.broad_sampler,
            balanced_dynamics_sampling=train_args.balanced_dynamics_sampling,
            sample_external_force=not train_args.disable_sampled_external_force,
        )
        mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        indices = torch.full(
            (batch_size,), -1, device=device, dtype=torch.long
        )
        target_permutation = torch.randperm(batch_size, device=device)
        target_indices = target_permutation[:retain_per_batch]
        # Consume the historical randint draw so subsequent simulator resets keep
        # the same RNG cadence; use globally unique retain IDs for true 128-scene
        # diversity instead of the discarded with-replacement draw.
        torch.randint(
            len(retain_bank), (retain_per_batch,), device=device
        )
        bank_ids_cpu = globally_unique_retain_ids[
            batch_id * retain_per_batch : (batch_id + 1) * retain_per_batch
        ]
        bank_ids = bank_ids_cpu.to(device=device)
        mask[target_indices] = True
        indices[target_indices] = bank_ids
        for field in fields(L2FState):
            destination = getattr(state, field.name)
            if any(stride == 0 for stride in destination.stride()):
                destination = destination.clone()
                setattr(state, field.name, destination)
            source = retain_bank.state[field.name][bank_ids_cpu].to(
                device=device, dtype=destination.dtype
            )
            destination[target_indices] = source
        states.append(state)
        retain_masks.append(mask)
        retain_indices.append(indices)
    state_bank = _concat_states(states)
    batch_indices = torch.arange(
        scenario_count, device=device, dtype=torch.long
    ).reshape(batch_count, batch_size)
    payload = {
        "format": "persistent-q2-training-bank-v2",
        "schedule_semantics": (
            "128 unique scenarios in fixed batches; each batch owns persistent "
            "shooting boundaries across routed optimizer updates"
        ),
        "seed": seed,
        "schedule_seed": schedule_seed,
        "state": _state_payload(state_bank),
        "retain_mask": torch.cat(tuple(retain_masks)).cpu(),
        "retain_indices": torch.cat(tuple(retain_indices)).cpu(),
        "batch_indices": batch_indices.cpu(),
        "update_batch_ids": update_batch_ids.cpu(),
    }
    torch.save(payload, path)
    visit_counts = torch.bincount(update_batch_ids, minlength=batch_count)
    manifest = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "scenario_count": scenario_count,
        "batch_count": batch_count,
        "batch_size": batch_size,
        "updates": updates,
        "cycles": cycles,
        "extra_batch_id": extra_batch_id,
        "visit_count_min": int(visit_counts.min().item()),
        "visit_count_max": int(visit_counts.max().item()),
        "scenario_exposure_min": int(visit_counts.min().item()),
        "scenario_exposure_max": int(visit_counts.max().item()),
        "retain_count": int(torch.cat(tuple(retain_masks)).sum().item()),
        "unique_retain_id_count": int(
            torch.unique(torch.cat(tuple(retain_indices))[torch.cat(tuple(retain_indices)) >= 0]).numel()
        ),
        "update_batch_ids": [int(value) for value in update_batch_ids.tolist()],
    }
    return state_bank, manifest


def _write_training_eval_summary(
    path: Path, evaluations: dict[str, dict[str, np.ndarray]]
) -> None:
    rows: list[dict[str, Any]] = []
    for horizon in (500, 2000, 5000):
        for channel in ("position", "velocity", "omega", "success"):
            key = f"h{horizon}_{channel}"
            rows.append(
                {
                    "metric": key,
                    "before": float(evaluations["before"][key].mean()),
                    "tbptt": float(evaluations["tbptt"][key].mean()),
                    "multiple_shooting": float(
                        evaluations["multiple_shooting"][key].mean()
                    ),
                }
            )
    _write_csv(path, rows)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.batch_size != 8 or args.training_scenarios != 128:
        raise ValueError("the locked experiment is fixed at 128 scenarios in batches of 8")
    started = time.perf_counter()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = args.eval_set if args.eval_set.is_absolute() else ROOT / args.eval_set
    train_args = _load_train_args(args.base_config)
    if int(train_args.seed) != args.seed:
        raise ValueError("base checkpoint/config seed does not match locked seed")
    device = torch.device(args.device)
    schedule_path = output_dir / "training_schedule_128x8.pt"
    training_state, schedule_manifest = _write_diverse_schedule(
        schedule_path,
        train_args=train_args,
        batch_size=args.batch_size,
        scenario_count=args.training_scenarios,
        cycles=args.cycles,
        updates=args.updates,
        extra_batch_id=args.extra_batch_id,
        seed=args.seed,
        schedule_seed=args.schedule_seed,
        device=device,
    )

    eval_payload = torch.load(eval_path, map_location="cpu")
    eval_state = _state_from_payload(
        eval_payload["state"], device=device, dtype=torch.float32
    )
    eval_groups = list(eval_payload["groups"])
    training_digests = {
        _scenario_sha256(training_state, index)
        for index in range(training_state.position.shape[0])
    }
    eval_digests = {
        _scenario_sha256(eval_state, index)
        for index in range(eval_state.position.shape[0])
    }
    overlap = training_digests.intersection(eval_digests)
    if len(training_digests) != args.training_scenarios:
        raise RuntimeError(
            "generated training bank does not contain 128 unique full scenarios"
        )
    if overlap:
        raise RuntimeError("training and fixed unseen evaluation sets overlap")
    schedule_manifest["eval_scenario_overlap_count"] = 0
    schedule_manifest["unique_full_scenario_count"] = len(training_digests)
    schedule_manifest["eval_set_path"] = str(eval_path.resolve())
    schedule_manifest["eval_set_sha256"] = _sha256(eval_path)
    (output_dir / "schedule_manifest.json").write_text(
        json.dumps(schedule_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    formal: dict[str, dict[str, Any]] = {}
    for mode in ("tbptt", "multiple-shooting"):
        arm_dir = output_dir / "formal" / mode.replace("-", "_")
        arm_dir.mkdir(parents=True, exist_ok=True)
        formal[mode] = _run_arm(
            mode=mode,
            output_dir=arm_dir,
            schedule_path=schedule_path,
            updates=args.updates,
            device=args.device,
            timeout_seconds=args.arm_timeout_seconds,
        )
    if any(
        int(formal[mode]["completed_updates"]) != args.updates
        for mode in formal
    ):
        raise RuntimeError("one arm did not complete the locked equal update budget")

    checkpoint_path = Path(train_args.init_checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    checkpoints = {
        "before": None,
        "tbptt": output_dir / "formal/tbptt/model.pt",
        "multiple_shooting": output_dir / "formal/multiple_shooting/model.pt",
    }
    unseen_evaluations: dict[str, Any] = {"groups": eval_groups}
    training_evaluations: dict[str, dict[str, np.ndarray]] = {}
    unseen_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for label, arm_checkpoint in checkpoints.items():
        policy = _load_arm_policy(checkpoint_path, arm_checkpoint, device=device)
        unseen_metrics = native_l2f_rollout(
            policy, eval_state, train_args, horizon=5000, backend=args.device
        )
        training_metrics = native_l2f_rollout(
            policy, training_state, train_args, horizon=5000, backend=args.device
        )
        unseen_evaluations[label] = unseen_metrics
        training_evaluations[label] = training_metrics
        unseen_rows.extend(
            _evaluation_rows([label] * len(eval_groups), eval_groups, unseen_metrics)
        )
        training_rows.extend(
            _evaluation_rows(
                [label] * args.training_scenarios,
                ["training"] * args.training_scenarios,
                training_metrics,
            )
        )
    _write_csv(output_dir / "unseen_paired_scenario_metrics.csv", unseen_rows)
    _write_csv(output_dir / "training_scenario_metrics.csv", training_rows)
    table, auxiliary = _summarize_results(
        unseen_evaluations,
        replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write_csv(output_dir / "compact_table.csv", table)
    _write_csv(output_dir / "auxiliary_metrics.csv", auxiliary)
    _write_training_eval_summary(
        output_dir / "training_metrics_summary.csv", training_evaluations
    )
    decision = _gate_decision(
        table, formal["tbptt"], formal["multiple-shooting"]
    )
    auxiliary_lookup = {
        (row["scenario_group"], row["metric"]): row for row in auxiliary
    }
    h500_velocity = auxiliary_lookup[("all", "h500_velocity")]
    decision["checks"]["h500_velocity_within_2pct"] = float(
        h500_velocity["multiple_shooting"]
    ) <= 1.02 * max(float(h500_velocity["tbptt"]), 1.0e-12)
    decision["decision"] = (
        "support" if all(decision["checks"].values()) else "not_support"
    )
    elapsed = time.perf_counter() - started
    report = {
        "decision": decision,
        "schedule": schedule_manifest,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training": formal,
        "elapsed_seconds": elapsed,
        "physical_steps_per_arm": args.updates * args.batch_size * 1000,
        "only_intended_change": (
            "replace one repeated 8-scene batch with 128 persistent-boundary "
            "scenarios exposed 8-9 times"
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown_table(
        output_dir / "compact_table.md", table, decision["decision"]
    )
    print(f"decision={decision['decision']} elapsed={elapsed:.1f}s", flush=True)
    print(f"wrote {output_dir / 'compact_table.md'}", flush=True)


if __name__ == "__main__":
    main()

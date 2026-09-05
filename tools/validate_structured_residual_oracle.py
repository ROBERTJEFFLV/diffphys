"""Pre-register an away-from-equilibrium structured residual action oracle.

This is a diagnostic gate between local-gain Phase B and residual Phase C.
It trains a disposable copy of the Phase-B policy on independent student
closed-loop DAgger banks, then measures the reachable whole-action RMS only on
an explicit normalized feedback-feature annulus.  Near-equilibrium Q2 bias is
reported separately and is never used as the parity target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import load_q2_policy  # noqa: E402
from env_l2f import L2FParams, L2FSimulator, L2FState  # noqa: E402
from structured_checkpoint import deployment_policy_hash  # noqa: E402
from structured_distillation import (  # noqa: E402
    DAggerScenarioBank, build_dagger_scenario_bank, collect_dagger_episode,
)
from structured_rollout import load_structured_policy  # noqa: E402


DEFAULT_Q2 = ROOT / "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    return device


def _move_bank(bank: DAggerScenarioBank, device: torch.device) -> DAggerScenarioBank:
    if device.type == "cpu":
        return bank
    state = L2FState(**{
        name: getattr(bank.state, name).to(device)
        for name in bank.state.__dataclass_fields__
    })
    return type(bank)(state, bank.tw_bin.to(device), bank.log_alpha_bin.to(device), bank.stratum)


def _bank_hash(bank: DAggerScenarioBank) -> str:
    digest = hashlib.sha256()
    for name in sorted(bank.state.__dataclass_fields__):
        value = getattr(bank.state, name).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _feedback_features(policy, episode) -> torch.Tensor:
    state = policy.initial_state(episode.observations[0])
    values = []
    for step in range(episode.observations.shape[0]):
        output = policy.forward_with_aux(
            episode.observations[step], state,
            applied_action=episode.executed_actions[step],
        )
        state = output.next_state
        values.append(output.auxiliary["feedback_features"])
    return torch.stack(values)


def _annulus_stats(episode, features: torch.Tensor, r_min: float,
                   r_max: float) -> dict[str, Any]:
    radius = torch.linalg.vector_norm(features, dim=-1)
    difference = episode.student_actions - episode.teacher_actions
    annulus = (radius >= r_min) & (radius <= r_max)
    near = radius < r_min
    count = annulus.sum(dim=0)
    per_scenario = torch.sqrt(
        (difference.square().mean(dim=-1) * annulus).sum(dim=0)
        / count.clamp_min(1)
    )
    near_count = near.sum(dim=0)
    near_rms = torch.sqrt(
        (difference.square().mean(dim=-1) * near).sum(dim=0)
        / near_count.clamp_min(1)
    )
    selected = difference[annulus]
    oracle_rms = (selected.square().mean().sqrt() if selected.numel() else
                  torch.tensor(float("inf"), device=difference.device))
    return {
        "oracle_rms": float(oracle_rms),
        "per_scenario_rms_max": float(per_scenario.max()),
        "annulus_count": int(annulus.sum()),
        "annulus_scenarios_with_samples": int((count > 0).sum()),
        "near_equilibrium_q2_bias_rms": float(
            torch.sqrt((difference.square().mean(dim=-1) * near).sum()
                       / near.sum().clamp_min(1))
        ),
        "near_equilibrium_count": int(near.sum()),
        "radius_min": float(radius.min()), "radius_max": float(radius.max()),
        "per_scenario_rms": [float(value) for value in per_scenario],
        "near_equilibrium_per_scenario_rms": [float(value) for value in near_rms],
    }


def _masked_train_step(policy, teacher, simulator, bank, *, horizon: int,
                       r_min: float, r_max: float) -> tuple[float, int]:
    episode = collect_dagger_episode(
        teacher, policy, simulator, bank, beta=0.0, horizon=horizon,
        episode_seed=17,
    )
    state = policy.initial_state(episode.observations[0])
    losses = []
    selected = 0
    for step in range(episode.observations.shape[0]):
        output = policy.forward_with_aux(
            episode.observations[step], state,
            applied_action=episode.executed_actions[step],
        )
        state = output.next_state
        features = output.auxiliary["feedback_features"]
        radius = torch.linalg.vector_norm(features, dim=-1)
        mask = (radius >= r_min) & (radius <= r_max)
        selected += int(mask.sum())
        if bool(mask.any()):
            losses.append(F.smooth_l1_loss(
                output.action[mask], episode.teacher_actions[step][mask]
            ))
        if (step + 1) % 25 == 0:
            state = state.detach()
    if not losses:
        return 0.0, 0
    loss = torch.stack(losses).mean()
    loss.backward()
    return float(loss.detach()), selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="passed Phase-B checkpoint")
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_Q2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--heldout-count", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=125)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--r-min", type=float, default=0.25)
    parser.add_argument("--r-max", type=float, default=3.0)
    parser.add_argument("--measurement-floor", type=float, default=1.3e-3)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    from structured_training_runtime import add_training_arguments
    add_training_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    if not (0.0 < args.r_min < args.r_max and args.measurement_floor > 0.0):
        raise ValueError("require 0 < r_min < r_max and positive measurement floor")
    device = _device(args.device)
    torch.manual_seed(args.seed)
    teacher, teacher_args = load_q2_policy(args.source_checkpoint, device=device, dtype=torch.float32)
    teacher.eval()
    source_payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    source_report = source_payload.get("report", {})
    if source_report.get("active_phase") != "B" or not source_report.get("phase_b_gate_passed", False):
        raise RuntimeError("residual oracle requires a passed Phase-B checkpoint")
    simulator = L2FSimulator(L2FParams(dt=float(teacher_args.get("dt", 0.01))))
    student, _ = load_structured_policy(args.checkpoint, device=device)
    if source_report.get("student_policy_deployment_hash") != deployment_policy_hash(student):
        raise RuntimeError("Phase-B report deployment hash does not match its checkpoint")
    config = replace(student.config, residual_scale=args.residual_scale, residual_trainable=True)
    candidate = type(student)(config).to(device)
    candidate.load_state_dict(student.state_dict(), strict=True)
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    for name, parameter in candidate.named_parameters():
        if name.startswith(("encoder.", "gru.", "residual_head.")):
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        (p for p in candidate.parameters() if p.requires_grad), lr=args.lr,
        weight_decay=1.0e-5,
    )
    train_cpu = build_dagger_scenario_bank(args.train_count, seed=args.seed + 401,
                                           dt=simulator.params.dt)
    from structured_training_runtime import TrainingSession
    session = TrainingSession(args, candidate, optimizer, stage="residual_oracle")
    train_bank = _move_bank(train_cpu, device)
    history = session.progress["history"]
    if args.final_evaluation:
        session.begin_final([args.seed + 140002])
        history = session.progress["history"]
    session.save()
    for update in range(session.updates, session.updates if args.final_evaluation else args.updates):
        if session.should_stop():
            break
        optimizer.zero_grad(set_to_none=True)
        loss, selected = _masked_train_step(
            candidate, teacher, simulator, train_bank, horizon=args.horizon,
            r_min=args.r_min, r_max=args.r_max,
        )
        if selected:
            norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(), 10.0)
            if not bool(torch.isfinite(norm)) or not math.isfinite(loss):
                session.save()
                raise RuntimeError("non-finite residual-oracle update")
            optimizer.step()
        session.record_update({"loss": loss, "annulus_samples": selected})
    heldout_seed = args.seed + (140002 if args.final_evaluation else 2300000)
    heldout_cpu = build_dagger_scenario_bank(args.heldout_count, seed=heldout_seed,
        dt=simulator.params.dt, per_cell=args.heldout_count // 16)
    heldout_bank = _move_bank(heldout_cpu, device)
    heldout_episode = collect_dagger_episode(
        teacher, candidate, simulator, heldout_bank, beta=0.0,
        horizon=args.horizon, episode_seed=args.seed + 499,
    )
    heldout_features = _feedback_features(candidate, heldout_episode)
    stats = _annulus_stats(heldout_episode, heldout_features, args.r_min, args.r_max)
    oracle_rms = float(stats["oracle_rms"])
    threshold = max(2.0 * oracle_rms, float(args.measurement_floor))
    allocator_values = heldout_episode.allocator_residual.detach().reshape(-1)
    allocator_finite = bool(torch.isfinite(allocator_values).all())
    allocator_baseline = (
        float(torch.quantile(allocator_values, 0.99))
        if allocator_values.numel() and allocator_finite else float("inf")
    )
    # This is registered before Phase C changes any policy weights.  It is a
    # release limit, not a number fitted after looking at the Phase-C result.
    # The relative form makes the contract auditable even when the absolute
    # residual is close to zero.
    allocator_relative_multiplier = 2.0
    allocator_baseline_for_relative = max(
        allocator_baseline, float(args.measurement_floor)
    )
    allocator_threshold = max(
        allocator_relative_multiplier * allocator_baseline_for_relative,
        float(args.measurement_floor),
    )
    oracle_gate_passed = bool(
        math.isfinite(oracle_rms) and stats["annulus_count"] > 0
        and stats["annulus_scenarios_with_samples"] == args.heldout_count
        and bool(heldout_episode.finite.all()) and allocator_finite
        and isinstance(source_report.get("student_policy_deployment_hash"), str)
        and bool(source_report.get("student_policy_deployment_hash"))
    )
    oracle_contract = {
        "schema_version": 1,
        "pre_registered": True,
        "registered": True,
        "source_phase": "B",
        "phase_b_deployment_hash": source_report.get("student_policy_deployment_hash"),
        "annulus": {
            "r_min": float(args.r_min), "r_max": float(args.r_max),
            "normalized_feedback_feature": True,
        },
        "action_rms_threshold": threshold,
        "whole_policy_action_rms_threshold": threshold,
        "preregistered_action_rms_threshold": threshold,
        "allocator_residual_p99_threshold": allocator_threshold,
        "residual_p99_threshold": allocator_threshold,
        "baseline_residual_p99": allocator_baseline_for_relative,
        "observed_baseline_residual_p99": allocator_baseline,
        "relative_multiplier": allocator_relative_multiplier,
        "threshold_definition": "max(2*heldout_oracle_rms,measurement_floor)",
        "allocator_threshold_definition": "max(relative_multiplier*phase_b_heldout_allocator_p99,measurement_floor)",
    }
    report = {
        "phase": "structured-residual-oracle",
        "active_phase": "residual_oracle",
        "screening_only": True, "promotion_gate_passed": False,
        "oracle_gate_passed": oracle_gate_passed,
        "phase_b_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "phase_b_deployment_hash": source_report.get("student_policy_deployment_hash"),
        "candidate_deployment_hash": deployment_policy_hash(candidate),
        "train_bank_hash": _bank_hash(train_cpu), "heldout_bank_hash": _bank_hash(heldout_cpu),
        "train_count": args.train_count, "heldout_count": args.heldout_count,
        "seed": args.seed, "bank_seeds": {"train": args.seed + 401, "heldout": heldout_seed},
        "horizon": args.horizon, "updates": args.updates,
        "annulus": {"r_min": args.r_min, "r_max": args.r_max,
                     "normalized_feedback_feature": True},
        "measurement_floor": args.measurement_floor,
        "heldout_oracle_rms": oracle_rms,
        "preregistered_action_rms_threshold": threshold,
        "threshold_definition": "max(2*heldout_oracle_rms,measurement_floor)",
        "structured_residual_oracle": oracle_contract,
        "pre_registered": True,
        "registered": True,
        "allocator_baseline_residual_p99": allocator_baseline_for_relative,
        "allocator_observed_baseline_residual_p99": allocator_baseline,
        "allocator_residual_p99_threshold": allocator_threshold,
        "allocator_relative_multiplier": allocator_relative_multiplier,
        "allocator_finite": allocator_finite,
        "q2_e0_bias": stats["near_equilibrium_q2_bias_rms"],
        "q2_e0_bias_count": stats["near_equilibrium_count"],
        "heldout_stats": stats, "history": history,
        "optimizer_scope": ["encoder", "gru", "residual_head"],
        "teacher_runtime_dependency": True,
    }
    if not args.final_evaluation:
        session.record_development(score=oracle_rms, passed=oracle_gate_passed, metrics=report)
        session.finish_development()
        return
    report["training_session"] = session.summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": "structured-residual-oracle-copy",
                "model": candidate.state_dict(), "config": asdict(candidate.config),
                "report": report}, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "oracle_gate_passed": oracle_gate_passed,
                      "threshold": threshold}, sort_keys=True))


if __name__ == "__main__":
    main()

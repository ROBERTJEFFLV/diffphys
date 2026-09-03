"""Held-out 99% capability calibration for the structured controller.

The formal calibration uses four predeclared roll-authority risk strata with
128 independent scenarios per stratum (512 total).  A second independent bank
validates the one-sided effectiveness interval.  The 4x4 cells remain rollout
diagnostics; they are not misrepresented as cell-conditional 99% coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import load_q2_policy  # noqa: E402
from env_l2f import L2FParams, L2FSimulator, L2FState  # noqa: E402
from structured_distillation import (  # noqa: E402
    build_dagger_scenario_bank,
    collect_dagger_episode,
    fit_effectiveness_conformal_q,
    one_sided_conformal_quantile,
)
from structured_policy import CAPABILITY_HI, CAPABILITY_LO  # noqa: E402
from structured_checkpoint import (  # noqa: E402
    CADENCE_SEMANTICS_VERSION,
    base_policy_hash,
    deployment_policy_hash,
)
from structured_rollout import load_structured_policy  # noqa: E402


DEFAULT_Q2 = (
    ROOT
    / "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt"
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    return device


def _move_bank(bank, device: torch.device):
    if device.type == "cpu":
        return bank
    state = L2FState(
        **{
            name: getattr(bank.state, name).to(device)
            for name in bank.state.__dataclass_fields__
        }
    )
    return type(bank)(
        state, bank.tw_bin.to(device), bank.log_alpha_bin.to(device), bank.stratum
    )


def _bank_hash(bank) -> str:
    digest = hashlib.sha256()
    for name in sorted(bank.state.__dataclass_fields__):
        value = getattr(bank.state, name).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _policy_config_hash(policy, *, include_q: bool = True) -> str:
    """Compatibility wrapper around the shared deployment hash."""
    return deployment_policy_hash(policy) if include_q else base_policy_hash(policy)


def _two_sided_metadata(episode, *, miscoverage: float,
                        phase_steps: tuple[int, ...]) -> list[dict]:
    rows = []
    for phase_step in phase_steps:
        index = phase_step
        for stratum in range(4):
            mask = episode.log_alpha_bin == stratum
            score = (
                (episode.capability_target_z[mask] - episode.capability_z_mean[index, mask]).abs()
                / episode.capability_z_log_scale[index, mask].exp().clamp_min(1.0e-8)
            ).amax(dim=-1)
            q = one_sided_conformal_quantile(
                score, miscoverage=miscoverage
            ).clamp_min(0.0)
            rows.append(
                {
                    "phase_step": phase_step,
                    "risk_stratum": stratum,
                    "samples": int(mask.sum()),
                    "q_two_sided": float(q),
                }
            )
    return rows


def _width_gate(episode, q: torch.Tensor, *, phase_step: int = 50) -> dict:
    index = phase_step
    mean = episode.capability_z_mean[index, :, :3]
    sigma = episode.capability_z_log_scale[index, :, :3].exp()
    multiplier = q[:3]
    lower = (mean - multiplier * sigma).clamp(-1.0, 1.0)
    upper = (mean + multiplier * sigma).clamp(-1.0, 1.0)
    log_lo = mean.new_tensor(CAPABILITY_LO).log()
    log_hi = mean.new_tensor(CAPABILITY_HI).log()
    half = 0.5 * (log_hi - log_lo)
    ratios = torch.exp(half[:3] * (upper - lower))
    limits = ratios.new_tensor((1.5, 2.0, 2.0))
    passed = (ratios <= limits).all(dim=-1)
    return {
        "phase_step": phase_step,
        "pass_fraction": float(passed.float().mean()),
        "ratio_p99": [
            float(torch.quantile(ratios[:, index], 0.99)) for index in range(3)
        ],
        "required_fraction": 0.99,
        "passed": bool(float(passed.float().mean()) >= 0.99),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal 512-scenario capability calibration")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_Q2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1707)
    parser.add_argument("--horizon", type=int, default=76)
    parser.add_argument("--miscoverage", type=float, default=0.01)
    parser.add_argument("--maximum-self-calibration-rounds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon < 76 or not 0.0 < args.miscoverage < 1.0:
        raise ValueError(
            "formal calibration needs horizon>=76 (through call75) and valid miscoverage"
        )
    device = _device(args.device)
    policy, source = load_structured_policy(args.checkpoint, device=device)
    teacher, teacher_args = load_q2_policy(
        args.source_checkpoint, device=device, dtype=torch.float32
    )
    policy.eval()
    teacher.eval()
    simulator = L2FSimulator(L2FParams(dt=float(teacher_args.get("dt", policy.config.dt))))
    phase_steps = (50, 75)
    if args.maximum_self_calibration_rounds < 2:
        raise ValueError("formal self-calibration requires at least two fresh rounds")

    iterations = []
    q_candidate = (
        policy.capability_conformal_q.detach().clone()
        if bool(policy.capability_calibration_valid.item())
        else torch.zeros(6, device=device)
    )
    calibration_cpu = None
    calibration_episode = None
    rows = []
    q = q_candidate
    converged = False
    # An uncalibrated policy first needs a bootstrap candidate.  That data is
    # not used as the final conformal set because installing q changes the
    # closed-loop state distribution.
    if not bool(policy.capability_calibration_valid.item()):
        bootstrap_cpu = build_dagger_scenario_bank(
            512, seed=args.seed, dt=simulator.params.dt, per_cell=32
        )
        bootstrap = _move_bank(bootstrap_cpu, device)
        bootstrap_episode = collect_dagger_episode(
            teacher, policy, simulator, bootstrap, beta=0.0,
            horizon=args.horizon, episode_seed=args.seed + 900,
        )
        q_candidate, _ = fit_effectiveness_conformal_q(
            bootstrap_episode,
            miscoverage=args.miscoverage,
            phase_steps=phase_steps,
        )
        policy.install_capability_conformal_q(q_candidate, sample_count=512)
        iterations.append(
            {
                "iteration": "bootstrap",
                "calibration_seed": args.seed,
                "calibration_dataset_hash": _bank_hash(bootstrap_cpu),
                "input_calibration_valid": False,
                "q_in": [0.0] * 6,
                "q_fit": [float(value) for value in q_candidate],
                "q_installed": [float(value) for value in q_candidate],
                "self_consistency_check": False,
            }
        )

    for iteration in range(args.maximum_self_calibration_rounds):
        calibration_seed = args.seed + 1 + iteration
        q_before = q_candidate.detach().clone()
        calibration_cpu = build_dagger_scenario_bank(
            512, seed=calibration_seed, dt=simulator.params.dt, per_cell=32
        )
        calibration = _move_bank(calibration_cpu, device)
        input_calibration_valid = bool(
            policy.capability_calibration_valid.item()
        )
        behavior_hash_in = _policy_config_hash(policy, include_q=True)
        calibration_episode = collect_dagger_episode(
            teacher, policy, simulator, calibration, beta=0.0,
            horizon=args.horizon, episode_seed=args.seed + 1000 + iteration,
        )
        q_fit, rows = fit_effectiveness_conformal_q(
            calibration_episode,
            miscoverage=args.miscoverage,
            phase_steps=phase_steps,
        )
        relative_change = float(
            ((q_fit[:3] - q_before[:3]).abs()
             / q_before[:3].abs().clamp_min(1.0)).max()
        )
        dominated = bool(
            (q_fit[:3] <= q_before[:3] + 1.0e-7).all().item()
        )
        if dominated:
            # Crucially, do not change q.  This calibration batch was sampled
            # under exactly the behavior that will be deployed.
            q = q_before
            q_candidate = q_before
            converged = True
        else:
            q_candidate = torch.maximum(q_before, q_fit)
            policy.install_capability_conformal_q(q_candidate, sample_count=512)
            q = q_candidate
        behavior_hash_out = _policy_config_hash(policy, include_q=True)
        iterations.append(
            {
                "iteration": iteration,
                "calibration_seed": calibration_seed,
                "calibration_dataset_hash": _bank_hash(calibration_cpu),
                "behavior_hash_in": behavior_hash_in,
                "behavior_hash_out": behavior_hash_out,
                "input_calibration_valid": input_calibration_valid,
                "q_in": [float(value) for value in q_before],
                "q_fit": [float(value) for value in q_fit],
                "q_installed": [float(value) for value in q_candidate],
                "q_relative_change": relative_change,
                "candidate_dominates_fit": dominated,
                "self_consistency_check": True,
            }
        )
        if dominated:
            break

    assert calibration_cpu is not None and calibration_episode is not None
    # The terminal validation bank is never reused to tune q.  It is collected
    # under the final installed policy and is a shift/width diagnostic only.
    validation_seed = args.seed + args.maximum_self_calibration_rounds + 100
    validation_cpu = build_dagger_scenario_bank(
        512, seed=validation_seed, dt=simulator.params.dt, per_cell=32
    )
    validation = _move_bank(validation_cpu, device)
    validation_episode = collect_dagger_episode(
        teacher, policy, simulator, validation, beta=0.0,
        horizon=args.horizon, episode_seed=args.seed + 2000,
    )
    # Reuse the final calibration set only to compute the already-fixed q and
    # validation diagnostics.  The validation bank does not feed the fit.
    q_check, rows = fit_effectiveness_conformal_q(
        calibration_episode,
        validation_episode=validation_episode,
        miscoverage=args.miscoverage,
        phase_steps=phase_steps,
    )
    q_consistent = bool((q_check[:3] <= q[:3] + 1.0e-7).all().item())
    width = _width_gate(validation_episode, q, phase_step=50)
    sufficient_samples = all(
        int(row["calibration_samples"]) >= 128 for row in rows
    )
    finite = bool(
        calibration_episode.finite.all() and validation_episode.finite.all()
    )
    promotion = bool(
        sufficient_samples and width["passed"] and finite
        and converged and q_consistent
    )
    if not promotion:
        # A failed calibration artifact must not silently use its empirical q
        # as if it had a formal coverage contract.
        policy.capability_calibration_valid.fill_(False)
    two_sided = _two_sided_metadata(
        calibration_episode,
        miscoverage=args.miscoverage,
        phase_steps=phase_steps,
    )
    calibration_metadata = {
        "cadence_semantics_version": CADENCE_SEMANTICS_VERSION,
        "cadence_semantics": {
            "call_index_completed_transitions": True,
            "call0_has_response": False,
            "publication_calls": [50, 75],
            "availability_t25": [0, 0, 0, 0, 0, 0],
            "publication_rule_after_first": "positive slow_cadence offsets from call50",
            "t50_call_index": 50,
        },
        "conditional_scope": "alpha_conditional_four_log_alpha_risk_strata",
        "score_definition": "joint one-sided standardized upper score; max over effectiveness dimensions and phase steps 50/75",
        "cell_scope": "4x4_rollout_diagnostic_only",
        "miscoverage": args.miscoverage,
        "phase_steps": list(phase_steps),
        "q_upper": [float(value) for value in q],
        "q_upper_by_phase_stratum": rows,
        "q_two_sided_by_phase_stratum": two_sided,
        "n_calibration": 512,
        "n_validation": 512,
        "self_calibration_iterations": iterations,
        "self_calibration_converged": converged,
        "q_recomputed_dominated_by_deployed_candidate": q_consistent,
        "final_calibration_dataset_hash": _bank_hash(calibration_cpu),
        "validation_dataset_hash": _bank_hash(validation_cpu),
        "validation_seed": validation_seed,
        "policy_config_hash": _policy_config_hash(policy, include_q=True),
        "calibrated_parameter_hash": _policy_config_hash(policy, include_q=True),
        "policy_config_hash_excludes": [],
        "width_gate": width,
        "finite": finite,
        "validation_shift_diagnostic": rows,
        "validation_coverage_is_not_proof": True,
        "sufficient_samples": sufficient_samples,
        "promotion_gate_passed": promotion,
    }
    previous_report = source.get("report", {})
    report = {
        **previous_report,
        "capability_calibration": calibration_metadata,
        # Capability calibration alone never promotes the complete controller.
        "migration_gate_passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "structured-recurrent-motor-policy",
            "model": policy.state_dict(),
            "config": source["config"],
            "fast_feedback_verified": policy.fast_feedback.verified,
            "report": report,
        },
        args.output,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(calibration_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(calibration_metadata, sort_keys=True))


if __name__ == "__main__":
    main()

"""Safe 64-scenario recurrent DAgger distillation into the structured student."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import load_q2_policy  # noqa: E402
from env_l2f import L2FParams, L2FSimulator  # noqa: E402
from structured_distillation import (  # noqa: E402
    DAGGER_BETA_SCHEDULE,
    build_dagger_scenario_bank,
    collect_dagger_episode,
    dagger_window_loss,
    conformal_coverage_by_stratum,
    fit_effectiveness_conformal_q,
    distillation_phase_components,
    layered_dagger_gate,
    phase_a_equilibrium_gate,
    set_distillation_phase,
)
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy  # noqa: E402
from structured_checkpoint import (  # noqa: E402
    CADENCE_SEMANTICS_VERSION,
    deployment_policy_hash,
    identifier_weights_hash,
    load_identifier_init_artifact,
    validate_causal_revalidation_report,
    validate_identifier_pretraining_report,
)
from tools.diagnose_causal_identifier_oracle import (  # noqa: E402
    DEFAULT_PROBE_V4_REPORT,
    probe_v5_eligibility,
)


DEFAULT_Q2 = ROOT / "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="64-scenario 4x4 recurrent DAgger structured-policy distillation")
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_Q2)
    parser.add_argument("--student-checkpoint", type=Path, default=None,
                        help="required when continuing phase B or C")
    parser.add_argument(
        "--identifier-init-artifact", type=Path, default=None,
        help="required for Phase A1: production identifier init artifact",
    )
    parser.add_argument(
        "--identifier-pretraining-report", type=Path, default=None,
        help="required for Phase A1: independent accuracy/provenance gate for the init",
    )
    parser.add_argument(
        "--probe-v4-report", type=Path, default=DEFAULT_PROBE_V4_REPORT,
        help="current formal v4 freeze record, revalidated immediately before Phase A1",
    )
    parser.add_argument(
        "--causal-revalidation-report", type=Path, default=None,
        help="post-A1 causal gate report required before Phase A2",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=126)
    parser.add_argument("--prefix", type=int, default=25)
    parser.add_argument("--updates-per-beta", type=int, default=1)
    parser.add_argument("--maximum-buffer-episodes", type=int, default=10)
    parser.add_argument("--delta-action-weight", type=float, default=0.1)
    parser.add_argument("--body-z-weight", type=float, default=0.25)
    parser.add_argument(
        "--capability-mean-weight", type=float, default=1.0,
        help="explicit normalized capability-mean supervision",
    )
    parser.add_argument(
        "--capability-nll-weight", type=float, default=0.0,
        help="heteroscedastic NLL weight; zero in A1 and positive in A2",
    )
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--identifier-dim", type=int, default=64)
    parser.add_argument("--slow-cadence", type=int, default=25)
    parser.add_argument("--identification-publish-start", type=int, default=100)
    parser.add_argument(
        "--motor-observer-bank-size", type=int, choices=(0, 15, 35), default=0,
        help="0 preserves legacy checkpoints; 15/35 enable fixed multi-tau v1/v2",
    )
    parser.add_argument(
        "--motor-observer-mode",
        choices=("legacy", "fixed_multi_tau_v1", "fixed_multi_tau_v2"),
        default="legacy",
    )
    parser.add_argument(
        "--motor-tau-grid-version", type=int, choices=(0, 1, 2), default=0,
    )
    parser.add_argument(
        "--burn-in-probe-amplitude", type=float, default=0.0,
        help=(
            "zero-mean motor-coordinate identification probe amplitude; "
            "the command still passes through the constrained allocator"
        ),
    )
    parser.add_argument(
        "--allocator-solver", choices=("smooth_dls", "box_qp"),
        default="smooth_dls",
        help="motor allocator; formal structured runs use the exact four-motor box QP",
    )
    parser.add_argument(
        "--allocator-rate-limit", type=float, default=0.0,
        help="post-burn-in command slew limit in normalized action units/second; 0 disables it",
    )
    parser.add_argument(
        "--identification-oracle-report", type=Path, default=None,
        help=(
            "formal K35 causal-identification report; required whenever "
            "--motor-observer-bank-size is non-zero"
        ),
    )
    parser.add_argument("--action-gate", type=float, default=1.3e-3)
    parser.add_argument("--omega-gate", type=float, default=5.0,
                        help="override with 1.5x the paired-Q2 omega gate")
    parser.add_argument("--residual-scale", type=float, default=0.0)
    parser.add_argument(
        "--phase", choices=("A", "A1", "A2", "C"), default="A1",
        help=(
            "A is a deprecated A1 alias; A1 fits identifier/mean, A2 fits "
            "only uncertainty scale, and phase B has its own local-JVP tool"
        ),
    )
    parser.add_argument(
        "--external-h250-report", type=Path, default=None,
        help="deprecated alias for --external-residual-oracle-report",
    )
    parser.add_argument(
        "--external-residual-oracle-report", type=Path, default=None,
        help="pre-registered residual oracle report authorizing Phase C",
    )
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return result


def _initial_observation(state):
    return torch.cat((state.position, state.velocity, state.rotation.reshape(state.position.shape[0], 9),
                      state.omega, torch.zeros_like(state.position), state.previous_action), dim=-1)


def _validate_external_h250_report(path: Path, student: StructuredRecurrentPolicy) -> dict:
    """Validate a held-out Phase-B report before enabling residual DAgger."""

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid external H250 report: {path}") from exc
    if not isinstance(report, dict):
        raise RuntimeError("external H250 report must be a JSON object")
    if report.get("phase") != "structured-residual-oracle":
        raise RuntimeError("external report is not a structured residual oracle")
    current_hash = deployment_policy_hash(student)
    if report.get("phase_b_deployment_hash") != current_hash:
        raise RuntimeError("external H250 report policy hash does not match Phase-B checkpoint")
    if not bool(report.get("oracle_gate_passed", False)):
        raise RuntimeError("external residual oracle does not pass its registered gate")
    # New oracle reports carry one immutable contract for both action and
    # allocator release checks.  Keep accepting the old flat screening shape
    # for the unit-level helper, but a real Phase-C artifact will only be
    # promotable when the nested contract is complete.
    oracle = report.get("structured_residual_oracle")
    if not isinstance(oracle, dict):
        oracle = {
            "schema_version": 0,
            "pre_registered": bool(report.get("pre_registered", False)),
            "phase_b_deployment_hash": report.get("phase_b_deployment_hash"),
            "annulus": report.get("annulus"),
            "action_rms_threshold": report.get("preregistered_action_rms_threshold"),
        }
    annulus = oracle.get("annulus", report.get("annulus"))
    threshold = float(report.get("preregistered_action_rms_threshold", float("inf")))
    threshold = float(oracle.get(
        "whole_policy_action_rms_threshold",
        oracle.get("action_rms_threshold", threshold),
    ))
    if (not isinstance(annulus, dict) or not (0.0 < float(annulus.get("r_min", 0.0))
            < float(annulus.get("r_max", 0.0))) or not math.isfinite(threshold)
            or threshold <= 0.0):
        raise RuntimeError("external residual oracle has invalid annulus/threshold")
    if oracle.get("phase_b_deployment_hash", report.get("phase_b_deployment_hash")) != current_hash:
        raise RuntimeError("external oracle contract policy hash does not match Phase-B checkpoint")
    return {
        "path": str(path.resolve()), "policy_hash": current_hash,
        "oracle_gate_passed": True, "r_min": float(annulus["r_min"]),
        "r_max": float(annulus["r_max"]), "threshold": threshold,
        "oracle_contract": dict(oracle),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_identification_oracle_report(
    path: Path,
    *,
    source_checkpoint: Path,
    config: StructuredPolicyConfig,
) -> dict:
    """Validate the immutable K35 causal-identification release artifact.

    A multi-tau observer is an architectural change, so a banked Phase-A run
    may not silently fall back to a screening report or to an artifact made
    with a different checkpoint/cadence contract.  This check is deliberately
    fail-closed: the oracle tool can add fields over time, but the release
    fields below must be present and true before any banked training starts.
    """
    if config.motor_observer_bank_size == 0:
        return {"required": False, "path": None}
    if not path.is_file():
        raise RuntimeError(
            "banked observer training requires --identification-oracle-report "
            f"pointing to an existing formal report: {path}"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid identification oracle report: {path}") from exc
    if not isinstance(report, dict):
        raise RuntimeError("identification oracle report must be a JSON object")
    required = {
        "diagnostic", "probe_amplitude",
        "tau_grid_sizes", "cadence_semantics_version", "cadence_semantics",
        "checkpoint_sha256", "code_sha256",
    }
    missing = sorted(required.difference(report))
    if missing:
        raise RuntimeError(
            "identification oracle report is missing release fields: "
            + ", ".join(missing)
        )
    if report["diagnostic"] != "causal-identifier-sequence-oracle":
        raise RuntimeError("identification oracle has an unexpected diagnostic kind")
    # The collector's final sequence-artifact gate is deliberately false
    # while the production identifier-init artifact is being produced.  A1
    # therefore accepts its immutable pretraining contract; later stages can
    # still require the stronger formal fields.  Never treat a screening
    # report as either contract.
    final_ready = report.get("formal_eligible") is True and report.get("gate_passed") is True
    pretraining_ready = (
        report.get("requested_formal_shape") is True
        and report.get("pretraining_gate_passed") is True
    )
    if not (final_ready or pretraining_ready):
        raise RuntimeError(
            "identification oracle is neither formally eligible nor ready for "
            "production identifier pretraining"
        )
    if pretraining_ready:
        checks = report.get("pretraining_checks")
        required_checks = {
            "coverage", "paired_safety", "physics_ceiling", "physics_finite",
            "physics_branch_support", "finite_collection",
            "no_identification_failure", "zero_q2_parity",
        }
        if (not isinstance(checks, dict)
                or any(checks.get(name) is not True for name in required_checks)):
            raise RuntimeError("identification oracle pretraining contract is incomplete")
    if abs(float(report["probe_amplitude"])) > 1.0e-12:
        raise RuntimeError("identification oracle must use passive v5 with probe amplitude 0.0")
    grid_sizes = report["tau_grid_sizes"]
    try:
        normalized_grid_sizes = sorted(set(int(value) for value in grid_sizes))
    except (TypeError, ValueError):
        normalized_grid_sizes = []
    if normalized_grid_sizes != [35]:
        raise RuntimeError("identification oracle must certify the fixed K35 grid")
    # v2 is required explicitly.  Accept the two names used by the diagnostic
    # report during its schema transition, but never infer v2 from K=35 alone.
    grid_version = report.get("representation_grid_version", report.get("motor_tau_grid_version"))
    if grid_version is None:
        representation = report.get("representation_grid")
        if isinstance(representation, dict):
            grid_version = representation.get("version")
    try:
        grid_version_value = int(grid_version)
    except (TypeError, ValueError):
        grid_version_value = -1
    if grid_version_value != 2:
        raise RuntimeError("identification oracle must certify representation grid v2")
    if report["cadence_semantics_version"] != CADENCE_SEMANTICS_VERSION:
        raise RuntimeError("identification oracle cadence semantics do not match this code")
    cadence = report["cadence_semantics"]
    if not isinstance(cadence, dict) or not bool(cadence.get("call_index_completed_transitions")):
        raise RuntimeError("identification oracle lacks completed-transition cadence semantics")
    if list(cadence.get("publication_calls", ())) != [100, 125]:
        raise RuntimeError("identification oracle active publications must be call100/call125")
    if list(cadence.get("availability_t25", ())) != [0, 0, 0, 0, 0, 0]:
        raise RuntimeError("identification oracle must mark every t25 capability unavailable")
    if int(cadence.get("t50_call_index", -1)) != 100:
        raise RuntimeError("identification oracle t50 call index is invalid")
    checkpoint_hash = str(report["checkpoint_sha256"])
    if checkpoint_hash != _sha256_file(source_checkpoint):
        raise RuntimeError("identification oracle checkpoint hash does not match source checkpoint")
    code_hash = str(report["code_sha256"])
    if len(code_hash) != 64 or any(char not in "0123456789abcdef" for char in code_hash.lower()):
        raise RuntimeError("identification oracle code hash is malformed")
    return {
        "required": True,
        "path": str(path.resolve()),
        "diagnostic": report["diagnostic"],
        "checkpoint_sha256": checkpoint_hash,
        "code_sha256": code_hash,
        "representation_grid_version": 2,
        "gate_passed": bool(final_ready),
        "pretraining_gate_passed": bool(pretraining_ready),
    }


@torch.no_grad()
def _annulus_action_gate(student, episode, *, r_min: float, r_max: float,
                          threshold: float) -> tuple[list[dict], bool, float]:
    state = student.initial_state(episode.observations[0])
    errors, masks = [], []
    for step in range(episode.observations.shape[0]):
        output = student.forward_with_aux(
            episode.observations[step], state,
            applied_action=episode.executed_actions[step],
        )
        state = output.next_state
        radius = torch.linalg.vector_norm(output.auxiliary["feedback_features"], dim=-1)
        errors.append((output.action - episode.teacher_actions[step]).square().mean(dim=-1))
        masks.append((radius >= r_min) & (radius <= r_max))
    error = torch.stack(errors)
    mask = torch.stack(masks)
    counts = mask.sum(dim=0)
    rms = torch.sqrt((error * mask).sum(dim=0) / counts.clamp_min(1))
    rows = []
    for index, value in enumerate(rms.tolist()):
        rows.append({
            "scenario_index": index, "annulus_action_rms": float(value),
            "annulus_samples": int(counts[index]),
            "gate_passed": int(counts[index] > 0 and math.isfinite(value) and value <= threshold),
        })
    finite = bool(torch.isfinite(error).all())
    return rows, bool(finite and all(row["gate_passed"] for row in rows)), float(rms.max())


def main() -> None:
    args = parse_args()
    if (args.horizon < args.prefix + 100 or args.updates_per_beta < 1
            or args.maximum_buffer_episodes < len(DAGGER_BETA_SCHEDULE)):
        raise ValueError(
            "horizon must contain prefix plus 100 training steps and replay "
            "must retain at least one episode from every DAgger round"
        )
    if args.phase in ("A", "A1"):
        if args.capability_nll_weight != 0.0:
            raise ValueError("Phase A1 requires --capability-nll-weight 0")
        if args.capability_mean_weight <= 0.0:
            raise ValueError("Phase A1 requires a positive capability mean weight")
        probe_eligibility = probe_v5_eligibility(
            args.probe_v4_report, q2_checkpoint=args.source_checkpoint
        )
        if not probe_eligibility.get("eligible"):
            raise RuntimeError(
                "Phase A1 requires the current formal v4 freeze record "
                f"(reason={probe_eligibility.get('reason')})"
            )
    elif args.phase == "A2":
        if args.capability_nll_weight <= 0.0:
            raise ValueError("Phase A2 requires a positive capability NLL weight")
        if args.capability_mean_weight != 0.0:
            raise ValueError("Phase A2 requires --capability-mean-weight 0")
    device = _device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    teacher, teacher_args = load_q2_policy(args.source_checkpoint, device=device, dtype=torch.float32)
    teacher.eval()
    simulator = L2FSimulator(L2FParams(dt=float(teacher_args.get("dt", 0.01))))
    # Training rounds share one fixed bank for paired comparisons.  Calibration,
    # validation and the final gate use independent deterministic banks.
    train_bank_cpus = [
        build_dagger_scenario_bank(64, seed=args.seed + 100 + round, dt=simulator.params.dt)
        for round in range(len(DAGGER_BETA_SCHEDULE))
    ]
    calibration_bank_cpu = build_dagger_scenario_bank(64, seed=args.seed + 1, dt=simulator.params.dt)
    validation_bank_cpu = build_dagger_scenario_bank(64, seed=args.seed + 2, dt=simulator.params.dt)
    final_bank_cpu = build_dagger_scenario_bank(64, seed=args.seed + 3, dt=simulator.params.dt)

    def move_bank(bank_cpu):
        if device.type == "cpu":
            return bank_cpu
        from env_l2f import L2FState
        state = L2FState(**{
            name: getattr(bank_cpu.state, name).to(device)
            for name in bank_cpu.state.__dataclass_fields__
        })
        return type(bank_cpu)(state, bank_cpu.tw_bin.to(device),
                              bank_cpu.log_alpha_bin.to(device), bank_cpu.stratum)

    train_banks = [move_bank(item) for item in train_bank_cpus]
    bank = train_banks[0]
    calibration_bank = move_bank(calibration_bank_cpu)
    validation_bank = move_bank(validation_bank_cpu)
    final_bank = move_bank(final_bank_cpu)
    if args.phase in ("A2", "C") and args.student_checkpoint is None:
        required = "phase-A1" if args.phase == "A2" else "phase-B local-gain"
        raise RuntimeError(f"phase {args.phase} requires a {required} checkpoint")
    checkpoint_payload = None
    causal_revalidation_validated = False
    if args.student_checkpoint is not None:
        checkpoint_payload = torch.load(
            args.student_checkpoint, map_location=device, weights_only=False
        )
        previous_phase = checkpoint_payload.get("report", {}).get("active_phase")
        expected_phase = {"A2": "A1", "C": "B"}.get(args.phase)
        if expected_phase is not None and previous_phase != expected_phase:
            raise RuntimeError(
                f"phase {args.phase} requires a phase-{expected_phase} checkpoint, got {previous_phase!r}"
            )
        if args.phase == "A2" and not bool(
            checkpoint_payload.get("report", {}).get("phase_a_mean_gate_passed", False)
        ):
            raise RuntimeError("phase A2 requires phase_a_mean_gate_passed=true")
        if args.phase == "A2":
            previous_report = checkpoint_payload.get("report", {})
            if previous_report.get("causal_revalidation_required") is not True:
                raise RuntimeError(
                    "phase A2 requires an explicit stale causal gate from A1"
                )
            if previous_report.get("causal_gate_status") != "stale_revalidation_required":
                raise RuntimeError(
                    "phase A2 requires causal_gate_status=stale_revalidation_required"
                )
            if args.causal_revalidation_report is None:
                raise RuntimeError(
                    "phase A2 requires --causal-revalidation-report after A1 identifier updates"
                )
            validate_causal_revalidation_report(
                    args.causal_revalidation_report,
                    phase_a1_checkpoint=args.student_checkpoint,
                    a1_report=previous_report,
                )
            causal_revalidation_validated = True
    if checkpoint_payload is None:
        config = StructuredPolicyConfig(
            hidden_dim=args.hidden_dim, identifier_dim=args.identifier_dim,
            dt=simulator.params.dt, slow_cadence=args.slow_cadence,
            identification_publish_start=args.identification_publish_start,
            motor_observer_bank_size=args.motor_observer_bank_size,
            motor_observer_mode=args.motor_observer_mode,
            motor_tau_grid_version=args.motor_tau_grid_version,
            burn_in_probe_amplitude=args.burn_in_probe_amplitude,
            allocator_solver=args.allocator_solver,
            allocator_rate_limit=args.allocator_rate_limit,
            residual_scale=args.residual_scale,
        )
    else:
        valid_fields = {field.name for field in fields(StructuredPolicyConfig)}
        config_values = checkpoint_payload.get("config", {})
        config = StructuredPolicyConfig(**{
            key: value for key, value in config_values.items() if key in valid_fields
        })
        if args.phase == "A2" and (
            config.motor_observer_bank_size != args.motor_observer_bank_size
            or config.motor_observer_mode != args.motor_observer_mode
            or config.motor_tau_grid_version != args.motor_tau_grid_version
            or config.identification_publish_start != args.identification_publish_start
            or abs(config.burn_in_probe_amplitude - args.burn_in_probe_amplitude) > 1.0e-12
            or config.allocator_solver != args.allocator_solver
            or abs(config.allocator_rate_limit - args.allocator_rate_limit) > 1.0e-12
        ):
            raise RuntimeError(
                "Phase A2 observer-bank/probe/publication/allocator config must match Phase A1"
            )
        if args.phase == "C":
            if args.residual_scale <= 0.0:
                raise RuntimeError("phase C requires --residual-scale > 0")
            config = replace(
                config,
                residual_scale=args.residual_scale,
                residual_trainable=True,
            )
    oracle_contract = _validate_identification_oracle_report(
        args.identification_oracle_report,
        source_checkpoint=args.source_checkpoint,
        config=config,
    ) if config.motor_observer_bank_size else {"required": False, "path": None}
    student = StructuredRecurrentPolicy(config).to(device)
    if checkpoint_payload is not None:
        student.load_state_dict(checkpoint_payload["model"], strict=True)
    identifier_contract = None
    identifier_pretraining_contract = None
    if args.phase == "A2" and checkpoint_payload is not None:
        identifier_contract = checkpoint_payload.get("report", {}).get(
            "identifier_init_artifact"
        )
    if args.phase in ("A", "A1"):
        if args.identifier_init_artifact is None:
            raise RuntimeError(
                "Phase A1 requires --identifier-init-artifact; refusing an unbound identifier init"
            )
        identifier_contract = load_identifier_init_artifact(
            args.identifier_init_artifact,
            student,
            q2_checkpoint=args.source_checkpoint,
            causal_oracle_report=args.identification_oracle_report,
            probe_v4_report=args.probe_v4_report,
        )
        if args.identifier_pretraining_report is None:
            raise RuntimeError(
                "Phase A1 requires --identifier-pretraining-report; "
                "a structurally valid artifact alone does not certify calls100/125 accuracy"
            )
        identifier_pretraining_contract = validate_identifier_pretraining_report(
            args.identifier_pretraining_report,
            identifier_artifact=args.identifier_init_artifact,
            identifier_contract=identifier_contract,
            policy=student,
            q2_checkpoint=args.source_checkpoint,
            causal_oracle_report=args.identification_oracle_report,
        )
    external_h250 = None
    if args.phase == "C":
        oracle_report = args.external_residual_oracle_report or args.external_h250_report
        if oracle_report is None:
            raise RuntimeError("phase C residual training requires --external-residual-oracle-report")
        external_h250 = _validate_external_h250_report(oracle_report, student)
    active_phase = "A1" if args.phase == "A" else args.phase
    active_parameters = set_distillation_phase(student, active_phase)
    if not active_parameters:
        raise RuntimeError("selected distillation phase has no trainable parameters")
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=args.lr, weight_decay=1.0e-5)
    history = []
    final_episodes = []
    # Keep a stratified recurrent replay across all DAgger rounds.  A plain
    # FIFO of the last two episodes made the identifier chase the newest bank
    # and forget the previous authority strata/distributions.
    replay_by_round: dict[int, list] = {}
    replay_slots_per_round = max(
        1, args.maximum_buffer_episodes // len(DAGGER_BETA_SCHEDULE)
    )
    replay_buffer = []
    for round_index, beta in enumerate(DAGGER_BETA_SCHEDULE, start=1):
        episodes = []
        for _ in range(args.updates_per_beta):
            episode = collect_dagger_episode(
                teacher, student, simulator, train_banks[round_index - 1], beta=beta,
                horizon=args.horizon, episode_seed=args.seed + round_index * 1000,
            )
            round_replay = replay_by_round.setdefault(round_index, [])
            round_replay.append(episode)
            del round_replay[:-replay_slots_per_round]
            replay_buffer = [
                stored
                for replay_round in sorted(replay_by_round)
                for stored in replay_by_round[replay_round]
            ]
            buffered = [
                dagger_window_loss(
                    student, stored, prefix=args.prefix,
                    delta_action_weight=args.delta_action_weight,
                    capability_weight=args.capability_nll_weight,
                    capability_mean_weight=args.capability_mean_weight,
                    phase=active_phase,
                    body_z_weight=args.body_z_weight,
                )
                for stored in replay_buffer
            ]
            loss = torch.stack([item[0] for item in buffered]).mean()
            components = {
                key: sum(item[1][key] for item in buffered) / len(buffered)
                for key in ("action", "full_action", "analytic_trim", "body_z",
                            "disturbance", "one_step_equilibrium",
                            "capability_nll", "capability_mean", "delta_action",
                            "same_latent_intercept_diagnostic")
            }
            components["beta"] = beta
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
            if not bool(torch.isfinite(gradient_norm).item()):
                raise RuntimeError("non-finite DAgger gradient")
            optimizer.step()
            episodes.append(episode)
            history.append({"round": round_index, "beta": beta, "loss": float(loss.detach()),
                            "gradient_norm": float(gradient_norm.detach()), **components})
            history[-1].update({
                "train_bank_seed": args.seed + 100 + (round_index - 1),
                "teacher_execution_count": int(episode.intervention_mask.sum().item()),
                "student_execution_count": int(episode.intervention_mask.numel() - episode.intervention_mask.sum().item()),
                "intervention_mode": "teacher_or_student_per_episode",
                "identification_failure_t50_count": int(episode.identification_failure_t50.sum().item()),
            })
        final_episodes = episodes
    # Calibration and validation are held out from the training bank.  The
    # final gate is a third fresh bank collected after the last update.
    calibration_episode = collect_dagger_episode(
        teacher, student, simulator, calibration_bank, beta=0.0,
        horizon=args.horizon, episode_seed=args.seed + 9100,
    )
    validation_episode = collect_dagger_episode(
        teacher, student, simulator, validation_bank, beta=0.0,
        horizon=args.horizon, episode_seed=args.seed + 9200,
    )
    conformal_q, conformal_risk_rows = fit_effectiveness_conformal_q(
        calibration_episode,
        validation_episode=validation_episode,
        miscoverage=0.10,
        phase_steps=(100, 125),
    )
    # The 64-scenario DAgger run installs a smoke calibration so phase B has a
    # nonzero, uncertainty-gated contextual path.  It is never labeled a 99%
    # promotion calibration; that requires the separate 512-scenario tool.
    if active_phase == "A2":
        student.install_capability_conformal_q(
            conformal_q, sample_count=calibration_bank.count
        )
    final_episode = collect_dagger_episode(
        teacher, student, simulator, final_bank, beta=0.0,
        horizon=args.horizon, episode_seed=args.seed + 9300,
    )
    if active_phase == "C":
        gate_rows, gate_passed, annulus_rms_max = _annulus_action_gate(
            student, final_episode, r_min=external_h250["r_min"],
            r_max=external_h250["r_max"], threshold=external_h250["threshold"],
        )
    else:
        gate_rows, gate_passed = layered_dagger_gate(
            [final_episode], action_rms_threshold=args.action_gate,
            omega_max=args.omega_gate,
        )
        annulus_rms_max = None
    conformal_rows = conformal_coverage_by_stratum(
        calibration_episode, validation_episode=validation_episode,
    )
    final_teacher_execution = int(final_episode.intervention_mask.sum().item())
    if active_phase == "A2":
        equilibrium_gate, equilibrium_gate_passed = phase_a_equilibrium_gate(
            final_episode
        )
    elif active_phase == "A1":
        # Width is intentionally not an A1 decision: the scale head is frozen
        # at its broad initialization.  A1 reports physical mean diagnostics,
        # but only A2 can produce the promotable equilibrium/uncertainty gate.
        equilibrium_gate, phase_a_mean_gate_passed = phase_a_equilibrium_gate(
            final_episode, require_identification_width=False
        )
        equilibrium_gate["gate_deferred_to"] = "A2 uncertainty/width validation"
        equilibrium_gate["identification_failure_t50_ignored_in_A1"] = int(
            final_episode.identification_failure_t50.sum().item()
        )
        equilibrium_gate_passed = False
    else:
        inherited = checkpoint_payload.get("report", {}) if checkpoint_payload else {}
        equilibrium_gate = inherited.get("equilibrium_gate", {"gate_passed": False})
        equilibrium_gate_passed = bool(equilibrium_gate.get("gate_passed", False))
    if active_phase == "C":
        # Any learned residual/GRU change changes the on-policy identifier
        # distribution.  A q fitted before this stage is not a deployment
        # calibration for the new controller.
        student.invalidate_capability_calibration()
    # Four samples/cell is deliberately smoke-only; this tool never promotes a
    # checkpoint to migration without the external H250/release gate.
    migration_gate_passed = False
    payload = {
        "phase": "structured-recurrent-dagger",
        "cadence_semantics_version": CADENCE_SEMANTICS_VERSION,
        "cadence_semantics": {
            "call_index_completed_transitions": True,
            "call0_has_response": False,
            "publication_calls": [100, 125],
            "availability_t25": [0, 0, 0, 0, 0, 0],
            "publication_rule_after_first": "positive slow_cadence offsets from call100",
            "t50_call_index": 100,
        },
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "teacher_is_runtime_dependency": False,
        "teacher_is_training_dependency": True,
        "device": str(device), "seed": args.seed, "scenario_count": bank.count,
        "bank_seeds": {"train": [args.seed + 100 + round for round in range(len(DAGGER_BETA_SCHEDULE))], "calibration": args.seed + 1,
                       "validation": args.seed + 2, "final_gate": args.seed + 3},
        "tw_bins": 4, "log_alpha_bins": 4, "scenarios_per_cell": 4,
        "horizon": args.horizon, "prefix": args.prefix, "window": args.horizon - args.prefix,
        "beta_schedule": list(DAGGER_BETA_SCHEDULE), "final_beta": DAGGER_BETA_SCHEDULE[-1],
        "config": asdict(config), "history": history, "layered_gate_passed": gate_passed,
        "identification_oracle": oracle_contract,
        "identifier_init_artifact": identifier_contract,
        "identifier_pretraining_report": identifier_pretraining_contract,
        "identifier_weights_sha256_before": (
            identifier_contract["identifier_weights_sha256"]
            if identifier_contract is not None else None
        ),
        "identifier_weights_sha256_after": identifier_weights_hash(student),
        "identifier_updated_in_stage": bool(active_phase == "A1"),
        # Updating the production identifier invalidates the causal gate that
        # justified its initialization.  A separate, hash-bound report is
        # required before A2 can fit uncertainty.
        "causal_gate_status": (
            "stale_revalidation_required" if active_phase == "A1"
            else ("revalidated" if causal_revalidation_validated
                  else "revalidation_required_before_a2")
        ),
        "causal_gate_stale": bool(active_phase == "A1"),
        "causal_revalidation_required": bool(
            active_phase == "A1" or not causal_revalidation_validated
        ),
        "causal_gate_revalidated": bool(causal_revalidation_validated),
        "aggregated_replay_buffer_episodes": len(replay_buffer),
        "layered_gate_rows": gate_rows,
        "conformal_coverage_by_stratum": conformal_rows,
        "effectiveness_conformal_q": [float(value) for value in conformal_q],
        "effectiveness_conformal_risk_rows": conformal_risk_rows,
        "conformal_scope": "alpha_conditional_four_log_alpha_risk_strata; joint phase100_125 effectiveness dimensions",
        "validation_interpretation": "shift diagnostic only; not a coverage proof",
        "teacher_same_latent_intercept_is_equilibrium": False,
        "teacher_same_latent_intercept_usage": "diagnostic_only",
        "capability_calibration_promotion_eligible": False,
        "intervention_markers": {
            "teacher_execution_total": int(sum(row["teacher_execution_count"] for row in history)),
            "student_execution_total": int(sum(row["student_execution_count"] for row in history)),
            "per_episode_mask": True,
            "final_two_beta0": list(DAGGER_BETA_SCHEDULE[-2:]) == [0.0, 0.0],
        },
        "final_teacher_execution_count": final_teacher_execution,
        "active_phase": active_phase,
        "active_phase_components": list(distillation_phase_components(active_phase)),
        "equilibrium_gate": equilibrium_gate,
        "equilibrium_gate_passed": equilibrium_gate_passed,
        "phase_a_mean_gate_passed": bool(
            active_phase == "A1" and phase_a_mean_gate_passed
        ) if active_phase == "A1" else bool(
            checkpoint_payload is not None
            and checkpoint_payload.get("report", {}).get("phase_a_mean_gate_passed", False)
        ),
        "capability_calibration_installed_in_stage": bool(active_phase == "A2"),
        "external_h250_gate": bool(external_h250 is not None),
        "external_h250_report": external_h250,
        # Carry the exact pre-registered contract through Phase C.  It is
        # intentionally not refit on Phase-C data; post-calibration evidence
        # is required to remeasure against this same annulus/threshold.
        "structured_residual_oracle": (
            external_h250["oracle_contract"] if external_h250 is not None else None
        ),
        "phase_c_deployment_hash": deployment_policy_hash(student),
        "phase_c_annulus": external_h250 if active_phase == "C" else None,
        "phase_c_annulus_action_rms_max": annulus_rms_max,
        "phase_c_whole_policy_gate_passed": bool(args.phase == "C" and gate_passed),
        "phase_c_gate_passed": bool(
            active_phase == "C" and gate_passed and final_teacher_execution == 0
            and external_h250 is not None
            and list(DAGGER_BETA_SCHEDULE[-2:]) == [0.0, 0.0]
        ),
        "active_parameter_names": list(active_parameters),
        "migration_gate_passed": migration_gate_passed,
        "migration_gate_reason": "prototype_smoke_only; conformal/release gate is not promoted",
        "supervision_contract": {
            "same_hidden_uT0": False,
            "same_latent_intercept": "diagnostic_only; not equilibrium and not a training target",
            "analytic_motor_trim_target": True,
            "analytic_body_z_target": True,
            "explicit_physical_disturbance_observer": True,
            "absolute_fast_state_in_slow_context": False,
            "delta_action": "teacher_action_minus_analytic_motor_trim",
            "delta_w": False,
            "migration_blocker": "local-JVP Phase B, Phase C, final calibration, and external release gates remain",
        },
        # This MVP intentionally does not claim a JVP/one-step solver or a
        # validated migration certificate.  Downstream release gates must
        # remain false until those independent checks are supplied.
        "unsupported_gate": False,
        "unsupported_gates": {"jvp": False, "one_step": False, "certificate": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": "structured-recurrent-motor-policy", "model": student.state_dict(),
                "config": asdict(config), "fast_feedback_verified": student.fast_feedback.verified,
                "report": payload}, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output), "report": str(args.report), "layered_gate_passed": gate_passed}, sort_keys=True))


if __name__ == "__main__":
    main()

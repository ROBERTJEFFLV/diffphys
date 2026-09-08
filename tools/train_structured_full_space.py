from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_l2f import L2FParams, L2FSimulator  # noqa: E402
from structured_distillation import build_dagger_scenario_bank  # noqa: E402
from full_space_shooting import (  # noqa: E402
    FullSpaceProblem,
    joint_jvp,
    solve_joint_sqp_step,
)
from structured_rollout import (  # noqa: E402
    ParameterVectorSpec,
    StructuredBoundaryCodec,
    StructuredClosedLoopState,
    build_action_probe_bank,
    functional_probe_actions,
    functional_trajectory_actions,
    functional_trajectory_diagnostics,
    initialize_exact_endpoints,
    load_structured_policy,
    make_segment_map,
    phase_space_contraction_risk_residual,
    rollout_structured_segment,
)
from structured_stability import default_phase_space_metric  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matrix-free full-space structured-policy shooting step.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--segment-steps", type=int, default=250)
    parser.add_argument(
        "--pre-rollout-steps", type=int, default=50,
        help="complete the identifier/burn-in before creating shooting nodes",
    )
    parser.add_argument("--segments", type=int, choices=(2, 4), default=2)
    parser.add_argument("--outer-steps", type=int, default=1)
    parser.add_argument("--damping", type=float, default=100.0)
    parser.add_argument("--penalty", type=float, default=10.0)
    parser.add_argument("--cg-iterations", type=int, default=4)
    parser.add_argument("--max-backtracks", type=int, default=8)
    parser.add_argument("--parameter-radius", type=float, default=0.05)
    parser.add_argument("--parameter-scale-floor", type=float, default=1.0e-2)
    parser.add_argument(
        "--trainable-prefix",
        action="append",
        default=None,
        help="repeatable explicit policy-parameter prefix; frozen parameters are always excluded",
    )
    parser.add_argument("--action-radius", type=float, default=0.01)
    parser.add_argument("--action-max-radius", type=float, default=0.05)
    parser.add_argument("--action-probe-stride", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--risk-beta", type=float, default=1.0e-2)
    parser.add_argument("--position-natural-frequency", type=float, default=1.5)
    parser.add_argument("--position-damping-ratio", type=float, default=0.9)
    parser.add_argument("--tilt-natural-frequency", type=float, default=5.0)
    parser.add_argument("--tilt-damping-ratio", type=float, default=1.0)
    parser.add_argument("--minimum-ratio", type=float, default=0.25)
    parser.add_argument("--maximum-ratio", type=float, default=2.0)
    parser.add_argument("--maximum-linearized-constraint", type=float, default=1.0e-3)
    parser.add_argument("--maximum-relative-kkt", type=float, default=1.0e-2)
    parser.add_argument("--maximum-rejections", type=int, default=3)
    parser.add_argument("--minimum-radius", type=float, default=1.0e-4)
    parser.add_argument(
        "--heldout-seed-offset", type=int, default=1_000_003,
        help="disjoint authority-stratified bank used only for nonlinear update acceptance",
    )
    parser.add_argument(
        "--heldout-maximum-risk-ratio", type=float, default=1.0,
        help="maximum candidate/reference phase-space risk on the held-out bank",
    )
    parser.add_argument("--allow-small-risk-smoke", action="store_true")
    parser.add_argument("--allow-failed-migration-gate", action="store_true")
    from structured_training_runtime import add_training_arguments
    add_training_arguments(parser)
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _initial_observation(state) -> torch.Tensor:
    return torch.cat(
        (
            state.position,
            state.velocity,
            state.rotation.reshape(state.position.shape[0], 9),
            state.omega,
            torch.zeros_like(state.position),
            state.previous_action,
        ),
        dim=-1,
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    if args.final_evaluation:
        raise ValueError("MS acceptance is development data; final release is the separate post-MS pipeline")
    if args.outer_steps < 1 or args.batch_size < 1:
        raise ValueError("outer steps and batch size must be positive")
    if args.heldout_seed_offset == 0 or args.heldout_maximum_risk_ratio <= 0.0:
        raise ValueError("held-out seed offset must be nonzero and risk ratio positive")
    effective_tail = int(torch.ceil(torch.tensor(
        (1.0 - float(args.alpha)) * float(args.batch_size)
    )).item())
    if effective_tail < 8 and not args.allow_small_risk_smoke:
        raise ValueError(
            "full-space risk update needs at least eight effective tail scenarios; "
            "use --allow-small-risk-smoke only for solver smoke tests"
        )
    device = _device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    policy, source = load_structured_policy(args.checkpoint, device=device)
    from structured_checkpoint import require_formal_identification_config
    if not args.allow_failed_migration_gate:
        require_formal_identification_config(policy.config)
    migration_passed = source.get("report", {}).get("migration_gate_passed") is True
    if not migration_passed and not args.allow_failed_migration_gate:
        raise RuntimeError(
            "structured checkpoint failed the Q2 migration gate; refusing long-horizon optimization"
        )
    if args.segment_steps % policy.config.slow_cadence != 0:
        raise ValueError("segment steps must be divisible by the slow cadence")
    if args.pre_rollout_steps % policy.config.slow_cadence != 0:
        raise ValueError("pre-rollout steps must align with the slow cadence")
    safe_prefixes = ("contextual_gain_head.", "residual_head.")
    if args.trainable_prefix is None:
        if not args.allow_failed_migration_gate:
            raise ValueError("formal full-space shooting requires an explicit trainable prefix")
    elif any(prefix not in safe_prefixes for prefix in args.trainable_prefix):
        raise ValueError("full-space trainable prefixes are restricted to contextual gain/residual")
    if args.trainable_prefix is not None:
        # ``requires_grad`` is not serialized by state_dict.  Make the CLI
        # allowlist authoritative after checkpoint loading: no identifier,
        # equilibrium or allocator parameter may silently enter the SQP step.
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(
                any(name.startswith(prefix) for prefix in args.trainable_prefix)
            )
    if args.action_probe_stride != 1 and not args.allow_small_risk_smoke:
        raise ValueError("formal action trust region must evaluate every reference action")

    from structured_training_runtime import TrainingSession
    session = TrainingSession(args, policy, None, stage="fullspace" + str(args.segments))
    simulator = L2FSimulator(L2FParams(dt=policy.config.dt))

    def scenario_state(seed: int):
        if args.allow_small_risk_smoke:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            return simulator.reset(
                args.batch_size,
                device=device,
                dtype=torch.float32,
                sample_dynamics=True,
                sampled_dynamics_level="broad",
                broad_sampler="physical-fit",
                balanced_dynamics_sampling=False,
                sample_external_force=True,
            )
        if args.batch_size % 16:
            raise ValueError(
                "formal full-space risk batch must be divisible by 16 for the "
                "predeclared 4x4 authority strata"
            )
        bank = build_dagger_scenario_bank(
            args.batch_size,
            seed=seed,
            dt=policy.config.dt,
            per_cell=args.batch_size // 16,
        )
        return type(bank.state)(**{
            name: getattr(bank.state, name).to(device=device, dtype=torch.float32)
            for name in bank.state.__dataclass_fields__
        })

    physical = scenario_state(args.seed)
    heldout_seed = args.seed + args.heldout_seed_offset
    heldout_physical = scenario_state(heldout_seed)
    risk_sampler = (
        "natural-physical-fit-smoke" if args.allow_small_risk_smoke
        else "authority-stratified-4x4-physical-fit"
    )

    def initial_closed(state):
        recurrent = policy.initial_state(_initial_observation(state))
        return StructuredClosedLoopState(physical=state, policy=recurrent)

    closed = initial_closed(physical)
    heldout_closed = initial_closed(heldout_physical)
    if args.pre_rollout_steps < 0:
        raise ValueError("pre-rollout steps must be non-negative")
    if args.pre_rollout_steps:
        with torch.no_grad():
            closed, _ = rollout_structured_segment(
                policy, simulator, closed, steps=args.pre_rollout_steps, collect=False
            )
            heldout_closed, _ = rollout_structured_segment(
                policy, simulator, heldout_closed,
                steps=args.pre_rollout_steps, collect=False,
            )
    identification_failed = closed.policy.identification_failed
    if (
        identification_failed is not None
        and bool(identification_failed.any().item())
        and not args.allow_failed_migration_gate
    ):
        raise RuntimeError(
            "identifier confidence failed before shooting; refusing to remove the burn-in guard"
        )
    boot_total = policy.config.identification_publish_start
    boot_completed = args.pre_rollout_steps >= boot_total
    if not boot_completed and not args.allow_failed_migration_gate:
        raise ValueError(
            "formal full-space shooting must start after the complete policy burn-in"
        )
    codec = StructuredBoundaryCodec(physical, policy, boot_completed=boot_completed)
    heldout_codec = StructuredBoundaryCodec(
        heldout_physical, policy, boot_completed=boot_completed
    )
    parameter_spec = ParameterVectorSpec.from_module(
        policy, trainable_only=True, allow_prefixes=args.trainable_prefix
    )
    if not parameter_spec.names:
        raise ValueError("the full-space trainable allowlist selected no parameters")
    theta = parameter_spec.flatten(policy).detach()
    parameter_scale = parameter_spec.trust_scale(
        policy, floor=args.parameter_scale_floor
    )
    segment_map = make_segment_map(
        policy,
        simulator,
        codec,
        parameter_spec,
        steps=args.segment_steps,
    )
    heldout_segment_map = make_segment_map(
        policy,
        simulator,
        heldout_codec,
        parameter_spec,
        steps=args.segment_steps,
    )
    initial = codec.pack(closed).detach()
    heldout_initial = heldout_codec.pack(heldout_closed).detach()
    saved_solver = session.progress.get("solver")
    if saved_solver is not None:
        initial, heldout_initial = saved_solver["initial"], saved_solver["heldout_initial"]
        theta, endpoints = saved_solver["theta"], saved_solver["endpoints"]
    else:
        endpoints = initialize_exact_endpoints(segment_map, initial, theta, args.segments)
    metric = default_phase_space_metric(
        dt=policy.config.dt,
        sample_steps=args.segment_steps,
        position_natural_frequency=args.position_natural_frequency,
        position_damping_ratio=args.position_damping_ratio,
        tilt_natural_frequency=args.tilt_natural_frequency,
        tilt_damping_ratio=args.tilt_damping_ratio,
        device=device,
        dtype=theta.dtype,
    )
    rows = session.progress.setdefault("solver_rows", [])
    saved_solver = saved_solver or {}
    damping = float(saved_solver.get("damping", args.damping))
    parameter_radius = float(saved_solver.get("parameter_radius", args.parameter_radius))
    action_radius = float(saved_solver.get("action_radius", args.action_radius))
    consecutive_rejections = int(saved_solver.get("consecutive_rejections", 0))
    session.save()
    for outer in range(session.updates + 1, args.outer_steps + 1):
        if session.should_stop():
            break
        problem = FullSpaceProblem(
            initial=initial,
            boundaries=endpoints,
            theta=theta,
            segment_map=segment_map,
            layout=codec.layout,
            fixed_boundary_mask=codec.fixed_boundary_mask,
            task_residual=lambda starts, ends, value: phase_space_contraction_risk_residual(
                codec, starts, ends,
                metric=metric.matrix,
                retention=metric.linear_model_retention,
                alpha=args.alpha,
                beta=args.risk_beta,
            ),
        )
        action_probes = build_action_probe_bank(
            policy, simulator, codec, parameter_spec, theta, initial, endpoints,
            steps=args.segment_steps, stride=args.action_probe_stride,
        )
        action_evaluator = lambda value, _nodes: functional_probe_actions(
            policy, codec, parameter_spec, value, action_probes
        )
        step = solve_joint_sqp_step(
            problem, linear_solver="legacy-cg",
            damping=damping,
            penalty=args.penalty,
            cg_iterations=args.cg_iterations,
            parameter_radius=parameter_radius,
            parameter_scale=parameter_scale,
            action_radius=action_radius,
            action_max_radius=args.action_max_radius,
            action_evaluator=action_evaluator,
            max_backtracks=args.max_backtracks,
            minimum_ratio=args.minimum_ratio,
        )
        row = asdict(step)
        row.pop("theta")
        row.pop("boundaries")
        row["outer_step"] = outer
        row["attempt_parameter_radius"] = parameter_radius
        row["attempt_action_radius"] = action_radius
        preliminary_gate = bool(
            step.accepted
            and not step.linear_solver_breakdown
            and args.minimum_ratio <= step.ratio <= args.maximum_ratio
            and step.linearized_constraint_relative <= args.maximum_linearized_constraint
            and step.kkt_stationarity_relative <= args.maximum_relative_kkt
        )
        row["preliminary_solver_gate_passed"] = preliminary_gate
        row["restoration_gate_passed"] = False
        row["heldout_gate_passed"] = False
        if preliminary_gate:
            proposed_theta = step.theta.detach()
            # Nonlinear restoration: trajectory-local nodes are recomputed for
            # the accepted policy.  The restored continuous rollout, not the
            # free-node trial, decides whether the policy update is retained.
            proposed_endpoints = initialize_exact_endpoints(
                segment_map, initial, proposed_theta, args.segments
            )
            restored_theta_direction = proposed_theta - theta
            restored_boundary_direction = proposed_endpoints - endpoints
            baseline_constraint = problem.defects(endpoints, theta).detach()
            baseline_task_for_model = problem.task_values(endpoints, theta).detach()
            restored_linear_constraint = baseline_constraint + joint_jvp(
                problem,
                restored_theta_direction,
                restored_boundary_direction,
                kind="constraint",
            ).detach()
            restored_linear_task = baseline_task_for_model + joint_jvp(
                problem,
                restored_theta_direction,
                restored_boundary_direction,
                kind="task",
            ).detach()
            restored_direction_norm2 = (
                restored_theta_direction.square().sum()
                + restored_boundary_direction.square().sum()
            )
            restored_predicted = (
                0.5 * (
                    baseline_task_for_model.square().sum()
                    - restored_linear_task.square().sum()
                )
                + 0.5 * float(args.penalty) * (
                    baseline_constraint.square().sum()
                    - restored_linear_constraint.square().sum()
                )
                - 0.5 * damping * restored_direction_norm2
            )
            restored_predicted_value = float(restored_predicted)
            restored = FullSpaceProblem(
                initial, proposed_endpoints, proposed_theta, segment_map, codec.layout,
                task_residual=problem.task_residual,
            )
            heldout_baseline_endpoints = initialize_exact_endpoints(
                heldout_segment_map, heldout_initial, theta, args.segments
            )
            heldout_proposed_endpoints = initialize_exact_endpoints(
                heldout_segment_map, heldout_initial, proposed_theta, args.segments
            )
            heldout_task = lambda starts, ends, value: phase_space_contraction_risk_residual(
                heldout_codec, starts, ends,
                metric=metric.matrix,
                retention=metric.linear_model_retention,
                alpha=args.alpha,
                beta=args.risk_beta,
            )
            with torch.no_grad():
                baseline_merit, _, baseline_task = problem.merit(
                    endpoints, theta, args.penalty
                )
                restored_merit, restored_defects, restored_task = restored.merit(
                    proposed_endpoints, proposed_theta, args.penalty
                )
                restored_actions, restored_failures = functional_trajectory_diagnostics(
                    policy, simulator, codec, parameter_spec, proposed_theta,
                    initial, proposed_endpoints, steps=args.segment_steps,
                )
                baseline_actions = functional_trajectory_actions(
                    policy, simulator, codec, parameter_spec, theta,
                    initial, endpoints, steps=args.segment_steps,
                )
                restored_action_rms = torch.sqrt(
                    (restored_actions - baseline_actions).square().mean()
                )
                restored_action_max = (
                    restored_actions - baseline_actions
                ).abs().reshape(-1, args.batch_size, 4).amax(dim=(0, 2)).max()
                restored_identification_failed = restored_failures.any()
                heldout_baseline = FullSpaceProblem(
                    heldout_initial, heldout_baseline_endpoints, theta,
                    heldout_segment_map, heldout_codec.layout,
                    task_residual=heldout_task,
                )
                heldout_proposed = FullSpaceProblem(
                    heldout_initial, heldout_proposed_endpoints, proposed_theta,
                    heldout_segment_map, heldout_codec.layout,
                    task_residual=heldout_task,
                )
                heldout_baseline_residual = heldout_baseline.task_values(
                    heldout_baseline_endpoints, theta
                )
                heldout_proposed_residual = heldout_proposed.task_values(
                    heldout_proposed_endpoints, proposed_theta
                )
                heldout_baseline_risk = 0.5 * heldout_baseline_residual.square().sum()
                heldout_proposed_risk = 0.5 * heldout_proposed_residual.square().sum()
                heldout_risk_ratio = heldout_proposed_risk / heldout_baseline_risk.clamp_min(
                    1.0e-12
                )
                heldout_baseline_actions = functional_trajectory_actions(
                    policy, simulator, heldout_codec, parameter_spec, theta,
                    heldout_initial, heldout_baseline_endpoints,
                    steps=args.segment_steps,
                )
                heldout_proposed_actions, heldout_proposed_failures = functional_trajectory_diagnostics(
                    policy, simulator, heldout_codec, parameter_spec, proposed_theta,
                    heldout_initial, heldout_proposed_endpoints,
                    steps=args.segment_steps,
                )
                heldout_action_delta = heldout_proposed_actions - heldout_baseline_actions
                heldout_action_rms = heldout_action_delta.square().mean().sqrt()
                heldout_action_max = heldout_action_delta.abs().max()
                heldout_identification_failed = heldout_proposed_failures.any()
            row.update(
                {
                    "restored_defect_norm": float(torch.linalg.vector_norm(restored_defects)),
                    "restored_task_norm": float(torch.linalg.vector_norm(restored_task)),
                    "baseline_task_norm": float(torch.linalg.vector_norm(baseline_task)),
                    "restored_merit": float(restored_merit),
                    "baseline_merit": float(baseline_merit),
                    "restored_action_rms": float(restored_action_rms),
                    "restored_action_max_per_scenario": float(restored_action_max),
                    "restored_identification_failed": bool(
                        restored_identification_failed.item()
                    ),
                    "restored_model_predicted_reduction": restored_predicted_value,
                    "heldout_seed": heldout_seed,
                    "heldout_baseline_risk": float(heldout_baseline_risk),
                    "heldout_proposed_risk": float(heldout_proposed_risk),
                    "heldout_risk_ratio": float(heldout_risk_ratio),
                    "heldout_action_rms": float(heldout_action_rms),
                    "heldout_action_max": float(heldout_action_max),
                    "heldout_identification_failed": bool(
                        heldout_identification_failed.item()
                    ),
                }
            )
            heldout_gate = bool(
                torch.isfinite(heldout_proposed_risk)
                and float(heldout_risk_ratio) <= args.heldout_maximum_risk_ratio
                and float(heldout_action_rms) <= action_radius
                and float(heldout_action_max) <= args.action_max_radius
                and not bool(heldout_identification_failed.item())
            )
            row["heldout_gate_passed"] = heldout_gate
            restoration_gate = bool(
                torch.isfinite(restored_merit)
                and float(restored_merit) < float(baseline_merit)
                and float(restored_action_rms) <= action_radius
                and float(restored_action_max) <= args.action_max_radius
                and float(torch.linalg.vector_norm(restored_defects)) <= 1.0e-6
                and not bool(restored_identification_failed.item())
                and heldout_gate
            )
            restored_actual = float(baseline_merit - restored_merit)
            restored_ratio = (
                restored_actual / restored_predicted_value
                if restored_predicted_value > 0.0 else 0.0
            )
            row["restored_actual_reduction"] = restored_actual
            row["restored_ratio"] = restored_ratio
            restoration_gate = bool(
                restoration_gate
                and args.minimum_ratio <= restored_ratio <= args.maximum_ratio
            )
            row["restoration_gate_passed"] = restoration_gate
            if restoration_gate:
                theta = proposed_theta
                endpoints = proposed_endpoints
        row["solver_gate_passed"] = bool(
            preliminary_gate and row["restoration_gate_passed"]
        )
        rows.append(row)
        stop_after_step = False
        if rows[-1]["solver_gate_passed"]:
            consecutive_rejections = 0
            accepted_ratio = float(row["restored_ratio"])
            if accepted_ratio > 0.75:
                damping = max(1.0e-12, 0.5 * damping)
                parameter_radius *= 1.5
                action_radius *= 1.5
        else:
            consecutive_rejections += 1
            damping *= 10.0
            parameter_radius *= 0.5
            action_radius *= 0.5
            if (
                consecutive_rejections >= args.maximum_rejections
                or min(parameter_radius, action_radius) < args.minimum_radius
            ):
                stop_after_step = True
        parameter_spec.assign_(policy, theta)
        session.progress["solver"] = {"initial": initial, "heldout_initial": heldout_initial,
            "theta": theta, "endpoints": endpoints, "damping": damping,
            "parameter_radius": parameter_radius, "action_radius": action_radius,
            "consecutive_rejections": consecutive_rejections}
        session.record_update({"accepted": row["solver_gate_passed"], "outer_step": outer})
        session.save()
        if stop_after_step:
            break

    parameter_spec.assign_(policy, theta)
    accepted_steps = sum(int(row["solver_gate_passed"]) for row in rows)
    if accepted_steps:
        policy.invalidate_capability_calibration()
    # This gate certifies only that the requested horizon admitted at least
    # one numerically acceptable full-space step.  It deliberately does not
    # re-promote the changed controller: any accepted theta update still has
    # to repeat calibration, post-calibration evidence and paired migration.
    formal_gate_passed = bool(
        accepted_steps >= 1
        and effective_tail >= 8
        and boot_completed
        and args.action_probe_stride == 1
        and not args.allow_small_risk_smoke
        and not args.allow_failed_migration_gate
    )
    # Keep immutable, pre-registered controller evidence (for example the
    # residual annulus/oracle definition) so the mandatory post-update
    # calibration and postcheck can re-evaluate the *same* claims.  All stale
    # pass bits below are explicitly reset.
    payload = {
        **source.get("report", {}),
        "phase": "matrix-free-full-space-trust-region-shooting-mvp",
        "solver_scope": (
            "algebraic matrix-free joint SQP MVP; independent shooting nodes, "
            "but no production block preconditioner"
        ),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_migration_gate_passed": migration_passed,
        # Any accepted policy change invalidates the closed-loop migration and
        # capability calibration evidence.  A separate frozen-policy release
        # gate must promote the newly written checkpoint.
        "migration_gate_passed": False,
        "postcheck_gate_passed": False,
        "equilibrium_gate_passed": False,
        "phase_b_gate_passed": False,
        "phase_c_gate_passed": False,
        "pre_update_controller_evidence_invalidated": bool(accepted_steps > 0),
        "requires_post_update_release_gate": True,
        "capability_calibration_requires_refresh": bool(
            sum(int(row["solver_gate_passed"]) for row in rows) > 0
        ),
        "failed_gate_override": bool(args.allow_failed_migration_gate and not migration_passed),
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "segment_steps": args.segment_steps,
        "pre_rollout_steps": args.pre_rollout_steps,
        "boot_completed_before_shooting": boot_completed,
        "segments": args.segments,
        "risk_alpha": args.alpha,
        "effective_tail_count": effective_tail,
        "risk_sampler": risk_sampler,
        "training_scenario_seed": args.seed,
        "heldout_scenario_seed": heldout_seed,
        "heldout_seed_offset": args.heldout_seed_offset,
        "heldout_scenario_count": args.batch_size,
        "heldout_maximum_risk_ratio": args.heldout_maximum_risk_ratio,
        "heldout_bank_used_for_direction": False,
        "heldout_bank_used_for_acceptance_only": True,
        "small_risk_smoke_override": bool(args.allow_small_risk_smoke),
        "boundary_state_dim": codec.state_dim,
        "policy_parameter_dim": int(theta.numel()),
        "candidate_metric_retention": metric.linear_model_retention,
        "candidate_metric_equation_residual": metric.equation_residual,
        "trainable_parameter_names": list(parameter_spec.names),
        "parameter_trust_norm": "block-scaled-rms",
        "rows": rows,
        "training_session": session.summary(),
        "acceptance_bank_is_development_data": True,
        "accepted_steps": accepted_steps,
        "formal_gate_passed": formal_gate_passed,
        "formal_gate_scope": (
            "solver-and-restored-rollout acceptance only; the updated policy "
            "is not a release checkpoint until calibration/postcheck/migration repeat"
        ),
        "consecutive_rejections_at_stop": consecutive_rejections,
        "final_damping": damping,
        "final_parameter_radius": parameter_radius,
        "final_action_radius": action_radius,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "structured-recurrent-motor-policy",
            "model": policy.state_dict(),
            "config": source["config"],
            "fast_feedback_verified": policy.fast_feedback.verified,
            "report": payload,
        },
        args.output,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "accepted_steps": payload["accepted_steps"], "rows": rows}, sort_keys=True))


if __name__ == "__main__":
    import os
    historical = (
        "--historical-q2-distillation" in sys.argv
        or os.environ.get("DIFFPHYS_HISTORICAL_Q2") == "1"
    )
    if not historical:
        from tools.train_response_control import main as response_task_main
        raise SystemExit(response_task_main(["--optimizer", "full-space-ms", *sys.argv[1:]]))
    if "--historical-q2-distillation" in sys.argv:
        sys.argv.remove("--historical-q2-distillation")
    main()

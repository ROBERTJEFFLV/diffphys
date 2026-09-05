"""Versioned teacher-free research training, evaluation, and resumable state."""
from __future__ import annotations

from dataclasses import asdict, fields, replace
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

from env_l2f import L2FParams, L2FSimulator
from response_policy import ARCHITECTURE, ResponseMotorPolicy, ResponsePolicyConfig
from response_task import (
    TaskLossConfig, observation, prediction_residual, rollout, sample_scenarios,
    scenario_costs, task_loss, trajectory_metrics,
)

ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "response-control-task-v1"
TRAIN_SEED_BASE = 31_000_007
DEVELOPMENT_SEEDS = (32_000_007, 32_010_007)
MS_ACCEPTANCE_SEED = 33_000_007
FINAL_SEEDS = (34_000_007, 34_010_007)
FINAL_HORIZONS = (125, 500, 1000)
FINAL_CLAIM = ROOT / "reports" / "response_control_v1_final_claim.json"
SOURCE_FILES = (
    "response_policy.py", "response_task.py", "response_shooting.py",
    "response_training.py", "tools/train_response_control.py",
    "env_l2f.py", "full_space_shooting.py",
    "tools/check_response_training_contract.py", "tools/check_response_preflight.py",
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_hash(policy) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_hash() -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update(name.encode("ascii"))
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def capture_rng() -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("this exact RNG resume requires CUDA")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def protocol(policy_config, loss_config, scenarios) -> dict:
    return {
        "version": PROTOCOL_VERSION, "architecture": ARCHITECTURE,
        "model_training_seed": 7, "train_scenario_seed_base": TRAIN_SEED_BASE,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "ms_acceptance_seed": MS_ACCEPTANCE_SEED,
        "final_seeds": list(FINAL_SEEDS), "final_horizons": list(FINAL_HORIZONS),
        "scenarios_per_bank": scenarios, "stratification": "4x4 TW/log-roll-authority",
        "policy": asdict(policy_config), "loss": asdict(loss_config),
        "observation": "world-p/world-v/R/body-omega/body-integral/executed-previous-action",
        "action_mapping": "normalized_per-airframe_hover_relative; zero is hover without external force",
        "action_mapping_requires_airframe_calibration_for_real_actuators": True,
        "startup": "memory updates from the first completed transition; no capability publication gate",
        "success": {"position": 0.05, "velocity": 0.10, "full_omega": 0.50,
                    "consecutive_steps": loss_config.steady_steps},
        "parameter_family": "registered physical-fit simulator, fixed dynamics per episode",
        "no_absolute_yaw_or_horizontal_attitude_objective": True,
        "formal_v5_freeze_not_a_prerequisite_or_certificate": True,
        "deployment_authorized": False,
    }


def binding(args, policy_config, loss_config) -> dict:
    return {
        "source_sha256": source_hash(), "protocol": protocol(policy_config, loss_config, args.scenarios),
        "optimizer": args.optimizer, "horizon": args.horizon, "segments": args.segments,
        "segment_steps": args.segment_steps, "lr": args.lr,
        "weight_decay": args.weight_decay, "gradient_clip": args.gradient_clip,
        "device": args.device, "dtype": args.dtype, "torch_version": str(torch.__version__),
        "cg_iterations": args.cg_iterations, "parameter_radius": args.parameter_radius,
        "action_radius": args.action_radius, "action_max_radius": args.action_max_radius,
        "development_every": args.development_every,
        "contract_sha256": None if args.contract_report is None else file_hash(args.contract_report),
    }


def save_training(path, policy, optimizer, progress, solver, run_binding, initial_model):
    atomic_torch(path, {
        "schema": PROTOCOL_VERSION, "architecture": ARCHITECTURE,
        "model": policy.state_dict(), "model_sha256": model_hash(policy),
        "policy_config": asdict(policy.config), "binding": run_binding,
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "rng": capture_rng(), "progress": copy.deepcopy(progress), "solver": solver,
        "initial_model": initial_model, "deployment_authorized": False,
        "formal_eligible": False,
    })


def load_policy_checkpoint(path, device, dtype):
    value = torch.load(path, map_location="cpu")
    if value.get("schema") != PROTOCOL_VERSION or value.get("architecture") != ARCHITECTURE:
        raise ValueError("not a response-task checkpoint; Q2/distillation weights are not accepted")
    policy = ResponseMotorPolicy(ResponsePolicyConfig(**value["policy_config"])).to(device=device, dtype=dtype)
    policy.load_state_dict(value["model"])
    # Float64/float32 conversion is allowed only for explicit weights-only
    # initialization; evaluation/resume validate the stored dtype separately.
    return policy, value


def task_gradient_norms(loss, policy):
    named = [(name, p) for name, p in policy.named_parameters() if not name.startswith("response_predictor.")]
    gradients = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
    squared = {"memory": 0.0, "controller": 0.0}
    for (name, _), gradient in zip(named, gradients):
        key = "controller" if name.startswith("controller.") else "memory"
        if gradient is not None:
            squared[key] += float(gradient.detach().double().square().sum())
    return {"task_" + name + "_gradient_norm": math.sqrt(value) for name, value in squared.items()}


def _reference_rollout(checkpoint, simulator, initial, horizon):
    # This import and checkpoint load exist ONLY in explicitly requested
    # evaluation. No training/profile/MS objective calls this function.
    from diagnostics.formal_rollout import load_q2_policy
    from policy_observation import build_policy_observation, initial_observation_state, update_position_integral
    reference, _ = load_q2_policy(checkpoint, device=initial.position.device, dtype=initial.position.dtype)
    reference.eval()
    settings = dict(mode="integral25", integral_input_frame="body", integral_input_multiplier=1.0,
                    noise_max=0.0, integral_limit=0.5, integral_leak=0.0, integral_clamp_mode="box")
    settings.update(getattr(reference, "q2_observation_settings", {}))
    options = {key: settings[key] for key in ("mode", "integral_input_frame", "integral_input_multiplier", "noise_max")}
    obs_state = initial_observation_state(initial.position.shape[0], device=initial.position.device,
                                          dtype=initial.position.dtype)
    hidden = reference.initial_hidden(initial.position.shape[0], device=initial.position.device,
                                      dtype=initial.position.dtype)
    state = initial
    arrays = {name: [] for name in ("positions", "velocities", "omegas", "actions", "action_deltas", "omega_deltas")}
    for _ in range(horizon):
        obs, observed_p = build_policy_observation(state, obs_state, **options)
        action, hidden = reference(obs, hidden)
        obs_state = update_position_integral(
            obs_state, observed_p, dt=simulator.params.dt, integral_limit=settings["integral_limit"],
            integral_leak=settings["integral_leak"], integral_clamp_mode=settings["integral_clamp_mode"],
        )
        after = simulator.step(state, action, grad_decay=1.0)
        for name, tensor in (
            ("positions", after.position), ("velocities", after.velocity), ("omegas", after.omega),
            ("actions", action), ("action_deltas", action - state.previous_action),
            ("omega_deltas", after.omega - state.omega),
        ):
            arrays[name].append(tensor)
        state = after
    return SimpleNamespace(**{name: torch.stack(values) for name, values in arrays.items()})


def _select_trace(trace, mask):
    return SimpleNamespace(**{
        name: getattr(trace, name)[:, mask]
        for name in ("positions", "velocities", "omegas", "actions", "action_deltas", "omega_deltas")
    })


@torch.no_grad()
def evaluate(
    policy, loss_config, *, seeds, horizons, scenarios, output=None,
    q2_checkpoint=None, split="development", save_trajectories=False,
):
    device, dtype = next(policy.parameters()).device, next(policy.parameters()).dtype
    simulator = L2FSimulator(L2FParams(dt=policy.config.dt))
    records, tensor_records = [], []
    for seed in seeds:
        initial, cells = sample_scenarios(scenarios, seed=seed, dt=policy.config.dt, device=device, dtype=dtype)
        for horizon in horizons:
            trace = rollout(policy, simulator, initial, horizon)
            metrics = trajectory_metrics(trace, loss_config)
            if not metrics["finite"] or not math.isfinite(metrics["task_objective"]):
                if output is not None:
                    atomic_torch(output.with_suffix(".nonfinite.pt"), {
                        "seed": seed, "horizon": horizon, "observations": trace.observations.cpu(),
                        "actions": trace.actions.cpu(), "deployment_authorized": False,
                    })
                raise RuntimeError("non-finite continuous evaluation; failing trajectory preserved")
            row = {"seed": seed, "horizon": horizon, "policy": metrics}
            for metric, cell in zip(metrics["scenarios"], cells.tolist()):
                metric["tw_bin"], metric["log_alpha_bin"] = cell
            if q2_checkpoint is not None:
                reference = _reference_rollout(q2_checkpoint, simulator, initial, horizon)
                row["q2"] = trajectory_metrics(reference, loss_config)
                # Pair every scene, not just aggregate means. These comparisons
                # are research metrics, not a replacement for the historical 5% gate.
                for candidate, baseline in zip(metrics["scenarios"], row["q2"]["scenarios"]):
                    candidate["q2_cost"] = baseline["task_cost"]
                    candidate["q2_success"] = baseline["success"]
            records.append(row)
            if save_trajectories:
                tensor_records.append({
                    "seed": seed, "horizon": horizon, "cells": cells.cpu(),
                    "initial_state": {f.name: getattr(initial, f.name).cpu() for f in fields(initial)},
                    "observations": trace.observations.cpu(), "actions": trace.actions.cpu(),
                    "positions": trace.positions.cpu(), "velocities": trace.velocities.cpu(),
                    "omegas": trace.omegas.cpu(),
                })
    report = {
        "protocol": PROTOCOL_VERSION, "split": split, "model_sha256": model_hash(policy),
        "source_sha256": source_hash(), "records": records,
        "score": sum(row["policy"]["task_objective"] for row in records) / len(records),
        "finite": all(row["policy"]["finite"] for row in records),
        "success_count": sum(row["policy"]["success_count"] for row in records),
        **{
            key: math.sqrt(sum(row["policy"][key] ** 2 for row in records) / len(records))
            for key in ("position_rms", "velocity_rms", "omega_rms")
        },
        **{
            key: sum(row["policy"][key] for row in records) / len(records)
            for key in ("steady_success_rate", "motor_saturation_fraction")
        },
        "q2_checkpoint_sha256": None if q2_checkpoint is None else file_hash(q2_checkpoint),
        "all_scenarios_retained": True, "fresh_continuous_rollout_from_initial_state": True,
        "unseen_parameter_draws_within_registered_family": True,
        "out_of_family_generalization_tested": False,
        "best_known_per_airframe_optimality_reference_available": False,
        "formal_eligible": False, "deployment_authorized": False,
    }
    if output is not None:
        atomic_json(output, report)
        if save_trajectories:
            atomic_torch(output.with_suffix(".trajectories.pt"), {
                "split": split, "model_sha256": report["model_sha256"],
                "records": tensor_records, "deployment_authorized": False,
            })
    return report


def train(args, policy_config, loss_config):
    if args.updates is None or not 1 <= args.updates < 1_000_000:
        raise ValueError("set an explicit --updates budget below 1000000 after profiling")
    if args.seed != 7:
        raise ValueError("this mechanism-screening protocol trains only initialization seed7")
    if args.q2_checkpoint is not None:
        raise ValueError("Q2 is an optional evaluation reference, not a training input")
    if FINAL_CLAIM.exists():
        raise RuntimeError("this protocol final set has been consumed; register a new independent protocol before further candidate development")
    if args.updates > 5:
        validate_training_contract(args.contract_report, policy_config, loss_config)
    if args.optimizer == "full-space-ms" and args.initialize_from is None and args.resume is None and not (args.work_dir / "latest.training.pt").exists():
        raise ValueError("MS requires a learned task checkpoint; do not start it from a random controller")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7)
    policy = ResponseMotorPolicy(policy_config).to(device=device, dtype=dtype)
    if args.initialize_from is not None:
        initialized, source = load_policy_checkpoint(args.initialize_from, device, dtype)
        if initialized.config != policy.config:
            raise ValueError("weights-only initialization requires the same policy configuration")
        policy.load_state_dict(initialized.state_dict())
    optimizer = None if args.optimizer == "full-space-ms" else torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    run_binding = binding(args, policy_config, loss_config)
    work = args.work_dir
    latest = work / "latest.training.pt"
    resume = args.resume if args.resume is not None else (latest if latest.exists() else None)
    initial_model = copy.deepcopy(policy.state_dict())
    progress = {
        "attempts": 0, "updates": 0, "elapsed_seconds": 0.0,
        "best_score": None, "best_update": None, "bad_checks": 0,
        "history": [], "development": [], "training_seeds": [],
        "baseline_score": None, "status": "training",
        "initialization_checkpoint_sha256": None if args.initialize_from is None else file_hash(args.initialize_from),
    }
    solver = {"damping": args.damping}
    if resume is not None:
        saved = torch.load(resume, map_location="cpu")
        if saved.get("schema") != PROTOCOL_VERSION or saved.get("binding") != run_binding:
            raise ValueError("resume binding changed; use an explicit new weights-only experiment, not hash editing")
        policy.load_state_dict(saved["model"])
        if optimizer is not None:
            optimizer.load_state_dict(saved["optimizer"])
        progress, solver, initial_model = saved["progress"], saved["solver"], saved["initial_model"]
        restore_rng(saved["rng"])
    work.mkdir(parents=True, exist_ok=True)
    atomic_json(work / "configuration.json", {
        "binding": run_binding,
        "execution": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "resume": None if resume is None else str(resume),
        "deployment_authorized": False,
    })
    simulator = L2FSimulator(L2FParams(dt=policy_config.dt))
    started, previous_seconds = time.monotonic(), progress["elapsed_seconds"]
    stop = {"requested": False}
    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, lambda signum, frame: stop.update(requested=True))

    def save(path=latest):
        progress["elapsed_seconds"] = previous_seconds + time.monotonic() - started
        save_training(path, policy, optimizer, progress, solver, run_binding, initial_model)

    def development():
        report = evaluate(
            policy, loss_config, seeds=DEVELOPMENT_SEEDS, horizons=(args.horizon,),
            scenarios=args.scenarios,
            output=work / "development" / ("%07d.json" % progress["attempts"]),
        )
        if progress["baseline_score"] is None:
            progress["baseline_score"] = report["score"]
        compact = {key: report[key] for key in (
            "score", "finite", "position_rms", "velocity_rms", "omega_rms",
            "steady_success_rate", "motor_saturation_fraction",
        )}
        compact.update(attempt=progress["attempts"], update=progress["updates"])
        progress["development"].append(compact)
        old_score = progress["best_score"]
        improved = old_score is None or report["score"] < old_score * (1 - args.min_relative_improvement)
        if improved:
            progress.update(best_score=report["score"], best_update=progress["updates"], bad_checks=0)
            save(work / "best.training.pt")
            atomic_json(work / "best.development.json", report)
        else:
            progress["bad_checks"] += 1
        print(json.dumps({"development": compact}), flush=True)

    try:
        if not progress["development"]:
            development()
            save()
        progress["status"] = "training"
        while progress["attempts"] < args.updates:
            elapsed = previous_seconds + time.monotonic() - started
            if stop["requested"] or elapsed >= args.max_seconds:
                progress["status"] = "interrupted" if stop["requested"] else "time_budget"
                break
            index = progress["attempts"]
            scenario_seed = TRAIN_SEED_BASE + index
            initial, _ = sample_scenarios(
                args.scenarios, seed=scenario_seed, dt=policy_config.dt, device=device, dtype=dtype
            )
            if optimizer is None:
                from response_shooting import task_shooting_step
                development_initial, _ = sample_scenarios(
                    args.scenarios, seed=MS_ACCEPTANCE_SEED, dt=policy_config.dt, device=device, dtype=dtype
                )
                previous_rejections = solver.get("consecutive_rejections", 0)
                record, solver = task_shooting_step(
                    policy, simulator, initial, development_initial, loss_config,
                    segment_steps=args.segment_steps, segments=args.segments,
                    damping=solver["damping"], cg_iterations=args.cg_iterations,
                    parameter_radius=args.parameter_radius, action_radius=args.action_radius,
                    action_max_radius=args.action_max_radius, debug_solver=args.ms_debug,
                )
                accepted = record["accepted"]
                solver["consecutive_rejections"] = 0 if accepted else previous_rejections + 1
            else:
                trace = rollout(policy, simulator, initial, args.horizon)
                control = task_loss(trace, loss_config)
                with torch.no_grad():
                    metrics = trajectory_metrics(trace, loss_config)
                if not metrics["finite"]:
                    raise RuntimeError("non-finite physical or recurrent state")
                auxiliary = prediction_residual(policy, trace.observations, trace.actions).square().sum() if loss_config.prediction_weight else control.new_zeros(())
                loss = control + loss_config.prediction_weight * auxiliary
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite physical task loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.gradient_clip)
                if not bool(torch.isfinite(gradient_norm)):
                    raise RuntimeError("non-finite task gradient")
                before_model, before_optimizer = copy.deepcopy(policy.state_dict()), copy.deepcopy(optimizer.state_dict())
                optimizer.step()
                if not all(bool(torch.isfinite(value).all()) for value in policy.state_dict().values()):
                    policy.load_state_dict(before_model); optimizer.load_state_dict(before_optimizer)
                    raise RuntimeError("non-finite optimizer result; restored pre-update state")
                record = {
                    "task_loss": float(control.detach()),
                    **{key: metrics[key] for key in (
                        "position_rms", "velocity_rms", "omega_rms",
                        "steady_success_rate", "motor_saturation_fraction",
                    )},
                    "gradient_norm": float(gradient_norm), "numerics_finite": True,
                    "accepted": True,
                }
                accepted = True
                del trace, control, auxiliary, loss
            progress["attempts"] += 1
            progress["updates"] += int(accepted)
            progress["training_seeds"].append(scenario_seed)
            record.update(attempt=progress["attempts"], update=progress["updates"], scenario_seed=scenario_seed)
            progress["history"].append(record)
            if progress["attempts"] % args.checkpoint_every == 0:
                save()
                print(json.dumps(record), flush=True)
            if optimizer is None and solver.get("consecutive_rejections", 0) >= args.maximum_ms_rejections:
                progress["status"] = "solver_rejections"
                break
            if progress["attempts"] % args.development_every == 0:
                development()
                save()
                if progress["updates"] >= args.minimum_updates and progress["bad_checks"] >= args.patience:
                    progress["status"] = "development_plateau"
                    break
        if progress["status"] == "training":
            progress["status"] = "update_budget"
        if progress["development"][-1]["attempt"] != progress["attempts"]:
            development()
        save()
    except BaseException as error:
        progress["status"] = "failed"
        progress["error"] = repr(error)
        save()
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    summary = {
        "protocol": PROTOCOL_VERSION, "status": progress["status"],
        "actual_attempts": progress["attempts"], "actual_updates": progress["updates"],
        "elapsed_seconds": progress["elapsed_seconds"], "best_update": progress["best_update"],
        "baseline_development_score": progress["baseline_score"],
        "best_development_score": progress["best_score"],
        "final_seeds_consumed": [], "formal_eligible": False, "deployment_authorized": False,
    }
    atomic_json(work / "training_report.json", summary)
    return summary


def profile(args, policy_config, loss_config):
    torch.set_num_threads(args.threads)
    torch.manual_seed(7)
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    policy = ResponseMotorPolicy(policy_config).to(device=device, dtype=dtype)
    initial, _ = sample_scenarios(args.scenarios, seed=TRAIN_SEED_BASE, dt=policy_config.dt,
                                  device=device, dtype=dtype)
    simulator = L2FSimulator(L2FParams(dt=policy_config.dt))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    trace = rollout(policy, simulator, initial, args.horizon)
    loss = task_loss(trace, loss_config)
    if loss_config.prediction_weight:
        loss = loss + loss_config.prediction_weight * prediction_residual(
            policy, trace.observations, trace.actions
        ).square().sum()
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward = time.monotonic() - start
    start = time.monotonic()
    loss.backward(); optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    report = {
        "protocol": PROTOCOL_VERSION, "scope": "one Adam task update, not an MS throughput estimate",
        "device": str(device), "dtype": args.dtype, "scenarios": args.scenarios,
        "horizon": args.horizon, "forward_seconds": forward,
        "backward_update_seconds": time.monotonic() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        "production_checkpoint_written": False, "deployment_authorized": False,
    }
    atomic_json(args.work_dir / "profile.json", report)
    return report


def freeze_candidate(args):
    source = args.checkpoint or args.work_dir / "best.training.pt"
    saved = torch.load(source, map_location="cpu")
    if saved.get("schema") != PROTOCOL_VERSION or saved["binding"]["source_sha256"] != source_hash():
        raise ValueError("candidate checkpoint source/protocol is stale")
    progress = saved["progress"]
    if progress["updates"] < args.minimum_updates:
        raise ValueError("candidate has not completed the requested minimum training budget")
    development = progress["development"][-1]
    if not development["finite"] or not development["score"] < progress["baseline_score"]:
        raise ValueError("candidate has not improved its continuous development task score")
    if development["update"] != progress["updates"]:
        raise ValueError("development evidence does not belong to the candidate update")
    target = args.work_dir / "candidate.pt"
    if target.exists() or (args.work_dir / "candidate.json").exists():
        raise FileExistsError("candidate already frozen; preserve it and use a new experiment directory")
    dtype = getattr(torch, saved["binding"]["dtype"])
    policy = ResponseMotorPolicy(ResponsePolicyConfig(**saved["policy_config"])).to(dtype=dtype)
    policy.load_state_dict(saved["model"])
    if model_hash(policy) != saved["model_sha256"]:
        raise ValueError("candidate model digest mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        torch.save(saved, stream)
        stream.flush(); os.fsync(stream.fileno())
    record = {
        "protocol": PROTOCOL_VERSION, "candidate_sha256": file_hash(target),
        "source_checkpoint_sha256": file_hash(source), "source_sha256": source_hash(),
        "model_sha256": saved["model_sha256"], "update": progress["updates"],
        "final_seeds": list(FINAL_SEEDS), "final_horizons": list(FINAL_HORIZONS),
        "binding": saved["binding"], "deployment_authorized": False,
    }
    atomic_json(args.work_dir / "candidate.json", record)
    return record


def evaluate_checkpoint(args):
    final = args.mode == "final"
    path = (args.work_dir / "candidate.pt") if final else (
        args.checkpoint or args.work_dir / "best.training.pt"
    )
    saved = torch.load(path, map_location="cpu")
    if saved.get("schema") != PROTOCOL_VERSION or saved["binding"]["source_sha256"] != source_hash():
        raise ValueError("checkpoint source/protocol is stale")
    policy, _ = load_policy_checkpoint(path, torch.device(args.device), getattr(torch, saved["binding"]["dtype"]))
    if model_hash(policy) != saved["model_sha256"]:
        raise ValueError("checkpoint model digest mismatch")
    loss_config = TaskLossConfig(**saved["binding"]["protocol"]["loss"])
    scenarios = saved["binding"]["protocol"]["scenarios_per_bank"]
    if final:
        if not args.consume_final:
            raise ValueError("final evaluation requires explicit --consume-final")
        record = json.loads((args.work_dir / "candidate.json").read_text())
        if (record["candidate_sha256"] != file_hash(path)
            or record["binding"] != saved["binding"]
            or record["source_sha256"] != source_hash()
            or record["final_seeds"] != list(FINAL_SEEDS)
            or record["final_horizons"] != list(FINAL_HORIZONS)):
            raise ValueError("frozen candidate record mismatch")
        FINAL_CLAIM.parent.mkdir(parents=True, exist_ok=True)
        # Stable protocol-wide location: changing a work directory or model
        # hash does not grant another independent use of these final seeds.
        with FINAL_CLAIM.open("x", encoding="utf-8") as stream:
            json.dump({"status": "claimed", "candidate_sha256": record["candidate_sha256"],
                       "seeds": list(FINAL_SEEDS), "source_sha256": source_hash()}, stream, indent=2)
            stream.flush(); os.fsync(stream.fileno())
    report = evaluate(
        policy, loss_config, seeds=FINAL_SEEDS if final else DEVELOPMENT_SEEDS,
        horizons=FINAL_HORIZONS if final else (args.horizon,), scenarios=scenarios,
        output=args.work_dir / ("final_evaluation.json" if final else "evaluation.json"),
        q2_checkpoint=args.q2_checkpoint, split="final" if final else "development",
        save_trajectories=True,
    )
    report["candidate_frozen"] = final
    report["final_evaluation_complete"] = final
    output = args.work_dir / ("final_evaluation.json" if final else "evaluation.json")
    atomic_json(output, report)
    if final:
        atomic_json(FINAL_CLAIM, {
            "status": "completed", "candidate_sha256": file_hash(path),
            "report_sha256": file_hash(output), "seeds": list(FINAL_SEEDS),
            "source_sha256": source_hash(), "deployment_authorized": False,
        })
    return report


def validate_training_contract(path, policy_config, loss_config):
    if path is None:
        raise ValueError("more than five updates require --contract-report from the one-time training contract check")
    record = json.loads(Path(path).read_text())
    digest = record.pop("evidence_sha256", None)
    expected = hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()
    if digest != expected or record.get("source_sha256") != source_hash():
        raise ValueError("training contract evidence is stale or changed")
    if record.get("policy_config") != asdict(policy_config) or record.get("loss_config") != asdict(loss_config):
        raise ValueError("training contract belongs to a different policy/objective")
    required = {"no_q2_targets", "response_recurrent_gradient", "control_head_gradient",
                "no_privileged_actor_input", "auxiliary_target_stop_gradient", "startup_connected"}
    if not record.get("passed") or set(record.get("checks", {})) != required or not all(record["checks"].values()):
        raise ValueError("training contract failed")
    for group in ("response_recurrent", "control_head"):
        value = record.get("gradient_groups", {}).get(group, {})
        if not value.get("finite") or not math.isfinite(value.get("norm", float("nan"))) or value["norm"] <= 0:
            raise ValueError("missing finite, nonzero task gradient for " + group)
    return record


@torch.no_grad()
def evaluate_hidden_reset(args):
    """One DEV counterfactual: change only recurrent memory at a fixed time."""
    source = args.checkpoint or args.work_dir / "best.training.pt"
    saved = torch.load(source, map_location="cpu")
    if saved.get("schema") != PROTOCOL_VERSION or saved["binding"]["source_sha256"] != source_hash():
        raise ValueError("checkpoint source/protocol is stale")
    policy, _ = load_policy_checkpoint(
        source, torch.device(args.device), getattr(torch, saved["binding"]["dtype"])
    )
    if model_hash(policy) != saved["model_sha256"]:
        raise ValueError("checkpoint model digest mismatch")
    output = args.work_dir / "hidden_reset.json"
    if output.exists():
        raise FileExistsError("preserve the existing hidden-reset experiment rather than overwrite it")
    config = TaskLossConfig(**saved["binding"]["protocol"]["loss"])
    count = saved["binding"]["protocol"]["scenarios_per_bank"]
    simulator = L2FSimulator(L2FParams(dt=policy.config.dt))
    rows, trajectories = [], []
    for seed in DEVELOPMENT_SEEDS:
        initial, cells = sample_scenarios(
            count, seed=seed, dt=policy.config.dt,
            device=next(policy.parameters()).device, dtype=next(policy.parameters()).dtype,
        )
        prefix = rollout(policy, simulator, initial, args.reset_step)
        normal_state = prefix.end
        reset_state = replace(normal_state, policy=replace(
            normal_state.policy, memory=torch.zeros_like(normal_state.policy.memory)
        ))
        normal = rollout(policy, simulator, normal_state, args.reset_horizon)
        reset = rollout(policy, simulator, reset_state, args.reset_horizon)
        normal_metrics, reset_metrics = trajectory_metrics(normal, config), trajectory_metrics(reset, config)
        rows.append({
            "seed": seed, "normal": normal_metrics, "hidden_reset": reset_metrics,
            "same_physical_state_and_nonmemory_recurrent_fields": True,
        })
        trajectories.append({
            "seed": seed, "cells": cells.cpu(),
            "prefix_observations": prefix.observations.cpu(), "prefix_actions": prefix.actions.cpu(),
            "normal_observations": normal.observations.cpu(), "normal_actions": normal.actions.cpu(),
            "reset_observations": reset.observations.cpu(), "reset_actions": reset.actions.cpu(),
        })
    report = {
        "protocol": PROTOCOL_VERSION, "split": "development-mechanism-check",
        "checkpoint_sha256": file_hash(source), "model_sha256": model_hash(policy),
        "reset_after_physical_transitions": args.reset_step,
        "subsequent_horizon": args.reset_horizon, "records": rows,
        "intervention": "zero memory once, then allow normal online updates; do not change integral/action history/physical state",
        "interpretation": "paired effect sizes, not a claim of optimality or an independent FINAL result",
        "final_seeds_consumed": [], "formal_eligible": False, "deployment_authorized": False,
    }
    atomic_json(output, report)
    atomic_torch(output.with_suffix(".trajectories.pt"), trajectories)
    return report

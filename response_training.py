"""One production path: exact rematerialized H500 BPTT and persistent Adam."""

from __future__ import annotations

from dataclasses import asdict, fields
import copy
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import random
import signal
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

from env_l2f import L2FParams, L2FSimulator, L2FState
from response_policy import ARCHITECTURE, ResponseMotorPolicy, ResponsePolicyConfig
from response_task import (
    TaskLossConfig,
    sample_scenarios,
    rollout,
    trajectory_metrics,
    task_loss_components,
    hard_risk_metrics,
)
from response_adjoints import collect_boundary_rollout, backward_actor
from response_execution import exit_class

ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "response-actor-only-exact-v2"
TRAIN_SEED_BASE = 31_000_007
DEVELOPMENT_SEEDS = (32_000_007, 32_010_007)
FINAL_SEEDS = (34_000_007, 34_010_007)
SOURCE_FILES = (
    "response_policy.py",
    "response_task.py",
    "response_adjoints.py",
    "response_training.py",
    "response_execution.py",
    "env_l2f.py",
    "tools/train_response_control.py",
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash():
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def model_hash(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        value = value.detach().cpu().contiguous()
        for metadata in (name, str(value.dtype), str(tuple(value.shape))):
            digest.update(metadata.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch(path, value):
    _atomic(path, lambda stream: torch.save(value, stream))


def atomic_json(path, value):
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic(path, lambda stream: stream.write(data.encode()))


def capture_rng():
    name, keys, pos, gaussian, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [name, keys.tolist(), pos, gaussian, cached],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("exact resume of CUDA RNG requires CUDA")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def migrate_actor_weights(policy, state):
    """Drop only the known unused head; all effective weights remain strict."""
    expected = policy.state_dict()
    extra = set(state) - set(expected)
    h, m = policy.config.hidden_dim, policy.config.memory_dim
    head_shapes = {
        "response_predictor.0.weight": (h, m + 23),
        "response_predictor.0.bias": (h,),
        "response_predictor.2.weight": (6, h),
        "response_predictor.2.bias": (6,),
    }
    if set(expected) - set(state) or (extra and extra != set(head_shapes)):
        raise ValueError("unknown or missing Actor keys; only response_predictor may be removed")
    for name in extra:
        if tuple(state[name].shape) != head_shapes[name]:
            raise ValueError("unexpected legacy prediction head shape")
    if any(
        state[n].shape != v.shape or not bool(torch.isfinite(state[n]).all())
        for n, v in expected.items()
    ):
        raise ValueError("Actor tensor shape mismatch or nonfinite weights")
    policy.load_state_dict({n: state[n] for n in expected}, strict=True)


def load_legacy_checkpoint(path):
    """Explicit trusted local import; map only NumPy's known private rename."""
    class NumpyCompatibleUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'numpy._core.multiarray' and name in ('_reconstruct', 'scalar'):
                return getattr(np.core.multiarray, name)
            return super().find_class(module, name)

    compatibility = SimpleNamespace(__name__='legacy_numpy_pickle',
                                    Unpickler=NumpyCompatibleUnpickler, load=pickle.load)
    return torch.load(path, map_location='cpu', weights_only=False, pickle_module=compatibility)


def optimizer_parameter_names(policy, optimizer):
    names = {id(p): n for n, p in policy.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def migrate_named_adam(optimizer, policy, saved, saved_names):
    """Explicit name metadata is mandatory; never infer old parameter IDs."""
    if not saved_names or len(saved_names) != len(saved["param_groups"]) or len(saved_names) != 1:
        raise ValueError("named single-group Adam metadata required for migration")
    old_group = saved["param_groups"][0]
    names = saved_names[0]
    if len(names) != len(old_group["params"]) or len(set(names)) != len(names):
        raise ValueError("invalid optimizer name order")
    lookup = dict(zip(names, old_group["params"]))
    new = optimizer.state_dict()
    current = optimizer_parameter_names(policy, optimizer)[0]
    if not set(current).issubset(lookup) or any(
        not n.startswith("response_predictor.") for n in set(names) - set(current)
    ):
        raise ValueError("optimizer parameter names do not match effective Actor")
    parameters = dict(policy.named_parameters())
    for n, new_id in zip(current, new["param_groups"][0]["params"]):
        state = copy.deepcopy(saved["state"].get(lookup[n], {}))
        for key, value in state.items():
            if torch.is_tensor(value):
                if key != "step" and value.shape != parameters[n].shape:
                    raise ValueError("Adam moment shape mismatch for " + n)
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("nonfinite Adam state")
        if state:
            new["state"][new_id] = state
    new["param_groups"][0].update({k: v for k, v in old_group.items() if k != "params"})
    optimizer.load_state_dict(new)


def load_policy_checkpoint(path, device, dtype):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema") != PROTOCOL_VERSION or value.get("architecture") != ARCHITECTURE:
        raise ValueError("not a current Actor-only checkpoint; use explicit weights initialization")
    policy = ResponseMotorPolicy(ResponsePolicyConfig(**value["policy_config"])).to(
        device=device, dtype=dtype
    )
    migrate_actor_weights(policy, value["model"])
    return policy, value


def sample_training_scenarios(
    scenarios,
    attempt_index,
    *,
    batches=2,
    dt=0.01,
    device=torch.device("cpu"),
    dtype=torch.float32,
    scenario_mode="fixed-airframe",
):
    if attempt_index < 0 or batches < 1:
        raise ValueError("invalid sampling index")
    seeds = [TRAIN_SEED_BASE + batches * attempt_index + i for i in range(batches)]
    if seeds[-1] >= min(DEVELOPMENT_SEEDS):
        raise ValueError("TRAIN seed range would enter reserved EVAL")
    states = [
        sample_scenarios(
            scenarios, seed=s, dt=dt, device=device, dtype=dtype, scenario_mode=scenario_mode
        )[0]
        for s in seeds
    ]
    return pool_states(states), seeds


def pool_states(states):
    return L2FState(
        **{f.name: torch.cat([getattr(s, f.name) for s in states]) for f in fields(states[0])}
    )


@torch.no_grad()
def gradient_norm(parameters):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        raise FloatingPointError("Actor has no task gradients")
    norm = torch.stack([g.double().square().sum() for g in grads]).sum().sqrt()
    value = float(norm)
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite Actor gradient norm")
    return value


@torch.no_grad()
def safe_global_clip(parameters, limit):
    parameters = list(parameters)
    value = gradient_norm(parameters)
    if limit <= 0 or not math.isfinite(limit):
        raise ValueError("gradient clip must be finite and positive")
    scale = min(1.0, limit / (value + 1e-6))
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(scale)
    return value


@torch.no_grad()
def adaptive_clip(parameters, limit):
    if limit == 0:
        return
    for p in parameters:
        if p.grad is None:
            continue
        axes = tuple(range(1, p.ndim)) if p.ndim > 1 else None
        pn = p.double().norm(dim=axes, keepdim=True).clamp_min(0.001)
        gn = p.grad.double().norm(dim=axes, keepdim=True).clamp_min(1e-6)
        p.grad.mul_((limit * pn / gn).clamp_max(1).to(p.grad.dtype))


def binding(args, policy_config, loss_config):
    return {
        "source_sha256": source_hash(),
        "algorithm": "exact-boundary-bptt-adam",
        "protocol": {
            "version": PROTOCOL_VERSION,
            "architecture": ARCHITECTURE,
            "policy": asdict(policy_config),
            "loss": asdict(loss_config),
            "scenarios_per_bank": args.scenarios,
            "scenario_mode": args.scenario_mode,
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "deployment_authorized": False,
        },
        **{
            n: getattr(args, n)
            for n in (
                "seed",
                "device",
                "dtype",
                "horizon",
                "window_steps",
                "lr",
                "gradient_clip",
                "gradient_scale",
                "agc",
                "threads",
            )
        },
        "training_banks": 2,
        "torch_version": str(torch.__version__),
    }


def _checkpoint(policy, optimizer, progress, run_binding):
    return {
        "schema": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "policy_config": asdict(policy.config),
        "model": policy.state_dict(),
        "model_sha256": model_hash(policy),
        "optimizer": optimizer.state_dict(),
        "optimizer_parameter_names": optimizer_parameter_names(policy, optimizer),
        "rng": capture_rng(),
        "next_update": progress["updates"],
        "progress": copy.deepcopy(progress),
        "binding": run_binding,
        "deployment_authorized": False,
    }


def _recover_log(path, committed_update):
    """Crash after append but before checkpoint must not duplicate update IDs."""
    if not path.exists():
        return
    lines = []
    for raw in path.read_text().splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            break
        if row["update"] > committed_update:
            break
        lines.append(raw + "\n")
    _atomic(path, lambda stream: stream.write("".join(lines).encode()))


def _append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@torch.no_grad()
def evaluate(policy, simulator, initial, horizon, loss_config):
    trace = rollout(policy, simulator, initial, horizon)
    report = trajectory_metrics(trace, loss_config)
    if not report["finite"]:
        raise FloatingPointError("nonfinite fixed EVAL")
    report.pop("scenarios")
    report["task_components"] = task_loss_components(trace, loss_config)
    risk = hard_risk_metrics(trace, loss_config)
    report["risk"] = risk
    return report


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(args, policy_config, loss_config):
    """A finite loss increase never vetoes an update; numerical failures abort."""
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    args.device = str(device)
    dtype = getattr(torch, args.dtype)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    policy = ResponseMotorPolicy(policy_config).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    simulator = L2FSimulator(L2FParams(dt=policy_config.dt))
    run_binding = binding(args, policy_config, loss_config)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(
        (work / name).exists() for name in ("latest.pt", "history.jsonl", "evaluation.jsonl")
    ):
        raise ValueError("work directory already contains a run; choose resume or a new directory")
    progress = {
        "updates": 0,
        "status": "training",
        "elapsed_seconds": 0.0,
        "best_score": None,
        "best_success_rate": None,
        "best_success_score": None,
    }
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved.get("schema") != PROTOCOL_VERSION or saved["binding"] != run_binding:
            raise ValueError(
                "resume requires identical Actor-only objective, source and optimizer configuration"
            )
        migrate_actor_weights(policy, saved["model"])
        if saved.get("model_sha256") != model_hash(policy):
            raise ValueError("checkpoint model digest mismatch")
        migrate_named_adam(
            optimizer, policy, saved["optimizer"], saved["optimizer_parameter_names"]
        )
        progress = copy.deepcopy(saved["progress"])
        if saved["next_update"] != progress["updates"]:
            raise ValueError("inconsistent checkpoint sampling index")
        restore_rng(saved["rng"])
    elif args.init_checkpoint or args.migrate_checkpoint:
        path = args.init_checkpoint or args.migrate_checkpoint
        # Explicit local legacy migration may contain old NumPy RNG tuples.
        saved = load_legacy_checkpoint(path)
        if saved.get("policy_config") != asdict(policy_config) and not (
            args.migrate_checkpoint and "policy_config" not in saved
        ):
            raise ValueError("initialization policy configuration mismatch")
        migrate_actor_weights(policy, saved["model"])
        progress["initialization"] = {
            "file_sha256": file_hash(path),
            "weights_only": bool(args.init_checkpoint),
        }
        if args.migrate_checkpoint:
            metadata = json.loads(Path(args.migration_metadata).read_text())
            if (
                metadata.get("checkpoint_sha256") != file_hash(path)
                or metadata.get("algorithm") != "actor-only-exact-bptt"
            ):
                raise ValueError("migration metadata must certify this exact Actor-only checkpoint")
            if metadata.get("binding") != {
                k: v for k, v in run_binding.items() if k != "source_sha256"
            }:
                raise ValueError("explicit legacy migration training semantics mismatch")
            migrate_named_adam(
                optimizer, policy, saved["optimizer"], metadata["optimizer_parameter_names"]
            )
            progress.update(updates=int(metadata["next_update"]))
            restore_rng(saved["rng"])
    for name in ("history.jsonl", "evaluation.jsonl"):
        _recover_log(work / name, progress["updates"])
    elapsed_before = progress["elapsed_seconds"]
    stop = []
    old_handlers = {}

    def interrupted(signum, frame):
        stop.append(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, interrupted)
    eval_initial = pool_states(
        [
            sample_scenarios(
                args.scenarios,
                seed=s,
                dt=policy_config.dt,
                device=device,
                dtype=dtype,
                scenario_mode=args.scenario_mode,
            )[0]
            for s in DEVELOPMENT_SEEDS
        ]
    )

    def save(name="latest.pt"):
        atomic_torch(work / name, _checkpoint(policy, optimizer, progress, run_binding))

    def evaluation():
        rng = capture_rng()
        try:
            report = evaluate(policy, simulator, eval_initial, args.horizon, loss_config)
        finally:
            restore_rng(rng)
        _append(work / "evaluation.jsonl", {"update": progress["updates"], **report})
        score, success = report["task_objective"], report["steady_success_rate"]
        cost_best = progress["best_score"] is None or score < progress["best_score"]
        success_best = (
            progress["best_success_rate"] is None
            or success > progress["best_success_rate"]
            or (success == progress["best_success_rate"] and score < progress["best_success_score"])
        )
        if cost_best:
            progress.update(best_score=score, best_update=progress["updates"])
        if success_best:
            progress.update(
                best_success_rate=success,
                best_success_score=score,
                best_success_update=progress["updates"],
            )
        progress["last_evaluated_update"] = progress["updates"]
        if cost_best:
            save("best.pt")
        if success_best:
            save("best_success.pt")

    try:
        if (
            progress.get("last_evaluated_update") != progress["updates"]
            and progress["updates"] == 0
        ):
            evaluation()
        save()
        while progress["updates"] < args.updates:
            if stop:
                progress["status"] = "interrupted"
                break
            if time.monotonic() - started >= args.max_seconds:
                progress["status"] = "time_budget"
                break
            progress["status"] = "training"
            before = (
                copy.deepcopy(policy.state_dict()),
                copy.deepcopy(optimizer.state_dict()),
                capture_rng(),
            )
            _sync(device)
            update_start = time.monotonic()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            try:
                initial, seeds = sample_training_scenarios(
                    args.scenarios,
                    progress["updates"],
                    dt=policy_config.dt,
                    device=device,
                    dtype=dtype,
                    scenario_mode=args.scenario_mode,
                )
                optimizer.zero_grad(set_to_none=True)
                record = collect_boundary_rollout(
                    policy,
                    simulator,
                    initial,
                    loss_config,
                    horizon=args.horizon,
                    window_steps=args.window_steps,
                )
                _sync(device)
                forward_done = time.monotonic()
                backward = backward_actor(
                    policy, simulator, record, loss_config, gradient_scale=args.gradient_scale
                )
                raw_norm = gradient_norm(policy.parameters()) if args.agc else None
                adaptive_clip(policy.parameters(), args.agc)
                pre_clip = safe_global_clip(policy.parameters(), args.gradient_clip)
                if raw_norm is None:
                    raw_norm = pre_clip
                optimizer.step()
                tensors = list(policy.parameters()) + [
                    v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)
                ]
                if not all(bool(torch.isfinite(x).all()) for x in tensors):
                    raise FloatingPointError("nonfinite Adam parameters or moments")
            except Exception:
                policy.load_state_dict(before[0])
                optimizer.load_state_dict(before[1])
                restore_rng(before[2])
                optimizer.zero_grad(set_to_none=True)
                raise
            _sync(device)
            progress["updates"] += 1
            progress["elapsed_seconds"] = elapsed_before + time.monotonic() - started
            row = {
                "update": progress["updates"],
                "train_seeds": seeds,
                **record.metrics,
                "raw_gradient_norm": raw_norm,
                "pre_global_clip_norm": pre_clip,
                "gradient_scale": args.gradient_scale,
                "boundary_exact": all(r["exact"] for r in backward["boundaries"]),
                "boundary_max_error": max(r["max_error"] for r in backward["boundaries"]),
                "forward_seconds": forward_done - update_start,
                "update_seconds": time.monotonic() - update_start,
                "cuda_peak_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                ),
            }
            _append(work / "history.jsonl", row)
            print(json.dumps(row, allow_nan=False), flush=True)
            del record, backward, before
            if progress["updates"] % args.development_every == 0:
                evaluation()
            if progress["updates"] % args.checkpoint_every == 0:
                save()
        else:
            progress["status"] = "update_budget"
    except Exception as error:
        progress.update(status="failed", error=type(error).__name__ + ": " + str(error))
        save("failure.pt")
        raise
    finally:
        progress["elapsed_seconds"] = elapsed_before + time.monotonic() - started
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if progress["status"] != "failed":
            save()
        atomic_json(
            work / "summary.json",
            {
                **progress,
                "exit_class": exit_class(progress["status"]),
                "final_seeds_consumed": [],
                "deployment_authorized": False,
            },
        )
    return progress


def evaluate_checkpoint(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved["binding"]["source_sha256"] != source_hash():
        raise ValueError("checkpoint source mismatch")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    dtype = getattr(torch, saved["binding"]["dtype"])
    policy, _ = load_policy_checkpoint(args.checkpoint, device, dtype)
    cfg = saved["binding"]["protocol"]
    initial = pool_states(
        [
            sample_scenarios(
                cfg["scenarios_per_bank"],
                seed=s,
                dt=policy.config.dt,
                device=device,
                dtype=dtype,
                scenario_mode=cfg["scenario_mode"],
            )[0]
            for s in DEVELOPMENT_SEEDS
        ]
    )
    report = evaluate(
        policy,
        L2FSimulator(L2FParams(dt=policy.config.dt)),
        initial,
        saved["binding"]["horizon"],
        TaskLossConfig(**cfg["loss"]),
    )
    atomic_json(Path(args.work_dir) / "evaluation.json", report)
    return report

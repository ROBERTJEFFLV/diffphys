"""One production path: configurable temporal BPTT decay and persistent Adam."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import time
import traceback

import numpy as np
import torch

from .env_raptor import (RaptorParams, RaptorState, environment_contract,
                        ACTION_CONVENTION)
from .response_noise import DisturbanceConfig, attach_disturbances, disturbance_report
from .response_policy import (ARCHITECTURE, AGILE_ARCHITECTURE,
                              ResponseMotorPolicy, ResponsePolicyConfig)
from .agile_checkpoint import import_hover_checkpoint
from .agile_curriculum import (agile_config_from_dict, load_agile_config,
                               curriculum_difficulty)
from .agile_mission import sample_agile_mission
from .agile_objective import AgileLossConfig, AGILE_OBJECTIVE_VERSION
from .response_task import (
    TaskLossConfig,
    TASK_OBJECTIVE_VERSION,
    sample_scenarios,
    rollout,
    trajectory_metrics,
    reference_episode_metrics,
    task_loss_components,
    hard_risk_metrics,
    tensors_finite,
)
from .response_adjoints import collect_rollout, backward_actor
from .response_execution import exit_class, critical_gc
from .compiled_physics import make_simulator, physics_backend_contract
from .scene_prefetch import PrefetchedSampler
from .response_groups import GroupBalanceConfig, GROUP_BALANCE_VERSION
from .response_sampling import sample_coverage, sampling_contract, validate_sampling
from .response_audit import AuditConfig, UpdateAudit, parameter_changes, json_report

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "raptor-multi-airframe-gaussian-v6"
TRAIN_SEED_BASE = 31_000_007
TRAINING_BANKS = 4
DEVELOPMENT_SEEDS = (32_000_007, 32_010_007)
TEST_SEEDS = (34_000_007, 34_010_007)
AGILE_TRAIN_SEED_BASE = 41_000_007
AGILE_VALIDATION_SEED = 42_000_007
AGILE_TEST_SEED = 44_000_007
AGILE_SAMPLING_PROTOCOL = {
    "version": "committed-update-cpu-mission-v1",
    "train_seed_base": AGILE_TRAIN_SEED_BASE,
    "validation_seed": AGILE_VALIDATION_SEED,
    "test_seed": AGILE_TEST_SEED,
    "eval_hover_fraction": 0.,
}
SOURCE_FILES = (
    "diffphys/__init__.py",
    "diffphys/response_policy.py",
    "diffphys/response_task.py",
    "diffphys/response_adjoints.py",
    "diffphys/response_groups.py",
    "diffphys/response_grad_probe.py",
    "diffphys/response_sampling.py",
    "diffphys/response_audit.py",
    "configs/physics_coverage.json",
    "diffphys/response_training.py",
    "diffphys/response_execution.py",
    "diffphys/compiled_physics.py",
    "diffphys/strict_inductor.py",
    "diffphys/scene_prefetch.py",
    "diffphys/env_raptor.py",
    "diffphys/response_noise.py",
    "diffphys/agile_mission.py",
    "diffphys/agile_objective.py",
    "diffphys/agile_checkpoint.py",
    "diffphys/agile_curriculum.py",
    "configs/agile_flight.json",
    "tools/train_response_control.py",
    "tools/replay_response_update.py",
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
    """Load only native B tensors; never silently discard an old W_c block."""
    expected = policy.state_dict()
    if set(state) != set(expected):
        raise ValueError("Actor keys do not match GRU16 hidden-only readout")
    if any(
        state[n].shape != v.shape or not bool(torch.isfinite(state[n]).all())
        for n, v in expected.items()
    ):
        raise ValueError("Actor tensor shape mismatch or nonfinite weights")
    policy.load_state_dict(state, strict=True)


def require_reference_checkpoint(value):
    """A shape match cannot certify old hover-centered or plus-frame semantics."""
    if (value.get("schema") != PROTOCOL_VERSION
            or value.get("architecture") not in (ARCHITECTURE, AGILE_ARCHITECTURE)):
        raise ValueError("incompatible Actor architecture or motor semantics")
    cfg = value["binding"]["protocol"]
    task = cfg.get("task", "hover")
    if task not in ("hover", "agile"):
        raise ValueError("unsupported checkpoint task configuration")
    policy_config = ResponsePolicyConfig(**value["policy_config"])
    architecture = AGILE_ARCHITECTURE if task == "agile" else ARCHITECTURE
    if (cfg.get("version") != PROTOCOL_VERSION
            or value["architecture"] != architecture or cfg.get("architecture") != architecture
            or policy_config.goal_conditioning != (task == "agile")
            or ResponsePolicyConfig(**cfg["policy"]) != policy_config):
        raise ValueError("checkpoint task, Actor architecture or motor semantics mismatch")
    objective = AGILE_OBJECTIVE_VERSION if task == "agile" else TASK_OBJECTIVE_VERSION
    if cfg.get("task_objective") != objective:
        raise ValueError("checkpoint task objective mismatch: old losses require weights-only initialization")
    if task == "agile":
        agile_config = agile_config_from_dict(cfg["agile"])
        if (AgileLossConfig(**cfg["loss"]) != agile_config.loss
                or cfg.get("mission_sampling") != AGILE_SAMPLING_PROTOCOL):
            raise ValueError("checkpoint agile objective or sampling configuration mismatch")
    else:
        TaskLossConfig(**cfg["loss"])
    params = RaptorParams(**cfg["environment_params"])
    if (cfg.get("environment") != environment_contract(params)
            or cfg.get("environment_source_sha256") != file_hash(ROOT / "diffphys/env_raptor.py")
            or cfg.get("noise_source_sha256") != file_hash(ROOT / "diffphys/response_noise.py")):
        raise ValueError("checkpoint environment/action contract mismatch")
    DisturbanceConfig(**cfg["disturbances"])
    if value["policy_config"]["dt"] != params.dt:
        raise ValueError("checkpoint sampler, policy and environment disagree")
    return params


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
    if set(current) != set(lookup):
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
    require_reference_checkpoint(value)
    policy = ResponseMotorPolicy(ResponsePolicyConfig(**value["policy_config"])).to(
        dtype=getattr(torch, value["binding"]["dtype"])
    )
    migrate_actor_weights(policy, value["model"])
    if value.get("model_sha256") != model_hash(policy):
        raise ValueError("checkpoint model digest mismatch")
    return policy.to(device=device, dtype=dtype), value


def pool_states(states):
    # Validate before allocating: an already-noisy bank cannot be pooled twice.
    if not states or any(s.noise_tape.shape[1] != 1 for s in states):
        raise ValueError("pool clean initial banks before sampling disturbances")
    result = RaptorState(**{f.name: torch.cat([getattr(s, f.name) for s in states])
                            for f in fields(states[0])})
    return replace(result, noise_row=torch.arange(len(result.mass), device=result.mass.device))


def sample_pool(scenarios, seeds, *, dt=.01, device="cpu", dtype=torch.float32,
                horizon=500, disturbances=DisturbanceConfig()):
    states = [sample_scenarios(scenarios, seed=s, dt=dt, dtype=dtype, horizon=horizon,
                              disturbances=DisturbanceConfig.clean()) for s in seeds]
    initial = pool_states(states)
    # Airframe/initial-state randomness does not change when noise settings change.
    noise_seed = int.from_bytes(hashlib.sha256(repr(tuple(seeds)).encode()).digest()[:8], "little")
    initial = attach_disturbances(initial, disturbances, seed=noise_seed, horizon=horizon)
    return initial.to(device, dtype)


def sample_training_scenarios(scenarios, attempt_index, *, dt=.01, device="cpu",
                              dtype=torch.float32, horizon=500,
                              disturbances=DisturbanceConfig(), sampling="random",
                              sampling_report=None):
    validate_sampling(sampling, TRAINING_BANKS*scenarios)
    if attempt_index < 0:
        raise ValueError("invalid sampling index")
    seeds = [TRAIN_SEED_BASE + TRAINING_BANKS*attempt_index + i for i in range(TRAINING_BANKS)]
    if seeds[-1] >= min(DEVELOPMENT_SEEDS):
        raise ValueError("TRAIN seed range would enter reserved EVAL")
    if sampling == "random":
        initial = sample_pool(scenarios, seeds, dt=dt, device=device, dtype=dtype,
                              horizon=horizon, disturbances=disturbances)
        report = {"mode": "random", "total_train": TRAINING_BANKS*scenarios}
    else:
        initial, report = sample_coverage(TRAINING_BANKS*scenarios, seeds, dt=dt, dtype=dtype)
        # Exactly one full noise/pulse tape for the selected pool, never candidates.
        noise_seed = int.from_bytes(hashlib.sha256(repr(tuple(seeds)).encode()).digest()[:8], "little")
        initial = attach_disturbances(initial, disturbances, seed=noise_seed, horizon=horizon).to(device, dtype)
    if sampling_report is not None:
        sampling_report.update(report)
    return initial, seeds


def sample_training_mission(initial, update, config, *, horizon, device="cpu",
                            dtype=torch.float32, sampling_report=None):
    """Attach the current update's task after delivery of the CPU physical pool.

    The worker prefetches physical parameters and immutable disturbance tapes
    only.  Task amplitude is determined from the committed update, so resume,
    failed-update rollback and update replay use the same continuous schedule.
    Mission generation has its own seed and does not advance the training RNG.
    """
    if initial.position.device.type != "cpu":
        raise ValueError("agile missions must be sampled on the delivered CPU physical pool")
    difficulty = curriculum_difficulty(update, config)
    seed = AGILE_TRAIN_SEED_BASE + update
    if seed >= AGILE_VALIDATION_SEED:
        raise ValueError("TRAIN mission seed range would enter reserved EVAL")
    initial, mission = sample_agile_mission(
        initial, config.mission, seed=seed, difficulty=difficulty, horizon=horizon)
    if sampling_report is not None:
        sampling_report["agile"] = {
            "schedule_update": update,
            "mission_seed": seed,
            "requested_difficulty": difficulty,
            "mission": mission.report,
        }
    return initial.to(device, dtype), mission.to(device, dtype)


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


def binding(args, policy_config, loss_config, *, agile_config=None):
    task = getattr(args, "task", "hover")
    if task == "agile" and agile_config is None:
        agile_config = load_agile_config(getattr(args, "agile_config", None))
    result = {
        "source_sha256": source_hash(),
        "algorithm": ("time-decayed-bptt-adam" if args.time_decay > 0 else "exact-horizon-bptt-adam")
                     + "+physics-group-fixed-cap",
        "protocol": {
            "version": PROTOCOL_VERSION,
            "architecture": AGILE_ARCHITECTURE if task == "agile" else ARCHITECTURE,
            "task": task,
            "policy": asdict(policy_config),
            "loss": asdict(loss_config),
            "task_objective": AGILE_OBJECTIVE_VERSION if task == "agile" else TASK_OBJECTIVE_VERSION,
            "scenarios_per_bank": args.scenarios,
            "disturbances": asdict(DisturbanceConfig.from_args(args)),
            "eval_scenarios_per_bank": args.eval_scenarios,
            "environment_source_sha256": file_hash(ROOT / "diffphys/env_raptor.py"),
            "noise_source_sha256": file_hash(ROOT / "diffphys/response_noise.py"),
            "environment_params": asdict(RaptorParams(dt=policy_config.dt)),
            "environment": environment_contract(RaptorParams(dt=policy_config.dt)),
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "test_seeds": list(TEST_SEEDS),
            "deployment_authorized": False,
        },
        **{
            n: getattr(args, n)
            for n in (
                "seed",
                "device",
                "dtype",
                "horizon",
                "time_decay",
                "lr",
                "gradient_clip",
                "gradient_scale",
                "threads",
            )
        },
        "group_balance": {"version": GROUP_BALANCE_VERSION,
                          **asdict(GroupBalanceConfig.from_args(args))},
        "update_audit": asdict(AuditConfig.from_args(args)),
        "training_banks": TRAINING_BANKS,
        "training_sampling": sampling_contract(getattr(args, "train_sampling", "random")),
        "torch_version": str(torch.__version__),
        "numerical_backend": physics_backend_contract(args.device),
        "numerical_runtime": {"cuda":torch.version.cuda, "cudnn":torch.backends.cudnn.version(),
                              "gpu":torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else None,
                              "tf32":False},
        "execution": {"version":"compiled-prefetch-v1", "cpu_sampling":"spawn-one-batch",
                      "worker_threads":args.threads, "interop_threads":torch.get_num_interop_threads(),
                      "gc":"defer-forward-backward", "compaction":"dynamic"},
    }
    if task == "agile":
        result["protocol"].update(
            agile=agile_config.to_dict(),
            mission_sampling=dict(AGILE_SAMPLING_PROTOCOL),
        )
    return result


def _checkpoint(policy, optimizer, progress, run_binding):
    return {
        "schema": PROTOCOL_VERSION,
        "architecture": policy.architecture,
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
def evaluate(policy, simulator, initial, horizon, loss_config, *, mission=None):
    if mission is not None:
        # collect_rollout uses the same continuous objective/reduction as TRAIN.
        # no_grad disables the Actor probe and retains no backward graph.
        record = collect_rollout(policy, simulator, initial, loss_config,
                                 horizon=horizon, time_decay=0., mission=mission)
        report = dict(record.metrics)
        if not report["finite"]:
            raise FloatingPointError("nonfinite fixed EVAL")
        report["loss_config"] = asdict(loss_config)
        report["disturbances"] = disturbance_report(initial)
        return report
    trace = rollout(policy, simulator, initial, horizon)
    report = trajectory_metrics(trace, loss_config)
    if not report["finite"]:
        raise FloatingPointError("nonfinite fixed EVAL")
    report.update(reference_episode_metrics(trace))
    report["task_components"] = task_loss_components(trace, loss_config)
    report["loss_config"] = asdict(loss_config)
    report["disturbances"] = disturbance_report(initial)
    risk = hard_risk_metrics(trace)
    report["risk"] = risk
    return report


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(args, policy_config, loss_config):
    """A finite loss increase never vetoes an update; numerical failures abort."""
    task = getattr(args, "task", "hover")
    if task not in ("hover", "agile"):
        raise ValueError("task must be hover or agile")
    if policy_config.goal_conditioning != (task == "agile"):
        raise ValueError("Actor goal conditioning must match the selected task")
    agile_config = None
    if task == "agile":
        agile_config = getattr(args, "agile_training_config", None)
        if agile_config is None:
            agile_config = load_agile_config(getattr(args, "agile_config", None))
        if loss_config != agile_config.loss:
            raise ValueError("agile loss must match the expanded agile configuration")
        if agile_config.mission.hold_seconds >= args.horizon * policy_config.dt:
            raise ValueError("agile horizon must exceed mission hold_seconds")
    elif not isinstance(loss_config, TaskLossConfig):
        raise ValueError("hover task requires the hover loss configuration")
    sampling = getattr(args, "train_sampling", "random")
    validate_sampling(sampling, TRAINING_BANKS*args.scenarios)
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
    simulator = make_simulator(RaptorParams(dt=policy_config.dt), device)
    group_config = GroupBalanceConfig.from_args(args)
    disturbances = DisturbanceConfig.from_args(args)
    audit_config = AuditConfig.from_args(args)
    if (group_config.layout == "coverage128"
            and TRAINING_BANKS*args.scenarios < 128*group_config.min_scenarios):
        raise ValueError("not enough TRAIN scenes for the requested per-cell minimum")
    if TRAINING_BANKS*args.scenarios < group_config.min_scenarios:
        raise ValueError("not enough unique TRAIN scenes for group balancing")
    run_binding = binding(args, policy_config, loss_config, agile_config=agile_config)
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
    }
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        require_reference_checkpoint(saved)
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
    elif args.init_checkpoint:
        if task == "agile":
            progress["initialization"] = import_hover_checkpoint(policy, args.init_checkpoint)
        else:
            saved = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
            # Explicit hover weights-only import retains the legacy interface
            # check. Missing goal_conditioning denotes the original hover Actor.
            old_policy = saved.get("policy_config", {})
            old_environment = saved.get("binding", {}).get("protocol", {}).get("environment", {})
            if (saved.get("architecture") != ARCHITECTURE
                    or any(old_policy.get(k, False if k == "goal_conditioning" else None) != v
                           for k, v in asdict(policy_config).items())
                    or old_environment.get("action_convention") != ACTION_CONVENTION):
                raise ValueError("initialization Actor interface/motor semantics mismatch")
            migrate_actor_weights(policy, saved.get("model", {}))
            if saved.get("model_sha256") != model_hash(policy):
                raise ValueError("checkpoint model digest mismatch")
            progress["initialization"] = {"file_sha256": file_hash(args.init_checkpoint), "weights_only": True}
    for name in ("history.jsonl", "evaluation.jsonl"):
        _recover_log(work / name, progress["updates"])
    audit = UpdateAudit(work, audit_config, evaluation_interval=args.development_every,
                        binding=run_binding, source_root=ROOT, source_files=SOURCE_FILES,
                        torch_writer=atomic_torch, json_writer=atomic_json)
    if progress.get("sampling_index", progress["updates"]) != progress["updates"]:
        raise ValueError("inconsistent checkpoint sampling progress")
    progress["sampling_index"] = progress["updates"]
    elapsed_before = progress["elapsed_seconds"]
    entry_gc = gc.isenabled()
    sampler = None
    record = backward = before = initial = mission = None
    gradients_for_audit = {}
    failure = None
    stop = []
    old_handlers = {}

    def interrupted(signum, frame):
        stop.append(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, interrupted)
    eval_initial = eval_mission = None

    def save(name="latest.pt"):
        atomic_torch(work / name, _checkpoint(policy, optimizer, progress, run_binding))

    def evaluation():
        rng = capture_rng()
        try:
            report = evaluate(policy, simulator, eval_initial, args.horizon, loss_config,
                              mission=eval_mission)
            if agile_config is not None:
                report.update(eval_split="validation", eval_difficulty=agile_config.fixed_eval_difficulty,
                              eval_mission_seed=AGILE_VALIDATION_SEED,
                              eval_hover_fraction=AGILE_SAMPLING_PROTOCOL["eval_hover_fraction"])
        finally:
            restore_rng(rng)
        _append(work / "evaluation.jsonl", {"update": progress["updates"], **report})
        audit.evaluation(report, progress["best_score"], progress)
        score = report["task_objective"]
        cost_best = progress["best_score"] is None or score < progress["best_score"]
        if cost_best:
            progress.update(best_score=score, best_update=progress["updates"])
        progress["last_evaluated_update"] = progress["updates"]
        # Every evaluated Actor has its own full checkpoint; never only latest.pt.
        evaluated = work / "checkpoints" / f"eval_{progress['updates']:08d}_{model_hash(policy)[:16]}.pt"
        if not evaluated.exists():
            atomic_torch(evaluated, _checkpoint(policy, optimizer, progress, run_binding))
        if cost_best:
            save("best.pt")

    try:
        eval_initial = sample_pool(args.eval_scenarios, DEVELOPMENT_SEEDS, dt=policy_config.dt,
                                   device="cpu" if agile_config is not None else device,
                                   dtype=dtype, horizon=args.horizon,
                                   disturbances=disturbances)
        if agile_config is not None:
            eval_initial, eval_mission = sample_agile_mission(
                eval_initial, replace(agile_config.mission,
                                      hover_fraction=AGILE_SAMPLING_PROTOCOL["eval_hover_fraction"]),
                seed=AGILE_VALIDATION_SEED,
                difficulty=agile_config.fixed_eval_difficulty, horizon=args.horizon)
            eval_initial = eval_initial.to(device, dtype)
            eval_mission = eval_mission.to(device, dtype)
        if (
            progress.get("last_evaluated_update") != progress["updates"]
            and progress["updates"] == 0
        ):
            evaluation()
        save()
        if progress["updates"] < args.updates:
            sampler = PrefetchedSampler(TRAINING_BANKS*args.scenarios, args.horizon,
                                        progress["updates"], args.updates, dtype=dtype,
                                        threads=args.threads, sampling=sampling,
                                        dt=policy_config.dt, disturbances=disturbances)
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
            record = backward = None
            gradients_for_audit = {}
            seeds, sampling_report = [], {}
            try:
                sampling_report = {}
                initial, seeds = sampler(
                    args.scenarios,
                    progress["updates"],
                    dt=policy_config.dt,
                    device="cpu" if agile_config is not None else device,
                    dtype=dtype,
                    disturbances=disturbances,
                    horizon=args.horizon,
                    sampling=sampling,
                    sampling_report=sampling_report,
                )
                mission = None
                if agile_config is not None:
                    mission_started = time.monotonic()
                    initial, mission = sample_training_mission(
                        initial, progress["updates"], agile_config, horizon=args.horizon,
                        device=device, dtype=dtype, sampling_report=sampling_report)
                    _sync(device)
                    sampler.last["agile_mission_and_upload_seconds"] = time.monotonic() - mission_started
                optimizer.zero_grad(set_to_none=True)
                with critical_gc():
                    record = collect_rollout(
                        policy,
                        simulator,
                        initial,
                        loss_config,
                        horizon=args.horizon,
                        time_decay=args.time_decay,
                        group_config=group_config,
                        **({"mission": mission} if mission is not None else {}),
                    )
                    _sync(device)
                    forward_done = time.monotonic()
                    backward = backward_actor(
                        policy, simulator, record, loss_config, gradient_scale=args.gradient_scale,
                    )
                    _sync(device)
                    backward_done = time.monotonic()
                    # Full backward and Probe.close are complete before restoring GC.
                    record = replace(record, costs=record.costs.detach())
                raw_norm = safe_global_clip(policy.parameters(), args.gradient_clip)
                # Clone after all gradient transformations, before Adam can act.
                gradients_for_audit = {name: None if p.grad is None else p.grad.detach().clone()
                                       for name, p in policy.named_parameters()}
                optimizer.step()
                tensors = list(policy.parameters()) + [
                    v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)
                ]
                if not tensors_finite(tensors):
                    raise FloatingPointError("nonfinite Adam parameters or moments")
                changes = parameter_changes(policy, before[0])
                if not math.isfinite(changes["l2"]):
                    raise FloatingPointError("nonfinite actual Actor parameter-step norm")
                _sync(device)
                optimizer_done = time.monotonic()
            except BaseException as error:
                if record is not None and record.probe is not None:
                    record.probe.close()
                # Preserve the attempted state BEFORE restoring Actor/Adam/RNG.
                try:
                    audit.record(update=progress["updates"]+1, before=before, policy=policy,
                                 optimizer=optimizer, names=optimizer_parameter_names(policy, optimizer),
                                 rng_after=capture_rng(), progress=progress,
                                 sampling={"seeds": seeds, "report": sampling_report},
                                 gradients=gradients_for_audit,
                                 groups={"layout": None if record is None else record.group_balance,
                                         "gradient": None if backward is None else backward.get("group_gradient")},
                                 changes=None, failure=type(error).__name__+": "+str(error))
                finally:
                    policy.load_state_dict(before[0])
                    optimizer.load_state_dict(before[1])
                    restore_rng(before[2])
                    optimizer.zero_grad(set_to_none=True)
                raise
            try:
                audit_path = audit.record(
                    update=progress["updates"]+1, before=before, policy=policy, optimizer=optimizer,
                    names=optimizer_parameter_names(policy, optimizer), rng_after=capture_rng(),
                    progress=progress, sampling={"seeds": seeds, "report": sampling_report},
                    gradients=gradients_for_audit,
                    groups={"layout": record.group_balance, "gradient": backward["group_gradient"]},
                    changes=changes)
            except BaseException:
                # Do not silently train without requested evidence if storage fails.
                policy.load_state_dict(before[0])
                optimizer.load_state_dict(before[1])
                restore_rng(before[2])
                optimizer.zero_grad(set_to_none=True)
                raise
            _sync(device)
            audit_done = time.monotonic()
            progress["updates"] += 1
            progress["sampling_index"] = progress["updates"]
            progress["elapsed_seconds"] = elapsed_before + time.monotonic() - started
            row = {
                "update": progress["updates"],
                **sampler.last,
                "train_seeds": seeds,
                "training_sampling": sampling_report,
                **record.metrics,
                "raw_gradient_norm": raw_norm,
                "pre_global_clip_norm": raw_norm,
                "gradient_scale": args.gradient_scale,
                "group_balance": json_report({k: v for k, v in record.group_balance.items()
                                              if k != "scene_group_ids"}),
                "group_gradient": json_report(backward["group_gradient"]),
                "parameter_changes": changes,
                "audit_capsule": audit_path,
                "gradient_entering_adam_norm": gradient_norm(policy.parameters()),
                "time_decay": args.time_decay,
                "disturbances": disturbance_report(initial),
                "forward_seconds": forward_done - update_start,
                "backward_seconds": backward_done - forward_done,
                "optimizer_seconds": optimizer_done - backward_done,
                "audit_seconds": audit_done - optimizer_done,
                "update_seconds": time.monotonic() - update_start,
                "cuda_peak_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                ),
            }
            _append(work / "history.jsonl", row)
            print(json.dumps(row, allow_nan=False), flush=True)
            record = backward = before = initial = mission = None
            gradients_for_audit = {}
            optimizer.zero_grad(set_to_none=True)
            if progress["updates"] % args.development_every == 0:
                evaluation()
            if progress["updates"] % args.checkpoint_every == 0:
                save()
        else:
            progress["status"] = "update_budget"
    except BaseException as error:
        progress.update(status="failed", error=type(error).__name__ + ": " + str(error))
        traceback.clear_frames(error.__traceback__)
        failure = error.with_traceback(None)
    finally:
        if record is not None and record.probe is not None:
            record.probe.close()
        record = backward = before = initial = mission = None
        gradients_for_audit = {}
        optimizer.zero_grad(set_to_none=True)
        cleanup = {}
        try:
            if sampler is not None:
                sampler.worker.close()
                cleanup.update(worker_closed=sampler.worker.closed, worker_pid=sampler.worker.pid,
                               worker_close_seconds=sampler.worker.close_seconds)
        except BaseException as error:
            if failure is None:
                failure = error
                progress.update(status="failed", error=type(error).__name__ + ": " + str(error))
        finally:
            (gc.enable if entry_gc else gc.disable)()
            cleanup.update(entry_gc_enabled=entry_gc, gc_enabled=gc.isenabled(),
                           closed_probes=all(not module._forward_hooks for module in policy.modules()))
            drain = time.monotonic()
            gc.collect()
            _sync(device)
            cleanup["final_gc_seconds"] = time.monotonic()-drain
            progress["elapsed_seconds"] = elapsed_before + time.monotonic() - started
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        save("failure.pt" if failure else "latest.pt")
        progress["elapsed_seconds"] = elapsed_before + time.monotonic() - started
        atomic_json(
            work / "summary.json",
            {**progress, "cleanup":cleanup, "exit_class":exit_class(progress["status"]),
             "final_seeds_consumed":[], "deployment_authorized":False},
        )
    if failure is not None:
        raise failure

    return progress


def evaluate_checkpoint(args):
    """Rebuild the saved task and its independent validation/test sampling split."""
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    params = require_reference_checkpoint(saved)
    checkpoint_source = saved["binding"]["source_sha256"]
    evaluator_source = source_hash()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dtype = getattr(torch, saved["binding"]["dtype"])
    torch.set_num_threads(saved["binding"]["threads"])
    policy, _ = load_policy_checkpoint(args.checkpoint, device, dtype)
    cfg = saved["binding"]["protocol"]
    task = cfg.get("task", "hover")
    split = getattr(args, "eval_split", "validation")
    if split not in ("validation", "test"):
        raise ValueError("eval_split must be validation or test")
    seeds = (cfg.get("development_seeds", list(DEVELOPMENT_SEEDS)) if split == "validation"
             else cfg.get("test_seeds", list(TEST_SEEDS)))
    agile_config = agile_config_from_dict(cfg["agile"]) if task == "agile" else None
    initial = sample_pool(cfg["eval_scenarios_per_bank"], seeds,
                          dt=policy.config.dt, device="cpu" if agile_config is not None else device,
                          dtype=dtype,
                          horizon=saved["binding"]["horizon"],
                          disturbances=DisturbanceConfig(**cfg["disturbances"]))
    mission = None
    if agile_config is not None:
        mission_seed = cfg["mission_sampling"][split + "_seed"]
        initial, mission = sample_agile_mission(
            initial, replace(agile_config.mission,
                             hover_fraction=cfg["mission_sampling"]["eval_hover_fraction"]),
            seed=mission_seed,
            difficulty=agile_config.fixed_eval_difficulty, horizon=saved["binding"]["horizon"])
        initial = initial.to(device, dtype)
        mission = mission.to(device, dtype)
    report = evaluate(
        policy,
        make_simulator(params, device),
        initial,
        saved["binding"]["horizon"],
        agile_config.loss if agile_config is not None else TaskLossConfig(**cfg["loss"]),
        mission=mission,
    )
    report.update(
        task=task,
        eval_split=split,
        eval_seeds=list(seeds),
        checkpoint_source_sha256=checkpoint_source,
        evaluator_source_sha256=evaluator_source,
        source_match=checkpoint_source == evaluator_source,
        numerical_backend=physics_backend_contract(device),
    )
    if agile_config is not None:
        report.update(eval_difficulty=agile_config.fixed_eval_difficulty,
                      eval_mission_seed=mission_seed,
                      eval_hover_fraction=cfg["mission_sampling"]["eval_hover_fraction"])
    output = "evaluation.json" if split == "validation" else "evaluation_test.json"
    atomic_json(Path(args.work_dir) / output, report)
    return report

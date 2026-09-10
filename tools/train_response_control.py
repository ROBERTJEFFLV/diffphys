"""Primary response-conditioned task-learning entry point, without Q2 labels."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from response_policy import ResponsePolicyConfig
from response_task import TaskLossConfig
from response_training import (
    evaluate_checkpoint, evaluate_hidden_reset, freeze_candidate, profile, protocol, train,
    training_batch_count, critic_configuration,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "profile", "evaluate", "hidden-reset", "freeze", "final"), default="train")
    parser.add_argument("--optimizer", choices=("task-adam", "short-window", "adam", "full-space-ms"), default="short-window")
    parser.add_argument("--work-dir", type=Path, default=Path("runs/response_risk_critic_v1/seed7"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--scenarios", type=int, default=64, help="scenarios per stratified bank; DEV size is unchanged")
    parser.add_argument("--scenario-mode", choices=("physical-fit", "fixed-airframe"), default="physical-fit")
    parser.add_argument("--phase1-probes", action="store_true",
                        help="record continuity/loss/window gradients; fail on continuity or nonfinite")
    parser.add_argument("--phase1-train-only", action="store_true",
                        help="nominal airframe: accept finite TRAIN objective improvement; no DEV or risk veto")
    parser.add_argument("--adam-train-batches", type=int, default=2,
                        help="bank count for legacy Adam; short-window always uses two, MS uses one")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--window-steps", type=int, default=50)
    parser.add_argument("--actor-proposal", choices=("physics-subspace", "smoothmax-adam"), default="physics-subspace",
                        help="bounded real TRAIN candidate search; Adam is a regression backend")
    parser.add_argument("--subspace-parameter-relative-step", type=float, default=1.e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--critic-epochs", type=int, default=1)
    parser.add_argument("--critic-batch-size", type=int, default=1024)
    parser.add_argument("--value-target-tau", type=float, default=.6,
                        help="task-value target update: new critic fraction, old target retains 1-tau")
    parser.add_argument("--value-gradient-clip", type=float, default=10.)
    from response_adjoints import STATE_SCALES
    parser.add_argument('--value-derivative-state-groups', choices=tuple(STATE_SCALES), nargs='+', default=tuple(STATE_SCALES))
    parser.add_argument('--value-derivative-boundaries', type=int, nargs='+', default=(),
                        help='default: every nonterminal window boundary')
    parser.add_argument('--value-derivative-holdout-scenes', type=int, default=16)
    parser.add_argument('--value-derivative-batch-size', type=int, default=32)
    parser.add_argument('--value-derivative-epsilon', type=float, default=1.e-8)
    parser.add_argument('--value-derivative-balance-mode', choices=('minibatch','fixed'), default='minibatch')
    parser.add_argument('--value-terminal-mode', choices=('critic','oracle_full_state','none'), default='critic')
    parser.add_argument('--value-critic-only', action='store_true', help='bounded frozen Actor fit/readiness; never step Actor')
    parser.add_argument('--value-warmup-max-fits', type=int, default=8)
    parser.add_argument('--value-warmup-max-seconds', type=float, default=120.)
    parser.add_argument('--value-ready-min-cosine', type=float, default=.9)
    parser.add_argument('--value-ready-max-relative-error', type=float, default=.5)
    parser.add_argument("--critic-dev-relative-tolerance", type=float, default=.002)
    from response_task import RiskConfig
    for name, default in asdict(RiskConfig()).items():
        parser.add_argument("--risk-" + name.replace("_", "-"), type=float, default=default)
    parser.add_argument("--risk-weight", type=float, default=1.)
    parser.add_argument("--risk-smoothmax-beta", type=float, default=10.)
    parser.add_argument("--hard-risk-relative-tolerance", type=float, default=.002)
    parser.add_argument("--hard-risk-absolute-tolerance", type=float, default=1.e-8)
    parser.add_argument("--hard-position-bound", type=float, help="optional flight radius in metres, separate from tracking error")
    parser.add_argument("--hard-velocity-bound", type=float, help="optional speed envelope in m/s")
    parser.add_argument("--critic-direction-samples", type=int, default=4)
    parser.add_argument("--critic-direction-epsilon", type=float, default=.02)
    parser.add_argument("--critic-direction-weight", type=float, default=.1)
    parser.add_argument("--critic-direction-temperature", type=float, default=.1)
    parser.add_argument("--critic-direction-min-gap", type=float, default=1.e-6)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--max-seconds", type=float, default=10800)
    parser.add_argument("--minimum-updates", type=int, default=300)
    parser.add_argument("--development-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-relative-improvement", type=float, default=.002)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--gradient-clip", type=float, default=10)
    parser.add_argument("--adam-max-loss-ratio", type=float, default=2.0)
    parser.add_argument("--adam-max-omega-ratio", type=float, default=2.0)
    parser.add_argument("--adam-max-dev-loss-ratio", type=float, default=2.0)
    parser.add_argument("--maximum-proposal-rejections", "--maximum-adam-rejections",
                        dest="maximum_adam_rejections", type=int, default=3,
                        help="consecutive Actor proposal rejection limit")
    parser.add_argument("--dt", type=float, default=.01)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--action-rate", type=float, default=50)
    parser.add_argument("--integral-limit", type=float, default=.5)
    parser.add_argument("--integral-leak", type=float, default=0)
    for name, default in asdict(TaskLossConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int if name == "steady_steps" else float,
                            default=None if name == "prediction_weight" else default)
    parser.add_argument("--segments", type=int, choices=(2, 4), default=2)
    parser.add_argument("--segment-steps", type=int, default=250)
    parser.add_argument("--damping", type=float, default=100)
    parser.add_argument("--linear-solver", choices=("petsc-minres", "legacy-cg"), default="petsc-minres")
    parser.add_argument("--kkt-rtol", type=float, default=1.0e-6)
    parser.add_argument("--kkt-atol", type=float, default=1.0e-10)
    parser.add_argument("--kkt-max-iterations", type=int, default=200)
    parser.add_argument("--kkt-preconditioner", choices=("curvature-diagonal", "block-diagonal"), default="curvature-diagonal")
    parser.add_argument("--kkt-curvature-probes", type=int, default=8)
    parser.add_argument("--kkt-monitor", action="store_true")
    parser.add_argument("--cg-iterations", type=int, default=16)
    parser.add_argument("--maximum-ms-rejections", type=int, default=3)
    parser.add_argument("--parameter-radius", type=float, default=.05)
    parser.add_argument("--action-radius", type=float, default=.01)
    parser.add_argument("--action-max-radius", type=float, default=.05)
    parser.add_argument("--contract-report", type=Path)
    parser.add_argument("--ms-debug", action="store_true")
    parser.add_argument("--reset-step", type=int, default=100)
    parser.add_argument("--reset-horizon", type=int, choices=(250, 500), default=250)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--q2-checkpoint", type=Path, help="optional comparison in evaluate/final only")
    parser.add_argument("--consume-final", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.value_critic_only and args.optimizer != 'task-adam':
        parser.error('value-critic-only requires task-adam')
    if args.weight_decay is None:
        args.weight_decay = 0. if args.optimizer == "task-adam" else 1.e-5
    if args.optimizer == "task-adam" and (args.scenario_mode != "fixed-airframe"
            or args.mode not in ("train", "profile", "evaluate") or args.phase1_train_only or args.weight_decay != 0):
        parser.error("task-adam Phase 1 requires fixed-airframe, train/profile/evaluate, no approval flag and zero weight decay")
    if args.phase1_train_only and (args.scenario_mode != "fixed-airframe" or not args.phase1_probes
                                  or args.mode not in ("train", "profile")):
        parser.error("TRAIN-only Phase 1 requires fixed-airframe, phase1-probes and train/profile mode")
    if (args.phase1_probes or args.scenario_mode == "fixed-airframe") and (
        args.optimizer not in ("short-window", "task-adam") or (args.optimizer == "short-window" and args.actor_proposal != "physics-subspace")
        or args.mode not in ("train", "profile", "evaluate")
    ):
        parser.error("Phase 1 uses the normal short-window subspace train/profile/evaluate chain")
    if args.horizon is None:
        args.horizon = 500 if args.optimizer in ("short-window", "task-adam") else 125
    if args.prediction_weight is None:
        args.prediction_weight = 0. if args.optimizer in ("short-window", "task-adam") else .01
    if args.optimizer in ("short-window", "task-adam"):
        try:
            critic_configuration(args)
        except ValueError as error:
            parser.error(str(error))
        if args.horizon % args.window_steps or args.horizon < args.window_steps:
            parser.error("short-window horizon must be divisible by --window-steps")
        if args.prediction_weight != 0 or args.huber_delta <= 0:
            parser.error("short-window uses performance + local/future risk with no auxiliary and requires Huber delta > 0")
    if not (math.isfinite(args.kkt_rtol) and 0 < args.kkt_rtol < 1
            and math.isfinite(args.kkt_atol) and args.kkt_atol >= 0
            and args.kkt_max_iterations > 0 and args.kkt_curvature_probes > 0):
        parser.error("KKT requires 0 < rtol < 1, finite atol >= 0, and positive iterations")
    if args.seed != 7:
        parser.error("this protocol trains only model initialization seed7")
    if min(args.horizon, args.threads, args.minimum_updates, args.development_every,
           args.checkpoint_every, args.patience, args.segment_steps, args.cg_iterations, args.maximum_ms_rejections, args.reset_step) < 1:
        parser.error("counts must be positive")
    if args.maximum_adam_rejections < 1 or any(
        not math.isfinite(value) or value < 1
        for value in (args.adam_max_loss_ratio, args.adam_max_omega_ratio, args.adam_max_dev_loss_ratio)
    ):
        parser.error("Adam catastrophe ratios must be finite and >=1; rejection limit must be positive")
    if args.adam_train_batches < 1:
        parser.error("--adam-train-batches must be positive")
    if args.scenarios < 16 or args.scenarios % 16:
        parser.error("--scenarios must be a positive multiple of 16")
    if min(args.max_seconds, args.lr, args.gradient_clip, args.damping,
           args.parameter_radius, args.action_radius, args.action_max_radius) <= 0:
        parser.error("budgets, learning rate, and solver radii must be positive")
    if not 0 <= args.min_relative_improvement < 1 or args.weight_decay < 0:
        parser.error("invalid optimizer/early-stop settings")
    if args.q2_checkpoint is not None and args.mode not in ("evaluate", "final"):
        parser.error("Q2 is not allowed in training, profiling, or candidate selection")
    if args.consume_final and args.mode != "final":
        parser.error("--consume-final is only valid for --mode final")
    if args.resume is not None and args.initialize_from is not None:
        parser.error("choose exact resume or weights-only initialization")
    if args.optimizer == "full-space-ms":
        args.horizon = args.segment_steps * args.segments
    return args


def main(argv=None):
    args = parse_args(argv)
    policy_config = ResponsePolicyConfig(**{
        name: getattr(args, name) for name in ResponsePolicyConfig.__dataclass_fields__
    })
    loss_config = TaskLossConfig(**{
        name: getattr(args, name) for name in TaskLossConfig.__dataclass_fields__
    })
    if args.dry_run:
        result = {"dry_run": True, "mode": args.mode, "optimizer": args.optimizer,
                  "protocol": protocol(policy_config, loss_config, args.scenarios, args.scenario_mode),
                  "horizon": args.horizon, "updates": args.updates,
                  "training_batches": training_batch_count(args),
                  "training_scenarios": args.scenarios * training_batch_count(args),
                  "window_steps": args.window_steps if args.optimizer in ("short-window", "task-adam") else None,
                  "critic_training_only": args.optimizer in ("short-window", "task-adam"),
                  "critic": asdict(critic_configuration(args))
                            if args.optimizer in ("short-window", "task-adam") else None,
                  "linear_solver": args.linear_solver, "kkt_rtol": args.kkt_rtol,
                  "kkt_max_iterations": args.kkt_max_iterations,
                  "teacher_checkpoint_required": False, "all_policy_parameters_trainable": True}
    elif args.mode == "train":
        result = train(args, policy_config, loss_config)
    elif args.mode == "profile":
        result = profile(args, policy_config, loss_config)
    elif args.mode == "hidden-reset":
        result = evaluate_hidden_reset(args)
    elif args.mode == "freeze":
        result = freeze_candidate(args)
    else:
        result = evaluate_checkpoint(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Primary response-conditioned task-learning entry point, without Q2 labels."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from response_policy import ResponsePolicyConfig
from response_task import TaskLossConfig
from response_training import (
    evaluate_checkpoint, evaluate_hidden_reset, freeze_candidate, profile, protocol, train,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "profile", "evaluate", "hidden-reset", "freeze", "final"), default="train")
    parser.add_argument("--optimizer", choices=("adam", "full-space-ms"), default="adam")
    parser.add_argument("--work-dir", type=Path, default=Path("runs/response_control_v1_seed7"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--scenarios", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=125)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--max-seconds", type=float, default=10800)
    parser.add_argument("--minimum-updates", type=int, default=300)
    parser.add_argument("--development-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-relative-improvement", type=float, default=.002)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=10)
    parser.add_argument("--dt", type=float, default=.01)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--action-rate", type=float, default=50)
    parser.add_argument("--integral-limit", type=float, default=.5)
    parser.add_argument("--integral-leak", type=float, default=0)
    for name, default in asdict(TaskLossConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int if name == "steady_steps" else float, default=default)
    parser.add_argument("--segments", type=int, choices=(2, 4), default=2)
    parser.add_argument("--segment-steps", type=int, default=250)
    parser.add_argument("--damping", type=float, default=100)
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
    if args.seed != 7:
        parser.error("this protocol trains only model initialization seed7")
    if min(args.horizon, args.threads, args.minimum_updates, args.development_every,
           args.checkpoint_every, args.patience, args.segment_steps, args.cg_iterations, args.maximum_ms_rejections, args.reset_step) < 1:
        parser.error("counts must be positive")
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
                  "protocol": protocol(policy_config, loss_config, args.scenarios),
                  "horizon": args.horizon, "updates": args.updates,
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

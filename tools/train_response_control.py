#!/usr/bin/env python3
"""Response Actor-only training, bounded profiling and fixed development evaluation."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from response_policy import ResponsePolicyConfig
from response_task import TaskLossConfig
from response_training import train, evaluate_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "profile", "evaluate"), default="train")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--scenario-mode", choices=("fixed-airframe", "physical-fit"), default="fixed-airframe"
    )
    parser.add_argument(
        "--scenarios", type=int, default=128, help="initial states per bank; TRAIN pools four banks"
    )
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument(
        "--backprop-mode", choices=("full", "windowed"), default="full",
        help="full retains the entire graph; windowed recomputes to save memory",
    )
    parser.add_argument("--window-steps", type=int, default=50)
    parser.add_argument(
        "--updates",
        type=int,
        default=50,
        help="total update index to reach, including resumed updates",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=1800.0,
        help="per invocation budget; finish current update",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument(
        "--gradient-scale",
        type=float,
        default=0.1,
        help="fixed full-gradient multiplier, independent of window length",
    )
    parser.add_argument(
        "--agc",
        type=float,
        default=0.0,
        help="optional unitwise adaptive clipping (e.g. .01); 0 disables",
    )
    parser.add_argument("--development-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--work-dir", type=Path, default=Path("runs/response_actor_only/seed7"))
    initialize = parser.add_mutually_exclusive_group()
    initialize.add_argument("--resume", type=Path)
    initialize.add_argument(
        "--init-checkpoint",
        type=Path,
        help="explicit weights-only initialization; fresh Adam and sampling",
    )
    initialize.add_argument(
        "--migrate-checkpoint", type=Path, help="explicit legacy Actor-only Adam/RNG migration"
    )
    parser.add_argument(
        "--migration-metadata",
        type=Path,
        help="audited JSON with checkpoint hash, semantics, parameter names and next_update",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="checkpoint for fixed development evaluation"
    )
    for config in (ResponsePolicyConfig(), TaskLossConfig()):
        for field in fields(config):
            default = getattr(config, field.name)
            parser.add_argument(
                "--" + field.name.replace("_", "-"), type=type(default), default=default
            )
    args = parser.parse_args(argv)
    for name in (
        "threads",
        "scenarios",
        "horizon",
        "window_steps",
        "development_every",
        "checkpoint_every",
    ):
        if getattr(args, name) < 1:
            parser.error(name + " must be positive")
    if args.scenarios % 16 or args.horizon % args.window_steps:
        parser.error("scenarios must be divisible by 16 and horizon by window-steps")
    if args.updates < 0:
        parser.error("updates must be nonnegative")
    for name in ("max_seconds", "lr", "gradient_clip", "gradient_scale"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name + " must be finite and positive")
    if not math.isfinite(args.agc) or args.agc < 0:
        parser.error("agc must be finite and nonnegative")
    if args.migrate_checkpoint and not args.migration_metadata:
        parser.error("migration requires explicit metadata")
    if args.mode == "evaluate" and not args.checkpoint:
        parser.error("evaluate requires --checkpoint")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "evaluate":
        result = evaluate_checkpoint(args)
    else:
        if args.mode == "profile":
            args.updates = min(args.updates, 1)
        policy_config = ResponsePolicyConfig(
            **{f.name: getattr(args, f.name) for f in fields(ResponsePolicyConfig)}
        )
        loss_config = TaskLossConfig(
            **{f.name: getattr(args, f.name) for f in fields(TaskLossConfig)}
        )
        result = train(args, policy_config, loss_config)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Single multi-airframe training/evaluation entry; no legacy simulator or optimizer arms."""
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
from response_groups import GroupBalanceConfig
from response_noise import DisturbanceConfig
from response_training import train, evaluate_checkpoint
from response_sampling import validate_sampling
from response_audit import AuditConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    parser.convert_arg_line_to_args = lambda line: line.split()
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenarios", type=int, default=128, help="scenes per bank; TRAIN pools four banks")
    parser.add_argument("--train-sampling", choices=("random", "coverage128"), default="random",
                        help="TRAIN only: original random pool or fixed 4x4x4x2 physical coverage")
    parser.add_argument("--eval-scenarios", type=int, default=128, help="scenes per fixed EVAL bank; two banks")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--time-decay", type=float, default=1., help="backward-only decay s^-1; 0 gives exact BPTT")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gradient-clip", type=float, default=10.)
    parser.add_argument("--gradient-scale", type=float, default=.1)
    parser.add_argument("--group-max-groups", type=int, default=None,
                        help="128 for coverage128; adaptive maximum (default 16) for legacy random pools")
    parser.add_argument("--group-min-scenarios", type=int, default=None,
                        help="minimum per group: default 16 for coverage128, 32 for adaptive")
    parser.add_argument("--group-clip-norm", type=float, default=1.,
                        help="fixed whole-Actor group cap AFTER gradient-scale/CVaR; never amplify")
    parser.add_argument("--group-gradient-epsilon", type=float, default=1e-12)
    parser.add_argument("--group-backward", choices=("probe", "vjp"), default="probe",
                        help="one-pass native Actor gradient probe; vjp is an explicit verification reference")
    parser.add_argument("--group-vjp-chunk-size", type=int, default=16,
                        help="used only by the vjp reference, not by the one-pass probe")
    parser.add_argument("--disable-disturbances", action="store_true", help="zero-noise regression fixture")
    parser.add_argument("--disable-pulses", action="store_true", help="disable only pulses for regression; retain harder initial states")
    parser.add_argument("--position-noise-std", type=float, default=.001, help="metres per acquired sample per axis")
    parser.add_argument("--velocity-noise-std", type=float, default=.002, help="m/s per acquired sample per axis")
    parser.add_argument("--attitude-noise-std", type=float, default=.001, help="SO(3) rotation-vector radians per axis")
    parser.add_argument("--omega-noise-std", type=float, default=.002, help="rad/s per acquired sample per axis")
    parser.add_argument("--velocity-delay-min", type=float, default=.010, help="episode latency lower bound in seconds")
    parser.add_argument("--velocity-delay-max", type=float, default=.030, help="episode latency upper bound, <=.030 seconds")
    parser.add_argument("--development-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--work-dir", type=Path, default=Path("runs/pulsed_recovery/seed7"))
    initialize = parser.add_mutually_exclusive_group()
    initialize.add_argument("--resume", type=Path)
    initialize.add_argument("--init-checkpoint", type=Path, help="explicit weights-only start, fresh Adam")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint for evaluation")
    for config in (ResponsePolicyConfig(), TaskLossConfig()):
        for field in fields(config):
            default = getattr(config, field.name)
            parser.add_argument("--"+field.name.replace("_", "-"), type=type(default), default=default)
    for field in fields(AuditConfig):
        default = getattr(AuditConfig(), field.name)
        parser.add_argument("--audit-" + field.name.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args(argv)
    if args.group_max_groups is None:
        args.group_max_groups = 128 if args.train_sampling == "coverage128" else 16
    if args.group_min_scenarios is None:
        args.group_min_scenarios = 16 if args.train_sampling == "coverage128" else 32
    for name in ("threads", "scenarios", "eval_scenarios", "horizon", "development_every", "checkpoint_every"):
        if getattr(args, name) < 1:
            parser.error(name+" must be positive")
    if args.updates < 0:
        parser.error("updates must be nonnegative")
    for name in ("max_seconds", "lr", "gradient_clip", "gradient_scale"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name+" must be finite and positive")
    if not math.isfinite(args.time_decay) or args.time_decay < 0:
        parser.error("time-decay must be finite and nonnegative")
    try:
        groups = GroupBalanceConfig.from_args(args) if args.mode == "train" else GroupBalanceConfig()
        DisturbanceConfig.from_args(args)
        AuditConfig.from_args(args)
        if args.mode == "train":
            validate_sampling(args.train_sampling, 4*args.scenarios)
    except ValueError as error:
        parser.error(str(error))
    if (args.mode == "train" and groups.layout == "coverage128"
            and args.scenarios * 4 < 128 * groups.min_scenarios):
        parser.error("coverage128 requires at least group-min-scenarios scenes in EACH cell")
    if args.mode == "train" and 4*args.scenarios < groups.min_scenarios:
        parser.error("four TRAIN banks must contain at least group-min-scenarios scenes")
    if args.mode == "evaluate" and not args.checkpoint:
        parser.error("evaluate requires --checkpoint")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "evaluate":
        result = evaluate_checkpoint(args)
    else:
        policy_config = ResponsePolicyConfig(**{f.name:getattr(args,f.name) for f in fields(ResponsePolicyConfig)})
        loss_config = TaskLossConfig(**{f.name:getattr(args,f.name) for f in fields(TaskLossConfig)})
        result = train(args, policy_config, loss_config)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

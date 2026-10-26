"""Run the unchanged production trainer until a cooperative stop signal."""
from __future__ import annotations
from dataclasses import asdict, fields
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.train_response_control import parse_args
from response_policy import ResponsePolicyConfig
from response_task import TaskLossConfig
from pass_stop import PassStopLossConfig
from response_training import train


def main():
    args = parse_args()
    if args.mode != 'train':
        raise ValueError('continuous runner supports training only')
    # Budgets are operational controls, excluded from source/objective bindings.
    args.updates = float('inf')
    args.max_seconds = float('inf')
    policy = ResponsePolicyConfig(**{f.name: getattr(args, f.name) for f in fields(ResponsePolicyConfig)})
    values = {f.name: getattr(args, f.name) for f in fields(TaskLossConfig)}
    loss = (PassStopLossConfig(**values, mission=asdict(args.mission))
            if args.mission_config is not None else TaskLossConfig(**values))
    result = train(args, policy, loss)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

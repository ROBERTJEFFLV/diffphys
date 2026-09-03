"""Explicitly migrate a legacy structured checkpoint to fixed multi-tau v1/v2.

The adapter is zero initialized, so the added observer bank is initially a
diagnostic state only.  This tool refuses unexpected missing/shape-mismatched
weights, invalidates capability calibration, and verifies action parity before
writing a new checkpoint.  It is not a Phase-A promotion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structured_checkpoint import deployment_policy_hash  # noqa: E402
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy  # noqa: E402


VERSION_SIZES = {1: 15, 2: 35}


def _config(payload: dict) -> StructuredPolicyConfig:
    valid = {field.name for field in fields(StructuredPolicyConfig)}
    values = payload.get("config", {})
    return StructuredPolicyConfig(**{key: value for key, value in values.items() if key in valid})


@torch.no_grad()
def _parity(source: StructuredRecurrentPolicy,
            target: StructuredRecurrentPolicy) -> dict[str, float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1707)
    observation = 0.1 * torch.randn(16, 25, generator=generator)
    observation[:, 6:15] = torch.eye(3).reshape(1, 9)
    observation[:, 21:25] = 0.0
    source_state = source.initial_state(observation)
    target_state = target.initial_state(observation)
    maximum_action_error = 0.0
    maximum_identifier_error = 0.0
    for _ in range(50):
        left = source.forward_with_aux(observation, source_state)
        right = target.forward_with_aux(observation, target_state)
        maximum_action_error = max(
            maximum_action_error, float((left.action - right.action).abs().max())
        )
        maximum_identifier_error = max(
            maximum_identifier_error,
            float((left.next_state.identifier - right.next_state.identifier).abs().max()),
        )
        source_state, target_state = left.next_state, right.next_state
        observation[:, 21:25] = left.action
    return {
        "maximum_action_error": maximum_action_error,
        "maximum_identifier_error": maximum_identifier_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-version", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    source_config = _config(payload)
    if source_config.motor_observer_bank_size != 0:
        raise RuntimeError("input is not a legacy K=0 structured checkpoint")
    source = StructuredRecurrentPolicy(source_config).eval()
    source.load_state_dict(payload["model"], strict=True)
    target_size = VERSION_SIZES[args.target_version]
    target_config = replace(
        source_config,
        motor_observer_bank_size=target_size,
        motor_observer_mode=f"fixed_multi_tau_v{args.target_version}",
        motor_tau_grid_version=args.target_version,
    )
    target = StructuredRecurrentPolicy(target_config).eval()
    source_state = source.state_dict()
    target_state = target.state_dict()
    unexpected = sorted(set(source_state) - set(target_state))
    mismatched = sorted(
        key for key in set(source_state) & set(target_state)
        if source_state[key].shape != target_state[key].shape
    )
    missing = sorted(set(target_state) - set(source_state))
    target_only = {
        "bank_adapter.weight",
        "motor_observer_bank.tau_rise",
        "motor_observer_bank.tau_fall",
    }
    if unexpected or mismatched or set(missing) != target_only:
        raise RuntimeError(
            f"unsafe checkpoint migration: unexpected={unexpected}, "
            f"mismatched={mismatched}, missing={missing}"
        )
    with torch.no_grad():
        for key, value in source_state.items():
            target_state[key].copy_(value)
    target.load_state_dict(target_state, strict=True)
    parity = _parity(source, target)
    if parity["maximum_action_error"] != 0.0 or parity["maximum_identifier_error"] != 0.0:
        raise RuntimeError(f"zero-adapter parity failed: {parity}")
    # Parity is measured before invalidation so it isolates the architectural
    # migration.  The written checkpoint deliberately cannot reuse a posterior
    # calibration collected under the old recurrent state definition.
    target.invalidate_capability_calibration()

    report = {
        "phase": "structured-multi-tau-explicit-upgrade",
        "source_checkpoint": str(args.input.resolve()),
        "source_deployment_hash": deployment_policy_hash(source),
        "target_deployment_hash": deployment_policy_hash(target),
        "source_motor_observer_bank_size": 0,
        "target_motor_observer_bank_size": target_size,
        "target_motor_tau_grid_version": args.target_version,
        "target_only_state_keys": missing,
        "zero_adapter_parity": parity,
        "capability_calibration_invalidated": True,
        "promotion_eligible": False,
    }
    output_payload = {
        "architecture": "structured-recurrent-motor-policy",
        "model": target.state_dict(),
        "config": asdict(target_config),
        "fast_feedback_verified": target.fast_feedback.verified,
        "report": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

"""Measure Q2's action intercept at the analytic physical equilibrium.

The diagnostic deliberately keeps the observation fixed at the exact
disturbance-dependent equilibrium and advances only the Q2 recurrent hidden
state.  Teacher actions are not executed.  Consequently, a nonzero
``teacher_action - analytic_motor_trim`` is an intrinsic recurrent/head
intercept mismatch rather than closed-loop state drift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import load_q2_policy  # noqa: E402
from env_l2f import L2FParams, L2FSimulator  # noqa: E402
from equilibrium_control import (  # noqa: E402
    analytic_equilibrium_target,
    materialize_equilibrium_state,
)
from policy_observation import (  # noqa: E402
    build_policy_observation,
    initial_observation_state,
)
from structured_distillation import (  # noqa: E402
    DAggerScenarioBank,
    build_dagger_scenario_bank,
)


DEFAULT_Q2 = (
    ROOT
    / "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt"
)
DEFAULT_REPORT = ROOT / "reports/q2_equilibrium_bias_seed1707.json"
DEFAULT_STEPS = (26, 51, 76, 251)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _move_bank(bank: DAggerScenarioBank, device: torch.device) -> DAggerScenarioBank:
    if device.type == "cpu":
        return bank
    state_type = type(bank.state)
    state = state_type(**{
        name: getattr(bank.state, name).to(device)
        for name in bank.state.__dataclass_fields__
    })
    return DAggerScenarioBank(
        state=state,
        tw_bin=bank.tw_bin.to(device),
        log_alpha_bin=bank.log_alpha_bin.to(device),
        stratum=bank.stratum,
    )


def _bank_hash(bank: DAggerScenarioBank) -> str:
    digest = hashlib.sha256()
    for name in sorted(bank.state.__dataclass_fields__):
        value = getattr(bank.state, name).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def measure_q2_equilibrium_bias(
    teacher: Any,
    simulator: L2FSimulator,
    bank: DAggerScenarioBank,
    *,
    steps: Sequence[int] = DEFAULT_STEPS,
) -> list[dict[str, float | int]]:
    """Return action-intercept statistics after fixed-input recurrent burn-in."""

    requested = tuple(int(step) for step in steps)
    if not requested or min(requested) < 1 or len(set(requested)) != len(requested):
        raise ValueError("steps must be distinct positive integers")
    target = analytic_equilibrium_target(
        bank.state, gravity=simulator.params.gravity
    )
    if not bool(target.feasible.all().item()):
        raise RuntimeError("diagnostic bank contains an infeasible analytic equilibrium")
    equilibrium_state = materialize_equilibrium_state(bank.state, target)
    observation_state = initial_observation_state(
        bank.count,
        device=equilibrium_state.position.device,
        dtype=equilibrium_state.position.dtype,
    )
    observation, _ = build_policy_observation(
        equilibrium_state,
        observation_state,
        mode="integral25",
        integral_input_frame="body",
        noise_max=0.0,
    )
    hidden = teacher.initial_hidden(
        bank.count,
        device=observation.device,
        dtype=observation.dtype,
    )
    wanted = set(requested)
    rows: list[dict[str, float | int]] = []
    with torch.no_grad():
        for step in range(1, max(requested) + 1):
            result = teacher.forward_with_aux(observation, hidden)
            if not isinstance(result, (tuple, list)) or len(result) < 2:
                raise TypeError("Q2 teacher must return at least (action, next_hidden)")
            action, hidden = result[0], result[1]
            if step not in wanted:
                continue
            error = action - target.motor_trim
            per_scenario_rms = error.square().mean(dim=-1).sqrt()
            rows.append({
                "step": step,
                "action_minus_trim_rms": float(error.square().mean().sqrt()),
                "per_scenario_rms_p50": float(torch.quantile(per_scenario_rms, 0.50)),
                "per_scenario_rms_p99": float(torch.quantile(per_scenario_rms, 0.99)),
                "action_minus_trim_max_abs": float(error.abs().max()),
            })
    rows.sort(key=lambda row: int(row["step"]))
    return rows


def build_report(
    teacher: Any,
    simulator: L2FSimulator,
    bank: DAggerScenarioBank,
    *,
    checkpoint: Path,
    seed: int,
    steps: Sequence[int] = DEFAULT_STEPS,
) -> dict[str, object]:
    rows = measure_q2_equilibrium_bias(
        teacher, simulator, bank, steps=steps
    )
    return {
        "diagnostic": "q2-analytic-equilibrium-action-intercept",
        "source_checkpoint": str(checkpoint.resolve()),
        "seed": int(seed),
        "scenario_count": bank.count,
        "authority_layout": "4x4 TW/log-alpha; 4 scenarios/cell",
        "dt": float(simulator.params.dt),
        "observation_mode": "integral25-body",
        "recurrent_protocol": (
            "fixed analytic equilibrium observation; advance Q2 hidden only; "
            "do not execute teacher action"
        ),
        "analytic_equilibrium_feasible_fraction": 1.0,
        "bank_sha256": _bank_hash(bank),
        "rows": rows,
        "interpretation": (
            "A structured controller whose feedback and O(e^2) residual vanish "
            "at physical e=0 intentionally cannot reproduce this Q2 intercept. "
            "Whole-action parity must therefore be evaluated away from e=0 and "
            "kept separate from the physical-equilibrium invariance gate."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_Q2)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--seed", type=int, default=1707)
    parser.add_argument("--scenario-count", type=int, default=64)
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scenario_count != 64:
        raise ValueError("the registered diagnostic requires exactly 64 scenarios")
    device = _device(args.device)
    teacher, teacher_args = load_q2_policy(
        args.source_checkpoint, device=device, dtype=torch.float32
    )
    teacher.eval()
    simulator = L2FSimulator(
        L2FParams(dt=float(teacher_args.get("dt", 0.01)))
    )
    bank_cpu = build_dagger_scenario_bank(
        args.scenario_count, seed=args.seed, dt=simulator.params.dt
    )
    bank = _move_bank(bank_cpu, device)
    report = build_report(
        teacher,
        simulator,
        bank,
        checkpoint=args.source_checkpoint,
        seed=args.seed,
        steps=args.steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

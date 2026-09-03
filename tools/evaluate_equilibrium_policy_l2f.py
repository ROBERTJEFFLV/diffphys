from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import load_q2_policy  # noqa: E402
from env_l2f import L2FParams, L2FSimulator, L2FState  # noqa: E402
from equilibrium_control import (  # noqa: E402
    EQUILIBRIUM_POLICY_ARCHITECTURE,
    EquilibriumCenteredPolicy,
)
from policy_observation import (  # noqa: E402
    build_policy_observation,
    initial_observation_state,
    update_position_integral,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate equilibrium-policy checkpoints with the native L2F simulator."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="LABEL=PATH; a Q2 checkpoint is converted to the new update-zero policy.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=17007)
    parser.add_argument("--scenarios", type=int, default=128)
    parser.add_argument("--horizons", default="500,2000,5000")
    parser.add_argument("--tail-steps", type=int, default=500)
    parser.add_argument("--cvar-fraction", type=float, default=0.20)
    parser.add_argument("--success-position", type=float, default=0.10)
    parser.add_argument("--success-velocity", type=float, default=0.10)
    parser.add_argument("--success-omega", type=float, default=0.50)
    return parser.parse_args()


def _parse_checkpoints(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError("each checkpoint must use LABEL=PATH")
        label, raw_path = value.split("=", 1)
        if not label or label in labels:
            raise ValueError("checkpoint labels must be non-empty and unique")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        labels.add(label)
        result.append((label, path))
    return result


def _load_policy(
    path: Path,
    *,
    device: torch.device,
) -> tuple[EquilibriumCenteredPolicy, str]:
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = payload.get("architecture", {}) if isinstance(payload, dict) else {}
    if metadata.get("architecture") == EQUILIBRIUM_POLICY_ARCHITECTURE:
        policy = EquilibriumCenteredPolicy(
            observation_dim=int(metadata.get("observation_dim", 25)),
            encoder_dim=int(metadata.get("encoder_dim", 192)),
            hidden_dim=int(metadata.get("hidden_dim", 192)),
            encoder_depth=int(metadata.get("encoder_depth", 2)),
        ).to(device=device, dtype=torch.float32)
        policy.load_state_dict(payload["model"], strict=True)
        return policy.eval(), "equilibrium-checkpoint"
    source, _ = load_q2_policy(path, device=device, dtype=torch.float32)
    return EquilibriumCenteredPolicy.from_motor_gru(source).eval(), "q2-converted-update-zero"


def _clone_state(state: L2FState) -> L2FState:
    return L2FState(
        **{
            field.name: getattr(state, field.name).detach().clone()
            for field in fields(L2FState)
        }
    )


def _cvar(value: torch.Tensor, fraction: float) -> float:
    count = max(1, int(math.ceil(value.numel() * fraction)))
    return float(torch.topk(value, count, largest=True).values.mean().item())


def _summarize(
    position: torch.Tensor,
    velocity: torch.Tensor,
    omega: torch.Tensor,
    *,
    horizon: int,
    tail_steps: int,
    cvar_fraction: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    window = min(horizon, tail_steps)
    p_scenario = position[horizon - window : horizon].mean(dim=0)
    v_scenario = velocity[horizon - window : horizon].mean(dim=0)
    w_scenario = omega[horizon - window : horizon].mean(dim=0)
    success = (
        (p_scenario < args.success_position)
        & (v_scenario < args.success_velocity)
        & (w_scenario < args.success_omega)
    )
    return {
        "horizon": horizon,
        "window_steps": window,
        "position_mean": float(p_scenario.mean().item()),
        "position_cvar": _cvar(p_scenario, cvar_fraction),
        "velocity_mean": float(v_scenario.mean().item()),
        "velocity_cvar": _cvar(v_scenario, cvar_fraction),
        "omega_mean": float(w_scenario.mean().item()),
        "omega_cvar": _cvar(w_scenario, cvar_fraction),
        "success_rate": float(success.float().mean().item()),
    }


def _rollout(
    policy: EquilibriumCenteredPolicy,
    simulator: L2FSimulator,
    initial: L2FState,
    *,
    horizons: list[int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    state = _clone_state(initial)
    hidden = policy.initial_hidden(
        state.position.shape[0],
        device=state.position.device,
        dtype=state.position.dtype,
    )
    observation_state = initial_observation_state(
        state.position.shape[0],
        device=state.position.device,
        dtype=state.position.dtype,
    )
    positions: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    omegas: list[torch.Tensor] = []
    with torch.inference_mode():
        for _ in range(max(horizons)):
            observation, observed_position = build_policy_observation(
                state,
                observation_state,
                mode="integral25",
                noise_max=0.0,
                integral_input_frame="body",
            )
            action, hidden = policy(observation, hidden)
            observation_state = update_position_integral(
                observation_state,
                observed_position,
                dt=simulator.params.dt,
                integral_limit=0.5,
                integral_leak=0.0,
            )
            state = simulator.step(state, action)
            positions.append(torch.linalg.vector_norm(state.position, dim=-1).cpu())
            velocities.append(torch.linalg.vector_norm(state.velocity, dim=-1).cpu())
            omegas.append(torch.linalg.vector_norm(state.omega, dim=-1).cpu())
    position = torch.stack(positions)
    velocity = torch.stack(velocities)
    omega = torch.stack(omegas)
    return [
        _summarize(
            position,
            velocity,
            omega,
            horizon=horizon,
            tail_steps=(100 if horizon == 500 else args.tail_steps),
            cvar_fraction=args.cvar_fraction,
            args=args,
        )
        for horizon in horizons
    ]


def main() -> None:
    args = _parse_args()
    checkpoints = _parse_checkpoints(args.checkpoint)
    horizons = sorted({int(value) for value in args.horizons.split(",") if value})
    if not horizons or horizons[0] < 1:
        raise ValueError("horizons must be positive")
    if not 0.0 < args.cvar_fraction <= 1.0:
        raise ValueError("cvar-fraction must be in (0,1]")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    simulator = L2FSimulator(L2FParams(dt=0.01))
    initial = simulator.reset(
        args.scenarios,
        device=device,
        dtype=torch.float32,
        sample_dynamics=True,
        sampled_dynamics_level="broad",
        broad_sampler="physical-fit",
        balanced_dynamics_sampling=False,
        sample_external_force=True,
    )
    rows: dict[str, Any] = {}
    for label, path in checkpoints:
        policy, load_kind = _load_policy(path, device=device)
        rows[label] = {
            "checkpoint": str(path),
            "load_kind": load_kind,
            "metrics": _rollout(
                policy,
                simulator,
                initial,
                horizons=horizons,
                args=args,
            ),
        }
    payload = {
        "simulator": "native differentiable L2F torch simulator",
        "seed": args.seed,
        "scenarios": args.scenarios,
        "horizons": horizons,
        "paired_initial_states": True,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

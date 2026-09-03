from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import clone_state, load_q2_policy  # noqa: E402
from diagnostics.scenarios import load_matlab_scenarios  # noqa: E402
from env_l2f import (  # noqa: E402
    L2FLossConfig,
    L2FParams,
    L2FSimulator,
    L2FState,
    apply_gradient_decay,
)
from l2f_cuda_backend import cuda_step, load_extension  # noqa: E402
from policy_observation import (  # noqa: E402
    PolicyObservationState,
    build_policy_observation,
    initial_observation_state,
    update_position_integral,
)
from temporal_decay import (  # noqa: E402
    current_decay_equivalent_alpha,
    resolve_step_gradient_decay,
)


DYNAMIC_FIELDS = (
    "position",
    "velocity",
    "rotation",
    "omega",
    "motor",
    "previous_action",
)
PRIMARY_CURVES = (
    "state_chain_grad",
    "hidden_chain_grad",
    "action_grad",
    "policy_local_grad",
)
THRESHOLDS = (1.0e-2, 1.0e-3, 1.0e-4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure terminal-loss temporal credit under current and NMI decay."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt",
    )
    parser.add_argument(
        "--scenario-csv",
        type=Path,
        default=ROOT / "diagnostic_inputs/h10000_paired_96m_20260804/manifests/scenario_reset_exact.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/temporal_decay_minimal/backward",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("cuda", "torch"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tbptt-segment", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-policy-vjp", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_fixed_scenarios(
    path: Path,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[int], L2FState]:
    scenario_ids, full_state = load_matlab_scenarios(path, device=device, dtype=torch.float32)
    if batch_size <= 0 or batch_size > len(scenario_ids):
        raise ValueError("batch size must be in [1, scenario count]")
    indices = torch.linspace(
        0,
        len(scenario_ids) - 1,
        steps=batch_size,
        device=device,
        dtype=torch.float64,
    ).round().to(dtype=torch.long)
    selected = L2FState(
        **{
            field.name: getattr(full_state, field.name).index_select(0, indices)
            for field in fields(L2FState)
        }
    )
    cpu_indices = [int(value) for value in indices.cpu().tolist()]
    return [scenario_ids[index] for index in cpu_indices], cpu_indices, selected


def _clone_for_gradient(state: L2FState) -> L2FState:
    values: dict[str, torch.Tensor] = {}
    for field in fields(L2FState):
        value = getattr(state, field.name).detach().clone()
        if field.name in DYNAMIC_FIELDS:
            value.requires_grad_(True)
        values[field.name] = value
    return L2FState(**values)


def _detach_dynamic_state(state: L2FState) -> L2FState:
    values: dict[str, torch.Tensor] = {}
    for field in fields(L2FState):
        value = getattr(state, field.name)
        if field.name in DYNAMIC_FIELDS:
            value = value.detach().requires_grad_(True)
        values[field.name] = value
    return L2FState(**values)


def _retain_state(state: L2FState) -> None:
    for name in DYNAMIC_FIELDS:
        value = getattr(state, name)
        if value.requires_grad:
            value.retain_grad()


def _grad_norm(tensor: torch.Tensor) -> float:
    grad = tensor.grad
    if grad is None:
        return 0.0
    return float(torch.linalg.vector_norm(grad.detach().double()).item())


def _state_grad_norm(state: L2FState) -> float:
    square = 0.0
    for name in DYNAMIC_FIELDS:
        value = _grad_norm(getattr(state, name))
        square += value * value
    return math.sqrt(square)


def _parameter_grad_norm(policy: torch.nn.Module) -> float:
    total = torch.zeros((), device=next(policy.parameters()).device, dtype=torch.float64)
    for parameter in policy.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().double().square().sum()
    return float(torch.sqrt(total).item())


def _local_policy_vjp_norm(
    policy: torch.nn.Module,
    observation: torch.Tensor,
    hidden_input: torch.Tensor,
    action_grad: torch.Tensor | None,
    hidden_output_grad: torch.Tensor | None,
) -> float:
    if action_grad is None and hidden_output_grad is None:
        return 0.0
    replay_action, replay_hidden = policy(observation, hidden_input)
    action_vjp = torch.zeros_like(replay_action) if action_grad is None else action_grad.detach()
    hidden_vjp = (
        torch.zeros_like(replay_hidden)
        if hidden_output_grad is None
        else hidden_output_grad.detach()
    )
    parameters = tuple(policy.parameters())
    gradients = torch.autograd.grad(
        (replay_action, replay_hidden),
        parameters,
        grad_outputs=(action_vjp, hidden_vjp),
        allow_unused=True,
    )
    total = torch.zeros((), device=observation.device, dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            total += gradient.detach().double().square().sum()
    return float(torch.sqrt(total).item())


def _normalized(values: list[float]) -> list[float]:
    reference = values[0]
    if not math.isfinite(reference) or reference <= 0.0:
        return [float("nan") for _ in values]
    return [value / reference for value in values]


def _last_lag_at_or_above(values: list[float], threshold: float) -> int:
    valid = [lag for lag, value in enumerate(values, start=1) if math.isfinite(value) and value >= threshold]
    return max(valid) if valid else 0


def _first_lag_below(values: list[float], threshold: float) -> int:
    for lag, value in enumerate(values, start=1):
        if math.isfinite(value) and value < threshold:
            return lag
    return 0


def _theoretical_horizon(step_decay: float, threshold: float, horizon: int) -> int:
    if step_decay >= 1.0:
        return horizon
    return min(horizon, max(0, int(math.floor(math.log(threshold) / math.log(step_decay)))))


def _run_variant(
    *,
    label: str,
    mode: str,
    state_step_decay: float,
    hidden_step_decay: float,
    policy: torch.nn.Module,
    checkpoint_args: dict[str, Any],
    initial_state: L2FState,
    horizon: int,
    dt: float,
    backend: str,
    tbptt_segment: int | None,
    skip_policy_vjp: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    policy.zero_grad(set_to_none=True)
    state = _clone_for_gradient(initial_state)
    sim = L2FSimulator(L2FParams(dt=dt))
    loss_config = L2FLossConfig()
    batch = state.position.shape[0]
    hidden = policy.initial_hidden(
        batch, device=state.position.device, dtype=state.position.dtype
    ).requires_grad_(True)
    observation_state = initial_observation_state(
        batch, device=state.position.device, dtype=state.position.dtype
    )
    state_nodes: list[L2FState] = []
    hidden_inputs: list[torch.Tensor] = []
    raw_hidden_outputs: list[torch.Tensor] = []
    action_nodes: list[torch.Tensor] = []
    replay_observations: list[torch.Tensor] = []
    replay_hidden_inputs: list[torch.Tensor] = []

    integral_frame = str(checkpoint_args.get("integral_input_frame", "body"))
    integral_multiplier = float(checkpoint_args.get("integral_input_multiplier", 1.0))
    integral_limit = float(checkpoint_args.get("integral_limit", 0.5))
    integral_leak = float(checkpoint_args.get("integral_leak", 0.0))

    for step in range(horizon):
        if tbptt_segment and step > 0 and step % tbptt_segment == 0:
            state = _detach_dynamic_state(state)
            hidden = hidden.detach().requires_grad_(True)
            observation_state = PolicyObservationState(
                integral_position=observation_state.integral_position.detach()
            )
        _retain_state(state)
        hidden.retain_grad()
        state_nodes.append(state)
        hidden_inputs.append(hidden)
        observation, observed_position = build_policy_observation(
            state,
            observation_state,
            mode="integral25",
            noise_max=0.0,
            integral_input_frame=integral_frame,
            integral_input_multiplier=integral_multiplier,
        )
        replay_observations.append(observation.detach())
        replay_hidden_inputs.append(hidden.detach())
        action, raw_hidden = policy(observation, hidden)
        action.retain_grad()
        raw_hidden.retain_grad()
        action_nodes.append(action)
        raw_hidden_outputs.append(raw_hidden)
        hidden = apply_gradient_decay(raw_hidden, hidden_step_decay)
        observation_state = update_position_integral(
            observation_state,
            observed_position,
            dt=dt,
            integral_limit=integral_limit,
            integral_leak=integral_leak,
        )
        if backend == "cuda":
            state = cuda_step(state, action, sim.params, grad_decay=state_step_decay)
        else:
            state = sim.step(state, action, grad_decay=state_step_decay)

    terminal_components = sim.tracking_components(state, loss_config)
    terminal_loss = torch.stack(tuple(terminal_components.values()), dim=0).sum(dim=0).mean()
    terminal_loss.backward()
    total_parameter_grad = _parameter_grad_norm(policy)

    chronological: list[dict[str, Any]] = []
    for time_index in range(horizon):
        state_node = state_nodes[time_index]
        row: dict[str, Any] = {
            "variant": label,
            "graph_mode": mode,
            "time_index": time_index,
            "lag": horizon - time_index,
            "state_chain_grad": _state_grad_norm(state_node),
            "position_grad": _grad_norm(state_node.position),
            "velocity_grad": _grad_norm(state_node.velocity),
            "rotation_grad": _grad_norm(state_node.rotation),
            "omega_grad": _grad_norm(state_node.omega),
            "motor_grad": _grad_norm(state_node.motor),
            "previous_action_grad": _grad_norm(state_node.previous_action),
            "hidden_chain_grad": _grad_norm(hidden_inputs[time_index]),
            "action_grad": _grad_norm(action_nodes[time_index]),
        }
        if skip_policy_vjp:
            row["policy_local_grad"] = float("nan")
        else:
            row["policy_local_grad"] = _local_policy_vjp_norm(
                policy,
                replay_observations[time_index],
                replay_hidden_inputs[time_index],
                action_nodes[time_index].grad,
                raw_hidden_outputs[time_index].grad,
            )
        chronological.append(row)

    rows = sorted(chronological, key=lambda row: int(row["lag"]))
    curve_names = (
        "state_chain_grad",
        "position_grad",
        "velocity_grad",
        "rotation_grad",
        "omega_grad",
        "motor_grad",
        "previous_action_grad",
        "hidden_chain_grad",
        "action_grad",
        "policy_local_grad",
    )
    normalized_by_curve = {
        name: _normalized([float(row[name]) for row in rows]) for name in curve_names
    }
    for index, row in enumerate(rows):
        for name in curve_names:
            row[f"normalized_{name}"] = normalized_by_curve[name][index]

    summary_rows: list[dict[str, Any]] = []
    for name in PRIMARY_CURVES:
        raw = [float(row[name]) for row in rows]
        normalized = normalized_by_curve[name]
        finite = all(math.isfinite(value) for value in raw)
        finite_normalized = [value for value in normalized if math.isfinite(value)]
        base: dict[str, Any] = {
            "variant": label,
            "graph_mode": mode,
            "curve": name,
            "lag1_gradient": raw[0],
            "max_gradient": max(raw),
            "max_normalized": max(finite_normalized) if finite_normalized else float("nan"),
            "peak_lag": 1 + max(range(len(raw)), key=lambda index: raw[index]),
            "all_finite": int(finite),
            "terminal_loss": float(terminal_loss.detach().item()),
            "total_policy_parameter_grad": total_parameter_grad,
            "state_step_decay": state_step_decay,
            "hidden_step_decay": hidden_step_decay,
        }
        for threshold in THRESHOLDS:
            key = f"{threshold:g}"
            base[f"last_lag_at_or_above_{key}"] = _last_lag_at_or_above(
                normalized, threshold
            )
            base[f"first_lag_below_{key}"] = _first_lag_below(
                normalized, threshold
            )
        summary_rows.append(base)

    elapsed = time.perf_counter() - started
    metadata = {
        "variant": label,
        "graph_mode": mode,
        "terminal_loss": float(terminal_loss.detach().item()),
        "terminal_position": float(terminal_components["position"].mean().detach().item()),
        "terminal_velocity": float(terminal_components["velocity"].mean().detach().item()),
        "terminal_omega": float(terminal_components["omega"].mean().detach().item()),
        "total_policy_parameter_grad": total_parameter_grad,
        "state_step_decay": state_step_decay,
        "hidden_step_decay": hidden_step_decay,
        "runtime_seconds": elapsed,
        "all_curve_values_finite": int(
            all(
                math.isfinite(float(row[name]))
                for row in rows
                for name in PRIMARY_CURVES
            )
        ),
    }
    return rows, summary_rows, metadata


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    variants = []
    for row in rows:
        key = (str(row["variant"]), str(row["graph_mode"]))
        if key not in variants:
            variants.append(key)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, curve in zip(axes.flat, PRIMARY_CURVES):
        for variant, graph_mode in variants:
            selected = [
                row
                for row in rows
                if row["variant"] == variant and row["graph_mode"] == graph_mode
            ]
            axis.plot(
                [int(row["lag"]) for row in selected],
                [max(float(row[f"normalized_{curve}"]), 1.0e-30) for row in selected],
                label=f"{variant}:{graph_mode}",
            )
        for threshold in THRESHOLDS:
            axis.axhline(threshold, color="grey", linewidth=0.6, linestyle="--")
        axis.set_yscale("log")
        axis.set_title(curve)
        axis.grid(True, which="both", alpha=0.2)
    axes[1, 0].set_xlabel("lag (steps)")
    axes[1, 1].set_xlabel("lag (steps)")
    axes[0, 0].set_ylabel("gradient / lag-1 gradient")
    axes[1, 0].set_ylabel("gradient / lag-1 gradient")
    axes[0, 0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.horizon != 500:
        raise ValueError("the preregistered minimal diagnostic uses horizon=500")
    if args.tbptt_segment <= 0 or args.horizon % args.tbptt_segment != 0:
        raise ValueError("TBPTT segment must be positive and divide the horizon")
    device = torch.device(args.device)
    if args.backend == "cuda":
        if device.type != "cuda":
            raise ValueError("CUDA backend requires a CUDA device")
        load_extension()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    policy, checkpoint_args = load_q2_policy(
        args.checkpoint, device=device, dtype=torch.float32
    )
    scenario_ids, scenario_indices, initial_state = _select_fixed_scenarios(
        args.scenario_csv, batch_size=args.batch_size, device=device
    )
    current_state_alpha = current_decay_equivalent_alpha(0.5)
    current_hidden_alpha = current_decay_equivalent_alpha(0.7)
    variants = (
        (
            "A_current",
            resolve_step_gradient_decay(
                mode="current", dt=args.dt, current_base=0.5, alpha=None
            ),
            resolve_step_gradient_decay(
                mode="current", dt=args.dt, current_base=0.7, alpha=None
            ),
            current_state_alpha,
            current_hidden_alpha,
        ),
        (
            "B_alpha4",
            resolve_step_gradient_decay(
                mode="nmi", dt=args.dt, current_base=0.5, alpha=4.0
            ),
            resolve_step_gradient_decay(
                mode="nmi", dt=args.dt, current_base=0.7, alpha=4.0
            ),
            4.0,
            4.0,
        ),
        (
            "C_alpha1",
            resolve_step_gradient_decay(
                mode="nmi", dt=args.dt, current_base=0.5, alpha=1.0
            ),
            resolve_step_gradient_decay(
                mode="nmi", dt=args.dt, current_base=0.7, alpha=1.0
            ),
            1.0,
            1.0,
        ),
        (
            "D_alpha0.25",
            resolve_step_gradient_decay(
                mode="nmi", dt=args.dt, current_base=0.5, alpha=0.25
            ),
            resolve_step_gradient_decay(
                mode="nmi", dt=args.dt, current_base=0.7, alpha=0.25
            ),
            0.25,
            0.25,
        ),
    )

    all_rows: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    run_metadata: list[dict[str, Any]] = []
    for label, state_decay, hidden_decay, state_alpha, hidden_alpha in variants:
        rows, summaries, metadata = _run_variant(
            label=label,
            mode="continuous_h500",
            state_step_decay=state_decay,
            hidden_step_decay=hidden_decay,
            policy=policy,
            checkpoint_args=checkpoint_args,
            initial_state=initial_state,
            horizon=args.horizon,
            dt=args.dt,
            backend=args.backend,
            tbptt_segment=None,
            skip_policy_vjp=args.skip_policy_vjp,
        )
        for row in rows:
            row["state_alpha"] = state_alpha
            row["hidden_alpha"] = hidden_alpha
        for row in summaries:
            row["state_alpha"] = state_alpha
            row["hidden_alpha"] = hidden_alpha
            for threshold in THRESHOLDS:
                key = f"{threshold:g}"
                row[f"theoretical_state_horizon_{key}"] = _theoretical_horizon(
                    state_decay, threshold, args.horizon
                )
                row[f"theoretical_hidden_horizon_{key}"] = _theoretical_horizon(
                    hidden_decay, threshold, args.horizon
                )
        metadata["state_alpha"] = state_alpha
        metadata["hidden_alpha"] = hidden_alpha
        all_rows.extend(rows)
        all_summary.extend(summaries)
        run_metadata.append(metadata)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline = variants[0]
    rows, summaries, metadata = _run_variant(
        label="A_current",
        mode=f"training_tbptt{args.tbptt_segment}",
        state_step_decay=baseline[1],
        hidden_step_decay=baseline[2],
        policy=policy,
        checkpoint_args=checkpoint_args,
        initial_state=initial_state,
        horizon=args.horizon,
        dt=args.dt,
        backend=args.backend,
        tbptt_segment=args.tbptt_segment,
        skip_policy_vjp=args.skip_policy_vjp,
    )
    for row in rows:
        row["state_alpha"] = baseline[3]
        row["hidden_alpha"] = baseline[4]
    for row in summaries:
        row["state_alpha"] = baseline[3]
        row["hidden_alpha"] = baseline[4]
        for threshold in THRESHOLDS:
            key = f"{threshold:g}"
            row[f"theoretical_state_horizon_{key}"] = _theoretical_horizon(
                baseline[1], threshold, args.horizon
            )
            row[f"theoretical_hidden_horizon_{key}"] = _theoretical_horizon(
                baseline[2], threshold, args.horizon
            )
    metadata["state_alpha"] = baseline[3]
    metadata["hidden_alpha"] = baseline[4]
    all_rows.extend(rows)
    all_summary.extend(summaries)
    run_metadata.append(metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "gradient_curves.csv", all_rows)
    _write_csv(args.output_dir / "credit_horizons.csv", all_summary)
    _plot_curves(args.output_dir / "gradient_curves.png", all_rows)
    metadata_payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "scenario_csv": str(args.scenario_csv.resolve()),
        "scenario_csv_sha256": _sha256(args.scenario_csv),
        "scenario_ids": scenario_ids,
        "scenario_indices": scenario_indices,
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "dt": args.dt,
        "backend": args.backend,
        "seed": args.seed,
        "current_semantics": {
            "state_expression": "0.5**dt",
            "hidden_expression": "0.7**dt",
            "state_equivalent_alpha": current_state_alpha,
            "hidden_equivalent_alpha": current_hidden_alpha,
        },
        "runs": run_metadata,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output_dir / 'gradient_curves.csv'}")
    print(f"wrote {args.output_dir / 'credit_horizons.csv'}")


if __name__ == "__main__":
    main()

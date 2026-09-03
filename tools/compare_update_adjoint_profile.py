from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_l2f import L2FLossConfig, L2FParams, L2FSimulator, L2FState  # noqa: E402
from l2f_full_cuda_backend import (  # noqa: E402
    METRIC_NAMES,
    full_cuda_rollout_metrics,
)
from model import MotorGRUPolicy  # noqa: E402


PARAM_GROUPS = {
    "encoder": ("encoder.",),
    "gru": ("gru.",),
    "head": ("motor_head.",),
}

DEBUG_NAMES = (
    "actions",
    "p_states",
    "v_states",
    "r_states",
    "w_states",
    "motor_states",
    "previous_action_states",
    "potentials",
    "hidden_adj_before",
    "hidden_adj_after",
    "lp_adj",
    "lv_adj",
    "lR_adj",
    "lw_adj",
    "lm_adj",
    "lpa_adj",
    "action_adj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cuda-full adjoint profiles before and after one candidate optimizer update."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--tail-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--state-grad-decay", type=float, default=None)
    parser.add_argument("--hidden-grad-decay", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--sample-dynamics", action="store_true", default=None)
    parser.add_argument("--fixed-dynamics", action="store_false", dest="sample_dynamics")
    parser.add_argument("--sampled-dynamics-level", default=None, choices=("small", "medium", "broad"))
    parser.add_argument("--broad-sampler", default=None, choices=("legacy", "physical", "physical-fit"))
    parser.add_argument("--init-optimizer-state", action="store_true")
    parser.add_argument("--terminal-loss-only", action="store_true")
    parser.add_argument("--noise-seed", type=int, default=0)
    return parser.parse_args()


def _ckpt_get(args_dict: dict[str, Any], key: str, default: Any) -> Any:
    return args_dict[key] if key in args_dict and args_dict[key] is not None else default


def _option(cli_value: Any, args_dict: dict[str, Any], key: str, default: Any) -> Any:
    return cli_value if cli_value is not None else _ckpt_get(args_dict, key, default)


def clone_state(state: L2FState) -> L2FState:
    values = {}
    for field in fields(L2FState):
        values[field.name] = getattr(state, field.name).detach().clone()
    return L2FState(**values)


def grad_norm_fp64(policy: MotorGRUPolicy) -> float:
    device = next(policy.parameters()).device
    total = torch.zeros((), device=device, dtype=torch.float64)
    for param in policy.parameters():
        if param.grad is not None:
            total = total + param.grad.detach().double().square().sum()
    return float(torch.sqrt(total).item())


def apply_fp64_global_clip(policy: MotorGRUPolicy, grad_clip: float) -> tuple[float, float, float]:
    before = grad_norm_fp64(policy)
    scale = 1.0
    if grad_clip > 0.0 and math.isfinite(before):
        scale = min(1.0, grad_clip / (before + 1.0e-12))
        for param in policy.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)
    after = grad_norm_fp64(policy)
    return before, after, scale


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def set_optimizer_hparams(optimizer: torch.optim.Optimizer, *, lr: float, weight_decay: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
        group["weight_decay"] = weight_decay


def param_snapshot(policy: MotorGRUPolicy) -> list[torch.Tensor]:
    return [param.detach().clone() for param in policy.parameters()]


def max_abs_param_delta(policy: MotorGRUPolicy, before: list[torch.Tensor]) -> float:
    values = [(param.detach() - old).abs().amax() for param, old in zip(policy.parameters(), before)]
    return float(torch.stack(values).amax().item()) if values else 0.0


def named_grad_norms(policy: MotorGRUPolicy) -> dict[str, float]:
    device = next(policy.parameters()).device
    groups = {name: torch.zeros((), device=device, dtype=torch.float64) for name in PARAM_GROUPS}
    total = torch.zeros((), device=device, dtype=torch.float64)
    max_abs = 0.0
    for name, param in policy.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total = total + grad.double().square().sum()
        max_abs = max(max_abs, float(grad.abs().max().item()))
        for group, prefixes in PARAM_GROUPS.items():
            if name.startswith(prefixes):
                groups[group] = groups[group] + grad.double().square().sum()
    out = {f"{group}_grad_norm": float(torch.sqrt(value).item()) for group, value in groups.items()}
    out["total_grad_norm"] = float(torch.sqrt(total).item())
    out["max_abs_grad"] = max_abs
    return out


def _safe_max_abs(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    value = torch.nan_to_num(
        tensor.detach().abs(),
        nan=float("inf"),
        posinf=float("inf"),
        neginf=float("inf"),
    ).max()
    return float(value.item())


def _safe_l2(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    value = torch.nan_to_num(
        tensor.detach(),
        nan=float("inf"),
        posinf=float("inf"),
        neginf=-float("inf"),
    )
    return float(torch.linalg.norm(value.double()).item())


def _max_over_steps(rows: list[dict[str, float | int | str]], column: str, end_step: int) -> float:
    values = [
        float(row[column])
        for row in rows
        if isinstance(row.get("rollout_step"), int) and int(row["rollout_step"]) <= end_step
    ]
    return max(values) if values else float("nan")


def run_profile(
    *,
    policy: MotorGRUPolicy,
    sim: L2FSimulator,
    initial_state: L2FState,
    loss_config: L2FLossConfig,
    horizon: int,
    tail_steps: int,
    state_step_decay: float,
    hidden_step_decay: float,
    clf_kappa: float,
    u_soft: float,
    lambda_clf: float,
    lambda_out: float,
    lambda_tail: float,
    lambda_du: float,
    lambda_ddu: float,
    lambda_sat: float,
    noise_seed: int,
    external_torque_max: float,
    action_noise_max: float,
    observation_noise_max: float,
    terminal_loss_only: bool,
    phase: str,
) -> tuple[torch.Tensor, dict[str, float | int | str], list[dict[str, float | int | str]]]:
    policy.train()
    policy.zero_grad(set_to_none=True)
    outputs = full_cuda_rollout_metrics(
        policy,
        clone_state(initial_state),
        sim.params,
        loss_config,
        horizon=horizon,
        tail_steps=tail_steps,
        state_step_decay=state_step_decay,
        hidden_step_decay=hidden_step_decay,
        clf_kappa=clf_kappa,
        u_soft=u_soft,
        lambda_clf=lambda_clf,
        lambda_out=lambda_out,
        lambda_tail=lambda_tail,
        lambda_du=lambda_du,
        lambda_ddu=lambda_ddu,
        lambda_sat=lambda_sat,
        noise_seed=noise_seed,
        external_torque_max=external_torque_max,
        action_noise_max=action_noise_max,
        observation_noise_max=observation_noise_max,
        terminal_loss_only=terminal_loss_only,
        collect_debug=True,
    )
    metrics_tensor = outputs[0]
    loss = metrics_tensor[0]
    loss.backward()

    debug = dict(zip(DEBUG_NAMES, outputs[7:24]))
    grad_stats = named_grad_norms(policy)
    summary: dict[str, float | int | str] = {
        "phase": phase,
        "loss": float(metrics_tensor[0].detach().item()),
        "tracking": float(metrics_tensor[1].detach().item()),
        "position": float(metrics_tensor[2].detach().item()),
        "velocity": float(metrics_tensor[3].detach().item()),
        METRIC_NAMES[4]: float(metrics_tensor[4].detach().item()),
        "omega": float(metrics_tensor[5].detach().item()),
        "clf": float(metrics_tensor[6].detach().item()),
        "outward": float(metrics_tensor[7].detach().item()),
        "tail": float(metrics_tensor[8].detach().item()),
        "du": float(metrics_tensor[9].detach().item()),
        "ddu": float(metrics_tensor[10].detach().item()),
        "sat": float(metrics_tensor[11].detach().item()),
        **grad_stats,
        "final_position_max_abs": _safe_max_abs(outputs[1]),
        "final_velocity_max_abs": _safe_max_abs(outputs[2]),
        "final_rotation_max_abs": _safe_max_abs(outputs[3]),
        "final_omega_max_abs": _safe_max_abs(outputs[4]),
        "final_motor_max_abs": _safe_max_abs(outputs[5]),
        "final_previous_action_max_abs": _safe_max_abs(outputs[6]),
    }

    rows: list[dict[str, float | int | str]] = []
    for rollout_step in range(1, horizon + 1):
        action_step = rollout_step - 1
        row: dict[str, float | int | str] = {
            "phase": phase,
            "rollout_step": rollout_step,
            "loss": summary["loss"],
            "potential_max_abs": _safe_max_abs(debug["potentials"][rollout_step]),
            "state_p_max_abs": _safe_max_abs(debug["p_states"][rollout_step]),
            "state_v_max_abs": _safe_max_abs(debug["v_states"][rollout_step]),
            "state_R_max_abs": _safe_max_abs(debug["r_states"][rollout_step]),
            "state_w_max_abs": _safe_max_abs(debug["w_states"][rollout_step]),
            "state_motor_max_abs": _safe_max_abs(debug["motor_states"][rollout_step]),
            "action_max_abs": _safe_max_abs(debug["actions"][action_step]),
            "max_lp_adj": _safe_max_abs(debug["lp_adj"][rollout_step]),
            "max_lv_adj": _safe_max_abs(debug["lv_adj"][rollout_step]),
            "max_lR_adj": _safe_max_abs(debug["lR_adj"][rollout_step]),
            "max_lw_adj": _safe_max_abs(debug["lw_adj"][rollout_step]),
            "max_lm_adj": _safe_max_abs(debug["lm_adj"][rollout_step]),
            "max_lpa_adj": _safe_max_abs(debug["lpa_adj"][rollout_step]),
            "max_action_adj": _safe_max_abs(debug["action_adj"][action_step]),
            "max_hidden_adj_before": _safe_max_abs(debug["hidden_adj_before"][action_step]),
            "max_hidden_adj_after": _safe_max_abs(debug["hidden_adj_after"][action_step]),
            "l2_lp_adj": _safe_l2(debug["lp_adj"][rollout_step]),
            "l2_lv_adj": _safe_l2(debug["lv_adj"][rollout_step]),
            "l2_lR_adj": _safe_l2(debug["lR_adj"][rollout_step]),
            "l2_lw_adj": _safe_l2(debug["lw_adj"][rollout_step]),
            "l2_lm_adj": _safe_l2(debug["lm_adj"][rollout_step]),
            "l2_lpa_adj": _safe_l2(debug["lpa_adj"][rollout_step]),
            "l2_action_adj": _safe_l2(debug["action_adj"][action_step]),
            "l2_hidden_adj_before": _safe_l2(debug["hidden_adj_before"][action_step]),
            "l2_hidden_adj_after": _safe_l2(debug["hidden_adj_after"][action_step]),
        }
        rows.append(row)

    for column in (
        "max_action_adj",
        "max_hidden_adj_after",
        "max_lp_adj",
        "max_lw_adj",
        "max_lm_adj",
        "max_lpa_adj",
    ):
        summary[f"early8_{column}"] = _max_over_steps(rows, column, min(8, horizon))
        summary[f"early32_{column}"] = _max_over_steps(rows, column, min(32, horizon))
        summary[f"early64_{column}"] = _max_over_steps(rows, column, min(64, horizon))
        summary[f"all_{column}"] = _max_over_steps(rows, column, horizon)

    return loss, summary, rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ratio(after: float, before: float) -> float:
    if not math.isfinite(after) or not math.isfinite(before):
        return float("nan")
    return after / max(abs(before), 1.0e-30)


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("cuda-full update adjoint comparison requires CUDA")
    device = torch.device("cuda")
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = vars(checkpoint_args)

    seed = int(_option(args.seed, checkpoint_args, "seed", 7))
    horizon = int(_option(args.horizon, checkpoint_args, "horizon", 500))
    batch_size = int(_option(args.batch_size, checkpoint_args, "batch_size", 256))
    tail_steps = int(_option(args.tail_steps, checkpoint_args, "tail_steps", 50))
    lr = float(_option(args.lr, checkpoint_args, "lr", 3.0e-4))
    weight_decay = float(_option(args.weight_decay, checkpoint_args, "weight_decay", 1.0e-5))
    grad_clip = float(_option(args.grad_clip, checkpoint_args, "grad_clip", 10.0))
    dt = float(_option(args.dt, checkpoint_args, "dt", 0.01))
    state_grad_decay = float(_option(args.state_grad_decay, checkpoint_args, "state_grad_decay", 0.5))
    hidden_grad_decay = float(_option(args.hidden_grad_decay, checkpoint_args, "hidden_grad_decay", 0.7))
    sample_dynamics = bool(_option(args.sample_dynamics, checkpoint_args, "sample_dynamics", False))
    sampled_dynamics_level = str(_option(args.sampled_dynamics_level, checkpoint_args, "sampled_dynamics_level", "small"))
    broad_sampler = str(_option(args.broad_sampler, checkpoint_args, "broad_sampler", "legacy"))

    params = L2FParams(
        dt=dt,
        max_initial_position=float(_ckpt_get(checkpoint_args, "max_initial_position", 1.0)),
        max_initial_velocity=float(_ckpt_get(checkpoint_args, "max_initial_velocity", 0.6)),
        max_initial_angle=float(_ckpt_get(checkpoint_args, "max_initial_angle", 0.45)),
        max_initial_omega=float(_ckpt_get(checkpoint_args, "max_initial_omega", 1.0)),
        disturbance_force_max=float(_ckpt_get(checkpoint_args, "disturbance_force_max", 0.0)),
        external_force_ratio=float(_ckpt_get(checkpoint_args, "external_force_ratio", 0.0)),
    )
    loss_config = L2FLossConfig(
        p_scale=float(_ckpt_get(checkpoint_args, "p_scale", 2.0)),
        v_scale=float(_ckpt_get(checkpoint_args, "v_scale", 3.0)),
        omega_scale=float(_ckpt_get(checkpoint_args, "omega_scale", 10.0)),
        huber_beta=float(_ckpt_get(checkpoint_args, "huber_beta", 1.0)),
        w_p=float(_ckpt_get(checkpoint_args, "w_p", 1.0)),
        w_v=float(_ckpt_get(checkpoint_args, "w_v", 0.3)),
        w_omega=float(_ckpt_get(checkpoint_args, "w_omega", 0.05)),
    )
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    source_dim = int(state_dict["encoder.0.weight"].shape[1])
    if source_dim != 40:
        raise RuntimeError(
            "cuda-full profiling only supports legacy40 observations; "
            f"checkpoint uses {source_dim}D"
        )
    policy = MotorGRUPolicy(
        observation_dim=40,
        encoder_dim=int(_ckpt_get(checkpoint_args, "encoder_dim", 192)),
        hidden_dim=int(_ckpt_get(checkpoint_args, "hidden_dim", 192)),
        encoder_depth=int(_ckpt_get(checkpoint_args, "encoder_depth", 2)),
    ).to(device)
    missing, _ = policy.load_compatible_state_dict(state_dict)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    if args.init_optimizer_state and missing:
        raise ValueError("cannot restore an action-only optimizer after adding auxiliary heads")
    if args.init_optimizer_state and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_to_device(optimizer, device)
    set_optimizer_hparams(optimizer, lr=lr, weight_decay=weight_decay)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sim = L2FSimulator(params)
    initial_state = sim.reset(
        batch_size,
        device=device,
        sample_dynamics=sample_dynamics,
        sampled_dynamics_level=sampled_dynamics_level,
        broad_sampler=broad_sampler,
    )

    common = {
        "sim": sim,
        "initial_state": initial_state,
        "loss_config": loss_config,
        "horizon": horizon,
        "tail_steps": tail_steps,
        "state_step_decay": state_grad_decay ** dt,
        "hidden_step_decay": hidden_grad_decay ** dt,
        "clf_kappa": float(_ckpt_get(checkpoint_args, "clf_kappa", 1.0)),
        "u_soft": float(_ckpt_get(checkpoint_args, "u_soft", 0.9)),
        "lambda_clf": float(_ckpt_get(checkpoint_args, "lambda_clf", 0.5)),
        "lambda_out": float(_ckpt_get(checkpoint_args, "lambda_out", 0.1)),
        "lambda_tail": float(_ckpt_get(checkpoint_args, "lambda_tail", 1.0)),
        "lambda_du": float(_ckpt_get(checkpoint_args, "lambda_du", 3.0e-3)),
        "lambda_ddu": float(_ckpt_get(checkpoint_args, "lambda_ddu", 3.0e-4)),
        "lambda_sat": float(_ckpt_get(checkpoint_args, "lambda_sat", 0.03)),
        "noise_seed": int(args.noise_seed),
        "external_torque_max": float(_ckpt_get(checkpoint_args, "external_torque_max", 0.0)),
        "action_noise_max": float(_ckpt_get(checkpoint_args, "action_noise_max", 0.0)),
        "observation_noise_max": float(_ckpt_get(checkpoint_args, "observation_noise_max", 0.0)),
        "terminal_loss_only": bool(args.terminal_loss_only),
    }

    before_loss, before_summary, before_rows = run_profile(policy=policy, phase="before", **common)
    del before_loss
    grad_before_clip, grad_after_clip, grad_scale = apply_fp64_global_clip(policy, grad_clip)
    before_params = param_snapshot(policy)
    optimizer.step()
    delta = max_abs_param_delta(policy, before_params)
    optimizer.zero_grad(set_to_none=True)

    after_loss, after_summary, after_rows = run_profile(policy=policy, phase="after", **common)
    del after_loss

    comparison: dict[str, float | int | str] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)) if isinstance(checkpoint, dict) else -1,
        "batch_size": batch_size,
        "horizon": horizon,
        "tail_steps": tail_steps,
        "seed": seed,
        "sample_dynamics": int(sample_dynamics),
        "sampled_dynamics_level": sampled_dynamics_level,
        "broad_sampler": broad_sampler,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "grad_norm_before_clip": grad_before_clip,
        "grad_norm_after_clip": grad_after_clip,
        "grad_scale": grad_scale,
        "max_abs_param_delta": delta,
        "loaded_optimizer_state": int(args.init_optimizer_state),
        "terminal_loss_only": int(args.terminal_loss_only),
        "loss_before": before_summary["loss"],
        "loss_after_same_batch": after_summary["loss"],
        "loss_ratio_after_before": ratio(float(after_summary["loss"]), float(before_summary["loss"])),
        "total_grad_norm_before": before_summary["total_grad_norm"],
        "total_grad_norm_after": after_summary["total_grad_norm"],
        "total_grad_norm_ratio_after_before": ratio(
            float(after_summary["total_grad_norm"]),
            float(before_summary["total_grad_norm"]),
        ),
    }
    for column in (
        "max_action_adj",
        "max_hidden_adj_after",
        "max_lp_adj",
        "max_lw_adj",
        "max_lm_adj",
        "max_lpa_adj",
    ):
        for window in ("early8", "early32", "early64", "all"):
            key = f"{window}_{column}"
            comparison[f"{key}_before"] = before_summary[key]
            comparison[f"{key}_after"] = after_summary[key]
            comparison[f"{key}_ratio_after_before"] = ratio(float(after_summary[key]), float(before_summary[key]))

    write_csv(output_dir / "profile_before.csv", before_rows)
    write_csv(output_dir / "profile_after.csv", after_rows)
    write_csv(output_dir / "summary.csv", [comparison, before_summary, after_summary])

    print(f"wrote {output_dir}")
    print(
        "loss "
        f"{float(before_summary['loss']):.6g} -> {float(after_summary['loss']):.6g}, "
        f"grad {float(before_summary['total_grad_norm']):.6g} -> {float(after_summary['total_grad_norm']):.6g}, "
        f"delta={delta:.6g}"
    )
    for column in ("max_action_adj", "max_hidden_adj_after", "max_lw_adj", "max_lm_adj", "max_lpa_adj"):
        key = f"early64_{column}"
        print(
            f"{key}: {float(before_summary[key]):.6g} -> {float(after_summary[key]):.6g} "
            f"(x{ratio(float(after_summary[key]), float(before_summary[key])):.3g})"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env_l2f import L2FLossConfig, L2FParams, L2FState, L2FSimulator
from l2f_full_cuda_backend import full_cuda_rollout_metrics
from model import MotorGRUPolicy


ADJOINT_NAMES = (
    "lp_adj",
    "lv_adj",
    "lR_adj",
    "lw_adj",
    "lm_adj",
    "lpa_adj",
    "action_adj",
    "hidden_adj_before_actor",
    "hidden_adj_after_actor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA-full H500 gradient root-cause audit.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-step", type=int, default=-1)
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--tail-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--grad-skip-threshold", type=float, default=1.0e4)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--skip-nonfinite-update", action="store_true")
    parser.add_argument("--ignore-optimizer-state", action="store_true")
    parser.add_argument("--fixed-dynamics", action="store_true")
    parser.add_argument("--broad-sampler", default="physical", choices=("legacy", "physical"))
    parser.add_argument("--state-grad-decay", type=float, default=1.0)
    parser.add_argument("--hidden-grad-decay", type=float, default=1.0)
    return parser.parse_args()


def clone_state(state: L2FState) -> L2FState:
    return L2FState(
        **{
            field.name: getattr(state, field.name).detach().clone()
            for field in fields(L2FState)
        }
    )


def load_policy(path: Path, device: torch.device) -> tuple[MotorGRUPolicy, dict]:
    policy = MotorGRUPolicy().to(device=device)
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    policy.load_state_dict(state_dict)
    return policy, checkpoint if isinstance(checkpoint, dict) else {}


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def tensor_max_abs(tensor: torch.Tensor) -> float:
    value = tensor.detach().abs().amax()
    return float(value.item())


def tensors_finite(tensors: Iterable[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def params_finite(policy: MotorGRUPolicy) -> bool:
    return tensors_finite(parameter.detach() for parameter in policy.parameters())


def optimizer_state_finite(optimizer: torch.optim.Optimizer) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all().item()):
                return False
    return True


def optimizer_moments_finite(optimizer: torch.optim.Optimizer) -> tuple[bool, bool]:
    exp_avg_finite = True
    exp_avg_sq_finite = True
    for state in optimizer.state.values():
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if torch.is_tensor(exp_avg) and not bool(torch.isfinite(exp_avg).all().item()):
            exp_avg_finite = False
        if torch.is_tensor(exp_avg_sq) and not bool(torch.isfinite(exp_avg_sq).all().item()):
            exp_avg_sq_finite = False
    return exp_avg_finite, exp_avg_sq_finite


def max_abs_param(policy: MotorGRUPolicy) -> float:
    return max(tensor_max_abs(parameter) for parameter in policy.parameters())


def param_snapshot(policy: MotorGRUPolicy) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in policy.parameters()]


def max_abs_param_delta(policy: MotorGRUPolicy, before: list[torch.Tensor]) -> float:
    values = [
        (parameter.detach() - previous).abs().amax()
        for parameter, previous in zip(policy.parameters(), before)
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).amax().item())


def max_abs_grad(policy: MotorGRUPolicy) -> float:
    values = [
        parameter.grad.detach().abs().amax()
        for parameter in policy.parameters()
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).amax().item())


def max_abs_grad_for(policy: MotorGRUPolicy, predicate) -> float:
    values: list[float] = []
    for name, parameter in policy.named_parameters():
        if predicate(name) and parameter.grad is not None:
            values.append(tensor_max_abs(parameter.grad))
    return max(values) if values else 0.0


def first_bad_grad_module(policy: MotorGRUPolicy, grad_norm_before_clip: float) -> str:
    for name, parameter in policy.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            return name
    if not math.isfinite(grad_norm_before_clip):
        return "global_norm_overflow"
    return ""


def grad_finite(policy: MotorGRUPolicy) -> bool:
    for parameter in policy.parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            return False
    return True


def grad_norm(policy: MotorGRUPolicy, dtype: torch.dtype = torch.float32) -> float:
    total = torch.zeros((), device=next(policy.parameters()).device, dtype=dtype)
    for parameter in policy.parameters():
        if parameter.grad is not None:
            grad = parameter.grad.detach().to(dtype=dtype)
            total = total + grad.square().sum()
    return float(torch.sqrt(total).item())


def apply_fp64_global_grad_clip(policy: MotorGRUPolicy, grad_clip: float) -> tuple[float, float, float]:
    before_clip = grad_norm(policy, torch.float64)
    scale = 1.0
    if grad_clip > 0.0 and math.isfinite(before_clip):
        scale = min(1.0, grad_clip / (before_clip + 1.0e-12))
        for parameter in policy.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    after_clip = grad_norm(policy, torch.float64)
    return before_clip, after_clip, scale


def first_nonfinite_adjoint(adjoint_series: dict[str, torch.Tensor], horizon: int) -> tuple[str, int]:
    for time_index in range(horizon, -1, -1):
        for name in ADJOINT_NAMES:
            tensor = adjoint_series[name]
            if name == "action_adj" and time_index >= horizon:
                continue
            if not bool(torch.isfinite(tensor[time_index]).all().item()):
                return name, time_index
    return "", -1


def write_adjoint_rows(
    writer: csv.DictWriter,
    *,
    probe_index: int,
    train_step: int,
    horizon: int,
    loss: float,
    adjoint_series: dict[str, torch.Tensor],
) -> None:
    first_name, first_time = first_nonfinite_adjoint(adjoint_series, horizon)
    for time_index in range(horizon, -1, -1):
        row: dict[str, float | int | str] = {
            "probe_index": probe_index,
            "train_step": train_step,
            "backward_time_index": time_index,
            "loss": loss,
            "first_nonfinite_adjoint_name": first_name,
            "first_nonfinite_adjoint_time": first_time,
        }
        for name in ADJOINT_NAMES:
            tensor = adjoint_series[name]
            if name == "action_adj" and time_index >= horizon:
                row[f"max_abs_{name}"] = float("nan")
                row[f"{name}_finite"] = 1
            else:
                current = tensor[time_index].detach()
                row[f"max_abs_{name}"] = tensor_max_abs(current)
                row[f"{name}_finite"] = int(bool(torch.isfinite(current).all().item()))
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gradient_path = output_dir / "gradient_audit.csv"
    adjoint_path = output_dir / "adjoint_audit.csv"

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    params = L2FParams()
    loss_config = L2FLossConfig()
    sim = L2FSimulator(params)
    policy, checkpoint = load_policy(Path(args.checkpoint), device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if not args.ignore_optimizer_state and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_to_device(optimizer, device)

    state_template = sim.reset(
        args.batch_size,
        device=device,
        sample_dynamics=not args.fixed_dynamics,
        sampled_dynamics_level="broad",
        broad_sampler=args.broad_sampler,
    )

    gradient_fields = [
        "probe_index",
        "train_step",
        "mode",
        "loss",
        "grad_norm_before_clip",
        "grad_norm_after_clip",
        "grad_norm_fp32_before_clip",
        "grad_norm_fp64_before_clip",
        "grad_norm_fp32_after_clip",
        "grad_norm_fp64_after_clip",
        "grad_finite",
        "grad_tensor_all_finite",
        "param_finite_before",
        "param_finite_after",
        "optimizer_state_finite",
        "optimizer_exp_avg_finite",
        "optimizer_exp_avg_sq_finite",
        "max_abs_param",
        "max_abs_param_delta",
        "max_abs_grad_before_clip",
        "max_abs_grad_after_clip",
        "grad_scale",
        "max_abs_grad_encoder_before_clip",
        "max_abs_grad_gru_ih_before_clip",
        "max_abs_grad_gru_hh_before_clip",
        "max_abs_grad_motor_head_before_clip",
        "max_abs_grad_encoder",
        "max_abs_grad_gru_ih",
        "max_abs_grad_gru_hh",
        "max_abs_grad_motor_head",
        "first_bad_grad_module",
        "skip_reason",
        "optimizer_step_applied",
    ]
    adjoint_fields = [
        "probe_index",
        "train_step",
        "backward_time_index",
        "loss",
        "first_nonfinite_adjoint_name",
        "first_nonfinite_adjoint_time",
    ]
    for name in ADJOINT_NAMES:
        adjoint_fields.extend((f"max_abs_{name}", f"{name}_finite"))

    with gradient_path.open("w", newline="") as gradient_handle, adjoint_path.open("w", newline="") as adjoint_handle:
        gradient_writer = csv.DictWriter(gradient_handle, fieldnames=gradient_fields)
        adjoint_writer = csv.DictWriter(adjoint_handle, fieldnames=adjoint_fields)
        gradient_writer.writeheader()
        adjoint_writer.writeheader()

        for probe_index in range(args.probes):
            train_step = args.checkpoint_step if args.checkpoint_step >= 0 else probe_index
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            param_finite_before = params_finite(policy)
            params_before = param_snapshot(policy)
            outputs = full_cuda_rollout_metrics(
                policy,
                clone_state(state_template),
                params,
                loss_config,
                horizon=args.horizon,
                tail_steps=args.tail_steps,
                state_step_decay=args.state_grad_decay,
                hidden_step_decay=args.hidden_grad_decay,
                clf_kappa=1.0,
                u_soft=0.9,
                lambda_clf=0.5,
                lambda_out=0.1,
                lambda_tail=1.0,
                lambda_du=3.0e-3,
                lambda_ddu=3.0e-4,
                lambda_sat=0.03,
                noise_seed=0,
                external_torque_max=0.0,
                action_noise_max=0.0,
                observation_noise_max=0.0,
                collect_debug=True,
            )
            metrics = outputs[0]
            loss = metrics[0]
            adjoint_series = {
                "hidden_adj_before_actor": outputs[15],
                "hidden_adj_after_actor": outputs[16],
                "lp_adj": outputs[17],
                "lv_adj": outputs[18],
                "lR_adj": outputs[19],
                "lw_adj": outputs[20],
                "lm_adj": outputs[21],
                "lpa_adj": outputs[22],
                "action_adj": outputs[23],
            }

            loss.backward()
            grad_tensor_all_finite = grad_finite(policy)
            before_clip = grad_norm(policy, torch.float32)
            max_grad_before_clip = max_abs_grad(policy)
            max_grad_encoder_before_clip = max_abs_grad_for(policy, lambda name: name.startswith("encoder."))
            max_grad_gru_ih_before_clip = max_abs_grad_for(policy, lambda name: name.startswith("gru.") and "ih" in name)
            max_grad_gru_hh_before_clip = max_abs_grad_for(policy, lambda name: name.startswith("gru.") and "hh" in name)
            max_grad_motor_head_before_clip = max_abs_grad_for(policy, lambda name: name.startswith("motor_head."))
            before_clip_fp64, after_clip_fp64, grad_scale = apply_fp64_global_grad_clip(policy, args.grad_clip)
            after_clip = grad_norm(policy, torch.float32)
            max_grad_after_clip = max_abs_grad(policy)
            finite_grad = grad_tensor_all_finite and math.isfinite(before_clip_fp64) and math.isfinite(after_clip_fp64)
            bad_module = first_bad_grad_module(policy, before_clip)
            skip_reason = ""
            if not grad_tensor_all_finite:
                skip_reason = bad_module or "grad_tensor_nonfinite"
            elif not math.isfinite(before_clip_fp64):
                skip_reason = "grad_norm_nonfinite"
            elif args.grad_skip_threshold > 0.0 and before_clip_fp64 > args.grad_skip_threshold:
                skip_reason = "grad_skip_threshold"
            should_step = bool(args.update) and skip_reason == ""
            if bool(args.update) and not should_step and not args.skip_nonfinite_update and skip_reason != "grad_skip_threshold":
                should_step = True
            if should_step:
                optimizer.step()
            param_finite_after = params_finite(policy)
            opt_finite = optimizer_state_finite(optimizer)
            opt_exp_avg_finite, opt_exp_avg_sq_finite = optimizer_moments_finite(optimizer)
            param_delta = max_abs_param_delta(policy, params_before)

            gradient_writer.writerow(
                {
                    "probe_index": probe_index,
                    "train_step": train_step,
                    "mode": "update" if args.update else "no_update",
                    "loss": float(loss.detach().item()),
                    "grad_norm_before_clip": before_clip,
                    "grad_norm_after_clip": after_clip,
                    "grad_norm_fp32_before_clip": before_clip,
                    "grad_norm_fp64_before_clip": before_clip_fp64,
                    "grad_norm_fp32_after_clip": after_clip,
                    "grad_norm_fp64_after_clip": after_clip_fp64,
                    "grad_finite": int(finite_grad),
                    "grad_tensor_all_finite": int(grad_tensor_all_finite),
                    "param_finite_before": int(param_finite_before),
                    "param_finite_after": int(param_finite_after),
                    "optimizer_state_finite": int(opt_finite),
                    "optimizer_exp_avg_finite": int(opt_exp_avg_finite),
                    "optimizer_exp_avg_sq_finite": int(opt_exp_avg_sq_finite),
                    "max_abs_param": max_abs_param(policy),
                    "max_abs_param_delta": param_delta,
                    "max_abs_grad_before_clip": max_grad_before_clip,
                    "max_abs_grad_after_clip": max_grad_after_clip,
                    "grad_scale": grad_scale,
                    "max_abs_grad_encoder_before_clip": max_grad_encoder_before_clip,
                    "max_abs_grad_gru_ih_before_clip": max_grad_gru_ih_before_clip,
                    "max_abs_grad_gru_hh_before_clip": max_grad_gru_hh_before_clip,
                    "max_abs_grad_motor_head_before_clip": max_grad_motor_head_before_clip,
                    "max_abs_grad_encoder": max_abs_grad_for(policy, lambda name: name.startswith("encoder.")),
                    "max_abs_grad_gru_ih": max_abs_grad_for(policy, lambda name: name.startswith("gru.") and "ih" in name),
                    "max_abs_grad_gru_hh": max_abs_grad_for(policy, lambda name: name.startswith("gru.") and "hh" in name),
                    "max_abs_grad_motor_head": max_abs_grad_for(policy, lambda name: name.startswith("motor_head.")),
                    "first_bad_grad_module": bad_module,
                    "skip_reason": skip_reason,
                    "optimizer_step_applied": int(should_step),
                }
            )
            write_adjoint_rows(
                adjoint_writer,
                probe_index=probe_index,
                train_step=train_step,
                horizon=args.horizon,
                loss=float(loss.detach().item()),
                adjoint_series=adjoint_series,
            )
            gradient_handle.flush()
            adjoint_handle.flush()

    print(f"wrote gradient audit: {gradient_path}")
    print(f"wrote adjoint audit: {adjoint_path}")


if __name__ == "__main__":
    main()

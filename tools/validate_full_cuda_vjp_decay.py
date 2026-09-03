from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import fields
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_l2f import (  # noqa: E402
    L2FLossConfig,
    L2FParams,
    L2FSimulator,
    L2FState,
    apply_gradient_decay,
)
from l2f_full_cuda_backend import (  # noqa: E402
    METRIC_NAMES,
    full_cuda_rollout_metrics,
)
from model import MotorGRUPolicy  # noqa: E402
from policy_observation import (  # noqa: E402
    LEGACY_OBSERVATION_MODE,
    build_policy_observation,
    initial_observation_state,
)


PARAM_GROUPS = {
    "encoder": ("encoder.",),
    "gru": ("gru.",),
    "head": ("motor_head.",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate cuda-full rollout VJP and temporal decay.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--horizons", default="4,8,16")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tail-steps", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=8.0e-3)
    parser.add_argument("--atol", type=float, default=8.0e-4)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--physical-dynamics", action="store_true")
    parser.add_argument("--sampled-dynamics-level", default="small", choices=("small", "medium", "broad"))
    parser.add_argument("--broad-sampler", default="physical", choices=("legacy", "physical", "physical-fit"))
    return parser.parse_args()


def clone_state(state: L2FState, *, requires_grad: bool = False) -> L2FState:
    values = {}
    for field in fields(L2FState):
        value = getattr(state, field.name)
        cloned = value.detach().clone()
        if requires_grad and field.name in {
            "position",
            "velocity",
            "rotation",
            "omega",
            "motor",
            "previous_action",
        }:
            cloned.requires_grad_(True)
        values[field.name] = cloned
    return L2FState(**values)


def state_tuple(state: L2FState) -> tuple[torch.Tensor, ...]:
    return (
        state.position,
        state.velocity,
        state.rotation,
        state.omega,
        state.motor,
        state.previous_action,
    )


def metric_vector(metrics: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.stack([metrics[name] for name in METRIC_NAMES]).to(device=device)


def zero_policy_grads(policy: MotorGRUPolicy) -> None:
    for param in policy.parameters():
        param.grad = None


def named_grads(policy: MotorGRUPolicy) -> dict[str, torch.Tensor]:
    out = {}
    for name, param in policy.named_parameters():
        out[name] = torch.zeros_like(param) if param.grad is None else param.grad.detach().clone()
    return out


def group_norms(grads: dict[str, torch.Tensor]) -> dict[str, float]:
    out = {}
    for group, prefixes in PARAM_GROUPS.items():
        total = torch.zeros((), device=next(iter(grads.values())).device)
        for name, grad in grads.items():
            if name.startswith(prefixes):
                total = total + grad.double().square().sum()
        out[group] = float(torch.sqrt(total).item())
    return out


def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a.detach() - b.detach()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = torch.maximum(a.detach().abs(), b.detach().abs()).clamp_min(1.0e-12)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    print(f"    {name:<28} max_abs={max_abs:.6e} max_rel={max_rel:.6e}")
    return max_abs, max_rel


def tracking_components_mean(sim: L2FSimulator, state: L2FState, loss_config: L2FLossConfig) -> dict[str, torch.Tensor]:
    components = sim.tracking_components(state, loss_config)
    return {name: value.mean() for name, value in components.items()}


def pytorch_rollout_loss(
    *,
    policy: MotorGRUPolicy,
    sim: L2FSimulator,
    initial_state: L2FState,
    loss_config: L2FLossConfig,
    horizon: int,
    tail_steps: int,
    state_decay: float,
    hidden_decay: float,
    terminal_loss_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, L2FState, list[torch.Tensor]]:
    state = clone_state(initial_state)
    hidden = None
    actions: list[torch.Tensor] = []
    tracking_sum = torch.zeros((), device=state.position.device)
    clf_sum = torch.zeros((), device=state.position.device)
    outward_sum = torch.zeros((), device=state.position.device)
    du_sum = torch.zeros((), device=state.position.device)
    ddu_sum = torch.zeros((), device=state.position.device)
    sat_sum = torch.zeros((), device=state.position.device)
    metric_sums = {
        "position": torch.zeros((), device=state.position.device),
        "velocity": torch.zeros((), device=state.position.device),
        "omega": torch.zeros((), device=state.position.device),
    }
    previous_potential = sim.tracking_potential(state, loss_config)
    previous_action_delta = None
    tail_potentials: list[torch.Tensor] = []
    observation_state = initial_observation_state(
        state.position.shape[0], device=state.position.device, dtype=state.position.dtype
    )

    for _ in range(horizon):
        observation, _ = build_policy_observation(
            state, observation_state, mode=LEGACY_OBSERVATION_MODE
        )
        action, hidden = policy(observation, hidden)
        action.retain_grad()
        actions.append(action)
        hidden = apply_gradient_decay(hidden, hidden_decay)
        action_delta = action - state.previous_action
        state = sim.step(state, action, grad_decay=state_decay)

        tracking_components = sim.tracking_components(state, loss_config)
        potential = sum(tracking_components.values())
        tracking_sum = tracking_sum + potential.mean()
        clf_target = (1.0 - sim.params.dt * 1.0) * previous_potential.detach()
        clf_sum = clf_sum + F.relu(potential - clf_target).square().mean()
        outward_sum = outward_sum + sim.outward_velocity_loss(state, loss_config)
        du_sum = du_sum + action_delta.square().mean()
        sat_sum = sat_sum + F.relu(action.abs() - 0.9).square().mean()
        if previous_action_delta is not None:
            ddu_sum = ddu_sum + (action_delta - previous_action_delta).square().mean()
        for name, value in tracking_components.items():
            metric_sums[name] = metric_sums[name] + value.mean()
        tail_potentials.append(potential)
        previous_potential = potential
        previous_action_delta = action_delta

    if terminal_loss_only:
        comps = tracking_components_mean(sim, state, loss_config)
        loss = sum(comps.values())
        metrics = {
            "loss": loss,
            "tracking": loss,
            "position": comps["position"],
            "velocity": comps["velocity"],
            "omega": comps["omega"],
            "clf": torch.zeros_like(loss),
            "outward": torch.zeros_like(loss),
            "tail": torch.zeros_like(loss),
            "du": torch.zeros_like(loss),
            "ddu": torch.zeros_like(loss),
            "sat": torch.zeros_like(loss),
        }
        return loss, metric_vector(metrics, state.position.device), state, actions

    horizon_f = float(horizon)
    tail_count = min(max(tail_steps, 1), horizon)
    tracking_loss = tracking_sum / horizon_f
    clf_loss = clf_sum / horizon_f
    outward_loss = outward_sum / horizon_f
    tail_loss = torch.stack(tail_potentials[-tail_count:]).mean()
    du_loss = du_sum / horizon_f
    ddu_loss = ddu_sum / float(max(horizon - 1, 1))
    sat_loss = sat_sum / horizon_f
    loss = (
        tracking_loss
        + 0.5 * clf_loss
        + 0.1 * outward_loss
        + 1.0 * tail_loss
        + 3.0e-3 * du_loss
        + 3.0e-4 * ddu_loss
        + 0.03 * sat_loss
    )
    metrics = {
        "loss": loss,
        "tracking": tracking_loss,
        "position": metric_sums["position"] / horizon_f,
        "velocity": metric_sums["velocity"] / horizon_f,
        "omega": metric_sums["omega"] / horizon_f,
        "clf": clf_loss,
        "outward": outward_loss,
        "tail": tail_loss,
        "du": du_loss,
        "ddu": ddu_loss,
        "sat": sat_loss,
    }
    return loss, metric_vector(metrics, state.position.device), state, actions


def cuda_rollout(
    *,
    policy: MotorGRUPolicy,
    sim: L2FSimulator,
    initial_state: L2FState,
    loss_config: L2FLossConfig,
    horizon: int,
    tail_steps: int,
    state_decay: float,
    hidden_decay: float,
    terminal_loss_only: bool,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    outputs = full_cuda_rollout_metrics(
        policy,
        initial_state,
        sim.params,
        loss_config,
        horizon=horizon,
        tail_steps=tail_steps,
        state_step_decay=state_decay,
        hidden_step_decay=hidden_decay,
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
        terminal_loss_only=terminal_loss_only,
        collect_debug=True,
    )
    return outputs, outputs[23].detach()


def action_grad_profile(actions: list[torch.Tensor]) -> torch.Tensor:
    rows = []
    for action in actions:
        if action.grad is None:
            rows.append(torch.zeros_like(action))
        else:
            rows.append(action.grad.detach())
    return torch.stack(rows, dim=0)


def run_case(
    *,
    base_policy: MotorGRUPolicy,
    sim: L2FSimulator,
    initial_state: L2FState,
    loss_config: L2FLossConfig,
    horizon: int,
    tail_steps: int,
    state_decay: float,
    hidden_decay: float,
    terminal_loss_only: bool,
    rtol: float,
    atol: float,
) -> bool:
    print(
        f"\ncase horizon={horizon} state_decay={state_decay:g} "
        f"hidden_decay={hidden_decay:g} terminal_only={int(terminal_loss_only)}"
    )
    torch_policy = copy.deepcopy(base_policy)
    cuda_policy = copy.deepcopy(base_policy)
    torch_policy.train()
    cuda_policy.train()

    torch_state = clone_state(initial_state)
    cuda_state = clone_state(initial_state)

    zero_policy_grads(torch_policy)
    torch_loss, torch_metrics, torch_final, torch_actions = pytorch_rollout_loss(
        policy=torch_policy,
        sim=sim,
        initial_state=torch_state,
        loss_config=loss_config,
        horizon=horizon,
        tail_steps=tail_steps,
        state_decay=state_decay,
        hidden_decay=hidden_decay,
        terminal_loss_only=terminal_loss_only,
    )
    torch_loss.backward()
    torch_action_adj = action_grad_profile(torch_actions)
    torch_grads = named_grads(torch_policy)

    zero_policy_grads(cuda_policy)
    cuda_outputs, cuda_action_adj = cuda_rollout(
        policy=cuda_policy,
        sim=sim,
        initial_state=cuda_state,
        loss_config=loss_config,
        horizon=horizon,
        tail_steps=tail_steps,
        state_decay=state_decay,
        hidden_decay=hidden_decay,
        terminal_loss_only=terminal_loss_only,
    )
    cuda_metrics = cuda_outputs[0]
    cuda_outputs[0][0].backward()
    cuda_grads = named_grads(cuda_policy)

    ok = True
    comparisons = [
        compare_tensors("metrics", torch_metrics, cuda_metrics),
        compare_tensors("final_position", torch_final.position, cuda_outputs[1]),
        compare_tensors("final_velocity", torch_final.velocity, cuda_outputs[2]),
        compare_tensors("final_rotation", torch_final.rotation, cuda_outputs[3]),
        compare_tensors("final_omega", torch_final.omega, cuda_outputs[4]),
        compare_tensors("final_motor", torch_final.motor, cuda_outputs[5]),
        compare_tensors("final_previous_action", torch_final.previous_action, cuda_outputs[6]),
        compare_tensors("action_adjoint", torch_action_adj, cuda_action_adj),
    ]
    for max_abs, max_rel in comparisons:
        ok = ok and (max_abs <= atol or max_rel <= rtol)

    print("    param group norms:")
    torch_group = group_norms(torch_grads)
    cuda_group = group_norms(cuda_grads)
    for group in PARAM_GROUPS:
        t = torch_group[group]
        c = cuda_group[group]
        rel = abs(t - c) / max(abs(t), abs(c), 1.0e-12)
        print(f"      {group:<8} torch={t:.6e} cuda={c:.6e} rel={rel:.6e}")
        ok = ok and (abs(t - c) <= atol or rel <= rtol)

    worst_param_abs = 0.0
    worst_param_rel = 0.0
    worst_param_name = ""
    for name in torch_grads:
        t = torch_grads[name]
        c = cuda_grads[name]
        diff = (t - c).abs()
        max_abs = float(diff.max().item())
        denom = torch.maximum(t.abs(), c.abs()).clamp_min(1.0e-12)
        max_rel = float((diff / denom).max().item())
        if max_abs > worst_param_abs:
            worst_param_abs = max_abs
            worst_param_rel = max_rel
            worst_param_name = name
    print(
        f"    worst param element {worst_param_name}: "
        f"max_abs={worst_param_abs:.6e} max_rel={worst_param_rel:.6e}"
    )
    ok = ok and (worst_param_abs <= atol or worst_param_rel <= rtol)

    action_norm = cuda_action_adj.double().square().sum(dim=(1, 2)).sqrt().detach().cpu()
    print("    cuda action-adjoint norm by step:")
    print("      " + " ".join(f"{float(v):.3e}" for v in action_norm))
    print(f"    result={'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    args = parse_args()
    loss_config = L2FLossConfig()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("cuda-full VJP validation requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    params = L2FParams()
    sim = L2FSimulator(params)
    horizons = [int(item) for item in args.horizons.split(",") if item.strip()]

    base_policy = MotorGRUPolicy(
        observation_dim=40, encoder_dim=192, hidden_dim=192, encoder_depth=2
    ).to(device)
    initial_state = sim.reset(
        args.batch_size,
        device=device,
        sample_dynamics=args.physical_dynamics,
        sampled_dynamics_level=args.sampled_dynamics_level,
        broad_sampler=args.broad_sampler,
    )

    cases = []
    for horizon in horizons:
        cases.append((horizon, 1.0, 1.0, False))
        cases.append((horizon, 0.5, 0.7, False))
    terminal_horizon = horizons[-1]
    cases.extend(
        [
            (terminal_horizon, 1.0, 1.0, True),
            (terminal_horizon, 0.5, 1.0, True),
            (terminal_horizon, 0.0, 1.0, True),
            (terminal_horizon, 1.0, 0.0, True),
        ]
    )

    all_ok = True
    for horizon, state_decay, hidden_decay, terminal_loss_only in cases:
        ok = run_case(
            base_policy=base_policy,
            sim=sim,
            initial_state=initial_state,
            loss_config=loss_config,
            horizon=horizon,
            tail_steps=min(args.tail_steps, horizon),
            state_decay=state_decay,
            hidden_decay=hidden_decay,
            terminal_loss_only=terminal_loss_only,
            rtol=args.rtol,
            atol=args.atol,
        )
        all_ok = all_ok and ok

    if args.strict and not all_ok:
        raise SystemExit("full cuda VJP decay validation failed")
    if all_ok:
        print("\nfull cuda VJP decay validation passed")
    else:
        print("\nfull cuda VJP decay validation found mismatches")


if __name__ == "__main__":
    main()

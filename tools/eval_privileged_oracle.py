from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_l2f import L2FLossConfig, L2FParams, L2FSimulator, L2FState, normalized_capability_target  # noqa: E402
from l2f_cuda_backend import cuda_step, load_extension  # noqa: E402
from policy_observation import physical_observation  # noqa: E402


DEFAULT_DYNAMIC_HARD_IDS = (
    53, 69, 97, 147, 161, 237, 351, 355, 396, 467, 514,
    569, 608, 636, 650, 677, 682, 732, 768, 834, 864, 904,
)


class IndependentPrivilegedTeacher(nn.Module):
    """A separate from-scratch privileged MLP for every scenario/restart."""

    def __init__(self, count: int, input_dim: int, hidden_dim: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.w1 = nn.Parameter(
            torch.randn(count, hidden_dim, input_dim, generator=generator) / math.sqrt(input_dim)
        )
        self.b1 = nn.Parameter(torch.zeros(count, hidden_dim))
        self.w2 = nn.Parameter(
            torch.randn(count, hidden_dim, hidden_dim, generator=generator) / math.sqrt(hidden_dim)
        )
        self.b2 = nn.Parameter(torch.zeros(count, hidden_dim))
        self.w_out = nn.Parameter(torch.randn(count, 4, hidden_dim, generator=generator) * 1.0e-3)
        self.b_out = nn.Parameter(torch.zeros(count, 4))

    def forward(self, privileged: torch.Tensor) -> torch.Tensor:
        if privileged.shape[0] != self.w1.shape[0]:
            raise ValueError("each teacher row must receive its own scenario row")
        hidden = F.leaky_relu(
            torch.bmm(self.w1, privileged.unsqueeze(-1)).squeeze(-1) + self.b1,
            negative_slope=0.05,
        )
        hidden = F.leaky_relu(
            torch.bmm(self.w2, hidden.unsqueeze(-1)).squeeze(-1) + self.b2,
            negative_slope=0.05,
        )
        logits = torch.bmm(self.w_out, hidden.unsqueeze(-1)).squeeze(-1) + self.b_out
        return torch.tanh(logits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone privileged reachability oracle.")
    parser.add_argument("--scenario-ids", default="")
    parser.add_argument("--scenario-csv", default="reports/privileged_oracle/dynamic_hard_22_scenarios.csv")
    parser.add_argument("--scenario-state-csv", default="reports/privileged_oracle/dynamic_hard_22_scenarios.csv")
    parser.add_argument("--output-dir", default="reports/privileged_oracle_standalone_u2000_r3")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--sim-backend", default="cuda", choices=("cuda", "torch"))
    parser.add_argument("--seed", type=int, default=2307)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--train-horizon", type=int, default=500)
    parser.add_argument("--eval-horizon", type=int, default=10000)
    parser.add_argument("--steady-window-steps", type=int, default=100)
    parser.add_argument("--steady-required-fraction", type=float, default=0.95)
    parser.add_argument("--success-position", type=float, default=0.05)
    parser.add_argument("--success-velocity", type=float, default=0.10)
    parser.add_argument("--success-omega", type=float, default=0.20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--tail-weight", type=float, default=0.02)
    parser.add_argument("--early-tail-weight", type=float, default=0.25)
    parser.add_argument("--final-tail-weight", type=float, default=1.0)
    parser.add_argument("--action-smooth-weight", type=float, default=3.0e-3)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--shooting-updates", type=int, default=2000)
    parser.add_argument("--shooting-restarts", type=int, default=3)
    parser.add_argument("--shooting-lr", type=float, default=2.0e-2)
    parser.add_argument("--skip-direct-shooting", action="store_true")
    parser.add_argument("--skip-matlab-validation", action="store_true")
    parser.add_argument("--allow-short-smoke", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def _parse_ids(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_id(row: dict[str, str]) -> int:
    for name in ("scenario_id", "sample_id", "sample_index"):
        if row.get(name, ""):
            return int(float(row[name]))
    raise ValueError("scenario CSV must contain scenario_id, sample_id, or sample_index")


def _ensure_default_state_csv(path: Path) -> None:
    if path.exists():
        return
    command = (
        "restoredefaultpath; rehash toolboxcache; "
        f"addpath(fullfile('{ROOT.as_posix()}','matlab_l2f')); "
        f"export_dynamic_hard_scenarios('{path.as_posix()}');"
    )
    subprocess.run(("matlab", "-batch", command), cwd=ROOT, check=True)


def _vector(rows: list[dict[str, str]], prefix: str, width: int) -> torch.Tensor:
    return torch.tensor(
        [[float(row[f"{prefix}_{index}"]) for index in range(width)] for row in rows],
        dtype=torch.float32,
    )


def _scalar(rows: list[dict[str, str]], name: str) -> torch.Tensor:
    return torch.tensor([float(row[name]) for row in rows], dtype=torch.float32)


def load_scenarios(args: argparse.Namespace, device: torch.device) -> tuple[L2FState, torch.Tensor]:
    input_path = Path(args.scenario_csv)
    state_path = Path(args.scenario_state_csv)
    _ensure_default_state_csv(state_path)
    input_rows = _read_csv(input_path) if input_path.exists() else []
    requested_ids = _parse_ids(args.scenario_ids)
    if not requested_ids:
        requested_ids = [_row_id(row) for row in input_rows] if input_rows else list(DEFAULT_DYNAMIC_HARD_IDS)
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("scenario IDs must be unique")
    state_rows = input_rows if input_rows and "position_0" in input_rows[0] else _read_csv(state_path)
    by_id = {_row_id(row): row for row in state_rows}
    missing = sorted(set(requested_ids) - set(by_id))
    if missing:
        raise ValueError(f"full initial state unavailable for scenario IDs: {missing}")
    rows = [by_id[value] for value in requested_ids]
    rotation = torch.tensor(
        [[[float(row[f"rotation_{r}{c}"]) for c in range(3)] for r in range(3)] for row in rows],
        dtype=torch.float32,
    )
    scalar_names = (
        "mass", "thrust_to_weight", "torque_to_inertia", "rotor_distance_factor",
        "inertia_factor", "motor_time_rising", "motor_time_falling",
        "rotor_torque_constant", "cbrt_mass", "force_std", "arm_length",
        "inertia_x", "inertia_y", "inertia_z", "alpha_roll_max",
        "alpha_pitch_max", "alpha_yaw_max", "eta_yaw", "jz_over_jxy",
        "dt_alpha_roll_max", "dt_alpha_yaw_max",
    )
    values: dict[str, torch.Tensor] = {
        "position": _vector(rows, "position", 3),
        "velocity": _vector(rows, "velocity", 3),
        "rotation": rotation,
        "omega": _vector(rows, "omega", 3),
        "motor": _vector(rows, "motor", 4),
        "previous_action": _vector(rows, "previous_action", 4),
        "external_force": _vector(rows, "external_force", 3),
        "thrust_coeff_c0": _vector(rows, "thrust_coeff_c0", 4),
        "thrust_coeff_c1": _vector(rows, "thrust_coeff_c1", 4),
        "thrust_coeff_c2": _vector(rows, "thrust_coeff_c2", 4),
        **{name: _scalar(rows, name) for name in scalar_names},
    }
    state = L2FState(**{name: value.to(device) for name, value in values.items()})
    return state, torch.tensor(requested_ids, device=device, dtype=torch.long)


def select_state(state: L2FState, indices: torch.Tensor) -> L2FState:
    return L2FState(**{
        name: getattr(state, name).index_select(0, indices).detach().clone()
        for name in L2FState.__dataclass_fields__
    })


def finite_state(state: L2FState) -> torch.Tensor:
    finite = torch.ones(state.position.shape[0], device=state.position.device, dtype=torch.bool)
    for name in ("position", "velocity", "rotation", "omega", "motor", "previous_action"):
        finite &= torch.isfinite(getattr(state, name).reshape(state.position.shape[0], -1)).all(dim=-1)
    return finite


def privileged_features(state: L2FState) -> torch.Tensor:
    return torch.cat(
        (
            physical_observation(state),
            state.motor,
            state.previous_action,
            normalized_capability_target(state),
            state.external_force,
        ),
        dim=-1,
    )


def step_sim(sim: L2FSimulator, state: L2FState, action: torch.Tensor, backend: str) -> L2FState:
    return cuda_step(state, action, sim.params, grad_decay=1.0) if backend == "cuda" else sim.step(state, action, grad_decay=1.0)


def _margins(history: deque[torch.Tensor], threshold: float) -> torch.Tensor:
    values = torch.stack(tuple(history), dim=0)
    return F.relu(values.norm(dim=-1) / float(threshold) - 1.0).square().mean(dim=0)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clip_independent_gradients(parameters: list[nn.Parameter], max_norm: float) -> float:
    count = parameters[0].shape[0]
    norm_sq = torch.zeros(count, device=parameters[0].device, dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            norm_sq += parameter.grad.detach().double().square().flatten(1).sum(dim=1)
    norms = norm_sq.sqrt()
    if max_norm > 0.0:
        scales = (float(max_norm) / (norms + 1.0e-12)).clamp(max=1.0)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scales.to(parameter.dtype).reshape(-1, *([1] * (parameter.ndim - 1))))
    return float(norms.max().item())


def train_teachers(
    teacher: IndependentPrivilegedTeacher,
    sim: L2FSimulator,
    initial_state: L2FState,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], torch.Tensor]:
    optimizer = torch.optim.Adam(teacher.parameters(), lr=args.lr)
    loss_config = L2FLossConfig()
    reached_update = torch.full(
        (initial_state.position.shape[0],), -1, device=initial_state.position.device, dtype=torch.long
    )
    rows: list[dict[str, object]] = []
    indices = torch.arange(initial_state.position.shape[0], device=initial_state.position.device)
    for update in range(1, args.updates + 1):
        state = select_state(initial_state, indices)
        dense_sum = torch.zeros(state.position.shape[0], device=state.position.device)
        smooth_sum = torch.zeros_like(dense_sum)
        position_window: deque[torch.Tensor] = deque(maxlen=args.steady_window_steps)
        omega_window: deque[torch.Tensor] = deque(maxlen=args.steady_window_steps)
        success_window: deque[torch.Tensor] = deque(maxlen=args.steady_window_steps)
        previous_action = state.previous_action
        early_position = torch.zeros_like(dense_sum)
        early_omega = torch.zeros_like(dense_sum)
        alive = torch.ones_like(dense_sum, dtype=torch.bool)
        for step in range(1, args.train_horizon + 1):
            action = teacher(privileged_features(state))
            state = step_sim(sim, state, action, args.sim_backend)
            components = sim.tracking_components(state, loss_config)
            dense_sum = dense_sum + sum(components.values())
            smooth_sum = smooth_sum + (action - previous_action).square().mean(dim=-1)
            previous_action = action
            alive &= finite_state(state)
            position_window.append(state.position)
            omega_window.append(state.omega)
            success_window.append(
                (state.position.norm(dim=-1) < args.success_position)
                & (state.velocity.norm(dim=-1) < args.success_velocity)
                & (state.omega.norm(dim=-1) < args.success_omega)
                & alive
            )
            if step == 250:
                early_position = _margins(position_window, args.success_position)
                early_omega = _margins(omega_window, args.success_omega)
        final_position = _margins(position_window, args.success_position)
        final_omega = _margins(omega_window, args.success_omega)
        loss_per_teacher = (
            dense_sum / float(args.train_horizon)
            + args.action_smooth_weight * smooth_sum / float(args.train_horizon)
            + args.tail_weight * (
                args.early_tail_weight * (early_position + early_omega)
                + args.final_tail_weight * (final_position + final_omega)
            )
        )
        loss = loss_per_teacher.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"teacher training became non-finite at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_independent_gradients(list(teacher.parameters()), args.grad_clip)
        optimizer.step()
        fraction = torch.stack(tuple(success_window), dim=0).float().mean(dim=0)
        steady = (fraction >= args.steady_required_fraction) & alive
        reached_update[(reached_update < 0) & steady] = update
        row = {
            "optimizer_update": update,
            "loss": float(loss.detach()),
            "dense_loss": float((dense_sum / args.train_horizon).mean().detach()),
            "early_position_margin": float(early_position.mean().detach()),
            "early_omega_margin": float(early_omega.mean().detach()),
            "final_position_margin": float(final_position.mean().detach()),
            "final_omega_margin": float(final_omega.mean().detach()),
            "action_smooth_loss": float((smooth_sum / args.train_horizon).mean().detach()),
            "grad_norm_max": grad_norm,
            "steady_restart_rate_H500": float(steady.float().mean()),
            "ever_reached_restart_rate_H500": float((reached_update >= 0).float().mean()),
        }
        rows.append(row)
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            print(
                f"teacher update={update}/{args.updates} loss={row['loss']:.6f} "
                f"steady_H500={row['steady_restart_rate_H500']:.3f}",
                flush=True,
            )
    return rows, reached_update


@torch.no_grad()
def evaluate_teacher_python(
    teacher: IndependentPrivilegedTeacher,
    sim: L2FSimulator,
    initial_state: L2FState,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    indices = torch.arange(initial_state.position.shape[0], device=initial_state.position.device)
    state = select_state(initial_state, indices)
    alive = torch.ones(state.position.shape[0], device=state.position.device, dtype=torch.bool)
    window: deque[torch.Tensor] = deque(maxlen=args.steady_window_steps)
    h500_fraction = torch.zeros(state.position.shape[0], device=state.position.device)
    h500_steady = torch.zeros_like(alive)
    for step in range(1, args.eval_horizon + 1):
        state = step_sim(sim, state, teacher(privileged_features(state)), args.sim_backend)
        alive &= finite_state(state)
        window.append(
            (state.position.norm(dim=-1) < args.success_position)
            & (state.velocity.norm(dim=-1) < args.success_velocity)
            & (state.omega.norm(dim=-1) < args.success_omega)
            & alive
        )
        if step == 500:
            h500_fraction = torch.stack(tuple(window)).float().mean(dim=0)
            h500_steady = (h500_fraction >= args.steady_required_fraction) & alive
        if step % 1000 == 0:
            print(f"teacher Python validation H{step}/{args.eval_horizon}", flush=True)
    final_fraction = torch.stack(tuple(window)).float().mean(dim=0)
    return {
        "h500_fraction": h500_fraction,
        "h500_steady": h500_steady,
        "final_fraction": final_fraction,
        "final_steady": (final_fraction >= args.steady_required_fraction) & alive,
        "survival": alive,
        "position_final": state.position.norm(dim=-1),
        "velocity_final": state.velocity.norm(dim=-1),
        "omega_final": state.omega.norm(dim=-1),
    }


def train_direct_shooting(
    sim: L2FSimulator,
    initial_state: L2FState,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    count = initial_state.position.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 8803)
    logits = nn.Parameter(
        0.02 * torch.randn(count, args.train_horizon, 4, generator=generator).to(initial_state.position.device)
    )
    optimizer = torch.optim.Adam((logits,), lr=args.shooting_lr)
    loss_config = L2FLossConfig()
    indices = torch.arange(count, device=initial_state.position.device)
    rows: list[dict[str, object]] = []
    for update in range(1, args.shooting_updates + 1):
        state = select_state(initial_state, indices)
        dense_sum = torch.zeros(count, device=state.position.device)
        smooth_sum = torch.zeros_like(dense_sum)
        p_window: deque[torch.Tensor] = deque(maxlen=args.steady_window_steps)
        w_window: deque[torch.Tensor] = deque(maxlen=args.steady_window_steps)
        previous = state.previous_action
        early_p = torch.zeros_like(dense_sum)
        early_w = torch.zeros_like(dense_sum)
        for step in range(1, args.train_horizon + 1):
            action = torch.tanh(logits[:, step - 1])
            state = step_sim(sim, state, action, args.sim_backend)
            dense_sum = dense_sum + sum(sim.tracking_components(state, loss_config).values())
            smooth_sum = smooth_sum + (action - previous).square().mean(dim=-1)
            previous = action
            p_window.append(state.position)
            w_window.append(state.omega)
            if step == 250:
                early_p = _margins(p_window, args.success_position)
                early_w = _margins(w_window, args.success_omega)
        final_p = _margins(p_window, args.success_position)
        final_w = _margins(w_window, args.success_omega)
        per_sequence = (
            dense_sum / args.train_horizon
            + args.action_smooth_weight * smooth_sum / args.train_horizon
            + args.tail_weight * (
                args.early_tail_weight * (early_p + early_w)
                + args.final_tail_weight * (final_p + final_w)
            )
        )
        loss = per_sequence.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_((logits,), args.grad_clip))
        optimizer.step()
        row = {
            "optimizer_update": update,
            "loss": float(loss.detach()),
            "early_position_margin": float(early_p.mean().detach()),
            "early_omega_margin": float(early_w.mean().detach()),
            "final_position_margin": float(final_p.mean().detach()),
            "final_omega_margin": float(final_w.mean().detach()),
            "grad_norm": grad_norm,
        }
        rows.append(row)
        if update == 1 or update % args.log_every == 0 or update == args.shooting_updates:
            print(f"direct shooting update={update}/{args.shooting_updates} loss={row['loss']:.6f}", flush=True)
    return torch.tanh(logits.detach()), rows


def export_matlab_artifact(
    path: Path,
    teacher: IndependentPrivilegedTeacher,
    teacher_ids: torch.Tensor,
    teacher_capability: torch.Tensor,
    shooting_actions: torch.Tensor | None,
    shooting_ids: torch.Tensor | None,
    args: argparse.Namespace,
) -> None:
    state = teacher.state_dict()
    payload: dict[str, np.ndarray] = {
        f"teacher_{name}": value.detach().cpu().numpy().astype(np.float64)
        for name, value in state.items()
    }
    payload.update({
        "teacher_scenario_id": teacher_ids.detach().cpu().numpy().astype(np.float64).reshape(-1, 1),
        "teacher_capability": teacher_capability.detach().cpu().numpy().astype(np.float64),
        "success_position": np.array([[args.success_position]], dtype=np.float64),
        "success_velocity": np.array([[args.success_velocity]], dtype=np.float64),
        "success_omega": np.array([[args.success_omega]], dtype=np.float64),
        "steady_window_steps": np.array([[args.steady_window_steps]], dtype=np.float64),
        "steady_required_fraction": np.array([[args.steady_required_fraction]], dtype=np.float64),
    })
    if shooting_actions is not None and shooting_ids is not None:
        payload["shooting_actions"] = shooting_actions.detach().cpu().numpy().astype(np.float64)
        payload["shooting_scenario_id"] = shooting_ids.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)
    else:
        payload["shooting_actions"] = np.empty((0, args.train_horizon, 4), dtype=np.float64)
        payload["shooting_scenario_id"] = np.empty((0, 1), dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(path, payload, do_compression=True)


def run_matlab_validation(artifact: Path, scenario_csv: Path, output_csv: Path, args: argparse.Namespace) -> bool:
    matlab_root = (ROOT / "matlab_l2f").as_posix().replace("'", "''")
    expression = (
        "restoredefaultpath; rehash toolboxcache; "
        f"addpath('{matlab_root}'); "
        f"run_privileged_oracle_eval('{artifact.as_posix()}','{scenario_csv.as_posix()}',"
        f"'{output_csv.as_posix()}',{args.eval_horizon});"
    )
    try:
        subprocess.run(("matlab", "-batch", expression), cwd=ROOT, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"MATLAB oracle validation failed: {exc}", file=sys.stderr)
        return False
    return output_csv.exists()


def write_results(
    scenario_ids: torch.Tensor,
    teacher_ids: torch.Tensor,
    reached_update: torch.Tensor,
    python_eval: dict[str, torch.Tensor],
    matlab_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> None:
    output_dir = Path(args.output_dir)
    matlab_by_id: dict[int, list[dict[str, str]]] = {}
    for row in matlab_rows:
        matlab_by_id.setdefault(int(float(row["scenario_id"])), []).append(row)
    teacher_ids_cpu = teacher_ids.detach().cpu().tolist()
    detailed: list[dict[str, object]] = []
    for index, scenario_id in enumerate(teacher_ids_cpu):
        detailed.append({
            "scenario_id": scenario_id,
            "restart": index % args.restarts,
            "python_steady_H500": int(python_eval["h500_steady"][index]),
            f"python_steady_H{args.eval_horizon}": int(python_eval["final_steady"][index]),
            "python_window_fraction_H500": float(python_eval["h500_fraction"][index]),
            f"python_window_fraction_H{args.eval_horizon}": float(python_eval["final_fraction"][index]),
            "first_reachable_update": int(reached_update[index]),
            "python_survival": int(python_eval["survival"][index]),
            "position_final": float(python_eval["position_final"][index]),
            "velocity_final": float(python_eval["velocity_final"][index]),
            "omega_final": float(python_eval["omega_final"][index]),
        })
    _write_csv(output_dir / "restart_results.csv", detailed)
    scenario_rows: list[dict[str, object]] = []
    for scenario_id in scenario_ids.detach().cpu().tolist():
        rows = matlab_by_id.get(int(scenario_id), [])
        validated = [row for row in rows if float(row[f"steady_H{args.eval_horizon}"]) > 0.5]
        teacher_validated = [row for row in validated if row["controller_type"] == "teacher"]
        shooting_validated = [row for row in validated if row["controller_type"] == "direct_shooting"]
        scenario_rows.append({
            "scenario_id": scenario_id,
            "status": "oracle_reachable" if validated else "unproven",
            "matlab_validation_available": int(bool(rows)),
            "matlab_teacher_successes_H10000": len(teacher_validated),
            "matlab_direct_shooting_successes_H10000": len(shooting_validated),
            "attempted_teacher_restarts": args.restarts,
            "attempted_direct_shooting": int(not args.skip_direct_shooting),
        })
    _write_csv(output_dir / "scenario_results.csv", scenario_rows)
    reachable = sum(row["status"] == "oracle_reachable" for row in scenario_rows)
    report = (
        "# Standalone privileged oracle summary\n\n"
        f"- Scenarios: {len(scenario_rows)}\n"
        f"- From-scratch teacher restarts/scenario: {args.restarts}\n"
        f"- Teacher optimizer updates: {args.updates}\n"
        f"- Direct-shooting optimizer updates: {0 if args.skip_direct_shooting else args.shooting_updates}\n"
        f"- MATLAB H{args.eval_horizon} oracle_reachable: {reachable}/{len(scenario_rows)}\n"
        "- Every remaining scenario is labeled unproven, never physically_infeasible.\n"
    )
    (output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)


def main() -> None:
    args = parse_args()
    if args.restarts < 3:
        raise ValueError("privileged oracle requires at least three independent restarts")
    if not args.allow_short_smoke and args.updates < 2000:
        raise ValueError("formal privileged teacher requires at least 2000 optimizer updates")
    if not args.skip_direct_shooting and not args.allow_short_smoke and args.shooting_updates < 2000:
        raise ValueError("formal direct shooting requires at least 2000 optimizer updates")
    if not args.allow_short_smoke and args.eval_horizon != 10000:
        raise ValueError("formal privileged oracle requires MATLAB H10000 validation")
    if args.train_horizon < 500 or args.steady_window_steps > 250:
        raise ValueError("oracle requires H500 training and a window no longer than H250")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.sim_backend == "cuda":
        if device.type != "cuda":
            raise ValueError("compact CUDA requires --device cuda")
        load_extension()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    scenarios, scenario_ids = load_scenarios(args, device)
    restart_indices = torch.arange(len(scenario_ids), device=device).repeat_interleave(args.restarts)
    initial_state = select_state(scenarios, restart_indices)
    teacher_ids = scenario_ids.index_select(0, restart_indices)
    sim = L2FSimulator(L2FParams(dt=0.01))
    teacher = IndependentPrivilegedTeacher(
        initial_state.position.shape[0],
        privileged_features(initial_state).shape[-1],
        args.hidden_dim,
        args.seed,
    ).to(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    learning_curve, reached_update = train_teachers(teacher, sim, initial_state, args)
    _write_csv(output_dir / "teacher_learning_curve.csv", learning_curve)
    python_eval = evaluate_teacher_python(teacher, sim, initial_state, args)

    scenario_csv = Path(args.scenario_state_csv).resolve()
    teacher_artifact = output_dir / "oracle_teacher_artifact.mat"
    export_matlab_artifact(
        teacher_artifact,
        teacher,
        teacher_ids,
        normalized_capability_target(initial_state),
        None,
        None,
        args,
    )
    teacher_matlab_rows: list[dict[str, str]] = []
    teacher_validation = output_dir / "matlab_teacher_validation.csv"
    if not args.skip_matlab_validation and run_matlab_validation(
        teacher_artifact, scenario_csv, teacher_validation.resolve(), args
    ):
        teacher_matlab_rows = _read_csv(teacher_validation)

    if teacher_matlab_rows:
        teacher_success_ids = {
            int(float(row["scenario_id"]))
            for row in teacher_matlab_rows
            if row["controller_type"] == "teacher"
            and float(row[f"steady_H{args.eval_horizon}"]) > 0.5
        }
        failed_indices = torch.tensor(
            [
                index
                for index, scenario_id in enumerate(scenario_ids.detach().cpu().tolist())
                if int(scenario_id) not in teacher_success_ids
            ],
            device=device,
            dtype=torch.long,
        )
    else:
        scenario_python_success = torch.zeros(len(scenario_ids), device=device, dtype=torch.bool)
        for scenario_index in range(len(scenario_ids)):
            start = scenario_index * args.restarts
            scenario_python_success[scenario_index] = python_eval["final_steady"][
                start:start + args.restarts
            ].any()
        failed_indices = torch.nonzero(~scenario_python_success, as_tuple=False).flatten()

    shooting_actions: torch.Tensor | None = None
    shooting_ids: torch.Tensor | None = None
    if not args.skip_direct_shooting:
        if failed_indices.numel() > 0:
            shooting_scenario_indices = failed_indices.repeat_interleave(args.shooting_restarts)
            shooting_state = select_state(scenarios, shooting_scenario_indices)
            shooting_ids = scenario_ids.index_select(0, shooting_scenario_indices)
            shooting_actions, shooting_curve = train_direct_shooting(sim, shooting_state, args)
            _write_csv(output_dir / "direct_shooting_learning_curve.csv", shooting_curve)

    artifact = output_dir / "oracle_artifact.mat"
    export_matlab_artifact(
        artifact,
        teacher,
        teacher_ids,
        normalized_capability_target(initial_state),
        shooting_actions,
        shooting_ids,
        args,
    )
    torch.save(
        {
            "teacher": teacher.state_dict(),
            "teacher_scenario_id": teacher_ids.detach().cpu(),
            "shooting_actions": None if shooting_actions is None else shooting_actions.detach().cpu(),
            "shooting_scenario_id": None if shooting_ids is None else shooting_ids.detach().cpu(),
            "args": vars(args),
        },
        output_dir / "oracle.pt",
    )
    matlab_output = output_dir / "matlab_validation.csv"
    matlab_rows = teacher_matlab_rows
    if not args.skip_matlab_validation and shooting_actions is not None:
        if run_matlab_validation(artifact, scenario_csv, matlab_output.resolve(), args):
            matlab_rows = _read_csv(matlab_output)
    elif teacher_matlab_rows:
        _write_csv(matlab_output, teacher_matlab_rows)
    write_results(scenario_ids, teacher_ids, reached_update, python_eval, matlab_rows, args)


if __name__ == "__main__":
    main()

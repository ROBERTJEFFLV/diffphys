#!/usr/bin/env python3
"""
Pygame evaluator/replay for the current diffphys response-conditioned controller.

Recommended location inside the repository:
    diffphys/tools/eval_pygame.py

Examples:
    pip install pygame

    # Development visualization (default; does not use the reserved FINAL seed set)
    python tools/eval_pygame.py \
        --checkpoint runs/response_risk_critic_v1/seed7/best.training.pt

    # Visualize a specific scenario from the same stratified bank used by eval
    python tools/eval_pygame.py \
        --checkpoint runs/response_risk_critic_v1/seed7/best.training.pt \
        --scenario-index 17 --horizon 500 --device cuda

    # Reserved final bank (use intentionally)
    python tools/eval_pygame.py \
        --checkpoint runs/response_risk_critic_v1/seed7/candidate.pt \
        --split final --horizon 1000 --scenario-index 0

The physics and policy are NOT reimplemented here. The script calls the repository's
response_task.rollout(), L2FSimulator and checkpoint loader, then replays the resulting
trajectory. Rendering therefore cannot perturb the policy or physical rollout.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


WINDOW_W = 1280
WINDOW_H = 760
SIDE_W = 350
VIEW_W = WINDOW_W - SIDE_W
VIEW_H = WINDOW_H

BG = (18, 20, 24)
PANEL = (27, 30, 36)
GRID = (54, 59, 68)
GRID_MAJOR = (75, 81, 92)
WHITE = (232, 235, 239)
MUTED = (161, 168, 179)
RED = (231, 91, 91)
GREEN = (93, 201, 128)
BLUE = (95, 151, 235)
YELLOW = (231, 190, 87)
CYAN = (87, 204, 213)
PURPLE = (184, 128, 231)
ORANGE = (231, 145, 83)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="response-task checkpoint, e.g. best.pt or latest.pt")
    p.add_argument("--repo-root", type=Path,
                   help="diffphys repository root; normally auto-detected")
    p.add_argument("--split", choices=("development", "final"), default="development",
                   help="seed family to replay; development is the safe default")
    p.add_argument("--seed-index", type=int, default=0,
                   help="0 or 1 within the selected split's official seed tuple")
    p.add_argument("--seed", type=int,
                   help="explicit seed override; normally leave unset")
    p.add_argument("--scenario-index", type=int, default=0,
                   help="scenario row within the checkpoint's official stratified bank")
    p.add_argument("--horizon", type=int,
                   help="physical steps; default is the checkpoint training/eval horizon")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--fps", type=int, default=60, help="display refresh rate")
    p.add_argument("--speed", type=float, default=1.0, help="initial replay speed")
    p.add_argument("--start-paused", action="store_true")
    p.add_argument("--allow-stale-source", action="store_true",
                   help="permit historical checkpoints whose source hash differs from current checkout")
    p.add_argument("--screenshot-dir", type=Path, default=Path("reports/pygame_eval"))
    args = p.parse_args()

    if args.seed_index not in (0, 1):
        p.error("--seed-index must be 0 or 1")
    if args.scenario_index < 0:
        p.error("--scenario-index must be non-negative")
    if args.horizon is not None and args.horizon < 1:
        p.error("--horizon must be positive")
    if args.threads < 1 or args.fps < 1 or not math.isfinite(args.speed) or args.speed <= 0:
        p.error("--threads/--fps must be positive and --speed must be finite and positive")
    return args


def find_repo_root(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.resolve())

    candidates.append(Path.cwd().resolve())
    here = Path(__file__).resolve()
    candidates.extend(here.parents[:5])

    for candidate in candidates:
        if (candidate / "response_task.py").is_file() and (candidate / "env_l2f.py").is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate diffphys. Put this file in diffphys/tools/, run it from the "
        "repository root, or pass --repo-root /path/to/diffphys."
    )


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return torch.device(name)


def checkpoint_loss_config(saved: dict[str, Any], TaskLossConfig):
    protocol = saved.get("binding", {}).get("protocol", {})
    loss = protocol.get("loss")
    if isinstance(loss, dict):
        return TaskLossConfig(**loss)
    return TaskLossConfig()


def load_exact_eval(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from env_l2f import L2FParams, L2FSimulator
    from response_task import TaskLossConfig, rollout, sample_scenarios, trajectory_metrics
    from response_training import (
        DEVELOPMENT_SEEDS,
        FINAL_SEEDS,
        PROTOCOL_VERSION,
        load_policy_checkpoint,
        model_hash,
        source_hash,
    )

    torch.set_num_threads(args.threads)
    device = choose_device(args.device)

    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    raw = torch.load(checkpoint, map_location="cpu")
    if raw.get("schema") != PROTOCOL_VERSION:
        raise ValueError(
            f"{checkpoint.name} is not a current response-task checkpoint "
            f"(expected schema {PROTOCOL_VERSION!r}, got {raw.get('schema')!r})"
        )

    binding = raw.get("binding", {})
    stored_source = binding.get("source_sha256")
    current_source = source_hash()
    if stored_source != current_source and not args.allow_stale_source:
        raise ValueError(
            "checkpoint source/protocol is stale relative to this checkout. "
            "Use the matching repository revision, or pass --allow-stale-source "
            "only for visualization of a historical checkpoint."
        )

    dtype_name = binding.get("dtype", "float32")
    if not hasattr(torch, dtype_name):
        raise ValueError(f"unsupported checkpoint dtype: {dtype_name}")
    dtype = getattr(torch, dtype_name)

    policy, saved = load_policy_checkpoint(checkpoint, device, dtype)
    policy.eval()
    if "model_sha256" in saved and model_hash(policy) != saved["model_sha256"]:
        raise ValueError("checkpoint model digest mismatch")

    loss_config = checkpoint_loss_config(saved, TaskLossConfig)

    protocol = binding.get("protocol", {})
    bank_size = int(protocol.get("scenarios_per_bank", 64))
    if bank_size < 16 or bank_size % 16:
        raise ValueError(f"checkpoint has invalid scenario-bank size: {bank_size}")
    if args.scenario_index >= bank_size:
        raise ValueError(
            f"--scenario-index {args.scenario_index} is outside checkpoint bank [0,{bank_size - 1}]"
        )

    official_seeds = FINAL_SEEDS if args.split == "final" else DEVELOPMENT_SEEDS
    seed = int(args.seed if args.seed is not None else official_seeds[args.seed_index])

    default_horizon = int(binding.get("horizon", 500))
    horizon = int(args.horizon if args.horizon is not None else default_horizon)

    # Important: roll out the COMPLETE official bank, then select one row for display.
    # This preserves the eval batch shape and avoids a visualization-only single-row
    # GEMM path producing slightly different floating-point arithmetic.
    initial, cells = sample_scenarios(
        bank_size, seed=seed, dt=policy.config.dt, device=device, dtype=dtype,
        scenario_mode=protocol.get("scenario_mode", "fixed-airframe"),
    )
    simulator = L2FSimulator(L2FParams(dt=policy.config.dt))

    with torch.no_grad():
        trace = rollout(policy, simulator, initial, horizon)
        bank_metrics = trajectory_metrics(trace, loss_config)

    i = args.scenario_index
    obs = trace.observations[:, i].detach().cpu().numpy()
    actions = obs[:, 21:25].copy()  # previous executed action aligned with each state

    positions = obs[:, 0:3].copy()
    velocities = obs[:, 3:6].copy()
    rotations = obs[:, 6:15].reshape(-1, 3, 3).copy()
    omegas = obs[:, 15:18].copy()

    # Official single-scene final-window criterion from response_task.trajectory_metrics().
    tail = min(int(loss_config.steady_steps), horizon)
    p_norm = np.linalg.norm(positions[1:], axis=1)
    v_norm = np.linalg.norm(velocities[1:], axis=1)
    w_norm = np.linalg.norm(omegas[1:], axis=1)
    finite_scene = (
        np.isfinite(positions).all()
        and np.isfinite(velocities).all()
        and np.isfinite(rotations).all()
        and np.isfinite(omegas).all()
        and np.isfinite(actions).all()
    )
    scene_success = bool(
        finite_scene
        and np.all(p_norm[-tail:] < 0.05)
        and np.all(v_norm[-tail:] < 0.10)
        and np.all(w_norm[-tail:] < 0.50)
    )

    def scalar(field_name: str) -> float:
        value = getattr(initial, field_name)[i].detach().cpu()
        return float(value.reshape(-1)[0])

    external_force = initial.external_force[i].detach().cpu().numpy().astype(float)
    mass = scalar("mass")
    arm_length = scalar("arm_length")

    cell = cells[i].detach().cpu().tolist()
    return {
        "root": root,
        "checkpoint": checkpoint,
        "saved": saved,
        "split": args.split,
        "seed": seed,
        "scenario_index": i,
        "bank_size": bank_size,
        "cell": cell,
        "horizon": horizon,
        "dt": float(policy.config.dt),
        "positions": positions,
        "velocities": velocities,
        "rotations": rotations,
        "omegas": omegas,
        "actions": actions,
        "external_force": external_force,
        "mass": mass,
        "arm_length": arm_length,
        "thrust_to_weight": scalar("thrust_to_weight"),
        "alpha_roll_max": scalar("alpha_roll_max"),
        "eta_yaw": scalar("eta_yaw"),
        "tau_rise": scalar("motor_time_rising"),
        "tau_fall": scalar("motor_time_falling"),
        "finite_scene": finite_scene,
        "scene_success": scene_success,
        "steady_steps": int(loss_config.steady_steps),
        "bank_metrics": bank_metrics,
        "source_stale": stored_source != current_source,
    }


class Camera:
    def __init__(self, data: dict[str, Any]) -> None:
        self.yaw = math.radians(43.0)
        self.pitch = math.radians(29.0)
        self.zoom = 1.0
        self.follow = False

        pos = np.asarray(data["positions"], dtype=float)
        finite = np.isfinite(pos).all(axis=1)
        if finite.any():
            radial = np.linalg.norm(pos[finite], axis=1)
            q = float(np.percentile(radial, 98))
        else:
            q = 1.0
        self.extent = float(np.clip(max(1.5, q + 0.6), 1.5, 20.0))
        self.base_scale = min(VIEW_W, VIEW_H) / (2.55 * self.extent)

    def project(self, points: np.ndarray, center: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        rel = np.nan_to_num(pts - center.reshape(1, 3), nan=0.0, posinf=50.0, neginf=-50.0)

        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)

        # Yaw about world z, then pitch for an orthographic/isometric camera.
        x1 = cy * rel[:, 0] + sy * rel[:, 1]
        y1 = -sy * rel[:, 0] + cy * rel[:, 1]
        screen_y_axis = cp * rel[:, 2] - sp * y1

        scale = self.base_scale * self.zoom
        sx = VIEW_W * 0.5 + scale * x1
        sy_screen = VIEW_H * 0.52 - scale * screen_y_axis
        return np.stack((sx, sy_screen), axis=1)


def safe_rotation(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        return np.eye(3)
    return r


def clamp_vec_length(v: np.ndarray, max_len: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if not np.isfinite(v).all():
        return np.zeros(3)
    n = float(np.linalg.norm(v))
    if n <= max_len or n < 1e-12:
        return v
    return v * (max_len / n)


def polyline(pg, surface, color, points, width=1):
    if len(points) < 2:
        return
    valid = []
    for p in points:
        if np.isfinite(p).all():
            valid.append((int(round(p[0])), int(round(p[1]))))
    if len(valid) >= 2:
        pg.draw.lines(surface, color, False, valid, width)


def draw_arrow(pg, surface, color, start, end, width=2):
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)
    if not (np.isfinite(s).all() and np.isfinite(e).all()):
        return
    si = (int(round(s[0])), int(round(s[1])))
    ei = (int(round(e[0])), int(round(e[1])))
    pg.draw.line(surface, color, si, ei, width)
    d = e - s
    n = float(np.linalg.norm(d))
    if n < 3:
        return
    u = d / n
    perp = np.array([-u[1], u[0]])
    head = 9.0
    p1 = e - head * u + 0.55 * head * perp
    p2 = e - head * u - 0.55 * head * perp
    pg.draw.polygon(surface, color, [ei, tuple(p1.astype(int)), tuple(p2.astype(int))])


def draw_text(surface, font, text, pos, color=WHITE):
    surface.blit(font.render(str(text), True, color), pos)


def draw_grid(pg, surface, camera: Camera, center: np.ndarray, enabled: bool):
    if not enabled:
        return
    extent = int(math.ceil(camera.extent))
    spacing = 0.5 if camera.extent <= 4.0 else 1.0
    values = np.arange(-extent, extent + 1e-9, spacing)

    for x in values:
        pts = camera.project(np.array([[x, -extent, 0.0], [x, extent, 0.0]]), center)
        major = abs(x - round(x)) < 1e-8
        polyline(pg, surface, GRID_MAJOR if major else GRID, pts, 1)
    for y in values:
        pts = camera.project(np.array([[-extent, y, 0.0], [extent, y, 0.0]]), center)
        major = abs(y - round(y)) < 1e-8
        polyline(pg, surface, GRID_MAJOR if major else GRID, pts, 1)

    origin = camera.project(np.zeros((1, 3)), center)[0]
    pg.draw.circle(surface, YELLOW, tuple(origin.astype(int)), 7, 2)
    pg.draw.circle(surface, YELLOW, tuple(origin.astype(int)), 2)


def draw_world_axes(pg, surface, camera: Camera, center: np.ndarray):
    origin = np.zeros(3)
    axes = (
        (np.array([0.45, 0, 0]), RED),
        (np.array([0, 0.45, 0]), GREEN),
        (np.array([0, 0, 0.45]), BLUE),
    )
    o = camera.project(origin[None, :], center)[0]
    for endpoint, color in axes:
        e = camera.project(endpoint[None, :], center)[0]
        draw_arrow(pg, surface, color, o, e, 2)


def draw_drone(pg, surface, data: dict[str, Any], camera: Camera, idx: int,
               center: np.ndarray, show_velocity: bool, show_force: bool):
    p = np.asarray(data["positions"][idx], dtype=float)
    if not np.isfinite(p).all():
        return

    r = safe_rotation(data["rotations"][idx])
    arm = max(float(data["arm_length"]), 1e-4)

    # Same body-frame convention as matlab_l2f/l2f_replay_3d.m:
    # use the first two rotation-matrix columns as the two vehicle arms.
    x_arm = r[:, 0] * arm
    y_arm = r[:, 1] * arm

    pairs = [
        (p - x_arm, p + x_arm, RED),
        (p - y_arm, p + y_arm, GREEN),
    ]
    for a, b, color in pairs:
        pts = camera.project(np.stack((a, b)), center)
        polyline(pg, surface, color, pts, 4)
        for q in pts:
            pg.draw.circle(surface, WHITE, tuple(q.astype(int)), 5, 2)

    body = camera.project(p[None, :], center)[0]
    pg.draw.circle(surface, WHITE, tuple(body.astype(int)), 6)

    # Shadow gives altitude cues without inventing any extra physics.
    shadow_world = np.array([p[0], p[1], 0.0])
    shadow = camera.project(shadow_world[None, :], center)[0]
    pg.draw.circle(surface, MUTED, tuple(shadow.astype(int)), 5, 1)

    if show_velocity:
        v = clamp_vec_length(np.asarray(data["velocities"][idx]), 4.0)
        endpoint = p + 0.25 * v
        pts = camera.project(np.stack((p, endpoint)), center)
        draw_arrow(pg, surface, CYAN, pts[0], pts[1], 2)

    if show_force:
        f = np.asarray(data["external_force"], dtype=float)
        mass = max(float(data["mass"]), 1e-9)
        # Display vector only: F/(mg) is dimensionless, mapped to 0.6 m visual length.
        ratio = f / (mass * 9.80665)
        ratio = clamp_vec_length(ratio, 2.0)
        endpoint = p + 0.6 * ratio
        pts = camera.project(np.stack((p, endpoint)), center)
        draw_arrow(pg, surface, ORANGE, pts[0], pts[1], 3)


def draw_trail(pg, surface, data: dict[str, Any], camera: Camera, idx: int,
               center: np.ndarray, enabled: bool):
    if not enabled or idx < 1:
        return
    path = np.asarray(data["positions"][: idx + 1], dtype=float)
    if len(path) > 1400:
        stride = int(math.ceil(len(path) / 1400))
        path = path[::stride]
    points = camera.project(path, center)
    polyline(pg, surface, PURPLE, points, 2)


def draw_action_bars(pg, surface, font_small, actions: np.ndarray, x: int, y: int, width: int):
    bar_h = 16
    gap = 8
    for j, value in enumerate(actions):
        yy = y + j * (bar_h + gap)
        draw_text(surface, font_small, f"u{j}", (x, yy - 1), MUTED)
        bx = x + 28
        bw = width - 28
        pg.draw.rect(surface, GRID, (bx, yy, bw, bar_h), border_radius=3)
        mid = bx + bw // 2
        pg.draw.line(surface, MUTED, (mid, yy), (mid, yy + bar_h), 1)
        val = float(np.clip(value, -1.0, 1.0))
        if val >= 0:
            rect = (mid, yy + 2, int((bw / 2 - 2) * val), bar_h - 4)
        else:
            w = int((bw / 2 - 2) * (-val))
            rect = (mid - w, yy + 2, w, bar_h - 4)
        if rect[2] > 0:
            pg.draw.rect(surface, CYAN, rect, border_radius=2)


def fmt_metric(metrics: dict[str, Any], key: str, fmt: str = ".4f") -> str:
    value = metrics.get(key)
    if value is None:
        return "n/a"
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def draw_panel(pg, surface, data: dict[str, Any], idx: int, font, font_small, font_big,
               speed: float, paused: bool):
    x0 = VIEW_W
    pg.draw.rect(surface, PANEL, (x0, 0, SIDE_W, WINDOW_H))
    pad = 18
    x = x0 + pad
    y = 18

    draw_text(surface, font_big, "diffphys eval", (x, y))
    y += 36
    draw_text(surface, font_small, data["checkpoint"].name, (x, y), MUTED)
    y += 23
    draw_text(
        surface, font_small,
        f"{data['split']} seed={data['seed']}  scene={data['scenario_index']}/{data['bank_size']-1}",
        (x, y), MUTED
    )
    y += 21
    draw_text(surface, font_small, f"stratum cell={data['cell']}", (x, y), MUTED)
    y += 27

    t = idx * data["dt"]
    state = "PAUSED" if paused else f"{speed:g}x"
    draw_text(surface, font, f"t = {t:7.3f} s   step {idx:4d}/{data['horizon']}   {state}", (x, y))
    y += 30

    p = np.asarray(data["positions"][idx])
    v = np.asarray(data["velocities"][idx])
    w = np.asarray(data["omegas"][idx])
    pn = float(np.linalg.norm(p)) if np.isfinite(p).all() else float("inf")
    vn = float(np.linalg.norm(v)) if np.isfinite(v).all() else float("inf")
    wn = float(np.linalg.norm(w)) if np.isfinite(w).all() else float("inf")

    current_ok = pn < 0.05 and vn < 0.10 and wn < 0.50
    color = GREEN if current_ok else YELLOW
    draw_text(surface, font, f"|p| {pn:8.4f} m", (x, y), color)
    y += 25
    draw_text(surface, font, f"|v| {vn:8.4f} m/s", (x, y), color)
    y += 25
    draw_text(surface, font, f"|w| {wn:8.4f} rad/s", (x, y), color)
    y += 26
    draw_text(surface, font_small, "instant band: p<.05, v<.10, w<.50", (x, y), MUTED)
    y += 31

    draw_text(surface, font, "executed motor command", (x, y))
    y += 27
    draw_action_bars(pg, surface, font_small, data["actions"][idx], x, y, SIDE_W - 2 * pad)
    y += 4 * 24 + 15

    draw_text(surface, font, "episode dynamics  [viz only]", (x, y))
    y += 25
    dyn = [
        f"mass/arm   {data['mass']:.4f} kg / {data['arm_length']:.4f} m",
        f"T/W / aR   {data['thrust_to_weight']:.3f} / {data['alpha_roll_max']:.1f} rad/s^2",
        f"eta / tau  {data['eta_yaw']:.4f} / {data['tau_rise']:.3f},{data['tau_fall']:.3f} s",
        "Fext       [" + ", ".join(f"{q:+.3f}" for q in data["external_force"]) + "] N",
    ]
    for line in dyn:
        draw_text(surface, font_small, line, (x, y), MUTED)
        y += 20
    y += 9

    metrics = data["bank_metrics"]
    draw_text(surface, font, "official-bank metrics", (x, y))
    y += 25
    rows = [
        ("task objective", fmt_metric(metrics, "task_objective")),
        ("position RMS", fmt_metric(metrics, "position_rms")),
        ("velocity RMS", fmt_metric(metrics, "velocity_rms")),
        ("omega RMS", fmt_metric(metrics, "omega_rms")),
        ("steady success", fmt_metric(metrics, "steady_success_rate", ".3f")),
        ("motor saturation", fmt_metric(metrics, "motor_saturation_fraction", ".4f")),
    ]
    for label, value in rows:
        draw_text(surface, font_small, f"{label:<17} {value}", (x, y), MUTED)
        y += 19

    y += 5
    final_label = "SCENE FINAL: PASS" if data["scene_success"] else "SCENE FINAL: FAIL"
    final_color = GREEN if data["scene_success"] else RED
    draw_text(surface, font, final_label, (x, y), final_color)
    y += 23
    draw_text(surface, font_small, f"last {min(data['steady_steps'], data['horizon'])} physical steps", (x, y), MUTED)
    if data["source_stale"]:
        y += 24
        draw_text(surface, font_small, "WARNING: source hash differs", (x, y), ORANGE)


def draw_progress(pg, surface, idx: int, horizon: int):
    margin = 18
    width = VIEW_W - 2 * margin
    y = WINDOW_H - 20
    pg.draw.rect(surface, GRID, (margin, y, width, 5), border_radius=2)
    frac = 0.0 if horizon <= 0 else idx / horizon
    pg.draw.rect(surface, CYAN, (margin, y, int(width * frac), 5), border_radius=2)


def run_pygame(args: argparse.Namespace, data: dict[str, Any]) -> None:
    try:
        import pygame as pg
    except ImportError as exc:
        raise RuntimeError(
            "Pygame is not installed. Install it in the same environment with: pip install pygame"
        ) from exc

    pg.init()
    pg.display.set_caption("diffphys Pygame Eval")
    surface = pg.display.set_mode((WINDOW_W, WINDOW_H), pg.RESIZABLE)
    clock = pg.time.Clock()
    font_small = pg.font.SysFont("consolas", 14)
    font = pg.font.SysFont("consolas", 16)
    font_big = pg.font.SysFont("consolas", 22, bold=True)

    camera = Camera(data)
    idx = 0
    sim_time = 0.0
    paused = bool(args.start_paused)
    speed = float(args.speed)

    show_trail = True
    show_grid_flag = True
    show_velocity = True
    show_force = True

    last_tick = time.perf_counter()
    running = True
    while running:
        now = time.perf_counter()
        real_dt = min(now - last_tick, 0.1)
        last_tick = now

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                elif event.key == pg.K_SPACE:
                    paused = not paused
                    sim_time = idx * data["dt"]
                elif event.key == pg.K_r:
                    idx = 0
                    sim_time = 0.0
                elif event.key == pg.K_HOME:
                    idx = 0
                    sim_time = 0.0
                    paused = True
                elif event.key == pg.K_END:
                    idx = data["horizon"]
                    sim_time = idx * data["dt"]
                    paused = True
                elif event.key == pg.K_LEFT:
                    idx = max(0, idx - 1)
                    sim_time = idx * data["dt"]
                    paused = True
                elif event.key == pg.K_RIGHT:
                    idx = min(data["horizon"], idx + 1)
                    sim_time = idx * data["dt"]
                    paused = True
                elif event.key == pg.K_f:
                    camera.follow = not camera.follow
                elif event.key == pg.K_t:
                    show_trail = not show_trail
                elif event.key == pg.K_g:
                    show_grid_flag = not show_grid_flag
                elif event.key == pg.K_v:
                    show_velocity = not show_velocity
                elif event.key == pg.K_x:
                    show_force = not show_force
                elif event.key in (pg.K_EQUALS, pg.K_KP_PLUS):
                    camera.zoom = min(camera.zoom * 1.15, 8.0)
                elif event.key in (pg.K_MINUS, pg.K_KP_MINUS):
                    camera.zoom = max(camera.zoom / 1.15, 0.15)
                elif event.key == pg.K_LEFTBRACKET:
                    speed = max(0.125, speed / 2.0)
                elif event.key == pg.K_RIGHTBRACKET:
                    speed = min(32.0, speed * 2.0)
                elif event.key == pg.K_p:
                    args.screenshot_dir.mkdir(parents=True, exist_ok=True)
                    out = args.screenshot_dir / (
                        f"{data['split']}_seed{data['seed']}_scene{data['scenario_index']}_step{idx}.png"
                    )
                    pg.image.save(surface, str(out))
                    print(f"[eval_pygame] screenshot: {out}", flush=True)
            elif event.type == pg.MOUSEWHEEL:
                factor = 1.12 ** event.y
                camera.zoom = float(np.clip(camera.zoom * factor, 0.15, 8.0))

        keys = pg.key.get_pressed()
        camera.yaw += (float(keys[pg.K_d]) - float(keys[pg.K_a])) * real_dt * 1.25
        camera.pitch += (float(keys[pg.K_w]) - float(keys[pg.K_s])) * real_dt * 0.9
        camera.pitch = float(np.clip(camera.pitch, math.radians(-75), math.radians(75)))

        if not paused:
            sim_time += real_dt * speed
            idx = min(data["horizon"], int(sim_time / data["dt"]))
            if idx >= data["horizon"]:
                paused = True
                sim_time = idx * data["dt"]

        # Render against the designed 1280x760 canvas, then scale if the user resized.
        canvas = pg.Surface((WINDOW_W, WINDOW_H))
        canvas.fill(BG)

        p = np.asarray(data["positions"][idx], dtype=float)
        center = p if camera.follow and np.isfinite(p).all() else np.zeros(3)

        draw_grid(pg, canvas, camera, center, show_grid_flag)
        draw_world_axes(pg, canvas, camera, center)
        draw_trail(pg, canvas, data, camera, idx, center, show_trail)
        draw_drone(pg, canvas, data, camera, idx, center, show_velocity, show_force)

        draw_text(
            canvas, font_small,
            "SPACE pause | LEFT/RIGHT step | R restart | A/D yaw W/S pitch | +/- zoom | "
            "[/] speed | F follow | T trail | G grid | V velocity | X force | P screenshot",
            (18, 15), MUTED
        )
        draw_progress(pg, canvas, idx, data["horizon"])
        draw_panel(pg, canvas, data, idx, font, font_small, font_big, speed, paused)

        current_size = surface.get_size()
        if current_size == (WINDOW_W, WINDOW_H):
            surface.blit(canvas, (0, 0))
        else:
            scaled = pg.transform.smoothscale(canvas, current_size)
            surface.blit(scaled, (0, 0))
        pg.display.flip()
        clock.tick(args.fps)

    pg.quit()


def main() -> int:
    args = parse_args()
    root = find_repo_root(args.repo_root)
    data = load_exact_eval(args, root)

    print(
        "[eval_pygame] "
        f"checkpoint={data['checkpoint']} split={data['split']} seed={data['seed']} "
        f"bank={data['bank_size']} scenario={data['scenario_index']} cell={data['cell']} "
        f"horizon={data['horizon']} dt={data['dt']:.6g} "
        f"scene_success={data['scene_success']}",
        flush=True,
    )
    print(
        "[eval_pygame] bank metrics: "
        + ", ".join(
            f"{k}={v}"
            for k, v in data["bank_metrics"].items()
            if isinstance(v, (bool, int, float))
        ),
        flush=True,
    )

    run_pygame(args, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

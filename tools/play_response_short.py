#!/usr/bin/env python3
"""Play a saved fixed-EVAL flight without Torch or policy inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


FIELDS = ("position", "velocity", "orientation", "omega", "action", "valid",
          "rotor_positions", "mass_kg", "arm_length_m", "thrust_to_weight",
          "torque_to_inertia", "position_limit")
FORCE_FIELDS = ("external_force_world", "pulse_force_world", "pulse_point_body", "pulse_active")
GOAL_FIELDS = ("command_position", "command_rotation", "command_is_stop",
               "mission_points", "pass_rotation", "stop_rotation")
REGION_FIELDS = ("command_phase", "command_min_speed_m_s", "region_radius_m")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_replay(path):
    path = Path(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("replay_type") != "short-eval-v1":
        raise ValueError("expected a short-EVAL replay")
    task = meta.get("task", "hover")
    if task not in ("hover", "pass-stop"):
        raise ValueError("unsupported short-EVAL task")
    filename = meta.get("trajectory_file")
    if (not isinstance(filename, str) or Path(filename).name != filename
            or filename in ("", ".", "..")):
        raise ValueError("invalid trajectory filename")
    data_path = path.parent / filename
    if _sha256(data_path) != meta.get("trajectory_sha256"):
        raise ValueError("short-EVAL trajectory checksum mismatch")
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in FIELDS}
        # Old short replays remain readable; they explicitly show that force
        # telemetry was not recorded instead of implying zero disturbance.
        present = [name in archive for name in FORCE_FIELDS]
        if any(present) and not all(present):
            raise ValueError("incomplete short-EVAL force telemetry")
        if all(present):
            arrays.update((name, archive[name]) for name in FORCE_FIELDS)
        if task == "pass-stop":
            if not all(name in archive for name in GOAL_FIELDS):
                raise ValueError("incomplete PASS/STOP typed command telemetry")
            arrays.update((name, archive[name]) for name in GOAL_FIELDS)
            present = [name in archive for name in REGION_FIELDS]
            minimum_speed_task = (meta.get("mission") or {}).get("version") == "pass-region-min-speed-v3"
            if (any(present) or minimum_speed_task) and not all(present):
                raise ValueError("incomplete PASS/STOP region telemetry")
            if all(present):
                arrays.update((name, archive[name]) for name in REGION_FIELDS)
    count = len(meta.get("scenes", ()))
    horizon = arrays["valid"].shape[0]
    if (count < 1 or horizon < 1 or arrays["valid"].shape != (horizon, count)
            or arrays["orientation"].shape != (horizon + 1, count, 4)
            or any(arrays[name].shape != (horizon + 1, count, 3)
                   for name in ("position", "velocity", "omega"))
            or arrays["action"].shape != (horizon, count, 4)
            or arrays["rotor_positions"].shape != (count, 4, 3)
            or not np.isfinite(arrays["position"]).all()
            or not np.isfinite(arrays["orientation"]).all()):
        raise ValueError("short-EVAL trajectory shape or value mismatch")
    if FORCE_FIELDS[0] in arrays and (
        arrays["external_force_world"].shape != (count, 3)
        or any(arrays[name].shape != (horizon, count, 3)
               for name in ("pulse_force_world", "pulse_point_body"))
        or arrays["pulse_active"].shape != (horizon, count)
        or any(not np.isfinite(arrays[name]).all()
               for name in FORCE_FIELDS[:3])
    ):
        raise ValueError("short-EVAL force telemetry shape or value mismatch")
    if task == "pass-stop":
        mission = meta.get("mission")
        if (not isinstance(mission, dict)
                or any(not isinstance(mission.get(name), (int, float))
                       or not math.isfinite(mission[name]) or mission[name] <= 0
                       for name in ("radius_m", "pose_tolerance_deg"))
                or arrays["command_position"].shape != (horizon, count, 3)
                or arrays["command_rotation"].shape != (horizon, count, 3, 3)
                or arrays["command_is_stop"].shape != (horizon, count)
                or arrays["command_is_stop"].dtype != np.bool_
                or arrays["mission_points"].shape != (count, 3, 3)
                or any(arrays[name].shape != (count, 3, 3)
                       for name in ("pass_rotation", "stop_rotation"))
                or any(not np.isfinite(arrays[name]).all() for name in GOAL_FIELDS)):
            raise ValueError("PASS/STOP typed telemetry shape or value mismatch")
        if REGION_FIELDS[0] in arrays:
            phase = arrays['command_phase']
            floor = arrays['command_min_speed_m_s']
            radius = arrays['region_radius_m']
            if (phase.shape != (horizon, count) or not np.issubdtype(phase.dtype, np.integer)
                    or not np.isin(phase, (0, 1, 2)).all()
                    or not np.array_equal(phase == 2, arrays['command_is_stop'])
                    or floor.shape != (horizon, count) or not np.isfinite(floor).all()
                    or (floor[phase != 2] <= 0).any() or (floor[phase == 2] != 0).any()
                    or radius.shape != (count,) or not np.isfinite(radius).all()
                    or (radius <= 0).any()):
                raise ValueError("PASS/STOP region telemetry shape or value mismatch")
    return meta, arrays


def goal_status(meta, arrays, frame, scene):
    """Public command data for the HUD; old replays keep their original protocol."""
    command_frame = min(frame, len(arrays['valid'])-1)
    mission = meta['mission']
    is_stop = bool(arrays['command_is_stop'][command_frame, scene])
    phase = (int(arrays['command_phase'][command_frame, scene])
             if 'command_phase' in arrays else (2 if is_stop else None))
    radius = (float(arrays['region_radius_m'][scene]) if 'region_radius_m' in arrays
              else float(mission.get('region_radius_arms', 0)*arrays['arm_length_m'][scene]
                         or mission['radius_m']))
    floor = (float(arrays['command_min_speed_m_s'][command_frame, scene])
             if 'command_min_speed_m_s' in arrays else None)
    speed = float(np.linalg.norm(arrays['velocity'][frame, scene]))
    return {'phase': phase,
            'phase_name': {0: '接近点2', 1: '球内穿越', 2: '终点停稳'}.get(phase, 'PASS #2（旧回放）'),
            'region_radius_m': radius, 'center_tolerance_m': float(mission['radius_m']),
            'minimum_speed_m_s': floor, 'speed_m_s': speed,
            'speed_shortfall_m_s': None if floor is None else max(0., floor-speed)}


def quaternion_rotation(q):
    w, x, y, z = q
    return np.array(((1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)),
                     (2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)),
                     (2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y))))


def force_at_frame(arrays, scene, frame):
    """Physical forces applied during frame -> frame + 1, or None if absent."""
    if FORCE_FIELDS[0] not in arrays:
        return None
    horizon = arrays["valid"].shape[0]
    applied = frame < horizon and bool(arrays["valid"][frame, scene])
    static = arrays["external_force_world"][scene]
    pulse = arrays["pulse_force_world"][frame, scene] if applied else np.zeros_like(static)
    point = arrays["pulse_point_body"][frame, scene] if applied else np.zeros_like(static)
    return {
        "applied": applied,
        "external_force_world": static if applied else np.zeros_like(static),
        "pulse_force_world": pulse,
        "pulse_point_body": point,
        "pulse_active": bool(arrays["pulse_active"][frame, scene]) if applied else False,
        "total_force_world": static + pulse if applied else np.zeros_like(static),
    }


class Playback:
    def __init__(self, meta):
        self.meta = meta
        self.index = 0
        self.seconds = 0.0
        self.paused = False
        self.speed = 1.0
        self.auto = False
        self.hold = 0.0

    @property
    def scene(self):
        return self.meta["scenes"][self.index]

    @property
    def end_seconds(self):
        return self.scene["valid_steps"] * self.meta["dt"]

    @property
    def frame(self):
        return min(self.scene["valid_steps"], int(self.seconds / self.meta["dt"] + 1e-7))

    def choose(self, index):
        self.index = index % len(self.meta["scenes"])
        self.seconds = 0.0
        self.hold = 0.0

    def seek(self, seconds):
        self.seconds = min(max(float(seconds), 0.0), self.end_seconds)
        self.hold = 0.0

    def advance(self, elapsed):
        if self.paused:
            return
        if self.seconds < self.end_seconds:
            self.seconds = min(self.end_seconds, self.seconds + elapsed * self.speed)
        elif self.auto:
            self.hold += elapsed
            if self.hold >= 1.5:
                self.choose(self.index + 1)


def run_player(path, *, screenshot=None, max_frames=None, fps=30):
    import pygame as pg

    meta, arrays = load_replay(path)
    typed = meta.get("task") == "pass-stop"
    pg.display.init()
    pg.font.init()
    screen = pg.display.set_mode((1500, 900))
    pg.display.set_caption(f"DiffPhys | fixed EVAL #{meta['checkpoint_update']} | {meta['duration_seconds']:g}s")
    font_path = pg.font.match_font("notosanscjksc,notosanscjk,dejavusans,arial")
    fonts = {size: pg.font.Font(font_path, size) for size in (14, 16, 18, 22, 28)}
    clock = pg.time.Clock()
    state = Playback(meta)
    yaw, elevation, zoom = -.78, .42, 1.0
    page = 0
    buttons = []
    timeline = pg.Rect(27, 805, 1446, 18)
    colors = {
        "background": (15, 21, 31), "panel": (23, 32, 45),
        "line": (60, 75, 91), "white": (231, 238, 246),
        "muted": (157, 178, 196), "green": (91, 220, 166),
        "red": (252, 115, 118), "gold": (255, 207, 93),
        "steady_force": (255, 166, 82), "pulse_force": (220, 137, 255),
    }

    def text(value, x, y, size=18, color=None):
        screen.blit(fonts[size].render(str(value), True, color or colors["white"]), (x, y))

    def project(points, limit):
        points = np.asarray(points)
        co, si = math.cos(yaw), math.sin(yaw)
        horizontal = co * points[..., 0] - si * points[..., 1]
        depth = si * points[..., 0] + co * points[..., 1]
        vertical = math.cos(elevation) * points[..., 2] - math.sin(elevation) * depth
        scale = 225 * zoom / max(limit, 1.0)
        return np.stack((510 + scale * horizontal, 425 - scale * vertical), axis=-1)

    def line(points, color, width=2):
        pixels = np.asarray(points)
        if len(pixels) > 1 and np.isfinite(pixels).all():
            pg.draw.lines(screen, color, False, pixels.astype(int).tolist(), width)

    def force_arrow(origin, force, gravity_fraction, limit, color):
        magnitude = float(np.linalg.norm(force))
        if magnitude <= 1e-10:
            return
        start = project(origin, limit)
        projected = project(origin + force / magnitude, limit) - start
        projected_norm = float(np.linalg.norm(projected))
        if projected_norm <= 1e-5:
            pg.draw.circle(screen, color, start.astype(int), 8, 2)
            return
        direction = projected / projected_norm
        length = float(np.clip(26 + 65 * gravity_fraction / .2, 26, 105))
        end = start + direction * length
        wing = np.array((-direction[1], direction[0]))
        pg.draw.line(screen, color, start.astype(int), end.astype(int), 4)
        pg.draw.polygon(screen, color, [end.astype(int),
                                        (end - 13*direction + 6*wing).astype(int),
                                        (end - 13*direction - 6*wing).astype(int)])

    def button(label, rect, action):
        rect = pg.Rect(rect)
        buttons.append((rect, action))
        pg.draw.rect(screen, (35, 49, 64), rect, border_radius=5)
        pg.draw.rect(screen, colors["line"], rect, width=1, border_radius=5)
        surface = fonts[16].render(label, True, colors["white"])
        screen.blit(surface, surface.get_rect(center=rect.center))

    def handle(action):
        nonlocal page, yaw, elevation, zoom
        if action == "previous":
            state.choose(state.index - 1)
            page = state.index // 18
        elif action == "next":
            state.choose(state.index + 1)
            page = state.index // 18
        elif action == "pause":
            state.paused = not state.paused
        elif action == "replay":
            state.seek(0)
        elif action == "auto":
            state.auto = not state.auto
        elif action == "page_back":
            page = max(0, page - 1)
        elif action == "page_forward":
            page = min((len(meta["scenes"]) - 1) // 18, page + 1)
        elif action == "slower":
            state.speed = max(.125, state.speed / 2)
        elif action == "faster":
            state.speed = min(8., state.speed * 2)
        elif action == "camera":
            yaw, elevation, zoom = -.78, .42, 1.0
        elif action.startswith("scene:"):
            state.choose(int(action.split(":", 1)[1]))

    def draw():
        buttons.clear()
        screen.fill(colors["background"])
        pg.draw.rect(screen, colors["panel"], (1002, 0, 498, 784))
        task_name = "PASS/STOP" if typed else "悬停"
        text(f"固定 EVAL · {task_name} · {meta['duration_seconds']:g} 秒轨迹", 27, 19, 28)
        text(f"Actor #{meta['checkpoint_update']}   {len(meta['scenes'])} 架   H{len(arrays['valid'])}",
             28, 57, 16, colors["muted"])
        scene = state.scene
        index = state.index
        frame = state.frame
        position = arrays["position"][frame, index]
        velocity = arrays["velocity"][frame, index]
        omega = arrays["omega"][frame, index]
        limit = scene["position_limit_m"]
        failed = not scene["completed"]
        pg.draw.rect(screen, colors["line"], (17, 105, 972, 672), width=1, border_radius=6)
        # Draw this aircraft's original world boundary and recorded public goals.
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        corners = signs * limit
        pixels = project(corners, limit)
        for a in range(8):
            for bit in (1, 2, 4):
                b = a ^ bit
                if b > a:
                    line((pixels[a], pixels[b]), (112, 84, 74), 1)
        if typed:
            task_status = goal_status(meta, arrays, frame, index)
            command_frame = min(frame, len(arrays["valid"]) - 1)
            goal_position = arrays["command_position"][command_frame, index]
            goal_rotation = arrays["command_rotation"][command_frame, index]
            is_stop = bool(arrays["command_is_stop"][command_frame, index])
            for point, label in zip(arrays["mission_points"][index], ("1 出生", "2 PASS", "3 STOP")):
                pixel = project(point, limit).astype(int)
                pg.draw.circle(screen, colors["muted"], pixel, 7, 2)
                text(label, pixel[0] + 12, pixel[1] - 12, 16, colors["muted"])
            theta = np.linspace(0, 2*np.pi, 49)
            for first, second in ((0, 1), (0, 2), (1, 2)):
                circle = np.zeros((len(theta), 3))
                circle[:, first], circle[:, second] = np.cos(theta), np.sin(theta)
                ball = arrays['mission_points'][index, 1] + circle*task_status['region_radius_m']
                line(project(ball, limit), colors['muted'], 1)
            goal = project(goal_position, limit).astype(int)
            pg.draw.circle(screen, colors["gold"], goal, 13, 2)
            ghost = goal_position + arrays["rotor_positions"][index] @ goal_rotation.T
            for rotor in ghost:
                pg.draw.line(screen, colors["gold"], goal, project(rotor, limit).astype(int), 2)
            # Target body z-axis makes the commanded full attitude visible.
            line(project((goal_position, goal_position + goal_rotation[:, 2]*limit*.22), limit),
                 colors["gold"], 3)
        else:
            goal = project(np.zeros(3), limit).astype(int)
            pg.draw.circle(screen, colors["gold"], goal, 9, 2)
        trail = arrays["position"][:frame+1:max(1, frame // 250), index]
        line(project(trail, limit), (75, 164, 175), 2)
        body_to_world = quaternion_rotation(arrays["orientation"][frame, index])
        rotor_world = position + arrays["rotor_positions"][index] @ body_to_world.T
        center = project(position, limit).astype(int)
        for rotor in rotor_world:
            end = project(rotor, limit).astype(int)
            pg.draw.line(screen, (228, 235, 239), center, end, 3)
            pg.draw.circle(screen, (83, 168, 241), end, 5, 2)
        pg.draw.circle(screen, colors["red"] if failed else colors["green"], center, 6)
        force = force_at_frame(arrays, index, frame)
        if force is None:
            text("旧回放未记录外力数据", 30, 120, 16, colors["muted"])
        elif typed and force["applied"]:
            text("固定点任务：外力关闭 · 金色为当前命令姿态", 30, 120, 16, colors["gold"])
        elif force["applied"]:
            gravity = scene["mass_kg"] * 9.81
            steady = force["external_force_world"]
            pulse = force["pulse_force_world"]
            steady_fraction = float(np.linalg.norm(steady) / gravity)
            pulse_fraction = float(np.linalg.norm(pulse) / gravity)
            force_arrow(position, steady, steady_fraction, limit, colors["steady_force"])
            if force["pulse_active"]:
                pulse_origin = position + body_to_world @ force["pulse_point_body"]
                force_arrow(pulse_origin, pulse, pulse_fraction, limit, colors["pulse_force"])
            text(f"持续外力 {np.linalg.norm(steady):.3f} N ({steady_fraction*100:.1f}% mg)  ·  重心",
                 30, 120, 16, colors["steady_force"])
            pulse_status = "施加中" if force["pulse_active"] else "当前未施加"
            text(f"偏心脉冲 {pulse_status}  {np.linalg.norm(pulse):.3f} N "
                 f"({pulse_fraction*100:.1f}% mg)",
                 30, 146, 16, colors["pulse_force"])
            text("箭头按重力比例缩放；脉冲施力点随机体旋转", 30, 172, 14, colors["muted"])
        else:
            text("飞行已终止；此时没有后续物理步或外力", 30, 120, 16, colors["muted"])
        target_label = "固定三点 · 按测量位置切换 · 姿态不阻止切换" if typed else "目标 (0, 0, 0) m"
        text(f"{target_label}   |   位置边界 ±{limit:.2f} m/轴", 29, 744, 16, colors["muted"])

        text("选择无人机", 1022, 20, 22)
        button("上一页", (1212, 17, 93, 30), "page_back")
        button("下一页", (1315, 17, 93, 30), "page_forward")
        text(f"第 {page+1}/{(len(meta['scenes'])-1)//18+1} 页", 1022, 54, 16, colors["muted"])
        for j in range(page*18, min((page+1)*18, len(meta["scenes"]))):
            row = meta["scenes"][j]
            rect = pg.Rect(1018, 82 + (j-page*18)*26, 460, 24)
            buttons.append((rect, f"scene:{j}"))
            if j == index:
                pg.draw.rect(screen, (55, 74, 91), rect, border_radius=3)
            color = colors["green"] if row["completed"] else colors["red"]
            status = "全程" if row["completed"] else f"{row['valid_steps']*meta['dt']:.2f}s 越界"
            text(f"#{j:03d}  {row['mass_kg']*1000:.0f} g  {status}",
                 rect.x + 7, rect.y + 2, 16, color)
        status = "越界终止" if failed else f"飞满 {meta['duration_seconds']:g} 秒"
        text(f"Scene #{index:03d}  {status}", 1022, 565, 22,
             colors["red"] if failed else colors["green"])
        text(f"时间 {state.seconds:.2f}/{state.end_seconds:.2f}s   速度 {state.speed:g}×", 1022, 601, 18)
        if typed:
            angle = math.degrees(math.acos(float(np.clip(
                (np.trace(body_to_world.T @ goal_rotation) - 1)/2, -1., 1.))))
            mode = task_status['phase_name']
            text(f"命令 {mode}   距离 {np.linalg.norm(position-goal_position):.3f} m", 1022, 633, 18)
            text(f"姿态误差 {angle:.2f}°   速率 {task_status['speed_m_s']:.3f} m/s", 1022, 661, 18)
            floor = task_status['minimum_speed_m_s']
            if floor is not None and not is_stop:
                shortfall = task_status['speed_shortfall_m_s']
                label = '达标' if shortfall == 0 else f'不足 {shortfall:.2f}'
                color = colors['muted'] if task_status['phase'] == 0 else colors['red' if shortfall else 'green']
                text(f"球内速率 ≥ {floor:.2f} m/s · {label}", 1022, 691, 18, color)
            else:
                text(f"|ω| {np.linalg.norm(omega):.3f} rad/s", 1022, 691, 18)
            text(f"球半径 {task_status['region_radius_m']:.3f} m · 中心容差 {task_status['center_tolerance_m']:g} m",
                 1022, 721, 16, colors["muted"])
            metrics = meta["metrics"]
            text(f"整组 EVAL：PASS {metrics['pass_fraction']:.1%} · 成功 {metrics['success_fraction']:.1%}",
                 1022, 749, 16, colors["muted"])
        else:
            text(f"|p| {np.linalg.norm(position):.3f} m    |v| {np.linalg.norm(velocity):.3f} m/s",
                 1022, 633, 18)
            text(f"|ω| {np.linalg.norm(omega):.3f} rad/s", 1022, 661, 18)
            text(f"推重比 {scene['thrust_to_weight']:.2f}    T/I {scene['torque_to_inertia']:.0f}",
                 1022, 695, 16, colors["muted"])
            text(f"整段 RMS：p {scene['position_rms_m']:.3f} m   v {scene['velocity_rms_m_s']:.3f} m/s",
                 1022, 724, 16, colors["muted"])
            text(f"ω {scene['omega_rms_rad_s']:.3f} rad/s", 1022, 749, 16, colors["muted"])

        pg.draw.rect(screen, (49, 64, 79), timeline, border_radius=5)
        usable = round(timeline.width * state.end_seconds / meta["duration_seconds"])
        pg.draw.rect(screen, (76, 117, 135), (timeline.x, timeline.y, usable, timeline.height), border_radius=5)
        marker = timeline.x + round(timeline.width * state.seconds / meta["duration_seconds"])
        pg.draw.circle(screen, colors["gold"], (marker, timeline.centery), 8)
        x = 27
        for label, action, width in (("上一架", "previous", 106), ("下一架", "next", 106),
                                     ("播放" if state.paused else "暂停", "pause", 86),
                                     ("重播", "replay", 83), ("自动轮播", "auto", 112),
                                     ("减速", "slower", 80), ("加速", "faster", 80),
                                     ("视角复位", "camera", 105)):
            button(label, (x, 842, width, 35), action)
            x += width + 9
        text("←/→ 换飞机 · 空格暂停 · 拖动时间轴 · 滚轮缩放 · 鼠标拖动旋转", 843, 849, 16,
             colors["muted"])
        pg.display.flip()

    running = True
    dragging = False
    seeking = False
    frames = 0
    while running:
        elapsed = min(clock.tick(fps) / 1000., .1)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                action = {pg.K_LEFT: "previous", pg.K_RIGHT: "next", pg.K_SPACE: "pause",
                          pg.K_r: "replay", pg.K_a: "auto", pg.K_MINUS: "slower",
                          pg.K_EQUALS: "faster", pg.K_c: "camera"}.get(event.key)
                if action:
                    handle(action)
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                if timeline.inflate(0, 25).collidepoint(event.pos):
                    seeking = True
                    state.seek((event.pos[0] - timeline.x) / timeline.width * meta["duration_seconds"])
                elif any(rect.collidepoint(event.pos) for rect, _ in buttons):
                    handle(next(action for rect, action in buttons if rect.collidepoint(event.pos)))
                elif event.pos[0] < 1000:
                    dragging = True
            elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
                dragging = seeking = False
            elif event.type == pg.MOUSEMOTION:
                if seeking:
                    state.seek((event.pos[0] - timeline.x) / timeline.width * meta["duration_seconds"])
                elif dragging:
                    yaw += event.rel[0] * .006
                    elevation = float(np.clip(elevation + event.rel[1] * .005, -.15, 1.3))
            elif event.type == pg.MOUSEWHEEL:
                zoom = float(np.clip(zoom * 1.12**event.y, .35, 4.0))
        if not seeking and screenshot is None:
            state.advance(elapsed)
        draw()
        frames += 1
        if screenshot is not None:
            pg.image.save(screen, str(screenshot))
            running = False
        if max_frames is not None and frames >= max_frames:
            running = False
    pg.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("max-frames must be positive")
    if not 10 <= args.fps <= 60:
        parser.error("fps must be in [10,60]")
    run_player(args.replay, screenshot=args.screenshot, max_frames=args.max_frames, fps=args.fps)


if __name__ == "__main__":
    main()

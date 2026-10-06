#!/usr/bin/env python3
"""Read-only training dashboard + manual launch of the existing saved-flight UI.

Run beside training, or on another computer with synced logs/replay files. No
Torch, policy, CUDA, checkpoint loading, new EVAL or trainer-side hook is used.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.response_monitor import (TrainingLogs, discover_replays, launch_replay,
                                    envelope, number)

BG = (15, 21, 31)
PANEL = (22, 31, 44)
LINE = (56, 72, 90)
WHITE = (230, 239, 247)
MUTED = (157, 178, 196)
BLUE = (100, 175, 244)
GOLD = (255, 207, 93)
RED = (252, 115, 118)
GREEN = (91, 220, 166)


def fmt(value, suffix="", digits=4):
    value = number(value)
    return "--" if value is None else f"{value:.{digits}g}{suffix}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="TRAIN directory; read-only, may not exist yet")
    parser.add_argument("--replay-root", type=Path, help="shallow scan of exported playlists; no evaluation is run")
    parser.add_argument("--replay", type=Path, action="append", default=[], help="exact exported playlist.json; repeatable")
    parser.add_argument("--poll-seconds", type=float, default=1., help="log poll interval (minimum 1 second)")
    parser.add_argument("--max-points", type=int, default=2000, help="bounded retained records per log, 100..10000")
    parser.add_argument("--font", type=Path, help="optional local TTF/OTF font")
    parser.add_argument("--screenshot", type=Path, help="save one dashboard frame and exit (does not launch replay)")
    parser.add_argument("--max-frames", type=int, help="bounded UI smoke test")
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds < 1:
        parser.error("poll-seconds must be finite and at least 1")
    if not 100 <= args.max_points <= 10000:
        parser.error("max-points must be in [100,10000]")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("max-frames must be positive")
    return args


def run_monitor(args):
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        import pygame as pg
    except ImportError as error:
        raise SystemExit("Install optional GUI dependencies: python -m pip install -r requirements-gui.txt") from error
    # Match the uploaded player's palette/layout, using a software display.
    # No OPENGL/SCALED flag and no busy-loop timer: rendering is outside training.
    pg.display.init()
    pg.font.init()
    screen = pg.display.set_mode((1500, 950), pg.RESIZABLE)
    pg.display.set_caption("DiffPhys | read-only training monitor + saved flight replay")
    canvas = pg.Surface((1500, 950))
    font_path = str(args.font) if args.font else pg.font.match_font("dejavusans,arial")
    fonts = {size: pg.font.Font(font_path, size) for size in (13, 15, 17, 20, 24, 30)}
    clock = pg.time.Clock()
    logs = TrainingLogs(args.run_dir, max_rows=args.max_points)
    root = args.replay_root if args.replay_root else args.run_dir
    replays, scan_errors = discover_replays(root, args.replay)
    selected = 0
    page = 0
    replay_process = None
    message = "Replays open only on request; never re-simulated by this dashboard."
    running, dirty, paused, logarithmic = True, True, False, False
    next_poll = 0.
    next_replay_scan = time.monotonic() + 5.
    frames = 0
    buttons = []
    scene_buttons = []
    status = logs.runtime_status()
    show_status = False
    status_scroll = 0
    alert_key = None

    def text(value, x, y, size=17, color=WHITE, width=None):
        value = str(value)
        if width:
            while value and fonts[size].size(value)[0] > width:
                value = value[:-4] + "..." if len(value) > 4 else ""
        canvas.blit(fonts[size].render(value, True, color), (x, y))

    def button(label, box, action, active=False):
        rect = pg.Rect(box)
        buttons.append((rect, action))
        pg.draw.rect(canvas, (38, 61, 80) if active else (32, 44, 59), rect, border_radius=6)
        pg.draw.rect(canvas, BLUE if active else LINE, rect, width=1, border_radius=6)
        surface = fonts[15].render(label, True, WHITE)
        canvas.blit(surface, surface.get_rect(center=rect.center))

    def status_lines():
        train, evaluation = logs.train.latest, logs.eval.latest
        body = [status.title, status.detail, "",
                "Last completed TRAIN: " + str(train.get("update", "--")),
                "  task objective: " + fmt(train.get("task_objective")) +
                " | gradient norm: " + fmt(train.get("pre_global_clip_norm")) +
                " | update duration: " + fmt(train.get("update_seconds"), " s"),
                "  position / velocity / omega RMS: " + fmt(train.get("position_rms"), " m") +
                " / " + fmt(train.get("velocity_rms"), " m/s") +
                " / " + fmt(train.get("omega_rms"), " rad/s"),
                "Last fixed EVAL: " + str(evaluation.get("update", "--")) +
                " | task objective: " + fmt(evaluation.get("task_objective")) +
                " | first-exit fraction: " + fmt(evaluation.get("raptor_share_terminated")),
                "", "Recorded error / traceback:"]
        if status.error:
            body.extend(status.error.expandtabs(4).splitlines())
        else:
            body.append("No error text recorded. Unknown exit causes are not inferred.")
        body += ["", "Log file: " + str(args.run_dir / "training.stdout.log"),
                 "This panel is read-only. Closing it does not restart or stop training."]
        lines = []
        # Preserve every character in long messages; the panel scrolls instead
        # of replacing the error with an ellipsis.
        for value in body:
            if not value:
                lines.append("")
            while value:
                left, right = 1, len(value)
                while left < right:
                    middle = (left + right + 1) // 2
                    if fonts[15].size(value[:middle])[0] <= 1388:
                        left = middle
                    else:
                        right = middle - 1
                lines.append(value[:left])
                value = value[left:]
        return lines

    def chart(key, label, rect, use_eval=True, log=False):
        rect = pg.Rect(rect)
        pg.draw.rect(canvas, PANEL, rect, border_radius=8)
        text(label, rect.x + 14, rect.y + 10, 17, width=rect.width - 28)
        series = [(logs.train.points(key), BLUE)]
        if use_eval:
            series.append((logs.eval.points(key), GOLD))
        series = [([(x, math.log10(y) if log else y) for x, y in points if not log or y > 0], color)
                  for points, color in series]
        all_points = [point for points, _ in series for point in points]
        area = pg.Rect(rect.x + 72, rect.y + 60, rect.width - 93, rect.height - 94)
        for k in range(3):
            yy = area.top + k * area.height // 2
            pg.draw.line(canvas, LINE, (area.left, yy), (area.right, yy))
        if not all_points:
            text("Waiting for recorded data", area.x + 35, area.y + 35, 15, MUTED)
            return
        xmin = min(p[0] for p in all_points)
        xmax = max(p[0] for p in all_points)
        ymin = min(p[1] for p in all_points)
        ymax = max(p[1] for p in all_points)
        # Normalize by maximum magnitude before subtracting: finite huge
        # gradients must not overflow the chart's y span.
        scale = max(abs(ymin), abs(ymax), 1e-300)
        lo, hi = ymin / scale, ymax / scale
        if lo == hi:
            lo, hi = lo - .05, hi + .05
        def pixel(point):
            x, y = point
            px = area.x + (x - xmin) / max(1., xmax - xmin) * area.width
            py = area.bottom - (y / scale - lo) / max(hi - lo, 1e-15) * area.height
            return (round(px), round(py))
        canvas.set_clip(rect)
        for points, color in series:
            points = envelope(points, max(1, area.width // 2))
            pixels = [pixel(p) for p in points]
            if len(pixels) > 1:
                pg.draw.lines(canvas, color, False, pixels, 2)
            for p in pixels if color == GOLD else pixels[-1:]:
                pg.draw.circle(canvas, color, p, 3)
        canvas.set_clip(None)
        for value, yy in ((ymax, area.y - 5), (ymin, area.bottom - 12)):
            label_y = f"10^{value:.1f}" if log else fmt(value, digits=3)
            text(label_y, rect.x + 7, yy, 13, MUTED, 63)
        text(f"#{int(xmin)}", area.x, area.bottom + 9, 13, MUTED)
        text(f"#{int(xmax)}", area.right - 65, area.bottom + 9, 13, MUTED)
        last = logs.train.latest.get(key)
        text("TRAIN " + fmt(last), rect.right - 190, rect.y + 33, 13, BLUE)

    def handle(action):
        nonlocal paused, logarithmic, page, selected, replays, scan_errors, message, replay_process
        nonlocal show_status, status_scroll
        if action == "pause":
            paused = not paused  # Pauses this reader only; never signals the trainer.
        elif action == "log":
            logarithmic = not logarithmic
        elif action == "status":
            show_status = not show_status
            status_scroll = 0
        elif action == "scan":
            old = replays[selected].path if replays else None
            replays, scan_errors = discover_replays(root, args.replay)
            selected = next((i for i, r in enumerate(replays) if r.path == old), 0)
            page = selected // 5
            message = scan_errors[0] if scan_errors else f"Found {len(replays)} exported replays."
        elif action == "back":
            page = max(0, page - 1)
        elif action == "more":
            page = min(max(0, (len(replays) - 1) // 5), page + 1)
        elif action == "open":
            if not replays:
                message = "No replay. Save a fixed-EVAL trace or export a long-EVAL flight."
            elif replay_process is not None and replay_process.poll() is None:
                message = "A replay window is already open. Close it before opening another."
            else:
                try:
                    replay_process = launch_replay(replays[selected].path)
                    message = f"Opened recorded Actor #{replays[selected].update}; not the latest live model."
                except (OSError, ValueError) as error:
                    message = str(error)

    try:
        while running:
            clock.tick(10)  # Event handling only; charts repaint on polling/input.
            now = time.monotonic()
            if now >= next_poll:
                if not paused:
                    logs.poll()
                status = logs.runtime_status()
                new_alert = (status.state, status.update, status.exit_code, status.error, status.detail)
                if status.state in ("failed", "stopped", "interrupted", "exited"):
                    if new_alert != alert_key:
                        show_status, status_scroll = True, 0
                        alert_key = new_alert
                elif status.state == "running" and alert_key is not None:
                    show_status, status_scroll, alert_key = False, 0, None
                next_poll = now + args.poll_seconds
                dirty = True
            if now >= next_replay_scan:
                previous_path = replays[selected].path if replays else None
                replays, scan_errors = discover_replays(root, args.replay)
                selected = next((i for i, entry in enumerate(replays)
                                 if entry.path == previous_path), 0)
                page = min(page, max(0, (len(replays) - 1) // 5))
                next_replay_scan = now + 5.
                dirty = True
            if replay_process is not None and replay_process.poll() is not None:
                message = ("Replay window closed." if replay_process.returncode == 0 else
                           "Replay exited with an error; see this terminal.")
                replay_process = None
                dirty = True
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.VIDEORESIZE:
                    screen = pg.display.set_mode((max(800, event.w), max(507, event.h)), pg.RESIZABLE)
                    dirty = True
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        if show_status:
                            show_status = False
                        else:
                            running = False
                    if show_status and event.key in (pg.K_PAGEUP, pg.K_PAGEDOWN):
                        status_scroll += -10 if event.key == pg.K_PAGEUP else 10
                    for key, action in ((pg.K_SPACE, "pause"), (pg.K_l, "log"),
                                        (pg.K_r, "scan"), (pg.K_e, "status")):
                        if event.key == key:
                            handle(action)
                    if event.key == pg.K_RETURN and not show_status:
                        handle("open")
                    dirty = True
                elif event.type == pg.MOUSEWHEEL and show_status:
                    status_scroll -= event.y * 3
                    dirty = True
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    xy = (event.pos[0] * 1500 / screen.get_width(), event.pos[1] * 950 / screen.get_height())
                    for rect, index in scene_buttons:
                        if rect.collidepoint(xy):
                            selected = index
                    for rect, action in buttons:
                        if rect.collidepoint(xy):
                            handle(action)
                    dirty = True
            if not dirty:
                continue
            buttons, scene_buttons = [], []
            canvas.fill(BG)
            text("DiffPhys | Training monitor", 24, 14, 30)
            text("READ ONLY  |  Blue: TRAIN  |  Gold: fixed EVAL  |  Short replay export runs separately", 24, 54, 15, GREEN)
            text(str(args.run_dir.resolve()), 24, 80, 15, MUTED, 1440)
            train, evaluation = logs.train.latest, logs.eval.latest
            age = logs.age()
            cards = [("TRAIN update", str(train.get("update", "--"))),
                     ("Fixed EVAL update", str(evaluation.get("update", "--"))),
                     ("Update time (logged)", fmt(train.get("update_seconds"), " s")),
                     ("CUDA peak (logged)", fmt(train.get("cuda_peak_bytes", 0) / 2**30, " GiB") if "cuda_peak_bytes" in train else "--")]
            for i, (label, value) in enumerate(cards):
                x = 24 + i * 369
                pg.draw.rect(canvas, PANEL, (x, 111, 355, 77), border_radius=7)
                text(label, x + 14, 119, 15, MUTED)
                text(value, x + 14, 145, 24, WHITE)
            status_color = RED if status.state == "failed" else GREEN if status.state == "running" else GOLD
            status_background = ((65, 26, 34) if status.state == "failed" else
                                 (20, 52, 39) if status.state == "running" else (55, 44, 24))
            pg.draw.rect(canvas, status_background, (24, 196, 1452, 30), border_radius=5)
            text(status.title + " | " + status.detail, 36, 201, 15, status_color, 1428)
            specs = [("task_objective", "Task objective", True), ("position_rms", "Position RMS [m]", True),
                     ("velocity_rms", "Velocity RMS [m/s]", True), ("omega_rms", "Omega RMS [rad/s]", True),
                     ("pre_global_clip_norm", "Gradient norm (post-group, pre-clip)", False),
                     ("update_seconds", "Update duration [s]", False)]
            for i, (key, label, use_eval) in enumerate(specs):
                chart(key, label, (24 + (i % 2) * 493, 234 + (i // 2) * 209, 478, 195),
                      use_eval, logarithmic and key in ("task_objective", "pre_global_clip_norm"))
            pg.draw.rect(canvas, PANEL, (1010, 234, 466, 614), border_radius=8)
            text("Recorded diagnostics", 1026, 245, 20)
            for yy, label, value in (
                (282, "EVAL first-exit fraction", fmt(evaluation.get("raptor_share_terminated") * 100, "%") if "raptor_share_terminated" in evaluation else "--"),
                (309, "TRAIN motor saturation", fmt(train.get("motor_saturation_fraction") * 100, "%") if "motor_saturation_fraction" in train else "--"),
                (336, "Forward / update", fmt(train.get("forward_seconds"), " s") + " / " + fmt(train.get("update_seconds"), " s")),
                (363, "Valid transitions / update", fmt(train.get("physical_transitions"), digits=7))):
                text(label, 1026, yy, 15, MUTED)
                text(value, 1290, yy, 15, WHITE, 166)
            text("Saved-flight replay", 1026, 410, 20)
            text("Opens saved 3D arrays. Dashboard never loads a checkpoint.", 1026, 441, 13, MUTED)
            for index in range(page * 5, min(len(replays), (page + 1) * 5)):
                entry = replays[index]
                yy = 469 + (index % 5) * 49
                rect = pg.Rect(1026, yy, 433, 44)
                scene_buttons.append((rect, index))
                pg.draw.rect(canvas, (38, 61, 80) if selected == index else (29, 40, 54), rect, border_radius=4)
                text(f"Actor #{entry.update} | {entry.seconds:g}s | {entry.scenes} scenes", 1034, yy + 3, 15, GOLD)
                text(str(entry.path.parent.parent.name), 1034, yy + 23, 13, MUTED, 417)
            if not replays:
                text("No exported playlist found.", 1034, 491, 17, MUTED)
                text("--replay /path/to/playback/playlist.json", 1034, 523, 13, MUTED)
                text("or --replay-root /path/to/long_eval", 1034, 548, 13, MUTED)
            button("<", (1026, 724, 40, 32), "back")
            button(">", (1074, 724, 40, 32), "more")
            button("Refresh list [R]", (1122, 724, 149, 32), "scan")
            button("Open replay", (1280, 724, 179, 32), "open")
            text(message, 1026, 770, 13, MUTED, 431)
            text("Check Actor/update; historical long EVAL may use another model.",
                 1026, 797, 13, GOLD)
            text("It never replaces these logged training metrics.", 1026, 819, 13, MUTED)
            text("TRAIN log age: " + ("waiting for file" if age is None else f"{age:.0f}s") + " | Not a process heartbeat", 24, 869, 15, MUTED)
            text("Last saved status: " + logs.saved_status(), 24, 893, 15, MUTED, 880)
            invalid = logs.train.invalid_lines + logs.eval.invalid_lines
            warning = ("Waiting / read error: " + (logs.train.error or logs.eval.error or logs.summary_error)
                       if logs.train.error or logs.eval.error or logs.summary_error else
                       f"Cache: TRAIN {len(logs.train.rows)} / EVAL {len(logs.eval.rows)} rows; ignored invalid lines: {invalid}")
            if logs.train.prefix_skipped or logs.eval.prefix_skipped:
                warning = "Recent file tail only (older prefix omitted). " + warning
            text(warning, 24, 922, 13, MUTED, 1444)
            button("Resume reader" if paused else "Pause reader", (1026, 866, 138, 36), "pause", paused)
            button("Log y: ON" if logarithmic else "Log y: OFF", (1172, 866, 128, 36), "log", logarithmic)
            button("Status / error [E]", (1308, 866, 151, 36), "status", show_status)
            if show_status:
                # Block clicks through the overlay into replay/reader controls.
                buttons, scene_buttons = [], []
                pg.draw.rect(canvas, PANEL, (24, 246, 1452, 596), border_radius=8)
                pg.draw.rect(canvas, status_color, (24, 246, 1452, 596), width=2, border_radius=8)
                text("Training status and recorded error", 44, 263, 20, status_color)
                button("Close panel [Esc / E]", (1250, 256, 206, 36), "status")
                lines = status_lines()
                visible_lines = 23
                status_scroll = min(max(0, status_scroll), max(0, len(lines) - visible_lines))
                canvas.set_clip(pg.Rect(44, 310, 1388, 480))
                for i, line in enumerate(lines[status_scroll:status_scroll + visible_lines]):
                    text(line, 44, 310 + i * 20, 15)
                canvas.set_clip(None)
                text(f"Lines {status_scroll + 1}-{min(len(lines), status_scroll + visible_lines)} / {len(lines)}"
                     " | Mouse wheel / Page Up / Page Down to scroll", 44, 810, 13, MUTED)
            if screen.get_size() == canvas.get_size():
                screen.blit(canvas, (0, 0))
            else:
                pg.transform.scale(canvas, screen.get_size(), screen)
            pg.display.flip()
            frames += 1
            dirty = False
            if args.screenshot:
                pg.image.save(screen, str(args.screenshot))
                running = False
            if args.max_frames and frames >= args.max_frames:
                running = False
    finally:
        # No signals to the trainer, and no forced kill of an independent replay.
        pg.quit()


def main(argv=None):
    run_monitor(parse_args(argv))


if __name__ == "__main__":
    main()

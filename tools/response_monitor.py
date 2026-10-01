"""Read-only, bounded log following for the optional Pygame monitor.

Standard library only. No Torch, simulator, checkpoint loading or writer locks.
The trainer does not import this module. A JSONL record is usable only after its
newline has arrived. Atomic resume rewrites clear the corresponding old curve.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

MAX_READ_BYTES = 256 * 1024
MAX_LINE_BYTES = 128 * 1024
MAX_ROWS = 2000
METRICS = (
    "task_objective", "position_rms", "velocity_rms", "omega_rms",
    "pre_global_clip_norm", "raw_gradient_norm", "update_seconds",
    "forward_seconds", "cuda_peak_bytes", "physical_transitions",
    "motor_saturation_fraction", "raptor_share_terminated",
    "raptor_episode_length_mean", "scenario_count",
)


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    return result if math.isfinite(result) else None


def metric_row(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    update = raw.get("update")
    if isinstance(update, bool) or not isinstance(update, int) or not 0 <= update <= 2**53:
        return None
    row = {"update": update}
    for key in METRICS:
        value = number(raw.get(key))
        if value is not None:
            row[key] = value
    if "pre_global_clip_norm" not in row and "raw_gradient_norm" in row:
        row["pre_global_clip_norm"] = row["raw_gradient_norm"]
    if isinstance(raw.get("finite"), bool):
        row["finite"] = raw["finite"]
    return row


class JsonlTail:
    """Keep at most max_rows SMALL records; read at most max_bytes per poll.

    A cold start reads only the file tail, not an entire multi-day run. Skipped
    prefixes and oversize/invalid lines are reported, not disguised as full data.
    An anchor near the read offset also detects ordinary copy-truncate/rewrite
    logs even when the file has regrown beyond the old offset between polls.
    """
    def __init__(self, path, *, max_rows=MAX_ROWS, max_bytes=MAX_READ_BYTES,
                 max_line_bytes=MAX_LINE_BYTES):
        if min(max_rows, max_bytes, max_line_bytes) < 1:
            raise ValueError("reader limits must be positive")
        self.path = Path(path)
        self.rows: deque[dict] = deque(maxlen=max_rows)
        self.max_bytes, self.max_line_bytes = max_bytes, max_line_bytes
        self.offset = 0
        self.pending = b""
        self.discard_line = False
        self.identity = None
        self.stamp = None
        self.anchor = b""
        self.mtime = None
        self.error = ""
        self.invalid_lines = 0
        self.prefix_skipped = False
        self.total_bytes_read = 0
        self.last_read_bytes = 0
        self.resets = 0

    def _reset(self, size):
        self.rows.clear()
        self.pending = b""
        self.invalid_lines = 0
        self.offset = max(0, size - self.max_bytes)
        self.prefix_skipped = self.offset > 0
        self.discard_line = self.prefix_skipped
        self.anchor = b""
        self.resets += 1

    def poll(self) -> bool:
        self.last_read_bytes = 0
        try:
            stat = self.path.stat()
            stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if stamp == self.stamp:
                self.error = ""
                return False
            with self.path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                reset = identity != self.identity or stat.st_size < self.offset
                if not reset and self.anchor:
                    stream.seek(self.offset - len(self.anchor))
                    reset = stream.read(len(self.anchor)) != self.anchor
                if reset:
                    self._reset(stat.st_size)
                self.identity = identity
                stream.seek(self.offset)
                chunk = stream.read(self.max_bytes)
                self.offset += len(chunk)
                self.last_read_bytes = len(chunk)
                self.total_bytes_read += len(chunk)
                stream.seek(max(0, self.offset - 128))
                self.anchor = stream.read(min(128, self.offset))
                # Capture the pre-read size, so a concurrent append is read next time.
                self.stamp = ((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                              if self.offset >= stat.st_size else None)
                self.mtime = stat.st_mtime
            self.error = ""
            data = self.pending + chunk
            self.pending = b""
            lines = data.split(b"\n")
            for raw in lines[:-1]:
                if self.discard_line:
                    self.discard_line = False
                    continue
                if not raw.strip():
                    continue
                if len(raw) > self.max_line_bytes:
                    self.invalid_lines += 1
                    continue
                try:
                    row = metric_row(json.loads(raw))
                except (ValueError, UnicodeError, RecursionError):
                    row = None
                if row is None:
                    self.invalid_lines += 1
                    continue
                if self.rows and row["update"] < self.rows[-1]["update"]:
                    self.rows.clear()  # Same-file rollback: no stale future updates.
                if self.rows and row["update"] == self.rows[-1]["update"]:
                    self.rows.pop()
                self.rows.append(row)
            tail = lines[-1]
            if len(tail) > self.max_line_bytes:
                if not self.discard_line:
                    self.invalid_lines += 1
                self.discard_line = True
            elif not self.discard_line:
                self.pending = tail
            return bool(chunk) or reset
        except (OSError, ValueError) as error:
            self.error = str(error)
            # A temporarily missing file must not be reported as trainer failure.
            return False

    @property
    def latest(self) -> dict:
        return self.rows[-1] if self.rows else {}

    def points(self, key: str) -> list[tuple[float, float]]:
        return [(row["update"], row[key]) for row in self.rows if key in row]


def envelope(points: list[tuple[float, float]], columns: int = 300):
    """Bound draw work while retaining minima AND maxima, including spikes."""
    if columns < 1:
        raise ValueError("columns must be positive")
    if len(points) <= 2 * columns:
        return points
    out = [points[0]]
    width = math.ceil(len(points) / columns)
    for start in range(0, len(points), width):
        group = points[start:start + width]
        lo = min(range(len(group)), key=lambda i: group[i][1])
        hi = max(range(len(group)), key=lambda i: group[i][1])
        out.extend(group[i] for i in sorted({lo, hi}))
    out.append(points[-1])
    return out


def small_json(path: Path, limit: int = 2 * 1024 * 1024) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("JSON metadata exceeds size limit")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


class TrainingLogs:
    def __init__(self, run_dir, *, max_rows=MAX_ROWS, max_bytes=MAX_READ_BYTES):
        self.run_dir = Path(run_dir)
        self.train = JsonlTail(self.run_dir / "history.jsonl", max_rows=max_rows, max_bytes=max_bytes)
        self.eval = JsonlTail(self.run_dir / "evaluation.jsonl", max_rows=max_rows, max_bytes=max_bytes)
        self.summary: dict = {}
        self.summary_stamp = None
        self.summary_error = ""

    def poll(self):
        self.train.poll()
        self.eval.poll()
        path = self.run_dir / "summary.json"
        try:
            stat = path.stat()
            stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if stamp != self.summary_stamp:
                self.summary = small_json(path, 256 * 1024)
                self.summary_stamp = stamp
                self.summary_error = ""
        except FileNotFoundError:
            self.summary, self.summary_stamp, self.summary_error = {}, None, ""
        except (OSError, ValueError, RecursionError) as error:
            self.summary_error = str(error)

    def age(self, now=None):
        if self.train.mtime is None:
            return None
        return max(0., (time.time() if now is None else now) - self.train.mtime)

    def saved_status(self):
        # This is the last saved summary, not a PID/heartbeat/liveness claim.
        saved = self.summary.get("updates")
        current = self.train.latest.get("update")
        if not isinstance(saved, int) or isinstance(saved, bool):
            return "not available"
        if current is not None and saved < current:
            return "older summary (run resumed)"
        return f"{self.summary.get('status', 'unknown')} at #{saved} (saved)"


@dataclass(frozen=True)
class ReplayEntry:
    path: Path
    update: int | str
    scenes: int
    seconds: float
    model_hash: str


def replay_entry(path) -> ReplayEntry:
    path = Path(path).resolve()
    meta = small_json(path)
    scenes = meta.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("playlist contains no scenes")
    seconds = number(meta.get("duration_seconds"))
    dt = number(meta.get("dt"))
    if seconds is None or seconds <= 0 or dt is None or dt <= 0:
        raise ValueError("invalid replay duration/dt")
    # Read only metadata. Never decompress the NPZ during monitor polling.
    trajectory_file = meta.get("trajectory_file")
    if trajectory_file is None:
        data_path = path.with_suffix(".npz")
    elif (not isinstance(trajectory_file, str)
          or Path(trajectory_file).name != trajectory_file
          or trajectory_file in ("", ".", "..")):
        raise ValueError("invalid replay trajectory filename")
    else:
        data_path = path.parent / trajectory_file
    if not data_path.is_file():
        raise ValueError("matching replay NPZ has not been exported")
    return ReplayEntry(path, meta.get("checkpoint_update", "?"), len(scenes), seconds,
                       str(meta.get("model_sha256", "unknown")))


def discover_replays(root, explicit=(), *, limit=64, directory_limit=128):
    """Shallow and bounded. Never recursively traverse runs or checkpoints."""
    candidates = [Path(p) for p in explicit]
    errors = []
    if root is not None:
        root = Path(root)
        candidates += [root / "playlist.json", root / "playback/playlist.json"]
        try:
            # Limit *entries inspected*, not just matching directories.
            with os.scandir(root) as entries:
                for index, entry in enumerate(entries):
                    if index >= directory_limit:
                        errors.append("Replay scan limited; pass a narrower --replay-root")
                        break
                    if entry.is_dir(follow_symlinks=False):
                        child = Path(entry.path)
                        candidates += [child / "playlist.json", child / "playback/playlist.json"]
        except OSError as error:
            errors.append(str(error))
    result, seen = [], set()
    for path in candidates:
        if len(result) >= limit:
            errors.append("Replay list limited; use --replay for an exact file")
            break
        if path in seen:
            continue
        seen.add(path)
        try:
            if path.is_file():
                result.append(replay_entry(path))
        except (OSError, ValueError, RecursionError) as error:
            errors.append(f"{path.name}: {error}")
    return result, errors


def launch_replay(path, *, python=sys.executable, max_frames=None):
    """Only launch the saved-array renderer. Never --run-dir, export or EVAL."""
    entry = replay_entry(path)
    meta = small_json(entry.path)
    player_name = ("play_response_short.py"
                   if meta.get("replay_type") == "short-eval-v1"
                   else "play_response_long.py")
    player = Path(__file__).with_name(player_name)
    command = [str(python), str(player), "--replay", str(entry.path), "--fps", "20"]
    if max_frames is not None:
        command += ["--max-frames", str(max_frames)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYGAME_HIDE_SUPPORT_PROMPT="1")
    return subprocess.Popen(command, env=env, shell=False)

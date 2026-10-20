"""Export and follow the latest ordinary fixed-EVAL rollout for replay.

The trainer's source-bound files are not modified. This sidecar re-evaluates a
committed checkpoint on the trainer's deterministic fixed EVAL pool, records
the resulting trajectory, and keeps only one replay per run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env_raptor import RaptorParams, RaptorSimulator
from response_noise import DisturbanceConfig
from response_task import TaskLossConfig
from response_training import (
    DEVELOPMENT_SEEDS,
    evaluate,
    load_policy_checkpoint,
    sample_pool,
    source_hash,
)


REPLAY_SCHEMA = "short-eval-v1"
REPLAY_SUBDIR = "short_eval"
METRIC_KEYS = ("task_objective", "position_rms", "velocity_rms", "omega_rms")


class _RecordingSimulator:
    """Transparent simulator proxy that snapshots each actual active step."""

    def __init__(self, simulator):
        self._simulator = simulator
        self.events = []

    def __getattr__(self, name):
        return getattr(self._simulator, name)

    def step(self, state, action):
        result = self._simulator.step(state, action)
        self.events.append((
            state.noise_row.detach(),
            action.detach(),
            result.position.detach(),
            result.velocity.detach(),
            result.orientation.detach(),
            result.omega.detach(),
        ))
        return result


def _cpu_array(tensor):
    return tensor.detach().cpu().contiguous().numpy()


def _trajectory_arrays(initial, events, horizon):
    """Reassemble the compact live-scene calls into the trainer's dense trace."""
    count = initial.position.shape[0]
    expected_rows = torch.arange(count, device=initial.noise_row.device)
    if not torch.equal(initial.noise_row, expected_rows):
        raise ValueError("short-EVAL replay requires the original pooled scene order")

    state = {
        "position": initial.position.detach(),
        "velocity": initial.velocity.detach(),
        "orientation": initial.orientation.detach(),
        "omega": initial.omega.detach(),
        "action": initial.previous_action.detach(),
    }
    frames = {name: [value.clone()] for name, value in state.items() if name != "action"}
    actions, valid = [], []
    false = torch.zeros(count, dtype=torch.bool, device=initial.position.device)

    for step in range(horizon):
        active = false.clone()
        if step < len(events):
            rows, action, position, velocity, orientation, omega = events[step]
            active[rows] = True
            state["action"] = state["action"].index_copy(0, rows, action)
            for name, value in (("position", position), ("velocity", velocity),
                                ("orientation", orientation), ("omega", omega)):
                state[name] = state[name].index_copy(0, rows, value)
        actions.append(state["action"].clone())
        valid.append(active)
        for name in frames:
            frames[name].append(state[name].clone())

    arrays = {name: _cpu_array(torch.stack(values)) for name, values in frames.items()}
    arrays.update(
        action=_cpu_array(torch.stack(actions)),
        valid=_cpu_array(torch.stack(valid)),
        # These are the forces used for transition t -> t+1, not inferred
        # from the resulting motion or the noisy policy observation.
        external_force_world=_cpu_array(initial.external_force),
        pulse_force_world=_cpu_array(initial.pulse_tape[:, :horizon, :3].transpose(0, 1)),
        pulse_point_body=_cpu_array(initial.pulse_tape[:, :horizon, 3:].transpose(0, 1)),
        pulse_active=_cpu_array(initial.pulse_active_tape[:, :horizon].transpose(0, 1)),
        rotor_positions=_cpu_array(initial.rotor_positions),
        mass_kg=_cpu_array(initial.mass),
        arm_length_m=_cpu_array(initial.arm_length),
        thrust_to_weight=_cpu_array(initial.thrust_to_weight),
        torque_to_inertia=_cpu_array(initial.torque_to_inertia),
        position_limit=_cpu_array(initial.position_limit),
    )
    return arrays


@torch.no_grad()
def capture_short_eval(policy, simulator, initial, horizon, loss_config):
    """Run the usual EVAL exactly once and retain its full state/action trace."""
    recorder = _RecordingSimulator(simulator)
    report = evaluate(policy, recorder, initial, horizon, loss_config)
    arrays = _trajectory_arrays(initial, recorder.events, horizon)
    expected = (horizon + 1, initial.position.shape[0])
    if arrays["position"].shape[:2] != expected or arrays["orientation"].shape[:2] != expected:
        raise RuntimeError("recorded short-EVAL state dimensions are inconsistent")
    if arrays["valid"].shape != (horizon, initial.position.shape[0]):
        raise RuntimeError("recorded short-EVAL validity dimensions are inconsistent")
    return report, arrays


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _scene_rows(arrays):
    valid = arrays["valid"]
    horizon, count = valid.shape
    rows = []
    p = arrays["position"][1:]
    v = arrays["velocity"][1:]
    w = arrays["omega"][1:]
    for index in range(count):
        alive = valid[:, index]
        steps = int(alive.sum())
        denominator = max(steps, 1)
        failure_step = None if steps == horizon else steps
        rows.append({
            "scene_id": index,
            "scene_index": index,
            "mass_kg": float(arrays["mass_kg"][index]),
            "arm_length_m": float(arrays["arm_length_m"][index]),
            "thrust_to_weight": float(arrays["thrust_to_weight"][index]),
            "torque_to_inertia": float(arrays["torque_to_inertia"][index]),
            "position_limit_m": float(arrays["position_limit"][index]),
            "completed": failure_step is None,
            "failure_reason": None if failure_step is None else "position_boundary",
            "failure_step": failure_step,
            "valid_steps": steps,
            "position_rms_m": float(np.sqrt((np.square(p[:, index]).sum(-1) * alive).sum() / denominator)),
            "velocity_rms_m_s": float(np.sqrt((np.square(v[:, index]).sum(-1) * alive).sum() / denominator)),
            "omega_rms_rad_s": float(np.sqrt((np.square(w[:, index]).sum(-1) * alive).sum() / denominator)),
        })
    return rows


def save_latest_replay(run_dir, *, checkpoint_update, model_sha256, source_sha256,
                       dt, report, arrays, logged_eval_update=None,
                       logged_metrics_match=None, capture_seconds=None,
                       cuda_peak_allocated_bytes=None):
    """Atomically publish a latest-only replay; leave at most one old archive."""
    output = Path(run_dir) / REPLAY_SUBDIR
    output.mkdir(parents=True, exist_ok=True)
    playlist = output / "playlist.json"
    trajectory_name = f"trajectory_{int(checkpoint_update)}.npz"
    final_npz = output / trajectory_name
    fd, temporary_npz = tempfile.mkstemp(dir=output, prefix="trajectory.", suffix=".npz")
    os.close(fd)
    try:
        with open(temporary_npz, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_npz, final_npz)
    finally:
        if os.path.exists(temporary_npz):
            os.unlink(temporary_npz)

    horizon = int(arrays["valid"].shape[0])
    metadata = {
        "replay_type": REPLAY_SCHEMA,
        "dt": float(dt),
        "duration_seconds": float(horizon * dt),
        "scenes": _scene_rows(arrays),
        "checkpoint_update": int(checkpoint_update),
        "model_sha256": str(model_sha256),
        "source_sha256": str(source_sha256),
        "logged_eval_update": logged_eval_update,
        "logged_metrics_match": logged_metrics_match,
        "metrics": report,
        "trajectory_file": trajectory_name,
        "trajectory_sha256": _sha256(final_npz),
        "trajectory_uncompressed_bytes": int(sum(value.nbytes for value in arrays.values())),
        "trajectory_compressed_bytes": int(final_npz.stat().st_size),
        "capture_seconds": capture_seconds,
        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
        "target_world_m": [0.0, 0.0, 0.0],
        "arena": "per-scene position boundary from the fixed-EVAL sampler",
    }
    _atomic_json(playlist, metadata)
    for path in output.glob("trajectory_*.npz"):
        if path != final_npz:
            path.unlink(missing_ok=True)
    return playlist


def _read_latest_eval(path):
    latest = None
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("update"), int):
                latest = row
    except FileNotFoundError:
        pass
    return latest


def _existing_update(path):
    try:
        return int(json.loads(Path(path).read_text()).get("checkpoint_update", -1))
    except (OSError, ValueError, TypeError):
        return -1


def _metrics_match(report, logged, tolerance=2e-5):
    if not isinstance(logged, dict):
        return None
    for name in METRIC_KEYS:
        actual, expected = report.get(name), logged.get(name)
        if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
            return False
        if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
            return False
    return True


def export_checkpoint(checkpoint_path, run_dir, *, device="cuda"):
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device)
    start = time.monotonic()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved["binding"]["source_sha256"] != source_hash():
        raise ValueError("short-EVAL replay needs the checkpoint's exact training source")
    dtype = getattr(torch, saved["binding"]["dtype"])
    policy, saved = load_policy_checkpoint(checkpoint_path, device, dtype)
    policy.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    protocol = saved["binding"]["protocol"]
    horizon = int(saved["binding"]["horizon"])
    simulator = RaptorSimulator(RaptorParams(**protocol["environment_params"]))
    initial = sample_pool(
        int(protocol["eval_scenarios_per_bank"]), DEVELOPMENT_SEEDS,
        dt=policy.config.dt, device=device, dtype=dtype, horizon=horizon,
        disturbances=DisturbanceConfig(**protocol["disturbances"]),
    )
    report, arrays = capture_short_eval(
        policy, simulator, initial, horizon, TaskLossConfig(**protocol["loss"])
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        cuda_peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    else:
        cuda_peak_allocated_bytes = None
    capture_seconds = time.monotonic() - start
    update = int(saved["progress"]["updates"])
    latest_logged = _read_latest_eval(Path(run_dir) / "evaluation.jsonl")
    logged = latest_logged if latest_logged and latest_logged.get("update") == update else None
    metrics_match = _metrics_match(report, logged)
    if metrics_match is False:
        raise ValueError("recomputed fixed-EVAL metrics disagree with the logged evaluation")
    playlist = save_latest_replay(
        run_dir,
        checkpoint_update=update,
        model_sha256=saved["model_sha256"],
        source_sha256=saved["binding"]["source_sha256"],
        dt=policy.config.dt,
        report=report,
        arrays=arrays,
        logged_eval_update=None if logged is None else update,
        logged_metrics_match=metrics_match,
        capture_seconds=capture_seconds,
        cuda_peak_allocated_bytes=cuda_peak_allocated_bytes,
    )
    return {
        "checkpoint_update": update,
        "report": {name: report[name] for name in METRIC_KEYS},
        "capture_seconds": capture_seconds,
        "trajectory_uncompressed_bytes": int(sum(value.nbytes for value in arrays.values())),
        "trajectory_compressed_bytes": (Path(run_dir) / REPLAY_SUBDIR /
                                         json.loads(playlist.read_text())["trajectory_file"]).stat().st_size,
        "logged_eval_update": None if logged is None else update,
        "logged_metrics_match": metrics_match,
        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
        "playlist": str(playlist),
    }


def watch(run_dir, *, device="cuda", poll_seconds=1.0, once=False):
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / "latest.pt"
    playlist = run_dir / REPLAY_SUBDIR / "playlist.json"
    initial_export_done = False
    last_exported = _existing_update(playlist)
    while True:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except (OSError, EOFError, RuntimeError):
            if once:
                raise
            time.sleep(poll_seconds)
            continue
        checkpoint_update = int(checkpoint["progress"]["updates"])
        latest_eval = _read_latest_eval(run_dir / "evaluation.jsonl")
        eval_update = None if latest_eval is None else int(latest_eval["update"])

        # Publish a replay for the current Actor immediately on startup; subsequent
        # exports are tied to the exact fixed-EVAL/checkpoint update pair.
        should_export_current = not initial_export_done and checkpoint_update > last_exported
        should_export_eval = (eval_update is not None and eval_update > last_exported
                              and checkpoint_update == eval_update)
        if should_export_current or should_export_eval:
            if once:
                result = export_checkpoint(checkpoint_path, run_dir, device=device)
                print(json.dumps(result, allow_nan=False), flush=True)
            else:
                # The GPU context belongs to a short-lived child; no CUDA memory
                # remains reserved between the trainer's fixed evaluations.
                try:
                    subprocess.run((sys.executable, str(Path(__file__).resolve()),
                                    "--run-dir", str(run_dir), "--device", str(device),
                                    "--once"), check=True)
                except subprocess.CalledProcessError as error:
                    print(json.dumps({"replay_export_failed_update": checkpoint_update,
                                      "exit_code": error.returncode}), flush=True)
                    # A failed diagnostic export must not turn into a rapid GPU
                    # retry loop beside the production trainer.
                    last_exported = checkpoint_update
            last_exported = max(last_exported, _existing_update(playlist))
            initial_export_done = True
        if once:
            return
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--once", action="store_true", help="export the current checkpoint and exit")
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds < .2:
        parser.error("poll-seconds must be finite and at least 0.2")
    torch.set_num_threads(1)
    watch(args.run_dir, device=args.device, poll_seconds=args.poll_seconds, once=args.once)


if __name__ == "__main__":
    main()

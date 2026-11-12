#!/usr/bin/env python3
"""Run the paired, budgeted from-zero experiment, analyze it, restore agile."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CUDA_PYTHON = "/tmp/diffphys-cuda-venv/bin/python"


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def restore_agile(folder, env):
    record = json.loads((folder / "paused_agile_run.json").read_text())
    if not record.get("restore_when_finished"):
        return {"status": "not_requested"}
    project = Path(record["project_root"])
    pointer = project / "runs/current_pass_stop_training.json"
    current = json.loads(pointer.read_text())
    work = Path(record["run_dir"])
    if current["run_dir"] != str(work):
        return {"status": "superseded_by_new_agile_run"}
    processes = json.loads((work / "processes.json").read_text())
    old_pid = processes.get("trainer")
    process = Path("/proc") / str(old_pid) / "cmdline"
    if process.exists() and str(work).encode() in process.read_bytes():
        return {"status": "already_running", "pid": old_pid}
    command = list(record["active_launch"]["command"])
    for flag in ("--init-checkpoint", "--resume"):
        if flag in command:
            index = command.index(flag)
            del command[index:index+2]
    command += ["--resume", str(work / "latest.pt")]
    with (work / "training.stdout.log").open("ab", buffering=0) as log:
        child = subprocess.Popen(command, cwd=project, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    active = dict(record["active_launch"], command=command, started_at_utc=timestamp())
    write(work / "active_launch.json", active)
    processes["trainer"] = child.pid
    if record.get("exporter_was_running"):
        export_command = [CUDA_PYTHON, "-u", str(project / "tools/short_eval_replay.py"),
                          "--run-dir", str(work), "--device", "cuda", "--poll-seconds", "2"]
        with (work / "replay_export.stdout.log").open("ab", buffering=0) as log:
            exporter = subprocess.Popen(export_command, cwd=project, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        processes["replay_exporter"] = exporter.pid
    write(work / "processes.json", processes)
    current.update(trainer_pid=child.pid, restarted_at_utc=active["started_at_utc"])
    write(pointer, current)
    return {"status": "resume_launched", "pid": child.pid, "run_dir": str(work)}


def run(folder):
    folder = Path(folder).resolve()
    source = folder / "source" if (folder / "source").is_dir() else ROOT
    manifest = json.loads((folder / "manifest.json").read_text())
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "TORCHINDUCTOR_COMPILE_THREADS": "1",
           "PYTORCH_ALLOC_CONF": "expandable_segments:True", "PYTHONUNBUFFERED": "1"}
    state = {"started_at_utc": timestamp(), "status": "running", "phase": "preparing",
             "updates_per_arm": manifest["updates_per_arm"]}
    active = [None]
    interrupted = []

    def stop(signum, frame):
        interrupted.append(signum)
        if active[0] is not None and active[0].poll() is None:
            active[0].send_signal(signal.SIGTERM)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    try:
        for arm, weight in (("baseline", 0.), ("time", 1.)):
            if interrupted:
                raise InterruptedError("paired experiment interrupted")
            work = folder / arm
            work.mkdir(exist_ok=True)
            bootstrap = ('import sys; sys.argv[0]="tools/train_response_control.py"; '
                         'from tools.run_hover_time_experiment import main; main()')
            command = [CUDA_PYTHON, "-u", "-c", bootstrap, "--hover-time-weight", str(weight),
                       "@" + str(source / "configs/response_raptor_multi_airframe.args"), "--updates",
                       str(manifest["updates_per_arm"]), "--max-seconds", "86400",
                       "--work-dir", str(work)]
            if (work / "latest.pt").exists():
                command += ["--resume", str(work / "latest.pt")]
            state.update(phase=arm, current_run=str(work))
            write(folder / "status.json", state)
            write(work / "active_launch.json", {"command": command, "started_at_utc": timestamp(),
                  "initialization": "from zero, seed 7", "experiment": str(folder)})
            with (work / "training.stdout.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                active[0] = child
                write(work / "processes.json", {"trainer": child.pid})
                write(folder / "processes.json", {"supervisor": os.getpid(), "trainer": child.pid})
                code = child.wait()
            active[0] = None
            summary_path = work / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            write(work / "training_exit.json", {"exit_code": code,
                  "finished_at_utc": timestamp(), "summary": summary})
            if interrupted:
                raise InterruptedError("paired experiment interrupted at a saved update")
            if code or summary.get("updates") != manifest["updates_per_arm"]:
                raise RuntimeError(f"{arm} stopped before its complete update budget: {code}, {summary}")
            state[arm + "_complete"] = True
        state["phase"] = "common_heldout_evaluation"
        write(folder / "status.json", state)
        command = [CUDA_PYTHON, str(source / "tools/analyze_hover_time_ablation.py"),
                   "--experiment-dir", str(folder)]
        with (folder / "analysis.stdout.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            active[0] = child
            code = child.wait()
        active[0] = None
        if code:
            raise RuntimeError("common held-out evaluation failed; see analysis.stdout.log")
        state.update(status="complete", phase="complete")
    except BaseException as error:
        state.update(status="interrupted" if interrupted else "failed",
                     error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        state["finished_at_utc"] = timestamp()
        if not (folder / "do_not_restore_agile").exists():
            try:
                state["agile_restoration"] = restore_agile(folder, env)
            except Exception as error:
                state["agile_restoration"] = {"status": "failed", "error": str(error)}
        write(folder / "status.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.experiment_dir)

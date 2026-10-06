"""Visible exit/error state without importing the trainer or altering run files."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from tools import response_monitor as monitor

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


def logs_at(run, update=133):
    write(
        run / "history.jsonl",
        dict(
            update=update,
            task_objective=1.8,
            raw_gradient_norm=0.103,
            update_seconds=5.5,
            finite=True,
        ),
    )
    write(
        run / "evaluation.jsonl",
        dict(update=100, task_objective=2.3, raptor_share_terminated=0.4),
    )
    logs = monitor.TrainingLogs(run)
    logs.poll()
    return logs


def status(logs):
    assert callable(
        getattr(logs, "runtime_status", None)
    ), "GUI lacks exit/error status"
    return logs.runtime_status()


def test_oom_exit_shows_reason_completed_update_and_exit_code(tmp_path):
    logs = logs_at(tmp_path)
    error = "OutOfMemoryError: CUDA out of memory. Process 123 used 2.85 GiB."
    write(tmp_path / "summary.json", dict(status="failed", updates=133, error=error))
    write(
        tmp_path / "training_exit.json",
        dict(exit_code=1, summary=dict(status="failed", updates=133, error=error)),
    )
    logs.poll()
    current = status(logs)
    assert current.state == "failed" and "FAILED" in current.title
    assert current.update == 133 and current.exit_code == 1
    assert current.error == error and "133" in current.detail


def test_normal_update_budget_is_visible_and_has_no_failure(tmp_path):
    logs = logs_at(tmp_path, 50)
    write(
        tmp_path / "training_exit.json",
        dict(exit_code=0, summary=dict(status="update_budget", updates=50)),
    )
    current = status(logs)
    assert current.state == "stopped" and current.exit_code == 0 and not current.error
    assert "update_budget" in current.detail


def test_user_interrupt_is_distinct_from_a_crash(tmp_path):
    logs = logs_at(tmp_path)
    write(tmp_path / "summary.json", dict(status="interrupted", updates=133))
    logs.poll()
    current = status(logs)
    assert current.state == "interrupted" and not current.error


def test_recorded_user_interrupt_with_signal_code_is_not_a_crash(tmp_path):
    logs = logs_at(tmp_path)
    write(
        tmp_path / "training_exit.json",
        dict(exit_code=130, summary=dict(status="interrupted", updates=133)),
    )
    current = status(logs)
    assert current.state == "interrupted" and current.exit_code == 130


def test_signal_exit_without_summary_is_not_reported_as_oom(tmp_path):
    logs = logs_at(tmp_path)
    write(tmp_path / "training_exit.json", dict(exit_code=-9))
    current = status(logs)
    assert current.state == "failed" and current.exit_code == -9
    assert "SIGKILL" in current.detail and "out of memory" not in current.error.lower()


def test_long_eval_silence_does_not_make_live_process_a_crash(tmp_path, monkeypatch):
    logs = logs_at(tmp_path)
    write(tmp_path / "processes.json", dict(training_supervisor=123))
    monkeypatch.setattr(monitor, "_live_training_pid", lambda *args: 123, raising=False)
    current = status(logs)
    assert current.state == "running" and current.pid == 123 and not current.error


def test_new_resume_does_not_inherit_old_failure_banner(tmp_path, monkeypatch):
    logs = logs_at(tmp_path, 134)
    now = datetime.now(timezone.utc)
    write(
        tmp_path / "active_launch.json",
        dict(started_at_utc=now.isoformat(), start_update=133),
    )
    write(
        tmp_path / "summary.json", dict(status="failed", updates=133, error="old OOM")
    )
    os.utime(tmp_path / "summary.json", ((now - timedelta(hours=1)).timestamp(),) * 2)
    write(
        tmp_path / "training_exit.json",
        dict(
            exit_code=1,
            finished_at_utc=(now - timedelta(minutes=5)).isoformat(),
            summary=dict(status="failed", updates=133, error="old OOM"),
        ),
    )
    write(tmp_path / "processes.json", dict(training_supervisor=123))
    monkeypatch.setattr(monitor, "_live_training_pid", lambda *args: 123, raising=False)
    logs.poll()
    current = status(logs)
    assert current.state == "running" and not current.error


def test_legacy_logs_do_not_invent_process_liveness(tmp_path):
    logs = logs_at(tmp_path)
    current = status(logs)
    assert current.state == "unknown" and current.pid is None
    assert "process" in current.detail.lower()


def test_registered_process_disappearing_has_visible_unrecorded_exit(
    tmp_path, monkeypatch
):
    logs = logs_at(tmp_path)
    write(tmp_path / "processes.json", dict(training_supervisor=123))
    monkeypatch.setattr(
        monitor, "_live_training_pid", lambda *args: None, raising=False
    )
    current = status(logs)
    assert current.state == "exited" and current.exit_code is None
    assert "no exit record" in current.detail.lower()


def test_python_traceback_is_shown_when_structured_error_is_missing(tmp_path):
    logs = logs_at(tmp_path)
    write(tmp_path / "training_exit.json", dict(exit_code=1))
    (tmp_path / "training.stdout.log").write_text(
        "x" * 100000
        + '\nTraceback (most recent call last):\n  File "train.py", line 2\nRuntimeError: corrupt checkpoint\n'
    )
    current = status(logs)
    assert "RuntimeError: corrupt checkpoint" in current.error
    assert len(current.error) <= 32768


def test_old_traceback_followed_by_successful_updates_is_not_new_error(
    tmp_path, monkeypatch
):
    logs = logs_at(tmp_path)
    write(tmp_path / "processes.json", dict(training_supervisor=123))
    monkeypatch.setattr(monitor, "_live_training_pid", lambda *args: None)
    (tmp_path / "training.stdout.log").write_text(
        "Traceback (most recent call last):\nRuntimeError: OLD\n"
        + json.dumps(dict(update=133, finite=True))
        + "\n"
    )
    current = status(logs)
    assert current.state == "exited" and not current.error


def test_exit_summary_takes_precedence_over_stale_cached_summary(tmp_path):
    logs = logs_at(tmp_path)
    write(tmp_path / "summary.json", dict(status="update_budget", updates=133))
    logs.poll()
    write(
        tmp_path / "training_exit.json",
        dict(
            exit_code=1, summary=dict(status="failed", updates=133, error="new failure")
        ),
    )
    current = status(logs)
    assert current.state == "failed" and current.error == "new failure"


def test_partial_exit_metadata_is_visible_without_crashing_reader(tmp_path):
    logs = logs_at(tmp_path)
    (tmp_path / "training_exit.json").write_text('{"exit_code":')
    current = status(logs)
    assert current.state == "unknown" and "metadata" in current.detail.lower()


def test_waiting_directory_and_unrelated_reused_pid_are_not_running(tmp_path):
    logs = monitor.TrainingLogs(tmp_path)
    logs.poll()
    current = status(logs)
    assert current.state == "waiting"
    write(tmp_path / "processes.json", dict(training_supervisor=os.getpid()))
    assert status(logs).state == "exited"  # this pytest PID is not that trainer


def test_failure_dashboard_renders_full_error_details_read_only(tmp_path):
    pytest.importorskip("pygame")
    logs_at(tmp_path)
    error = "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 MiB."
    write(
        tmp_path / "training_exit.json",
        dict(exit_code=1, summary=dict(status="failed", updates=133, error=error)),
    )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    image = tmp_path / "failed.png"
    env = dict(
        os.environ,
        SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy",
        PYGAME_HIDE_SUPPORT_PROMPT="1",
        CUDA_VISIBLE_DEVICES="",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/monitor_response_training.py"),
            "--run-dir",
            str(tmp_path),
            "--screenshot",
            str(image),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert image.stat().st_size > 1000
    import pygame

    # A crash must be visibly red on the dashboard, not only present in logs.
    rendered = pygame.image.load(str(image))
    assert rendered.get_at((1450, 211))[:3] == (65, 26, 34)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p != image}

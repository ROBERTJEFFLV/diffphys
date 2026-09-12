"""Training exit semantics shared with launchers; business stops are not crashes."""
from __future__ import annotations


def exit_class(status: str, returncode: int = 0) -> str:
    if returncode != 0 or status == "failed":
        return "crash"
    if status in ("update_budget", "time_budget"):
        return "budget"
    if status == "interrupted":
        return "interrupted"
    return "running" if status == "training" else "unknown"

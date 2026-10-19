"""Training exit semantics shared with launchers; business stops are not crashes."""
from __future__ import annotations

from contextlib import contextmanager
import gc


def exit_class(status: str, returncode: int = 0) -> str:
    if returncode != 0 or status == "failed":
        return "crash"
    if status in ("update_budget", "time_budget"):
        return "budget"
    if status == "interrupted":
        return "interrupted"
    return "running" if status == "training" else "unknown"


@contextmanager
def critical_gc():
    """Defer automatic cyclic GC only until the completed graph is released."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        (gc.enable if enabled else gc.disable)()

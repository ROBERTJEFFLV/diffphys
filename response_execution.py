"""Training exit semantics shared with launchers; business stops are not crashes."""
from __future__ import annotations


BUSINESS_STOPS = frozenset({
    "adam_rejections", "adam_development_rollback", "solver_rejections",
    "development_plateau", "proposal_plateau",
})


def exit_class(status: str, returncode: int = 0) -> str:
    if returncode != 0 or status in ("failed", "adam_nonfinite_proposal"):
        return "crash"
    if status in BUSINESS_STOPS:
        return "business_stop"
    if status in ("update_budget", "time_budget"):
        return "budget"
    if status == "interrupted":
        return "interrupted"
    return "running" if status == "training" else "unknown"


def training_summary(progress: dict, protocol: str) -> dict:
    return {
        "protocol": protocol, "status": progress["status"], "exit_class": exit_class(progress["status"]),
        "actual_attempts": progress["attempts"], "actual_updates": progress["updates"],
        "elapsed_seconds": progress["elapsed_seconds"], "best_update": progress["best_update"],
        "baseline_development_score": progress["baseline_score"], "best_development_score": progress["best_score"],
        "rollback": progress.get("rollback"), "final_seeds_consumed": [],
        "formal_eligible": False, "deployment_authorized": False,
    }

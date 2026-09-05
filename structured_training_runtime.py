"""Durable training progress and one-time, candidate-bound final evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import signal
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from structured_checkpoint import CADENCE_SEMANTICS_VERSION, runtime_contract_hash, sha256_file

ROOT = Path(__file__).resolve().parent
SCHEMA = "structured_training_resume_v1"


def add_training_arguments(parser) -> None:
    parser.add_argument("--training-state", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--development-every", type=int, default=50)
    parser.add_argument("--minimum-updates", type=int, default=300)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.002)
    parser.add_argument("--max-seconds", type=float, default=10800.0)
    parser.add_argument("--final-evaluation", action="store_true",
                        help="evaluate the already frozen development candidate; perform no updates")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def training_code_hash() -> str:
    names = ["structured_training_runtime.py", "structured_checkpoint.py",
             "structured_distillation.py", "structured_local_distillation.py"]
    names += [str(path.relative_to(ROOT)) for path in sorted((ROOT / "tools").glob("*structured*.py"))]
    hashes = {name: sha256_file(ROOT / name) for name in names}
    hashes["runtime"] = runtime_contract_hash()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def equilibrium_score(gate: dict) -> float:
    mapping = {"body_z_angle_p99_degrees": "body_z_p99_degrees",
               "body_z_angle_max_degrees": "body_z_max_degrees",
               "disturbance_error_rms_over_g": "disturbance_rms_over_g",
               "disturbance_error_p99_over_g": "disturbance_p99_over_g"}
    thresholds = gate["thresholds"]
    ratios = []
    for row in gate["rows"]:
        for key, value in row.items():
            target = mapping.get(key, key)
            if target in thresholds and isinstance(value, (int, float)):
                ratios.append(float(value) / max(float(thresholds[target]), 1e-12))
        ratios.extend(float(v) / thresholds["capability_axis_z_rms"] for v in row["capability_axis_z_rms"])
        if row["motor_observer_p95"] is None or row["motor_observer_max"] is None:
            return float("inf")
    return max(ratios, default=float("inf"))


class TrainingSession:
    """Checkpoints are training artifacts, never downstream promotion records."""

    def __init__(self, args, policy, optimizer, *, stage: str) -> None:
        if int(args.seed) != 7:
            raise ValueError("this registered training run permits model seed 7 only")
        self.args, self.policy, self.optimizer, self.stage = args, policy, optimizer, stage
        self.path = args.training_state or args.output.with_suffix(".training.pt")
        self.best_path = self.path.with_suffix(".best.pt")
        self.candidate_path = self.path.with_suffix(".candidate.pt")
        self.candidate_record = self.path.with_suffix(".candidate.json")
        self.development_report = args.report.with_name(args.report.stem + "_development.json")
        self.final_claim = ROOT / "reports/structured_seed7_final_claims" / (stage + ".json")
        self.started = time.monotonic()
        self.previous_elapsed = 0.0
        self.stop_requested = False
        ignored = {"training_state", "checkpoint_every", "development_every", "minimum_updates",
                   "patience", "minimum_relative_improvement", "max_seconds", "final_evaluation",
                   "updates", "updates_per_beta", "iterations", "outer_steps", "dry_run"}
        configuration = {k: str(v) if isinstance(v, Path) else v
                         for k, v in vars(args).items() if k not in ignored}
        sources = {}
        for name in ("source_checkpoint", "checkpoint", "student_checkpoint", "causal_oracle_report",
                     "identifier_init_artifact", "identification_oracle_report", "probe_v4_report"):
            value = getattr(args, name, None)
            if value is not None and Path(value).is_file():
                sources[name] = sha256_file(value)
        self.binding = {"stage": stage, "configuration": configuration, "source_sha256": sources,
                        "policy_config": asdict(policy.config), "code_sha256": training_code_hash(),
                        "runtime_contract_sha256": runtime_contract_hash(),
                        "cadence_semantics_version": CADENCE_SEMANTICS_VERSION}
        self.progress: dict[str, Any] = {"updates": 0, "history": [], "development": [],
            "best_score": None, "best_passed": False, "bad_checks": 0, "status": "training"}
        if self.path.is_file():
            self.restore(self.path)
        elif args.final_evaluation:
            raise RuntimeError("final evaluation requires a saved training session")
        if self.final_claim.is_file() and not args.final_evaluation:
            raise RuntimeError("this stage already consumed final data; its final bank cannot be reused for tuning")
        if not args.final_evaluation:
            for signum in (signal.SIGINT, signal.SIGTERM):
                signal.signal(signum, self._request_stop)

    def _request_stop(self, signum, frame) -> None:
        self.stop_requested = True

    @property
    def elapsed(self) -> float:
        return self.previous_elapsed + time.monotonic() - self.started

    @property
    def updates(self) -> int:
        return int(self.progress["updates"])

    def restore(self, path: Path) -> None:
        payload = torch.load(path, map_location=next(self.policy.parameters()).device, weights_only=False)
        if payload.get("schema") != SCHEMA or payload.get("binding") != self.binding:
            raise RuntimeError("training checkpoint source/config/code binding is stale: " + str(path))
        self.policy.load_state_dict(payload["model"], strict=True)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.progress = payload["progress"]
        self.previous_elapsed = float(payload["elapsed_seconds"])
        self.started = time.monotonic()
        rng = payload["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])

    def save(self, path: Path | None = None) -> None:
        atomic_torch(path or self.path, {"schema": SCHEMA, "binding": self.binding,
            "architecture": "structured-recurrent-motor-policy", "config": asdict(self.policy.config),
            "model": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "progress": self.progress, "elapsed_seconds": self.elapsed,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []},
            "formal_eligible": False, "deployment_authorized": False})

    def record_update(self, row: dict) -> None:
        self.progress["updates"] += 1
        row = {**row, "update": self.updates, "elapsed_seconds": self.elapsed}
        self.progress["history"].append(row)
        if self.updates % self.args.checkpoint_every == 0:
            self.save()
            print(json.dumps({"stage": self.stage, **row}), flush=True)

    def development_due(self) -> bool:
        return self.updates == 1 or self.updates % self.args.development_every == 0

    def record_development(self, *, score: float, passed: bool, metrics: dict) -> None:
        finite_score = float(score) if math.isfinite(score) else None
        record = {"update": self.updates, "score": finite_score, "passed": bool(passed),
                  "metrics": metrics, "elapsed_seconds": self.elapsed}
        self.progress["development"].append(record)
        best = self.progress["best_score"]
        improved = finite_score is not None and (best is None or score < best * (1.0 - self.args.minimum_relative_improvement)
                                                or (passed and not self.progress["best_passed"]))
        if improved:
            self.progress.update(best_score=finite_score, best_passed=bool(passed),
                                 best_update=self.updates, bad_checks=0)
            self.save(self.best_path)
        else:
            self.progress["bad_checks"] += 1
        self.save()
        atomic_json(self.development_report, self.summary())
        print(json.dumps({"stage": self.stage, "development": record}), flush=True)

    def should_stop(self) -> bool:
        if self.stop_requested or self.elapsed >= self.args.max_seconds:
            return True
        return self.updates >= self.args.minimum_updates and (
            self.progress["best_passed"] or self.progress["bad_checks"] >= self.args.patience)

    def finish_development(self, *, complete_schedule: bool = True) -> dict:
        ready = bool(self.progress["best_passed"] and self.updates >= self.args.minimum_updates and complete_schedule)
        reason = ("candidate_ready" if ready else "interrupted" if self.stop_requested
                  else "walltime_budget_exhausted" if self.elapsed >= self.args.max_seconds
                  else "development_plateau" if self.progress["bad_checks"] >= self.args.patience
                  else "update_budget_exhausted")
        self.progress["status"] = reason
        self.save()
        if ready:
            best = torch.load(self.best_path, map_location="cpu", weights_only=False)
            atomic_torch(self.candidate_path, best)
            atomic_json(self.candidate_record, {"stage": self.stage,
                "candidate_sha256": sha256_file(self.candidate_path), "binding": self.binding,
                "selected_update": best["progress"]["updates"], "development_gate_passed": True,
                "executed_training_updates": self.updates, "training_elapsed_seconds": self.elapsed,
                "validation_consumed": [], "final_consumed": []})
        payload = self.summary()
        atomic_json(self.development_report, payload)
        return payload

    def begin_final(self, seeds: list[int]) -> None:
        if not self.args.final_evaluation:
            raise RuntimeError("final evaluation requires its explicit mode")
        record = json.loads(self.candidate_record.read_text())
        if (record["binding"] != self.binding or record["development_gate_passed"] is not True
                or record["candidate_sha256"] != sha256_file(self.candidate_path)):
            raise RuntimeError("the frozen development candidate is missing or stale")
        self.restore(self.candidate_path)
        self.final_selection = record
        self.final_claim.parent.mkdir(parents=True, exist_ok=True)
        with self.final_claim.open("x") as stream:
            json.dump({**record, "seeds": seeds, "status": "claimed_before_collection",
                       "report": str(self.args.report.resolve())}, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())

    def summary(self) -> dict:
        return {"stage": self.stage, "status": self.progress["status"], "seed": 7,
                "actual_updates": getattr(self, "final_selection", {}).get("executed_training_updates", self.updates),
                "selected_update": self.updates, "elapsed_seconds": self.elapsed,
                "training_checkpoint": str(self.path.resolve()),
                "best_checkpoint": str(self.best_path.resolve()),
                "development_gate_passed": bool(self.progress["best_passed"]),
                "candidate_ready": self.progress["status"] == "candidate_ready",
                "best_score": self.progress["best_score"], "best_update": self.progress.get("best_update"),
                "history": list(self.progress["history"]), "development": list(self.progress["development"]),
                "binding": self.binding, "formal_gate_passed": False,
                "pretraining_gate_passed": False, "deployment_authorized": False,
                "validation_consumed": [], "final_consumed": []}

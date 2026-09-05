"""Reproduce v5 physics/observer ceilings using training seeds only.

No optimizer, validation claim, engineering-validation bank, or blind bank is
used. The privileged fits score feasibility, not deployable identifier accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from joblib import Parallel, delayed
from tools.diagnose_causal_identifier_oracle import (
    FORMAL_TRAIN_SEEDS as TRAIN_SEEDS, _collect_one, _coverage_summary,
    analytic_privileged_ceiling as _physics_ceiling, _finite_collection,
)
from tools.diagnose_probe_v5 import DEFAULT_CHECKPOINT, code_hashes, file_hash
from probe_contract_v5 import CONTRACT_SHA256, Q2_SHA256


def audit_seed(checkpoint: Path, seed: int) -> dict:
    torch.set_num_threads(1)
    row = _collect_one(checkpoint, seed, 0.0, 128, 126)
    return {"seed": seed, "scenarios": 128, "horizon": 126,
            "zero_parity": row["zero_parity"], "finite": _finite_collection(row),
            "coverage": _coverage_summary([row]), "physics": _physics_ceiling([row])}


def json_safe(value):
    """Represent undefined diagnostic conditions as null, retaining failures."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if file_hash(args.checkpoint) != Q2_SHA256:
        raise ValueError("this audit is registered for the frozen Q2 checkpoint")
    rows = Parallel(n_jobs=args.n_jobs)(delayed(audit_seed)(args.checkpoint, seed) for seed in TRAIN_SEEDS)
    report = {"diagnostic": "passive-identification-train-ceilings-v5",
              "contract_sha256": CONTRACT_SHA256, "code_sha256": code_hashes(),
              "audit_code_sha256": file_hash(Path(__file__)), "q2_sha256": Q2_SHA256,
              "train": rows, "validation_consumed": [], "blind_consumed": [],
              "formal_eligible": False, "optimizer_updates": 0,
              "scope": "privileged physical feasibility and continuous-observer ceiling; not learned mean accuracy",
              "train_passed": all(r["finite"] and r["zero_parity"]["gate_passed"]
                                  and r["coverage"]["passed"] and r["physics"]["gate_passed"] for r in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "train_passed": report["train_passed"]}))


if __name__ == "__main__":
    main()

"""Paired closed-loop validation of the registered collective motor-lag probe.

Default runs consume training banks only.  --formal-freeze claims the single
validation attempt before any validation rollout.  Neither path uses blind data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probe_contract_v5 import (AMPLITUDE, CONTRACT_SHA256, DWELL_STEPS, ACTIVE_STEPS,
    FORMAL_HORIZON, FORMAL_SCENARIOS, TRAIN_SEEDS, VALIDATION_SEED, Q2_SHA256,
    ProbeState, apply_probe, metadata)
from diagnostics.formal_rollout import load_q2_policy
from env_l2f import L2FSimulator, L2FParams
from policy_observation import build_policy_observation, initial_observation_state, update_position_integral
from structured_distillation import build_dagger_scenario_bank
from tools.diagnose_causal_identifier_oracle import _q2_settings, _clone_state
from tools.diagnose_probe_v4 import shared_tau_diagnostics

DEFAULT_CHECKPOINT = ROOT / "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt"
DEFAULT_REPORT = ROOT / "reports/probe_v5_formal.json"
CLAIM_PATH = ROOT / "reports/probe_v5_validation_claim.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes() -> dict:
    return {name: file_hash(ROOT / name) for name in (
        "probe_contract_v5.py", "tools/diagnose_probe_v5.py", "structured_distillation.py",
        "tools/diagnose_probe_v4.py", "tools/diagnose_causal_identifier_oracle.py",
        "identification_information.py", "structured_policy.py", "identification_features.py",
        "env_l2f.py", "model.py", "policy_observation.py", "diagnostics/formal_rollout.py")}


@torch.no_grad()
def collect_pair(checkpoint: Path, seed: int, scenarios: int, horizon: int,
                 active_probe: bool = False) -> dict:
    torch.set_num_threads(1)
    policy, args = load_q2_policy(checkpoint, device="cpu", dtype=torch.float32)
    settings = _q2_settings(args, policy)
    if settings["noise_max"] != 0 or settings["dt"] != 0.01:
        raise ValueError("registered paired protocol requires deterministic Q2 observations and dt=.01")
    bank = build_dagger_scenario_bank(scenarios, seed=seed, dt=settings["dt"], per_cell=scenarios // 16)
    simulator = L2FSimulator(L2FParams(dt=settings["dt"]))

    def arm(amplitude: float | None) -> dict:
        physical = _clone_state(bank.state)
        hidden = policy.initial_hidden(scenarios, device="cpu", dtype=torch.float32)
        obs_state = initial_observation_state(scenarios, device="cpu", dtype=torch.float32)
        probe = ProbeState.initial(physical.previous_action)
        rows = {name: [] for name in ("position", "velocity", "omega", "action", "motor_before", "motor_after", "requested", "residual", "aborted")}
        for call in range(horizon):
            obs, position = build_policy_observation(physical, obs_state,
                mode=settings["mode"], noise_max=settings["noise_max"],
                integral_input_frame=settings["integral_input_frame"],
                integral_input_multiplier=settings["integral_input_multiplier"])
            base, hidden = policy(obs, hidden)
            if amplitude is None:
                action, requested = base, torch.zeros_like(base[:, :1])
            else:
                action, probe, requested = apply_probe(base, probe, call,
                    position=physical.position, velocity=physical.velocity,
                    omega=physical.omega, body_z=physical.rotation[:, :, 2], amplitude=amplitude)
            obs_state = update_position_integral(obs_state, position, dt=settings["dt"],
                integral_limit=settings["integral_limit"], integral_leak=settings["integral_leak"],
                integral_clamp_mode=settings["integral_clamp_mode"])
            end = simulator.step(physical, action, grad_decay=1.0)
            for name in ("position", "velocity", "omega"):
                rows[name].append(getattr(end, name).norm(dim=-1))
            for name, value in (("action", action), ("motor_before", physical.motor),
                                ("motor_after", end.motor), ("requested", requested),
                                ("residual", action - base), ("aborted", probe.aborted)):
                rows[name].append(value.clone())
            physical = end
        return {name: torch.stack(values) for name, values in rows.items()}

    baseline, zero, probe = arm(None), arm(0.0), arm(AMPLITUDE if active_probe else 0.0)
    parity = max(float((baseline[k] - zero[k]).abs().max()) for k in ("position", "velocity", "omega", "action", "motor_after"))
    window = slice(DWELL_STEPS, DWELL_STEPS + ACTIVE_STEPS)
    identification_window = slice(0, 100)
    tau = shared_tau_diagnostics(probe["action"][identification_window], probe["motor_before"][identification_window],
                                 motor_after=probe["motor_after"][identification_window], support_steps=100)
    # Historical quantity is an excitation proxy, NOT a statistical Fisher
    # matrix for the deployable observation model (which has nuisance terms).
    for scene in tau["per_scene"]:
        for branch in scene["branches"].values():
            branch["weighted_excitation_proxy"] = branch.pop("weighted_fisher_information")
    retention = probe["residual"][window].square().sum((0, 2)) / (
        4 * probe["requested"][window].square().sum((0, 2))).clamp_min(1e-15)
    supported = torch.tensor([row["gate_passed"] for row in tau["per_scene"]])
    eligible = supported & ~probe["aborted"].any(0)
    if active_probe:
        eligible = eligible & (retention >= .90)
    cells = (bank.tw_bin * 4 + bank.log_alpha_bin).long()
    coverage = [float(eligible[cells == cell].float().mean()) for cell in range(16)]
    metrics = paired_metrics(baseline, probe, cells)
    checks = {
        "zero_parity": parity <= 1e-7,
        "finite": all(bool(torch.isfinite(v).all()) for a in (baseline, probe) for v in a.values()),
        "paired_safety": all(v["passed"] for v in metrics.values()),
        "coverage": float(eligible.float().mean()) >= 0.90 and min(coverage) >= 0.50,
        "shared_tau_support": tau["gate_passed"],
        "exact_collective_residual": bool((probe["residual"] - probe["residual"][..., :1]).abs().max() <= 1.2e-7),
    }
    return {"seed": seed, "scenarios": scenarios, "horizon": horizon,
            "active_probe": active_probe, "checks": checks,
            "passed": all(checks.values()), "zero_parity_max": parity,
            "coverage": float(eligible.float().mean()), "coverage_by_cell": coverage,
            "aborted_count": int(probe["aborted"].any(0).sum()),
            "retention_min": float(retention.min()) if active_probe else None,
            "shared_tau": tau, "metrics": metrics}


def paired_metrics(zero: dict, probe: dict, cells: torch.Tensor) -> dict:
    """Pooled time/scene p99; never average per-scene quantiles.

    Preserve the v4 endpoint/tail gates, add the entire trajectory and every
    authority cell.  Aborted scenes remain in every safety comparison.
    """
    rows = {}
    for scope, selection in [("all", torch.ones_like(cells, dtype=torch.bool)),
                             *[(f"cell{c}", cells == c) for c in range(16)]]:
        for window, times in (("h75", 74), ("h125", 124),
                              ("tail75_125", slice(74, 125)), ("full", slice(None))):
            for name in ("position", "velocity", "omega"):
                a, b = probe[name][times][..., selection].double(), zero[name][times][..., selection].double()
                for statistic in ("mean", "p99"):
                    actual = float(a.mean() if statistic == "mean" else a.quantile(.99))
                    baseline = float(b.mean() if statistic == "mean" else b.quantile(.99))
                    allowed = 1.05 * baseline + 1e-7
                    rows[f"{scope}/{window}/{name}/{statistic}"] = {
                        "actual": actual, "baseline": baseline, "allowed": allowed,
                        "ratio": actual / max(baseline, 1e-12), "passed": actual <= allowed}
        a, b = probe["action"][:, selection], zero["action"][:, selection]
        actual, baseline = float((a.abs() >= .999).float().mean()), float((b.abs() >= .999).float().mean())
        rows[f"{scope}/saturation"] = {"actual": actual, "baseline": baseline,
            "allowed": baseline + .001, "passed": actual <= baseline + .001}
    for name, floor in (("position", 5.0), ("velocity", 20.0), ("omega", 5.0)):
        actual, baseline = float(probe[name].max()), float(zero[name].max())
        allowed = max(floor, 1.5 * baseline)
        rows[f"peak/{name}"] = {"actual": actual, "baseline": baseline,
                                "allowed": allowed, "passed": actual <= allowed}
    return rows


def eligibility(path: Path = DEFAULT_REPORT, *, q2_checkpoint: Path = DEFAULT_CHECKPOINT) -> dict:
    result = {"eligible": False, "reason": "v5 probe has no valid formal freeze"}
    try:
        report = json.loads(path.read_text())
        if (report["contract"] != metadata() or report["contract_sha256"] != CONTRACT_SHA256
                or report["code_sha256"] != code_hashes() or report["q2_sha256"] != Q2_SHA256
                or file_hash(q2_checkpoint) != Q2_SHA256 or report["formal_eligible"] is not True
                or report["validation_consumed"] != [VALIDATION_SEED]
                or report["blind_consumed"] != []):
            return result
        claim = json.loads(CLAIM_PATH.read_text())
        if claim["report_sha256"] != file_hash(path) or claim["contract_sha256"] != CONTRACT_SHA256:
            return result
        rows = report["train"] + [report["validation"]]
        if [r["seed"] for r in rows] != [*TRAIN_SEEDS, VALIDATION_SEED]:
            return result
        for row in rows:
            expected_checks = {"zero_parity", "finite", "paired_safety", "coverage", "shared_tau_support", "exact_collective_residual"}
            if (row.get("active_probe") is not False
                    or row["scenarios"] != FORMAL_SCENARIOS or row["horizon"] != FORMAL_HORIZON
                    or set(row["checks"]) != expected_checks or any(v is not True for v in row["checks"].values())):
                return result
            expected_metrics = {f"{scope}/{window}/{axis}/{stat}"
                for scope in ["all", *[f"cell{c}" for c in range(16)]]
                for window in ("h75", "h125", "tail75_125", "full")
                for axis in ("position", "velocity", "omega") for stat in ("mean", "p99")}
            expected_metrics.update(f"{scope}/saturation" for scope in ["all", *[f"cell{c}" for c in range(16)]])
            expected_metrics.update(f"peak/{axis}" for axis in ("position", "velocity", "omega"))
            if set(row["metrics"]) != expected_metrics:
                return result
            for key, value in row["metrics"].items():
                if type(value["actual"]) not in (float, int) or type(value["baseline"]) not in (float, int):
                    return result
                actual, baseline = float(value["actual"]), float(value["baseline"])
                if key.startswith("peak/"):
                    limit = max({"position": 5., "velocity": 20., "omega": 5.}[key.split("/")[1]], 1.5 * baseline)
                else:
                    limit = baseline + .001 if key.endswith("saturation") else 1.05 * baseline + 1e-7
                if not (0 <= actual <= limit and 0 <= baseline < float("inf")):
                    return result
        return {"eligible": True, "reason": "frozen collective v5 probe", "frozen_sha256": CONTRACT_SHA256}
    except (OSError, ValueError, KeyError, TypeError):
        return result


def run(args: argparse.Namespace) -> dict:
    if getattr(args, "active_probe", False) and args.formal_freeze:
        raise ValueError("active probe experiments cannot freeze the passive-first protocol")
    if args.scenarios < 16 or args.scenarios % 16 or args.horizon < 125:
        raise ValueError("require complete 4x4 authority cells and horizon>=125")
    if file_hash(args.checkpoint) != Q2_SHA256:
        raise ValueError("v5 requires the unchanged canonical Q2 checkpoint")
    report = {"contract": metadata(), "contract_sha256": CONTRACT_SHA256,
              "code_sha256": code_hashes(), "q2_sha256": Q2_SHA256,
              "validation_consumed": [], "blind_consumed": [], "formal_eligible": False}
    report["train"] = Parallel(n_jobs=args.n_jobs)(delayed(collect_pair)(
        args.checkpoint, seed, args.scenarios, args.horizon,
        getattr(args, "active_probe", False)) for seed in TRAIN_SEEDS)
    report["train_passed"] = all(row["passed"] for row in report["train"])
    if args.formal_freeze and report["train_passed"]:
        if args.scenarios != FORMAL_SCENARIOS or args.horizon != FORMAL_HORIZON:
            raise ValueError("formal freeze requires the exact registered size and horizon")
        claim = {"contract_sha256": CONTRACT_SHA256, "code_sha256": report["code_sha256"],
                 "report": str(args.output.resolve()), "status": "validation_claimed"}
        CLAIM_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CLAIM_PATH.open("x") as f:
            json.dump(claim, f, indent=2); f.flush(); os.fsync(f.fileno())
        report["validation_consumed"] = [VALIDATION_SEED]
        report["validation"] = collect_pair(args.checkpoint, VALIDATION_SEED, args.scenarios, args.horizon)
        report["formal_eligible"] = report["validation"]["passed"]
    report["formal"] = {"eligible": report["formal_eligible"], "gate_passed": report["formal_eligible"]}
    report["gate_passed"] = report["formal_eligible"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/probe_v5_train.json")
    parser.add_argument("--scenarios", type=int, default=FORMAL_SCENARIOS)
    parser.add_argument("--horizon", type=int, default=FORMAL_HORIZON)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--formal-freeze", action="store_true")
    parser.add_argument("--active-probe", action="store_true",
                        help="training-only experiment; never a formal passive-first release")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new report path")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    if report["validation_consumed"]:
        claim = json.loads(CLAIM_PATH.read_text())
        claim.update(status="validation_completed", report_sha256=file_hash(args.output))
        CLAIM_PATH.write_text(json.dumps(claim, indent=2))
    print(json.dumps({"train_passed": report["train_passed"],
        "formal_eligible": report["formal_eligible"], "validation_consumed": report["validation_consumed"]}))


if __name__ == "__main__":
    main()

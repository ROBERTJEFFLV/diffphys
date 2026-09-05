from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from probe_contract_v5 import ProbeState, apply_probe, CONTRACT_SHA256, metadata
from identification_information import advance_information, capability_support
from tools import diagnose_probe_v5 as diagnostic


def passing_freeze(path: Path, monkeypatch) -> None:
    """Synthetic evidence for consumer tests; never consumes the real ledger."""
    metrics = diagnostic.paired_metrics(
        {**{key: torch.ones(125, 64) for key in ("position", "velocity", "omega")}, "action": torch.zeros(125, 64, 4)},
        {**{key: torch.ones(125, 64) for key in ("position", "velocity", "omega")}, "action": torch.zeros(125, 64, 4)},
        torch.arange(16).repeat_interleave(4))
    def row(seed):
        return {"seed": seed, "scenarios": 64, "horizon": 125, "active_probe": False,
                "checks": {k: True for k in ("zero_parity", "finite", "paired_safety", "coverage", "shared_tau_support", "exact_collective_residual")},
                "metrics": metrics}
    report = {"contract": metadata(), "contract_sha256": CONTRACT_SHA256,
              "code_sha256": diagnostic.code_hashes(), "q2_sha256": diagnostic.Q2_SHA256,
              "formal_eligible": True, "gate_passed": True,
              "formal": {"eligible": True, "gate_passed": True},
              "validation_consumed": [7707], "blind_consumed": [],
              "train": [row(seed) for seed in diagnostic.TRAIN_SEEDS], "validation": row(7707)}
    path.write_text(json.dumps(report))
    claim = path.with_suffix(".claim.json")
    claim.write_text(json.dumps({"contract_sha256": CONTRACT_SHA256,
                                "report_sha256": diagnostic.file_hash(path)}))
    monkeypatch.setattr(diagnostic, "CLAIM_PATH", claim)


def test_passive_command_is_bitwise_base_and_collective_does_not_clip_per_motor():
    base = torch.tensor([[.999, -.2, .3, -.4]], dtype=torch.float64)
    args = dict(position=torch.zeros(1, 3), velocity=torch.zeros(1, 3),
                omega=torch.zeros(1, 3), body_z=torch.tensor([[0., 0., 1.]]))
    zero, _, _ = apply_probe(base, ProbeState.initial(base), 25, **args)
    assert torch.equal(zero, base)
    active, state, _ = apply_probe(base, ProbeState.initial(base), 25, amplitude=.005, **args)
    torch.testing.assert_close(active - base, torch.full_like(base, .001))
    assert float(active.max()) <= 1.
    assert state.residual.shape == (1, 1)


def test_probe_abort_is_sticky_and_an_empty_intersection_never_leaves_the_box():
    base = torch.zeros(1, 4)
    args = dict(position=torch.zeros(1, 3), velocity=torch.zeros(1, 3),
                omega=torch.zeros(1, 3), body_z=torch.tensor([[0., 0., 1.]]))
    action, state, _ = apply_probe(base, ProbeState.initial(base), 25, amplitude=.005,
                                  lower=torch.ones_like(base), upper=-torch.ones_like(base), **args)
    assert state.aborted.all() and torch.equal(action, base)
    action, state, _ = apply_probe(base, state, 30, amplitude=.005, **args)
    assert state.aborted.all() and torch.equal(action, base)


def test_collective_history_cannot_authorize_angular_or_inertia_axes():
    ledger = torch.zeros(1, 49)
    for t in range(100):
        value = .2 if t % 4 < 2 else -.2
        ledger = advance_information(ledger, torch.full((1, 4), value),
            torch.tensor([[0., 0., value]]), torch.ones(1, 3), torch.ones(1, 3), torch.ones(1, 1))
    support = capability_support(ledger)
    assert support[:, (0, 4, 5)].all()
    assert not support[:, 1:4].any()
    ledger[0, 3] = float("nan")
    assert not capability_support(ledger).any()


def test_formal_consumer_rejects_changed_evidence_and_never_accepts_train_only(tmp_path, monkeypatch):
    path = tmp_path / "freeze.json"
    passing_freeze(path, monkeypatch)
    assert diagnostic.eligibility(path)["eligible"]
    value = json.loads(path.read_text())
    value["train"][0]["metrics"]["all/full/omega/mean"]["actual"] = 1.051
    path.write_text(json.dumps(value))
    claim = json.loads(diagnostic.CLAIM_PATH.read_text())
    claim["report_sha256"] = diagnostic.file_hash(path)
    diagnostic.CLAIM_PATH.write_text(json.dumps(claim))
    assert not diagnostic.eligibility(path)["eligible"]
    value["formal_eligible"] = False
    path.write_text(json.dumps(value))
    assert not diagnostic.eligibility(path)["eligible"]


def test_paired_gate_keeps_an_aborted_bad_scene_in_p99():
    baseline = {key: torch.ones(125, 64) for key in ("position", "velocity", "omega")}
    baseline["action"] = torch.zeros(125, 64, 4)
    actual = {k: v.clone() for k, v in baseline.items()}
    actual["omega"][:, 0] = 10.
    rows = diagnostic.paired_metrics(baseline, actual, torch.arange(16).repeat_interleave(4))
    assert not rows["all/full/omega/p99"]["passed"]

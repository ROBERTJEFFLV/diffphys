from __future__ import annotations

import json
from argparse import Namespace

import torch

from tools.diagnose_probe_v3 import (
    PRE_REGISTERED_WAVEFORM,
    PROBE_AMPLITUDE,
    SEARCH_SEED,
    clip_residual,
    generate_candidate_family,
    lag_design,
    projected_coriolis_residual_ratio,
    run,
    search_probe_v3,
    standardized_rank_condition,
    validate_waveform,
    waveform_sha256,
)


def test_preregistered_probe_v3_is_fail_closed_and_reproducible() -> None:
    checks = validate_waveform()
    assert checks["gate_passed"]
    assert checks["gates"]["modal_covariance_ratio_ge_0p35"]
    assert checks["gates"]["multi_rotational_overlap_ge_8"]
    assert checks["gates"]["lag_design_rank12"]
    first = search_probe_v3(seed=SEARCH_SEED)
    second = search_probe_v3(seed=SEARCH_SEED)
    assert first is not None and second is not None
    assert first["waveform"] == second["waveform"]
    assert first["sha256"] == waveform_sha256()
    assert len(first["waveform"]) == 50
    assert all(len(row) == 4 for row in first["waveform"])
    assert set(value for row in first["waveform"] for value in row) <= {-1, 0, 1}
    table = torch.tensor(PRE_REGISTERED_WAVEFORM)
    assert torch.equal(table[:25].sum(0), torch.zeros(4, dtype=torch.int64))
    assert torch.equal(table[25:].sum(0), torch.zeros(4, dtype=torch.int64))


def test_probe_v3_fault_injection_fails_the_zero_sum_gate() -> None:
    faulty = [list(row) for row in PRE_REGISTERED_WAVEFORM]
    faulty[0][0] = 0
    checks = validate_waveform(faulty)
    assert not checks["gate_passed"]
    assert not checks["gates"]["per_block_zero_sum"]


def test_probe_v3_standardized_lag_design_is_full_rank() -> None:
    design = lag_design(torch.tensor(PRE_REGISTERED_WAVEFORM, dtype=torch.float64))
    result = standardized_rank_condition(design)
    assert result["feature_order"] == ["z_t", "z_t_minus_4", "z_t_minus_12"]
    assert result["rank"] == 12
    assert result["condition"] < 30.0


def test_residual_clip_respects_annulus_and_action_bounds() -> None:
    q2 = torch.tensor([[1.0, -1.0, 0.0, 0.25]])
    previous = torch.tensor([[0.004, -0.004, 0.005, -0.005]])
    requested = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    residual, lower, upper = clip_residual(requested, q2, previous)
    assert bool((residual >= lower).all() and (residual <= upper).all())
    assert bool((residual.abs() <= PROBE_AMPLITUDE + 1e-7).all())
    assert bool(((q2 + residual) >= -1.0).all() and ((q2 + residual) <= 1.0).all())
    assert float(residual[0, 0]) == 0.0  # q2 is at the deployable upper boundary


def test_coriolis_projection_ratio_faults_are_detected() -> None:
    a = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    collinear = a.clone()
    orthogonal = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]])
    assert float(projected_coriolis_residual_ratio(a, collinear)[0]) < 1.0e-7
    assert float(projected_coriolis_residual_ratio(a, orthogonal)[0]) > 1.0 - 1.0e-7


def test_dry_run_publishes_waveform_without_checkpoint(tmp_path) -> None:
    output = tmp_path / "probe.json"
    args = Namespace(checkpoint=None, output=output, scenarios=16, horizon=125,
                     n_jobs=1, dry_run=True)
    result = run(args)
    assert result["design_gate_passed"]
    assert not result["gate_passed"]  # no Q2 scoring means no frozen candidate
    output.write_text(json.dumps(result))
    assert json.loads(output.read_text())["waveform_sha256"] == waveform_sha256()


def test_bounded_search_scores_every_family_member_before_validation(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "q2.pt"
    checkpoint.write_bytes(b"q2")
    calls = []

    def fake_rollout(_checkpoint, seed, scenarios, horizon, waveform):
        calls.append(int(seed))
        return {"seed": int(seed)}

    monkeypatch.setattr("tools.diagnose_probe_v3._rollout_one", fake_rollout)
    monkeypatch.setattr("tools.diagnose_probe_v3._formal_gate",
                        lambda rows: (False, {"rows": len(rows), "seed_results": []}))
    monkeypatch.setattr("tools.diagnose_probe_v3._candidate_quality", lambda rows, design: (1,))
    args = Namespace(checkpoint=checkpoint, output=tmp_path / "report.json", scenarios=16,
                     horizon=125, n_jobs=1, dry_run=False)
    result = run(args)
    assert len(calls) == len(generate_candidate_family()) * 4
    assert sorted(set(calls)) == [3707, 4707, 5707, 6707]
    assert result["search"]["status"] == "stop_no_frozen_candidate"
    assert result["formal_frozen_sha256"] is None
    assert len(result["formal"]["candidate_scores"]) == len(generate_candidate_family())

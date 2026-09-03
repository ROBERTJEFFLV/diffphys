from __future__ import annotations

from types import SimpleNamespace

import torch

import tools.revalidate_structured_identifier as revalidation


def test_revalidation_defaults_use_disjoint_large_seed_ranges(tmp_path) -> None:
    args = revalidation.parse_args([
        "--phase-a1-checkpoint", str(tmp_path / "a1.pt"),
        "--phase-a1-report", str(tmp_path / "a1.json"),
        "--identifier-init-artifact", str(tmp_path / "identifier.pt"),
        "--identifier-pretrain-report", str(tmp_path / "pretrain.json"),
        "--output", str(tmp_path / "revalidated.pt"),
        "--report", str(tmp_path / "revalidated.json"),
    ])
    # A1 uses seeds near seed+100 and the pretrainer uses ranges near +100,
    # +10k, and +20k.  Revalidation must not silently reuse those banks.
    assert args.teacher_forced_seed_offset == 30001
    assert args.on_policy_seed_offset == 40002
    assert args.teacher_forced_seed_offset != args.on_policy_seed_offset


def test_revalidation_rejects_latched_identifier_failure(monkeypatch) -> None:
    episode = SimpleNamespace(
        intervention_mask=torch.ones(3, 2, dtype=torch.bool),
        finite=torch.ones(3, 2, dtype=torch.bool),
        identification_failure_t50=torch.tensor([False, True]),
    )
    monkeypatch.setattr(
        revalidation, "collect_dagger_episode",
        lambda *args, **kwargs: episode,
    )
    gate = {
        "rows": [
            {"phase_step": 50, "capability_z_rms": 0.0,
             "effectiveness_z_rms": 0.0,
             "capability_axis_z_rms": [0.0] * 6, "passed": True},
            {"phase_step": 75, "capability_z_rms": 0.0,
             "effectiveness_z_rms": 0.0,
             "capability_axis_z_rms": [0.0] * 6, "passed": True},
        ],
        "gate_passed": True,
    }
    monkeypatch.setattr(
        revalidation, "phase_a_equilibrium_gate",
        lambda *args, **kwargs: (gate, True),
    )
    bank = SimpleNamespace(count=2, stratum=("a", "b"))
    result = revalidation._bank_result(
        object(), object(), object(), bank,
        beta=1.0, horizon=125, episode_seed=7,
    )
    assert result["identification_failure_t50"] == 1
    assert result["phase_a_gate_passed"] is False

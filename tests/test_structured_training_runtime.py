from __future__ import annotations

import random
from argparse import Namespace

import numpy as np
import pytest
import torch

from structured_distillation import build_dagger_scenario_bank
from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy
from structured_training_runtime import TrainingSession


def setup_session(tmp_path, *, final=False):
    args = Namespace(seed=7, output=tmp_path / "model.pt", report=tmp_path / "report.json",
        training_state=None, checkpoint_every=1, development_every=1, minimum_updates=1,
        patience=3, minimum_relative_improvement=0.001, max_seconds=600,
        final_evaluation=final, lr=0.001)
    model = StructuredRecurrentPolicy(StructuredPolicyConfig(hidden_dim=4, identifier_dim=4))
    optimizer = torch.optim.AdamW(model.capability_head.parameters(), lr=args.lr)
    session = TrainingSession(args, model, optimizer, stage="unit_test")
    session.final_claim = tmp_path / "final_claim.json"
    return model, optimizer, session


def update(model, optimizer):
    x = torch.randn(5, 4) * (random.random() + float(np.random.rand()))
    loss = model.capability_head(x).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss)


def test_optimizer_rng_and_replay_resume_match_uninterrupted_training(tmp_path):
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model, optimizer, session = setup_session(tmp_path)
    session.progress["replay"] = {"executed_actions": torch.randn(3, 2, 4), "next_round": 3}
    session.record_update({"loss": update(model, optimizer)})
    expected_loss = update(model, optimizer)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    restored, restored_optimizer, resumed = setup_session(tmp_path)
    assert resumed.updates == 1
    assert resumed.progress["replay"]["next_round"] == 3
    assert update(restored, restored_optimizer) == expected_loss
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert restored_optimizer.state_dict()["state"]


def test_failed_development_does_not_authorize_final_data(tmp_path):
    model, optimizer, session = setup_session(tmp_path)
    session.record_update({"loss": update(model, optimizer)})
    session.record_development(score=20.0, passed=False, metrics={"motor_p95": 0.02})
    result = session.finish_development()
    assert not result["candidate_ready"]
    assert not session.candidate_path.exists()
    assert not session.args.report.exists()
    assert session.path.is_file()


def test_final_claim_is_exclusive_and_bound_to_frozen_candidate(tmp_path):
    model, optimizer, session = setup_session(tmp_path)
    session.record_update({"loss": update(model, optimizer)})
    session.record_development(score=0.5, passed=True, metrics={"unit_fixture": True})
    assert session.finish_development()["candidate_ready"]
    _, _, final = setup_session(tmp_path, final=True)
    final.begin_final([10007, 20007])
    with pytest.raises(FileExistsError):
        final.begin_final([10007, 20007])
    data = torch.load(final.candidate_path, weights_only=False)
    data["model"]["capability_head.bias"] += 1
    torch.save(data, final.candidate_path)
    with pytest.raises(RuntimeError, match="stale"):
        final.begin_final([10007, 20007])


def test_scenario_generation_does_not_reseed_the_model(tmp_path):
    torch.manual_seed(7)
    before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    bank = build_dagger_scenario_bank(64, seed=107)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    for after, expected in zip(torch.cuda.get_rng_state_all() if cuda_before else [], cuda_before):
        torch.testing.assert_close(after, expected, rtol=0, atol=0)
    assert all(int(((bank.tw_bin == i) & (bank.log_alpha_bin == j)).sum()) == 4
               for i in range(4) for j in range(4))


def test_production_pretrainer_persists_before_any_final_evaluation(tmp_path, monkeypatch):
    from tools import pretrain_structured_identifier as trainer
    if not trainer.DEFAULT_Q2.is_file():
        pytest.skip("canonical Q2 fixture is unavailable")
    monkeypatch.setattr(trainer, "validate_pretraining_gates",
                        lambda **kwargs: {"probe_v4": {}, "causal_oracle": {}})
    observed_seeds = []
    original_bank = trainer.build_dagger_scenario_bank

    def track_bank(*args, **kwargs):
        observed_seeds.append(kwargs["seed"])
        return original_bank(*args, **kwargs)

    monkeypatch.setattr(trainer, "build_dagger_scenario_bank", track_bank)
    args = trainer.parse_args(["--device", "cpu", "--updates", "1", "--scenarios", "16",
        "--output", str(tmp_path / "identifier_init.pt"), "--report", str(tmp_path / "report.json")])
    result = trainer.run(args)
    assert result["actual_updates"] == 1
    assert observed_seeds == [107, 2000007]
    assert not result["candidate_ready"]
    assert not (tmp_path / "identifier_init.pt").exists()
    saved = torch.load(tmp_path / "identifier_init.training.pt", weights_only=False)
    assert saved["optimizer"]["state"]
    assert saved["progress"]["updates"] == 1

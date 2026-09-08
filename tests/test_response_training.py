from __future__ import annotations

from dataclasses import asdict
import torch
import pytest

import response_training as training
from response_policy import ResponsePolicyConfig
from response_task import TaskLossConfig
from tools.train_response_control import parse_args


def arguments(path, updates):
    return parse_args([
        "--optimizer", "adam",
        "--device", "cpu", "--work-dir", str(path), "--updates", str(updates),
        "--horizon", "4", "--scenarios", "16", "--memory-dim", "4", "--hidden-dim", "8",
        "--minimum-updates", "1", "--checkpoint-every", "1", "--development-every", "2",
        "--prediction-weight", "0", "--steady-steps", "4",
    ])


def run_tiny(args):
    config = ResponsePolicyConfig(memory_dim=4, hidden_dim=8)
    loss = TaskLossConfig(prediction_weight=0, steady_steps=4)
    return training.train(args, config, loss)


def test_training_without_q2_and_exact_optimizer_rng_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "FINAL_CLAIM", tmp_path / "unused-final-claim.json")
    def forbidden(*args, **kwargs):
        raise AssertionError("training must not use a Q2 reference")
    monkeypatch.setattr(training, "_reference_rollout", forbidden)
    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    run_tiny(arguments(continuous, 2))
    run_tiny(arguments(resumed, 1))
    run_tiny(arguments(resumed, 2))
    first = torch.load(continuous / "latest.training.pt", map_location="cpu")
    second = torch.load(resumed / "latest.training.pt", map_location="cpu")
    assert first["progress"]["updates"] == second["progress"]["updates"] == 2
    assert all(torch.equal(first["model"][name], second["model"][name]) for name in first["model"])
    assert torch.equal(first["rng"]["torch"], second["rng"]["torch"])
    assert first["optimizer"]["state"].keys() == second["optimizer"]["state"].keys()
    assert second["deployment_authorized"] is False
    assert not (resumed / "candidate.pt").exists()


def test_q2_and_old_behavior_training_flags_are_not_accepted():
    with pytest.raises(SystemExit):
        parse_args(["--updates", "1", "--q2-checkpoint", "absent-q2.pt"])
    with pytest.raises(SystemExit):
        parse_args(["--updates", "1", "--trainable-prefix", "residual_head."])
    with pytest.raises(SystemExit):
        parse_args(["--updates", "1", "--allow-failed-migration-gate"])


def test_protocol_splits_are_disjoint_and_do_not_reuse_v5_validation():
    p = training.protocol(ResponsePolicyConfig(), TaskLossConfig(), 64)
    used = set(p["development_seeds"]) | set(p["final_seeds"]) | {p["ms_acceptance_seed"]}
    assert len(used) == 5
    assert 7707 not in used
    assert all(seed >= training.TRAIN_SEED_BASE + 1_000_000 for seed in used)
    assert p["action_mapping_requires_airframe_calibration_for_real_actuators"]
    assert not p["deployment_authorized"]


def test_final_claim_prevents_further_candidate_training(tmp_path, monkeypatch):
    claim = tmp_path / "claimed.json"
    claim.write_text('{"status":"claimed"}')
    monkeypatch.setattr(training, "FINAL_CLAIM", claim)
    with pytest.raises(RuntimeError, match="final set has been consumed"):
        run_tiny(arguments(tmp_path / "run", 1))


def test_dry_run_has_no_teacher_dependency_or_filesystem_writes(tmp_path, capsys):
    from tools.train_response_control import main
    assert main(["--dry-run", "--work-dir", str(tmp_path / "new")]) == 0
    assert not (tmp_path / "new").exists()
    assert '"teacher_checkpoint_required": false' in capsys.readouterr().out


def test_fullspace_selects_complete_startup_horizon():
    args = parse_args(["--optimizer", "full-space-ms", "--segments", "4", "--segment-steps", "250"])
    assert args.horizon == 1000

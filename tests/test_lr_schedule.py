"""Learning-rate scheduling must follow committed Adam updates and survive resume."""
from __future__ import annotations

import json

import pytest
import torch

import response_training
from test_reference_training import args_for, run
from tools.replay_response_update import compare_values, replay


COSINE_FLAGS = (
    "--lr", "0.001", "--lr-schedule", "cosine",
    "--lr-decay-updates", "4", "--lr-min", "0.00001",
)


def load(path):
    return torch.load(path / "latest.pt", map_location="cpu", weights_only=True)


def test_cosine_uses_current_update_lr_and_holds_minimum_after_decay(tmp_path):
    run(args_for(tmp_path, 6, COSINE_FLAGS))
    rows = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert [row["learning_rate"] for row in rows] == pytest.approx([
        .001, .000855017856687341, .000505, .000154982143312659, .00001, .00001,
    ])
    assert [row["next_learning_rate"] for row in rows] == pytest.approx([
        .000855017856687341, .000505, .000154982143312659, .00001, .00001, .00001,
    ])
    saved = load(tmp_path)
    assert saved["optimizer"]["param_groups"][0]["lr"] == .00001
    assert saved["lr_scheduler"]["last_epoch"] == 4
    assert saved["binding"]["lr_scheduler"] == {
        "type": "CosineAnnealingLR", "T_max": 4, "eta_min": .00001,
    }
    for path in (tmp_path / "audit/updates").glob("*.pt"):
        assert replay(path, "adam")["passed"]


@pytest.mark.parametrize("split", [2, 5])
def test_cosine_resume_matches_continuous_before_and_after_minimum(tmp_path, split):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run(args_for(full, 6, COSINE_FLAGS))
    run(args_for(resumed, split, COSINE_FLAGS))
    run(args_for(resumed, 6, (*COSINE_FLAGS, "--resume", str(resumed / "latest.pt"))))
    first, second = load(full), load(resumed)
    for key in ("model", "optimizer", "rng", "lr_scheduler"):
        assert compare_values(first[key], second[key])[0], key
    expected = [json.loads(line)["learning_rate"] for line in (full / "history.jsonl").read_text().splitlines()]
    actual = [json.loads(line)["learning_rate"] for line in (resumed / "history.jsonl").read_text().splitlines()]
    assert actual == expected


@pytest.mark.parametrize("failure_kind", ["numeric", "audit"])
def test_failed_update_does_not_advance_scheduler_or_change_lr(tmp_path, monkeypatch, failure_kind):
    run(args_for(tmp_path, 1, COSINE_FLAGS))
    before = load(tmp_path)
    if failure_kind == "numeric":
        original = torch.optim.Adam.step

        def fail(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            self.param_groups[0]["params"][0].data.fill_(float("nan"))
            return result

        monkeypatch.setattr(torch.optim.Adam, "step", fail)
    else:
        def fail(*args, **kwargs):
            raise OSError("audit disk full")

        monkeypatch.setattr(response_training.UpdateAudit, "record", fail)
    with pytest.raises((FloatingPointError, OSError)):
        run(args_for(tmp_path, 2, (*COSINE_FLAGS, "--resume", str(tmp_path / "latest.pt"))))
    after = torch.load(tmp_path / "failure.pt", weights_only=True)
    for key in ("model", "optimizer", "rng", "lr_scheduler"):
        assert compare_values(before[key], after[key])[0], key
    assert after["progress"]["updates"] == 1


@pytest.mark.parametrize("override", [
    ("--lr-decay-updates", "5"), ("--lr-min", ".00002"), ("--lr-schedule", "constant"),
])
def test_exact_resume_rejects_changed_schedule(tmp_path, override):
    run(args_for(tmp_path, 0, COSINE_FLAGS))
    with pytest.raises(ValueError, match="configuration"):
        run(args_for(tmp_path, 1, (*COSINE_FLAGS, *override, "--resume", str(tmp_path / "latest.pt"))))


def test_cosine_resume_rejects_missing_schedule_state(tmp_path):
    run(args_for(tmp_path, 0, COSINE_FLAGS))
    saved = load(tmp_path)
    saved.pop("lr_scheduler")
    broken = tmp_path / "missing_schedule.pt"
    torch.save(saved, broken)
    with pytest.raises(ValueError, match="scheduler"):
        run(args_for(tmp_path, 1, (*COSINE_FLAGS, "--resume", str(broken))))


@pytest.mark.parametrize("override", [
    ("--lr-decay-updates", "0"), ("--lr-decay-updates", "-1"),
    ("--lr-min", "-0.001"), ("--lr-min", "nan"),
    ("--lr-min", "inf"), ("--lr-min", ".002"),
])
def test_invalid_cosine_configuration_rejected(tmp_path, override):
    with pytest.raises(SystemExit):
        args_for(tmp_path, 0, (*COSINE_FLAGS, *override))

"""Bounded training-chain checks; these do not establish learned flight ability."""
from dataclasses import fields
import json

import pytest
import torch

from diffphys.agile_curriculum import AgileCurriculumConfig, AgileTrainingConfig
from diffphys.agile_mission import AgileMissionConfig
from diffphys.agile_objective import AGILE_OBJECTIVE_VERSION
from diffphys.response_policy import AGILE_ARCHITECTURE, ResponsePolicyConfig
from diffphys.response_task import TaskLossConfig
from diffphys.response_training import (
    DEVELOPMENT_SEEDS, TEST_SEEDS, train, evaluate_checkpoint,
    require_reference_checkpoint,
)
from tools.replay_response_update import compare_values, replay
from tools.train_response_control import parse_args


@pytest.fixture
def agile_config_file(tmp_path):
    # Tiny geometry allows a short physical rollout; schedule changes at each
    # of the first updates so a stale prefetched task would break this test.
    config = AgileTrainingConfig(
        mission=AgileMissionConfig(hold_seconds=.04, half_span_fraction=1e-6,
                                   bend_fraction=1e-8, tilt_degrees=5.),
        curriculum=AgileCurriculumConfig(start_difficulty=.2, end_difficulty=.8,
                                         ramp_updates=2),
    )
    path = tmp_path / "agile.json"
    path.write_text(json.dumps(config.to_dict()))
    return path


def args_for(path, config, updates=2, extra=()):
    return parse_args([
        "--task", "agile", "--agile-config", str(config),
        "--device", "cpu", "--dtype", "float64", "--scenarios", "8",
        "--eval-scenarios", "2", "--horizon", "12", "--memory-dim", "8",
        "--development-every", "1", "--checkpoint-every", "1",
        "--updates", str(updates), "--max-seconds", "60", "--work-dir", str(path),
        *extra,
    ])


def run(args):
    policy = ResponsePolicyConfig(**{
        entry.name: getattr(args, entry.name) for entry in fields(ResponsePolicyConfig)
    })
    loss = (args.agile_training_config.loss if args.task == "agile" else
            TaskLossConfig(**{entry.name: getattr(args, entry.name)
                              for entry in fields(TaskLossConfig)}))
    return train(args, policy, loss)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_task_cli_controls_architecture_and_default_decay(tmp_path, agile_config_file):
    args = args_for(tmp_path / "agile", agile_config_file, 0)
    assert args.goal_conditioning and args.time_decay == 0.
    assert args.lambda_R is None  # Legacy losses do not enter the agile objective.
    explicit = args_for(tmp_path / "explicit", agile_config_file, 0,
                        ("--time-decay", "1"))
    assert explicit.time_decay == 1.
    hover = parse_args(["--epsilon-p", ".01", "--epsilon-a", ".01", "--lambda-R", "25"])
    assert not hover.goal_conditioning and hover.time_decay == 1.
    with pytest.raises(SystemExit):
        parse_args(["--goal-conditioning", "False"])
    # Standalone evaluation reads its task/loss from the checkpoint, so neither
    # a current task JSON file nor legacy loss flags are necessary at parse time.
    evaluation = parse_args(["--mode", "evaluate", "--checkpoint", "saved.pt",
                             "--agile-config", "/missing/agile.json"])
    assert not hasattr(evaluation, "agile_training_config")


def test_agile_resume_replay_and_fixed_evaluation_match(tmp_path, agile_config_file):
    full, split = tmp_path / "full", tmp_path / "split"
    run(args_for(full, agile_config_file, 2))
    run(args_for(split, agile_config_file, 1))
    run(args_for(split, agile_config_file, 2, ("--resume", str(split / "latest.pt"))))
    first, second = [torch.load(path / "latest.pt", weights_only=True)
                     for path in (full, split)]
    for key in ("model", "optimizer", "rng"):
        assert compare_values(first[key], second[key])[0], key
    assert second["architecture"] == AGILE_ARCHITECTURE
    assert second["schema"] == "raptor-multi-airframe-gaussian-v6"
    assert second["policy_config"]["goal_conditioning"] is True
    assert second["binding"]["protocol"]["task_objective"] == AGILE_OBJECTIVE_VERSION
    assert "curriculum" not in second["progress"]  # No pass/fail scheduler state.
    assert second["progress"]["sampling_index"] == 2
    history = rows(split / "history.jsonl")
    sampled = [row["training_sampling"]["agile"] for row in history]
    assert [sample["schedule_update"] for sample in sampled] == [0, 1]
    assert [sample["requested_difficulty"] for sample in sampled] == pytest.approx([.2, .5])
    assert all(row["scenario_count"] == 32 for row in history)
    assert all(sample["mission"]["aircraft_rows_retained"] == 32 for sample in sampled)
    assert all(row["raw_gradient_norm"] > 0 for row in history)
    validation = rows(split / "evaluation.jsonl")
    assert all(row["eval_difficulty"] == 1. and row["eval_hover_fraction"] == 0.
               for row in validation)
    assert all(row["mission_sampling"]["planned_hover_count"] == 0 for row in validation)
    assert all(row["scenario_count"] == 4 for row in validation)
    assert second["progress"]["best_score"] == min(row["task_objective"] for row in validation)
    assert json.loads((split / "summary.json").read_text())["final_seeds_consumed"] == []

    capsule = sorted((split / "audit/updates").glob("*.pt"))[-1]
    before_bytes = capsule.read_bytes()
    replayed = replay(capsule, "full")
    assert replayed["passed"], replayed
    assert capsule.read_bytes() == before_bytes

    saved_bytes = (split / "latest.pt").read_bytes()
    evaluation_args = parse_args([
        "--mode", "evaluate", "--checkpoint", str(split / "latest.pt"),
        "--device", "cpu", "--disable-disturbances", "--work-dir", str(tmp_path / "evaluation"),
        "--agile-config", "/missing/current-config.json",
    ])
    actual = evaluate_checkpoint(evaluation_args)
    assert actual["task"] == "agile"
    assert actual["task_objective"] == validation[-1]["task_objective"]
    assert actual["disturbances"] == validation[-1]["disturbances"]
    assert actual["eval_seeds"] == list(DEVELOPMENT_SEEDS)
    evaluation_args.eval_split = "test"
    test = evaluate_checkpoint(evaluation_args)
    assert test["eval_seeds"] == list(TEST_SEEDS)
    assert set(test["eval_seeds"]).isdisjoint(actual["eval_seeds"])
    assert test["eval_mission_seed"] != actual["eval_mission_seed"]
    assert (tmp_path / "evaluation/evaluation_test.json").exists()
    assert (split / "latest.pt").read_bytes() == saved_bytes


def test_agile_resume_rejects_task_or_schedule_changes(tmp_path, agile_config_file):
    work = tmp_path / "source"
    run(args_for(work, agile_config_file, 0))
    checkpoint = work / "latest.pt"
    saved = torch.load(checkpoint, weights_only=True)
    require_reference_checkpoint(saved)
    changed = json.loads(agile_config_file.read_text())
    changed["curriculum"]["ramp_updates"] += 1
    modified = tmp_path / "changed.json"
    modified.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="configuration"):
        run(args_for(work, modified, 1, ("--resume", str(checkpoint))))
    with pytest.raises(ValueError, match="configuration"):
        run(args_for(work, agile_config_file, 1, (
            "--resume", str(checkpoint), "--task", "hover",
            "--epsilon-p", ".01", "--epsilon-a", ".01", "--lambda-R", "25",
        )))
    saved["policy_config"]["goal_conditioning"] = False
    with pytest.raises(ValueError, match="architecture|semantics"):
        require_reference_checkpoint(saved)


def test_hover_checkpoint_initializes_agile_weights_with_fresh_state(tmp_path, agile_config_file):
    hover = tmp_path / "hover"
    run(args_for(hover, agile_config_file, 1, (
        "--task", "hover", "--epsilon-p", ".01", "--epsilon-a", ".01", "--lambda-R", "25",
    )))
    source = torch.load(hover / "latest.pt", weights_only=True)
    assert source["optimizer"]["state"] and source["progress"]["updates"] == 1
    agile = tmp_path / "agile"
    run(args_for(agile, agile_config_file, 0, ("--init-checkpoint", str(hover / "latest.pt"))))
    imported = torch.load(agile / "latest.pt", weights_only=True)
    for name, expected in source["model"].items():
        actual = imported["model"][name]
        if name == "response_memory.weight_ih":
            torch.testing.assert_close(actual[:, :16], expected, rtol=0, atol=0)
            assert torch.count_nonzero(actual[:, 16:]) == 0
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert imported["optimizer"]["state"] == {}
    assert imported["progress"]["updates"] == imported["progress"]["sampling_index"] == 0
    provenance = imported["progress"]["initialization"]
    assert provenance["weights_only"] is True
    assert provenance["source_model_sha256"] == source["model_sha256"]
    assert provenance["input_columns_copied"] == 16 and provenance["input_columns_added"] == 20
    assert not torch.equal(imported["rng"]["torch"], source["rng"]["torch"])

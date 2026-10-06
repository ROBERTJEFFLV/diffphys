from __future__ import annotations

from loss_fixtures import test_loss

from dataclasses import replace
import json

import numpy as np
import torch
from unittest.mock import patch

from env_raptor import RaptorParams, RaptorSimulator
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig, rollout, sample_scenarios
from tools.short_eval_replay import capture_short_eval, save_latest_replay
from tools.response_monitor import launch_replay, replay_entry
from tools.play_response_short import FORCE_FIELDS, force_at_frame, load_replay


def test_short_eval_capture_matches_the_regular_rollout_and_keeps_one_latest(tmp_path):
    torch.manual_seed(17)
    horizon = 8
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).eval()
    simulator = RaptorSimulator(RaptorParams(dt=policy.config.dt))
    initial = sample_scenarios(3, seed=19, horizon=horizon)

    with torch.no_grad():
        expected = rollout(policy, simulator, initial, horizon)
    report, arrays = capture_short_eval(
        policy, simulator, initial, horizon, test_loss()
    )

    np.testing.assert_array_equal(arrays["position"], torch.cat(
        (initial.position[None], expected.positions)
    ).numpy())
    np.testing.assert_array_equal(arrays["velocity"], torch.cat(
        (initial.velocity[None], expected.velocities)
    ).numpy())
    np.testing.assert_array_equal(arrays["omega"], torch.cat(
        (initial.omega[None], expected.omegas)
    ).numpy())
    np.testing.assert_array_equal(arrays["action"], expected.actions.numpy())
    np.testing.assert_array_equal(arrays["valid"], expected.valid.numpy())
    np.testing.assert_array_equal(arrays["orientation"][0], initial.orientation.numpy())
    np.testing.assert_array_equal(arrays["external_force_world"], initial.external_force.numpy())
    np.testing.assert_array_equal(arrays["pulse_force_world"],
                                  initial.pulse_tape[:, :horizon, :3].transpose(0, 1).numpy())
    np.testing.assert_array_equal(arrays["pulse_point_body"],
                                  initial.pulse_tape[:, :horizon, 3:].transpose(0, 1).numpy())
    np.testing.assert_array_equal(arrays["pulse_active"],
                                  initial.pulse_active_tape[:, :horizon].transpose(0, 1).numpy())
    np.testing.assert_allclose(np.linalg.norm(arrays["orientation"], axis=-1), 1., atol=1e-6)
    assert arrays["orientation"].shape == (horizon + 1, 3, 4)
    assert report["finite"]

    playlist = save_latest_replay(
        tmp_path, checkpoint_update=50, model_sha256="model-50",
        source_sha256="source", dt=.01, report=report, arrays=arrays,
    )
    meta = json.loads(playlist.read_text())
    first_data = playlist.parent / meta["trajectory_file"]
    assert meta["replay_type"] == "short-eval-v1"
    assert meta["checkpoint_update"] == 50
    assert first_data.is_file()
    with np.load(first_data, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["position"], arrays["position"])

    save_latest_replay(
        tmp_path, checkpoint_update=100, model_sha256="model-100",
        source_sha256="source", dt=.01, report=report, arrays=arrays,
    )
    meta = json.loads(playlist.read_text())
    assert meta["checkpoint_update"] == 100
    assert sorted(path.name for path in playlist.parent.glob("trajectory_*.npz")) == [
        meta["trajectory_file"]
    ]
    entry = replay_entry(playlist)
    assert entry.update == 100
    with patch("tools.response_monitor.subprocess.Popen") as popen:
        launch_replay(playlist)
        command = popen.call_args.args[0]
        assert command[1].endswith("play_response_short.py")
    loaded_meta, loaded_arrays = load_replay(playlist)
    assert loaded_meta["checkpoint_update"] == 100
    np.testing.assert_array_equal(loaded_arrays["orientation"], arrays["orientation"])
    np.testing.assert_array_equal(loaded_arrays["pulse_force_world"], arrays["pulse_force_world"])

    old_arrays = {name: value for name, value in arrays.items() if name not in FORCE_FIELDS}
    old_playlist = save_latest_replay(
        tmp_path / "old", checkpoint_update=50, model_sha256="old",
        source_sha256="source", dt=.01, report=report, arrays=old_arrays,
    )
    _, loaded_old = load_replay(old_playlist)
    assert force_at_frame(loaded_old, 0, 0) is None


def test_short_eval_force_telemetry_uses_the_applied_transition_and_stops_at_termination():
    torch.manual_seed(31)
    horizon = 120  # The first scheduled pulse occurs by step 99.
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).eval()
    simulator = RaptorSimulator(RaptorParams(dt=policy.config.dt))
    initial = sample_scenarios(2, seed=37, horizon=horizon)
    _, arrays = capture_short_eval(
        policy, simulator, initial, horizon, test_loss()
    )
    assert arrays["pulse_active"].any()
    scene, frame = np.argwhere(arrays["pulse_active"] & arrays["valid"])[0, ::-1]
    applied = force_at_frame(arrays, int(scene), int(frame))
    assert applied["applied"] and applied["pulse_active"]
    np.testing.assert_array_equal(applied["pulse_force_world"], arrays["pulse_force_world"][frame, scene])
    np.testing.assert_array_equal(applied["pulse_point_body"], arrays["pulse_point_body"][frame, scene])
    np.testing.assert_array_equal(applied["total_force_world"],
                                  arrays["external_force_world"][scene] + arrays["pulse_force_world"][frame, scene])
    assert force_at_frame(arrays, int(scene), horizon)["applied"] is False

    stopped = dict(arrays, valid=arrays["valid"].copy())
    stopped["valid"][frame, scene] = False
    assert force_at_frame(stopped, int(scene), int(frame))["applied"] is False


def test_terminated_scene_stays_frozen_in_the_saved_short_trace():
    torch.manual_seed(23)
    horizon = 5
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).eval()
    simulator = RaptorSimulator()
    initial = sample_scenarios(2, seed=29, horizon=horizon)
    position = initial.position.clone()
    position[0] = 0
    velocity = initial.velocity.clone()
    velocity[0] = torch.tensor((10., 0., 0.))
    limit = initial.position_limit.clone()
    limit[0] = .05
    initial = replace(initial, position=position, velocity=velocity, position_limit=limit)

    with torch.no_grad():
        expected = rollout(policy, simulator, initial, horizon)
    _, arrays = capture_short_eval(
        policy, simulator, initial, horizon, test_loss()
    )
    np.testing.assert_array_equal(arrays["valid"], expected.valid.numpy())
    np.testing.assert_array_equal(arrays["position"], torch.cat(
        (initial.position[None], expected.positions)
    ).numpy())
    assert arrays["valid"][:, 0].tolist() == [True, False, False, False, False]
    np.testing.assert_array_equal(arrays["orientation"][1:, 0], np.repeat(
        arrays["orientation"][1:2, 0], horizon, axis=0
    ))

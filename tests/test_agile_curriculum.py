"""Preset amplitude depends on committed updates, with no performance gates."""
from __future__ import annotations

from dataclasses import asdict
import inspect
import json

import pytest

from diffphys.agile_curriculum import (
    AgileCurriculumConfig,
    agile_config_from_dict,
    curriculum_difficulty,
    load_agile_config,
)


def _write(tmp_path, value):
    path = tmp_path / "agile.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_default_config_is_explicit_valid_and_roundtrips(tmp_path):
    config = load_agile_config()
    assert config.version == "agile-flight-training-v1"
    assert config.mission.tilt_degrees == 60.
    assert config.mission.thrust_ratio == 1.2
    assert config.mission.hold_seconds == 1.
    assert config.mission.hover_fraction == .1
    assert config.fixed_eval_difficulty == 1.
    assert config.curriculum.start_difficulty == .25
    assert config.curriculum.end_difficulty == 1.
    assert config.curriculum.ramp_updates == 1000
    assert config.curriculum.interpolation == "smoothstep"
    assert load_agile_config(_write(tmp_path, config.to_dict())) == config


def test_saved_config_reconstruction_does_not_read_current_file(tmp_path, monkeypatch):
    import diffphys.agile_curriculum as module
    config = load_agile_config()
    snapshot = config.to_dict()
    assert isinstance(snapshot["mission"]["center_fraction"], list)
    monkeypatch.setattr(module, "DEFAULT_CONFIG", tmp_path / "absent.json")
    assert agile_config_from_dict(snapshot) == config
    assert agile_config_from_dict(asdict(config)) == config


def test_omitted_nested_fields_expand_defaults():
    config = agile_config_from_dict({"version": "agile-flight-training-v1",
                                    "mission": {}, "loss": {}, "curriculum": {}})
    assert config.mission.tilt_degrees == 60.
    assert config.curriculum.start_difficulty == .25


@pytest.mark.parametrize("section,key,value", [
    (None, "typo", 1),
    ("mission", "tilt_degree", 30),
    ("loss", "unknown_loss_field", 2),
    ("curriculum", "levels", [.25, .5, 1]),
    ("curriculum", "success_rate", .8),
    ("curriculum", "consecutive_evaluations", 2),
    ("mission", "tilt_degrees", "60"),
    ("mission", "tilt_degrees", True),
    ("mission", "center_fraction", [0, False, 0]),
    ("mission", "screen_batch_size", 128.0),
    ("curriculum", "warmup_updates", 1.0),
    ("curriculum", "ramp_updates", 0),
    ("curriculum", "ramp_updates", True),
    ("curriculum", "start_difficulty", -0.1),
    ("curriculum", "end_difficulty", .2),
    ("curriculum", "end_difficulty", 1.1),
    ("curriculum", "fixed_difficulty", True),
    ("curriculum", "fixed_difficulty", -0.1),
    ("curriculum", "interpolation", "adaptive"),
    ("curriculum", "interpolation", 1),
    (None, "fixed_eval_difficulty", .5),
    (None, "version", "unknown"),
])
def test_unknown_keys_invalid_types_and_invalid_values_fail(tmp_path, section, key, value):
    data = load_agile_config().to_dict()
    target = data if section is None else data[section]
    target[key] = value
    with pytest.raises(ValueError):
        load_agile_config(_write(tmp_path, data))


def test_missing_sections_duplicate_json_and_nonfinite_values_fail(tmp_path):
    data = load_agile_config().to_dict()
    del data["loss"]
    with pytest.raises(ValueError, match="loss"):
        load_agile_config(_write(tmp_path, data))
    path = tmp_path / "invalid.json"
    path.write_text('{"version":"agile-flight-training-v1","version":"other"}')
    with pytest.raises(ValueError, match="duplicate"):
        load_agile_config(path)
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="finite"):
        load_agile_config(path)


def test_smoothstep_schedule_is_bounded_monotone_and_has_smooth_endpoints():
    config = AgileCurriculumConfig(warmup_updates=10, ramp_updates=1000)
    values = [curriculum_difficulty(index, config) for index in range(1020)]
    assert values[0] == values[10] == .25
    assert values[510] == .625
    assert values[1010] == values[-1] == 1.
    assert all(.25 <= value <= 1 for value in values)
    assert all(a <= b for a, b in zip(values, values[1:]))
    first = values[11] - values[10]
    last = values[1010] - values[1009]
    middle = values[511] - values[510]
    assert 0 < first < .01 * middle
    assert 0 < last < .01 * middle
    assert curriculum_difficulty(10**100, config) == 1.


def test_linear_schedule_and_fixed_override_have_explicit_semantics():
    linear = AgileCurriculumConfig(start_difficulty=.2, end_difficulty=.8,
                                    warmup_updates=20, ramp_updates=100, interpolation="linear")
    assert curriculum_difficulty(0, linear) == .2
    assert curriculum_difficulty(20, linear) == .2
    assert curriculum_difficulty(70, linear) == pytest.approx(.5)
    assert curriculum_difficulty(120, linear) == .8
    fixed = AgileCurriculumConfig(fixed_difficulty=.6)
    assert [curriculum_difficulty(index, fixed) for index in (0, 1, 500, 100000)] == [.6] * 4


def test_resume_same_update_has_identical_amplitude_without_scheduler_state():
    config = load_agile_config()
    saved_binding = json.loads(json.dumps(config.to_dict()))
    restored = agile_config_from_dict(saved_binding)
    for update in (0, 37, 499, 500, 1000, 10000):
        assert curriculum_difficulty(update, restored) == curriculum_difficulty(update, config)
        assert curriculum_difficulty(update, config.curriculum) == curriculum_difficulty(update, config)


def test_no_metric_report_or_mutable_state_can_drive_the_schedule():
    import diffphys.agile_curriculum as module
    assert tuple(inspect.signature(curriculum_difficulty).parameters) == ("update", "config")
    assert not hasattr(module, "new_curriculum_state")
    assert not hasattr(module, "update_curriculum")
    config = AgileCurriculumConfig()
    order = (900, 0, 501, 501, 20, 900, 0)
    values = [curriculum_difficulty(index, config) for index in order]
    assert values[0] == values[5]
    assert values[1] == values[6]
    assert values[2] == values[3]


@pytest.mark.parametrize("update", [-1, True, 1.0, "1", {"level": 0}])
def test_invalid_update_rejected(update):
    with pytest.raises(ValueError, match="update"):
        curriculum_difficulty(update, AgileCurriculumConfig())

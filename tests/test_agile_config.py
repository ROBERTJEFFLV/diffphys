"""Fixed three-point configuration: strict snapshots without a curriculum."""
from __future__ import annotations

from dataclasses import asdict
import json

import pytest

from diffphys.agile_config import (
    agile_config_from_dict,
    load_agile_config,
)


def _write(tmp_path, value):
    path = tmp_path / "agile.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_default_config_is_explicit_valid_and_roundtrips(tmp_path):
    config = load_agile_config()
    assert config.version == "agile-flight-training-v4"
    assert config.mission.tilt_degrees == 105.
    assert config.mission.thrust_ratio == 1.2
    assert config.mission.half_span_fraction == .70
    assert config.mission.bend_fraction == .38
    assert config.mission.duration_search_factor == 1.05
    assert not hasattr(config.mission, "hold_seconds")
    assert not hasattr(config.loss, "hold_weight")
    assert config.mission.hover_fraction == 0.
    assert set(config.to_dict()) == {"version", "mission", "loss"}
    assert load_agile_config(_write(tmp_path, config.to_dict())) == config


def test_saved_config_reconstruction_does_not_read_current_file(tmp_path, monkeypatch):
    import diffphys.agile_config as module
    config = load_agile_config()
    snapshot = config.to_dict()
    assert isinstance(snapshot["mission"]["center_fraction"], list)
    monkeypatch.setattr(module, "DEFAULT_CONFIG", tmp_path / "absent.json")
    assert agile_config_from_dict(snapshot) == config
    assert agile_config_from_dict(asdict(config)) == config


def test_omitted_nested_fields_expand_defaults():
    config = agile_config_from_dict({"version": "agile-flight-training-v4",
                                    "mission": {}, "loss": {}})
    assert config.mission.tilt_degrees == 105.
    assert config.mission.hover_fraction == 0.


@pytest.mark.parametrize("section,key,value", [
    (None, "typo", 1),
    ("mission", "tilt_degree", 30),
    ("loss", "unknown_loss_field", 2),
    (None, "curriculum", {}),
    (None, "curriculum", {"start_difficulty": .25, "end_difficulty": 1.}),
    (None, "difficulty", .5),
    ("mission", "difficulty", .5),
    ("mission", "success_rate", .8),
    ("mission", "tilt_degrees", "60"),
    ("mission", "tilt_degrees", True),
    ("mission", "center_fraction", [0, False, 0]),
    ("mission", "screen_batch_size", 128.0),
    ("mission", "duration_search_factor", "1.05"),
    ("mission", "duration_search_factor", True),
    ("mission", "duration_search_factor", 1.),
    ("mission", "duration_search_factor", 2.01),
    ("mission", "hold_seconds", 1.),
    ("mission", "shrink_factors", [1., .5]),
    ("loss", "hold_weight", 1.),
    ("loss", "rest_position_scale_fraction", .04),
    ("loss", "terminal_position_weight", -1.),
    ("mission", "tilt_degrees", 180.),
    (None, "fixed_eval_difficulty", .5),
    (None, "version", "agile-flight-training-v1"),
    (None, "version", "agile-flight-training-v2"),
    (None, "version", "agile-flight-training-v3"),
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
    path.write_text('{"version":"agile-flight-training-v4","version":"other"}')
    with pytest.raises(ValueError, match="duplicate"):
        load_agile_config(path)
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="finite"):
        load_agile_config(path)



"""Strict task configuration and a preset continuous maneuver-amplitude schedule.

TRAIN draws the physical/noise pool, then samples a mission whose difficulty is
a deterministic function of committed optimizer updates. No flight metric,
event, pass/fail test or extra mutable scheduler state affects that function.
Fixed difficulty-1 validation stays separate for comparable continuous losses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
import types
from typing import Union, get_args, get_origin, get_type_hints

from .agile_mission import AgileMissionConfig
from .agile_objective import AgileLossConfig

AGILE_TRAINING_VERSION = "agile-flight-training-v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/agile_flight.json"


def _number(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number (not bool or string)")
    return float(value)


@dataclass(frozen=True)
class AgileCurriculumConfig:
    start_difficulty: float = .25
    end_difficulty: float = 1.
    warmup_updates: int = 0
    ramp_updates: int = 1000
    interpolation: str = "smoothstep"
    fixed_difficulty: float | None = None

    def __post_init__(self):
        start = _number(self.start_difficulty, "start_difficulty")
        end = _number(self.end_difficulty, "end_difficulty")
        if not 0 <= start <= end <= 1:
            raise ValueError("schedule difficulties must satisfy 0 <= start <= end <= 1")
        if type(self.warmup_updates) is not int or self.warmup_updates < 0:
            raise ValueError("warmup_updates must be a nonnegative integer")
        if type(self.ramp_updates) is not int or self.ramp_updates < 1:
            raise ValueError("ramp_updates must be a positive integer")
        if type(self.interpolation) is not str or self.interpolation not in ("smoothstep", "linear"):
            raise ValueError("interpolation must be smoothstep or linear")
        if self.fixed_difficulty is not None:
            fixed = _number(self.fixed_difficulty, "fixed_difficulty")
            if not 0 <= fixed <= 1:
                raise ValueError("fixed_difficulty must be in [0,1] or null")
            object.__setattr__(self, "fixed_difficulty", fixed)
        object.__setattr__(self, "start_difficulty", start)
        object.__setattr__(self, "end_difficulty", end)


@dataclass(frozen=True)
class AgileTrainingConfig:
    version: str = AGILE_TRAINING_VERSION
    mission: AgileMissionConfig = field(default_factory=AgileMissionConfig)
    loss: AgileLossConfig = field(default_factory=AgileLossConfig)
    curriculum: AgileCurriculumConfig = field(default_factory=AgileCurriculumConfig)
    fixed_eval_difficulty: float = 1.

    def __post_init__(self):
        if self.version != AGILE_TRAINING_VERSION:
            raise ValueError("unsupported agile training config version")
        if (not isinstance(self.mission, AgileMissionConfig)
                or not isinstance(self.loss, AgileLossConfig)
                or not isinstance(self.curriculum, AgileCurriculumConfig)):
            raise ValueError("agile mission, loss and curriculum must use their typed configs")
        if _number(self.fixed_eval_difficulty, "fixed_eval_difficulty") != 1.:
            raise ValueError("fixed_eval_difficulty must remain 1 for comparable validation")
        object.__setattr__(self, "fixed_eval_difficulty", 1.)

    def to_dict(self) -> dict:
        """Return all effective fields as a canonical JSON-compatible snapshot."""
        return _json_tree(asdict(self))


def _json_tree(value):
    """Convert declared tuple arrays without coercing strings/bools into numbers."""
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("agile config object keys must be strings")
        return {key: _json_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_tree(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError("agile config must contain only JSON-compatible values")


def _typed_value(value, annotation, name: str):
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is float:
        return _number(value, name)
    if annotation is int:
        if type(value) is not int:
            raise ValueError(name + " must be an integer")
        return value
    if annotation is bool:
        if type(value) is not bool:
            raise ValueError(name + " must be boolean")
        return value
    if annotation is str:
        if type(value) is not str:
            raise ValueError(name + " must be a string")
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        choices = [kind for kind in args if kind is not type(None)]
        if len(choices) == 1:
            return _typed_value(value, choices[0], name)
    if origin is tuple:
        if not isinstance(value, list):
            raise ValueError(name + " must be a JSON array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_typed_value(item, args[0], name) for item in value)
        if len(value) != len(args):
            raise ValueError(name + " has the wrong number of entries")
        return tuple(_typed_value(item, kind, name) for item, kind in zip(value, args))
    raise ValueError("unsupported type for " + name)


def _section(config_type, value, name: str, *, defaults=None):
    if not isinstance(value, dict):
        raise ValueError(name + " must be a JSON object")
    known = {entry.name for entry in fields(config_type)}
    unknown = set(value) - known
    if unknown:
        raise ValueError(name + " contains unknown keys: " + ", ".join(sorted(unknown)))
    annotations = get_type_hints(config_type)
    converted = {key: _typed_value(item, annotations[key], name + "." + key)
                 for key, item in value.items()}
    try:
        return config_type(**{**({} if defaults is None else asdict(defaults)), **converted})
    except (TypeError, ValueError) as error:
        raise ValueError("invalid " + name + ": " + str(error)) from error


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _nonfinite_json(value):
    raise ValueError("JSON values must be finite: " + value)


def load_agile_config(path: str | Path | None = None) -> AgileTrainingConfig:
    """Read required task sections, reject typos/types, expand omitted defaults."""
    location = DEFAULT_CONFIG if path is None else Path(path)
    value = json.loads(location.read_text(encoding="utf-8"),
                       object_pairs_hook=_unique_object, parse_constant=_nonfinite_json)
    return agile_config_from_dict(value)


def agile_config_from_dict(value: dict) -> AgileTrainingConfig:
    """Reconstruct a saved binding without consulting the current config file.

    Tuple arrays from dataclasses.asdict are explicitly canonicalized; numeric
    strings, booleans, unknown fields and unsupported versions remain errors.
    """
    value = _json_tree(value)
    required = {"version", "mission", "loss", "curriculum"}
    if not isinstance(value, dict):
        raise ValueError("agile training config must be a JSON object")
    if missing := required - set(value):
        raise ValueError("agile config is missing sections: " + ", ".join(sorted(missing)))
    if unknown := set(value) - required - {"fixed_eval_difficulty"}:
        raise ValueError("agile config contains unknown keys: " + ", ".join(sorted(unknown)))
    return AgileTrainingConfig(
        version=value["version"],
        mission=_section(AgileMissionConfig, value["mission"], "mission"),
        loss=_section(AgileLossConfig, value["loss"], "loss",
                      defaults=AgileTrainingConfig().loss),
        curriculum=_section(AgileCurriculumConfig, value["curriculum"], "curriculum"),
        fixed_eval_difficulty=value.get("fixed_eval_difficulty", 1.),
    )


def _curriculum(config: AgileCurriculumConfig | AgileTrainingConfig) -> AgileCurriculumConfig:
    if isinstance(config, AgileTrainingConfig):
        return config.curriculum
    if not isinstance(config, AgileCurriculumConfig):
        raise ValueError("a typed agile curriculum config is required")
    return config


def curriculum_difficulty(update: int, config: AgileCurriculumConfig | AgileTrainingConfig) -> float:
    """Return the scheduled amplitude from committed updates alone.

    After warmup, u = clamp((update-warmup)/ramp, 0, 1). Smoothstep uses
    u*u*(3-2*u), with zero slope at both endpoints. Resuming the same update
    and saved config reproduces exactly the same amplitude without extra state.
    """
    if type(update) is not int or update < 0:
        raise ValueError("committed update must be a nonnegative integer")
    schedule = _curriculum(config)
    if schedule.fixed_difficulty is not None:
        return schedule.fixed_difficulty
    elapsed = min(max(update-schedule.warmup_updates, 0), schedule.ramp_updates)
    fraction = elapsed / schedule.ramp_updates
    if schedule.interpolation == "smoothstep":
        fraction = fraction * fraction * (3. - 2. * fraction)
    return schedule.start_difficulty + (schedule.end_difficulty-schedule.start_difficulty) * fraction

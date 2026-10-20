"""Versioned settings for one fixed-point task, not a difficulty schedule."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

VERSION = "event-waypoints-al-v1"
GOAL_CONVENTION = "fixed-current-next-final-pose-direction-type-pending-v1"


@dataclass(frozen=True)
class WaypointConfig:
    version: str = VERSION
    arrival_radius_m: float = .1
    tilt_degrees: float = 105.
    half_span_fraction: float = .7
    bend_fraction: float = .38
    boundary_margin_fraction: float = .2
    passage_speed_m_s: float = 2.
    no_slowdown_half_width_m: float = .5
    # Fixed geometry rule: keep the transit band separate from the final point.
    # This never depends on the Actor, optimization update or success rate.
    band_max_leg_fraction: float = .4
    pose_half_width_m: float = .1
    pose_tolerance_degrees: float = 1.
    terminal_position_m: float = .1
    terminal_speed_m_s: float = .1
    terminal_attitude_degrees: float = 5.
    terminal_omega_rad_s: float = .5
    terminal_acceleration_m_s2: float = .25
    terminal_angular_acceleration_rad_s2: float = 1.
    terminal_motor_rate_s: float = .1
    maximum_omega_rad_s: float = 12.
    # Supervision only. Does not change the 100 Hz state/Actor rollout.
    audit_substeps: int = 2
    root_iterations: int = 36
    rho: float = 2.
    dual_interval: int = 8
    dual_updates_per_bank: int = 4
    event_sigma: float = .001
    event_directions: int = 4

    def __post_init__(self):
        if self.version != VERSION:
            raise ValueError("unsupported waypoint config version")
        ints = ("audit_substeps", "root_iterations", "dual_interval",
                "dual_updates_per_bank", "event_directions")
        for f in fields(self):
            x = getattr(self, f.name)
            if f.name == "version":
                continue
            if f.name in ints:
                if type(x) is not int or x < 1:
                    raise ValueError(f.name + " must be a positive integer")
            elif type(x) not in (float, int) or not math.isfinite(x) or x <= 0:
                raise ValueError(f.name + " must be finite and positive")
        if not 0 < self.tilt_degrees < 180:
            raise ValueError("requested midpoint tilt must be in (0,180)")
        if not 0 < self.boundary_margin_fraction < 1:
            raise ValueError("boundary margin must be in (0,1)")
        if max(self.half_span_fraction, self.bend_fraction) >= 1-self.boundary_margin_fraction:
            raise ValueError("waypoints must lie inside the unchanged world boundary")
        if not 0 < self.band_max_leg_fraction < .5:
            raise ValueError("band_max_leg_fraction must be in (0,.5)")
        if self.pose_half_width_m < self.arrival_radius_m:
            raise ValueError("pose band must cover the whole arrival ball")
        if self.no_slowdown_half_width_m <= self.pose_half_width_m:
            raise ValueError("no-slowdown band must contain pose band")
        if max(self.pose_tolerance_degrees, self.terminal_attitude_degrees) >= 180:
            raise ValueError("attitude tolerance must be below 180 degrees")
        if self.audit_substeps > 10 or not 12 <= self.root_iterations <= 48:
            raise ValueError("audit_substeps <=10 and 12<=root_iterations<=48 required")

    def snapshot(self):
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path):
        def unique(pairs):
            out = {}
            for key, value in pairs:
                if key in out:
                    raise ValueError("duplicate config key: " + key)
                out[key] = value
            return out
        value = json.loads(Path(path).read_text(), object_pairs_hook=unique)
        if not isinstance(value, dict) or set(value)-{f.name for f in fields(cls)}:
            raise ValueError("unknown waypoint config fields")
        return cls(**value)

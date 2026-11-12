"""Finite-horizon hover settling-time loss shared by TRAIN, EVAL and ablations."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .env_raptor import RaptorSimulator

TASK_OBJECTIVE_VERSION = "hover-motion-and-settling-time-v3"


@dataclass(frozen=True)
class HoverTimeConfig:
    weight: float = 1.0
    position_m: float = .1
    speed_m_s: float = .1
    omega_rad_s: float = .5

    def __post_init__(self):
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("time weight must be finite and nonnegative")
        for name in ("position_m", "speed_m_s", "omega_rad_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be positive and finite")


def settling_proxy(ratio, alive, *, dt):
    """Per-scene seconds proxy, including failed futures at the full horizon."""
    if (ratio.ndim != 2 or ratio.shape != alive.shape or len(ratio) < 2
            or alive.dtype != torch.bool or not math.isfinite(dt) or dt <= 0):
        raise ValueError("time proxy needs [H+1,N] ratios/masks and positive dt")
    quality = 1 / (1 + (ratio - 1).clamp_min(0))
    quality = torch.where(alive, quality, 0.)
    future = quality.flip(0).cummin(0).values.flip(0)
    return dt * (1 - future[:-1]).sum(0)


@torch.no_grad()
def settling_metrics(ratio, alive, *, dt):
    inside = (ratio <= 1) & alive
    index = torch.arange(len(ratio), device=ratio.device)[:, None]
    last_outside = torch.where(~inside, index, -1).amax(0)
    horizon = len(ratio) - 1
    duration = (last_outside + 1).clamp_max(horizon) * dt
    return {
        "hover_time_proxy_seconds": float(settling_proxy(ratio, alive, dt=dt).mean()),
        "hover_settling_time_seconds": float(duration.float().mean()),
        "hover_success_fraction": float(inside[-1].float().mean()),
    }


def _quality_samples(trace, config, *, birth=False):
    if birth:
        p, v, w = (trace.initial.position[None], trace.initial.velocity[None],
                   trace.initial.omega[None])
    else:
        p, v, w = trace.positions, trace.velocities, trace.omegas
    ratio = torch.maximum(torch.maximum(p.norm(dim=-1) / config.position_m,
                                       v.norm(dim=-1) / config.speed_m_s),
                          w.norm(dim=-1) / config.omega_rad_s)
    alive = ~RaptorSimulator.position_terminated(p, trace.initial.position_limit)
    return ratio, alive


def _report(base, ratio, alive, config, dt):
    report = dict(base)
    report["hover_objective"] = base["task_objective"]
    report.update(settling_metrics(ratio, alive, dt=dt))
    report["hover_time_weight"] = config.weight
    contribution = config.weight * report["hover_time_proxy_seconds"] / ((len(ratio)-1)*dt)
    report["task_objective"] = base["task_objective"] + contribution
    report["task_components"] = {**base["task_components"], "hover_time": contribution}
    return report


class HoverTimeObjective:
    """Observe the retained physical graph without additional simulator steps."""

    def __init__(self, config: HoverTimeConfig, dt: float):
        self.config = config
        self.dt = dt
        self.samples = []
        self.masks = []

    def observe(self, trace):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.config.weight > 0):
            if not self.samples:
                ratio, alive = _quality_samples(trace, self.config, birth=True)
                self.samples.append(ratio)
                self.masks.append(alive)
            ratio, alive = _quality_samples(trace, self.config)
            self.samples.append(ratio)
            self.masks.append(alive)

    def finish(self, costs, metrics):
        ratio, alive = torch.cat(self.samples), torch.cat(self.masks)
        report = _report(metrics, ratio, alive, self.config, self.dt)
        if costs is not None and self.config.weight:
            costs = costs + self.config.weight * settling_proxy(ratio, alive, dt=self.dt) / ((len(ratio)-1)*self.dt)
            report["task_objective"] = float(costs.detach().mean())
        if costs is not None and not bool(torch.isfinite(costs).all()):
            raise FloatingPointError("nonfinite hover settling-time objective")
        return costs, report

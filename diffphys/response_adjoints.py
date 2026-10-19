"""One retained full-horizon graph; physical-group Actor gradients and Time Decay."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import torch

from .response_task import (TaskLossConfig, FlightStatistics, initialize, rollout,
                           step_cost_components, uniform_scene_weights)
from .response_groups import (GroupBalanceConfig, group_gradient_coefficients,
                             backward_group_gradients, physics_group_layout)
from .response_grad_probe import GroupGradientProbe
from .agile_objective import (AGILE_COMPONENTS, AgileLossConfig, agile_terminal,
                              agile_metrics, prepare_agile_schedule)

# Streaming metrics/cost reduction preserves the original 50-step reduction order.
# This is NOT truncated BPTT: closed-loop tensors are never detached between chunks.
METRIC_CHUNK = 50


@dataclass(frozen=True)
class RolloutRecord:
    costs: torch.Tensor
    weights: torch.Tensor
    metrics: dict
    group_coefficients: torch.Tensor
    group_balance: dict
    group_config: GroupBalanceConfig
    loss_config: TaskLossConfig | AgileLossConfig
    probe: GroupGradientProbe | None = None


def collect_rollout(policy, simulator, initial, config, *, horizon=500, time_decay=1.,
                    group_config: GroupBalanceConfig = GroupBalanceConfig(), mission=None):
    if horizon < 1:
        raise ValueError("horizon must be positive")
    closed = initialize(policy, initial, mission=mission)
    if (mission is not None) != isinstance(config, AgileLossConfig):
        raise ValueError("agile mission and loss configuration must be supplied together")
    schedule = (prepare_agile_schedule(mission, config, horizon=horizon, dt=simulator.params.dt)
                if mission is not None else None)
    statistics = (FlightStatistics(initial, horizon, config, component_names=AGILE_COMPONENTS)
                  if mission is not None else FlightStatistics(initial, horizon, config))
    probe = (GroupGradientProbe(policy, *physics_group_layout(initial, group_config))
             if group_config.backward_backend == "probe" and torch.is_grad_enabled() else None)
    chunks = []
    measurements = {}
    try:
        with probe.capture() if probe is not None else nullcontext():
            for start in range(0, horizon, METRIC_CHUNK):
                trace = rollout(policy, simulator, closed, min(METRIC_CHUNK, horizon-start),
                                time_decay=time_decay, actor_probe=probe, record_observations=False,
                                mission=mission, agile_config=config if mission is not None else None,
                                horizon=horizon, agile_schedule=schedule)
                components = (trace.agile_components if mission is not None else
                              step_cost_components(trace, config, start=start, horizon=horizon))
                chunks.append(sum(components.values()))
                statistics.add(trace, start, components=components)
                if mission is not None:
                    for name, value in trace.agile_measurements.items():
                        measurements[name] = measurements.get(name, 0.)+value.sum(0)
                closed = trace.end
        costs = torch.cat(chunks).sum(0)
        if mission is not None:
            final = agile_terminal(closed.physical, mission, config)
            costs = costs+sum(final.values())
            statistics.add_terminal(final)
        if not bool(torch.isfinite(costs).all()):
            raise FloatingPointError("nonfinite full-horizon costs")
        weights = uniform_scene_weights(costs)
        if torch.is_grad_enabled():
            coefficients, report = group_gradient_coefficients(weights, initial, group_config)
        else:
            # Evaluation needs neither physical gradient cells nor their quota.
            # Its objective remains the same mean over all original scenes.
            coefficients, report = weights[None], {"evaluation_only": True}
        metrics = statistics.finish(costs.detach(), weights)
        if mission is not None:
            metrics.update(agile_metrics(closed.physical, mission, config,
                                         dt=simulator.params.dt, horizon=horizon, measurements=measurements))
            metrics["mission_sampling"] = mission.report
        return RolloutRecord(costs, weights, metrics,
                             coefficients, report, group_config, config, probe)
    except BaseException:
        if probe is not None:
            probe.close()
        raise


def backward_actor(policy, simulator, record, config, *, gradient_scale=.1):
    if config != record.loss_config:
        if record.probe is not None:
            record.probe.close()
        raise ValueError("recorded task objective changed")
    parameters = [p for p in policy.parameters() if p.requires_grad]
    if record.probe is not None:
        gradients, report = record.probe.backward(record.costs, record.group_coefficients,
                                                  parameters, record.group_config,
                                                  gradient_scale=gradient_scale)
    else:
        gradients, report = backward_group_gradients(record.costs, record.group_coefficients,
                                                    parameters, record.group_config,
                                                    gradient_scale=gradient_scale)
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient
    return {"group_gradient": report}

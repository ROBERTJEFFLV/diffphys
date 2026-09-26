"""One retained full-horizon graph; physical-group Actor gradients and Time Decay."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from response_task import TaskLossConfig, FlightStatistics, initialize, rollout, step_costs, risk_weights
from response_groups import GroupBalanceConfig, group_gradient_coefficients, backward_group_gradients

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
    loss_config: TaskLossConfig


def collect_rollout(policy, simulator, initial, config, *, horizon=500, time_decay=1.,
                    group_config: GroupBalanceConfig = GroupBalanceConfig()):
    if horizon < 1:
        raise ValueError("horizon must be positive")
    closed = initialize(policy, initial)
    statistics = FlightStatistics(initial, horizon, config)
    chunks = []
    for start in range(0, horizon, METRIC_CHUNK):
        trace = rollout(policy, simulator, closed, min(METRIC_CHUNK, horizon-start), time_decay=time_decay)
        chunks.append(step_costs(trace, config, start=start, horizon=horizon))
        statistics.add(trace, start)
        closed = trace.end
    costs = torch.cat(chunks).sum(0)
    if not bool(torch.isfinite(costs).all()):
        raise FloatingPointError("nonfinite full-horizon costs")
    weights = risk_weights(costs, config)
    coefficients, report = group_gradient_coefficients(weights, initial, group_config)
    return RolloutRecord(costs, weights, statistics.finish(costs.detach(), weights),
                         coefficients, report, group_config, config)


def backward_actor(policy, simulator, record, config, *, gradient_scale=.1):
    if config != record.loss_config:
        raise ValueError("recorded task objective changed")
    parameters = [p for p in policy.parameters() if p.requires_grad]
    gradients, report = backward_group_gradients(record.costs, record.group_coefficients,
                                                parameters, record.group_config,
                                                gradient_scale=gradient_scale)
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient
    return {"group_gradient": report}

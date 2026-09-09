"""Unified task-value Monte Carlo supervision and short-window Actor Adam.

Task costs always come from response_task.step_costs, including absolute time
weights. This path has no risk scalarization, candidate search or approval gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
import time

import torch
from torch import nn

from response_critic import critic_features, detach_closed_state, _finite_state
from response_phase1 import compare_boundary, task_loss_components, terminal_state_gradient
from response_task import (
    HardRiskConfig, RiskConfig, TaskLossConfig, TaskTrajectory, initialize,
    hard_risk_metrics, physical_risk_metrics, risk_weights, rollout, step_costs, trajectory_metrics,
)

TASK_VALUE_OBJECTIVE = "task-value-v1-exact-step-cost-suffix"


class TaskValueCritic(nn.Module):
    """Signed scalar cumulative task value; no output transform or learned units."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 256), nn.SiLU(),
                                     nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, inputs):
        return self.network(inputs)


@dataclass(frozen=True)
class TaskValueConfig:
    window_steps: int = 50
    lr: float = 1.e-3
    epochs: int = 1
    batch_size: int = 1024
    target_tau: float = .6  # new critic fraction; old target retains 1-tau
    gradient_clip: float = 10.

    def __post_init__(self):
        if min(self.window_steps, self.epochs, self.batch_size) < 1:
            raise ValueError("task-value counts must be positive")
        if any(not math.isfinite(v) or v <= 0 for v in (self.lr, self.gradient_clip)):
            raise ValueError("task-value learning rate and gradient clip must be positive and finite")
        if not math.isfinite(self.target_tau) or not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must be in (0, 1]")


@dataclass(frozen=True)
class TaskValueRecord:
    trajectory: TaskTrajectory
    inputs: torch.Tensor
    returns: torch.Tensor
    weights: torch.Tensor
    boundaries: dict


@torch.no_grad()
def collect_task_trajectory(policy, simulator, initial, horizon, window_steps, config):
    if horizon < 1 or window_steps < 1 or horizon % window_steps:
        raise ValueError("horizon must be a positive multiple of window_steps")
    inputs = [critic_features(initialize(policy, initial), 0, horizon)]
    boundaries = {}
    def observe(step, closed):
        inputs.append(critic_features(closed, step, horizon))
        if step % window_steps == 0:
            boundaries[step] = detach_closed_state(closed)
    trace = rollout(policy, simulator, initial, horizon, boundary_observer=observe)
    costs = step_costs(trace, config)
    returns = torch.cat((costs.flip(0).cumsum(0).flip(0), torch.zeros_like(costs[:1])))
    features = torch.stack(inputs).detach()
    if not _finite_state((features, returns)):
        raise FloatingPointError("nonfinite task trajectory or suffix labels")
    return TaskValueRecord(trace, features, returns.detach(), risk_weights(costs.sum(0), config), boundaries)


def _flat(gradients, parameters):
    return torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                      for p, g in zip(parameters, gradients)])


def accumulate_task_gradients(policy, target, simulator, initial, horizon, window_steps,
                              config, record, *, probes=True):
    """Average ten task gradients; only the caller performs an optimizer step."""
    if horizon < 1 or window_steps < 1 or horizon % window_steps:
        raise ValueError("horizon must be a positive multiple of window_steps")
    if any(p.requires_grad for p in target.parameters()):
        raise ValueError("target parameters must be frozen, with input gradients enabled")
    weights = record.weights.detach()
    policy.zero_grad(set_to_none=True)
    closed = detach_closed_state(initialize(policy, initial))
    parameters = list(policy.parameters())
    count = horizon // window_steps
    windows, surrogate = [], 0.
    for start in range(0, horizon, window_steps):
        end = start + window_steps
        trace = rollout(policy, simulator, closed, window_steps)
        boundary = compare_boundary(record.boundaries[end], trace.end, end)
        local = (weights * step_costs(trace, config, start=start, horizon=horizon).sum(0)).sum()
        terminal = local.new_zeros(())
        if end < horizon:
            # Network output ALREADY has cumulative task units. Do not multiply
            # by H-end, normalize it as risk, or detach the terminal state.
            terminal = (weights * target(critic_features(trace.end, end, horizon)).squeeze(-1)).sum()
        objective = local + terminal
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError("nonfinite window task surrogate")
        row = {"start": start, "end": end, "boundary": boundary,
               "local_task": float(local.detach()), "terminal_value": float(terminal.detach())}
        if probes:
            row.update(terminal_state_gradient(target, trace.end, end, horizon, mean_risk=False))
            local_grads = torch.autograd.grad(local, parameters, retain_graph=True, allow_unused=True)
            row['local_task_gradient_norm'] = float(_flat(local_grads, parameters).norm(dtype=torch.float64))
        grads = torch.autograd.grad(objective, parameters, allow_unused=True)
        row['task_gradient_norm'] = float(_flat(grads, parameters).norm(dtype=torch.float64))
        if not _finite_state(row):
            raise FloatingPointError("nonfinite short-window task gradient")
        for p, g in zip(parameters, grads):
            if g is not None:
                if p.grad is None:
                    p.grad = g.detach() / count
                else:
                    p.grad.add_(g.detach(), alpha=1. / count)
        surrogate += float(objective.detach()) / count
        windows.append(row)
        closed = detach_closed_state(trace.end)
        del trace, local, terminal, objective, grads
    return {"surrogate_loss": surrogate, "windows": windows}


@torch.no_grad()
def task_metrics(trace, config):
    metrics = trajectory_metrics(trace, config)
    metrics.update(physical_risk_metrics(trace, RiskConfig(), config))
    metrics.update(hard_risk_metrics(trace, RiskConfig(), config, HardRiskConfig()))
    metrics['task_loss_components'] = task_loss_components(trace, config)
    if not metrics['finite'] or not _finite_state(metrics):
        raise FloatingPointError("nonfinite continuous task metrics")
    return metrics


class TaskValueTrainer:
    """Persistent supervised Critic and target; Actor Adam is caller-owned."""

    def __init__(self, policy, example, horizon, config=TaskValueConfig()):
        if horizon < 1 or horizon % config.window_steps:
            raise ValueError("horizon must be divisible by window_steps")
        self.config = config
        self.critic = TaskValueCritic(critic_features(example, 0, horizon).shape[-1]).to(next(policy.parameters()))
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr)
        self.completed_fits = 0
        self.stage = 'idle'

    def state_dict(self):
        return {'objective': TASK_VALUE_OBJECTIVE, 'config': asdict(self.config),
                'critic': self.critic.state_dict(), 'target': self.target.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'completed_fits': self.completed_fits}

    def load_state_dict(self, saved):
        if saved.get('objective') != TASK_VALUE_OBJECTIVE or saved.get('config') != asdict(self.config):
            raise ValueError("task-value objective/config changed; start a new experiment")
        if not _finite_state(saved):
            raise ValueError("nonfinite task-value checkpoint")
        self.critic.load_state_dict(saved['critic'])
        self.target.load_state_dict(saved['target'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.completed_fits = saved['completed_fits']

    @torch.no_grad()
    def prediction_metrics(self, inputs, returns):
        predicted = torch.cat([self.critic(x).squeeze(-1) for x in inputs.split(self.config.batch_size)])
        if not _finite_state((predicted, returns)):
            raise FloatingPointError("nonfinite task-value prediction")
        error = predicted - returns
        return {'loss': float(nn.functional.smooth_l1_loss(predicted, returns)),
                'mae': float(error.abs().mean()), 'rmse': float(error.square().mean().sqrt()),
                'target_range': [float(returns.min()), float(returns.max())],
                'target_mean': float(returns.mean()),
                'prediction_range': [float(predicted.min()), float(predicted.max())]}

    def fit(self, record):
        inputs = record.inputs.detach().flatten(0, 1)
        returns = record.returns.detach().flatten()
        before = self.prediction_metrics(inputs, returns)
        snapshot = copy.deepcopy(self.state_dict())
        gradient_norms = []
        try:
            for _ in range(self.config.epochs):
                for indices in torch.randperm(returns.numel(), device=returns.device).split(self.config.batch_size):
                    self.optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.smooth_l1_loss(self.critic(inputs[indices]).squeeze(-1), returns[indices])
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite task-value fit loss")
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
                    gradient_norms.append(float(norm))
                    self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            after = self.prediction_metrics(inputs, returns)
            with torch.no_grad():
                for target, current in zip(self.target.parameters(), self.critic.parameters()):
                    target.lerp_(current, self.config.target_tau)
            if not _finite_state(self.state_dict()):
                raise FloatingPointError("nonfinite fitted task-value state")
            self.completed_fits += 1
        except Exception:
            self.load_state_dict(snapshot)
            self.optimizer.zero_grad(set_to_none=True)
            raise
        return {'critic_loss_before': before['loss'], 'critic_loss_after': after['loss'],
                'critic_before': before, 'critic_after': after,
                'critic_gradient_norm_max': max(gradient_norms), 'critic_fits': self.completed_fits}

    def update(self, policy, optimizer, simulator, initial, horizon, loss_config, *, gradient_clip, probes=True):
        clock = time.monotonic()
        self.stage = 'forward'
        record = collect_task_trajectory(policy, simulator, initial, horizon, self.config.window_steps, loss_config)
        metrics = task_metrics(record.trajectory, loss_config)
        forward_seconds = time.monotonic() - clock
        self.stage = 'critic'
        fitted = self.fit(record)
        fit_seconds = time.monotonic() - clock - forward_seconds
        self.stage = 'short_window_backward'
        actor = accumulate_task_gradients(policy, self.target, simulator, initial, horizon,
            self.config.window_steps, loss_config, record, probes=probes)
        norm = nn.utils.clip_grad_norm_(policy.parameters(), gradient_clip, error_if_nonfinite=True)
        after_clip = float(torch.cat([p.grad.flatten() for p in policy.parameters() if p.grad is not None]).norm(dtype=torch.float64))
        backward_seconds = time.monotonic() - clock - forward_seconds - fit_seconds
        self.stage = 'adam'
        old_model, old_optimizer = copy.deepcopy(policy.state_dict()), copy.deepcopy(optimizer.state_dict())
        before = nn.utils.parameters_to_vector(policy.parameters()).detach().clone()
        try:
            optimizer.step()
            if not _finite_state((policy.state_dict(), optimizer.state_dict())):
                raise FloatingPointError("nonfinite Actor Adam state")
        except Exception:
            policy.load_state_dict(old_model)
            optimizer.load_state_dict(old_optimizer)
            raise
        step_norm = float((nn.utils.parameters_to_vector(policy.parameters()).detach() - before).norm(dtype=torch.float64))
        self.stage = 'idle'
        return {'updated': True, 'finite': True, 'train_before': metrics, **fitted, **actor,
                'gradient_norm_before_clip': float(norm), 'gradient_norm_after_clip': after_clip,
                'parameter_step_norm': step_norm, 'forward_seconds': forward_seconds,
                'critic_seconds': fit_seconds, 'backward_seconds': backward_seconds,
                'update_seconds': time.monotonic() - clock}

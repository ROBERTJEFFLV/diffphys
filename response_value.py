"""Unified task-value Monte Carlo supervision and short-window Actor Adam.

Task costs always come from response_task.step_costs, including absolute time
weights. This path has no risk scalarization, candidate search or approval gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from contextlib import contextmanager
import copy
import math
import time

import torch
from torch import nn

from response_critic import critic_features, detach_closed_state, _finite_state
from response_phase1 import (compare_boundary, task_loss_components, terminal_state_gradient,
                            derivative_gradient_metrics)
from response_task import (
    HardRiskConfig, RiskConfig, TaskLossConfig, TaskTrajectory, ResponseClosedLoopState, initialize,
    hard_risk_metrics, physical_risk_metrics, risk_weights, rollout, step_costs, trajectory_metrics,
)

TASK_VALUE_OBJECTIVE = "task-value-v2-memory-cosine-lognorm"

# x_hat=x/s in critic_features: dV/dx_hat=s*dV/dx. Select one field,
# never silently enable unvalidated full-state supervision.
DERIVATIVE_STATE_SCALES = {
    'policy.memory': 1., 'policy.integral': .5,
    'physical.position': 5., 'physical.velocity': 5., 'physical.omega': 10.,
    'policy.previous_velocity': 5., 'policy.previous_omega': 10.,
}


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
    derivative_state_group: str = 'policy.memory'
    derivative_boundaries: tuple[int, ...] = ()  # first/middle/last nonterminal window
    derivative_samples: int = 32
    derivative_holdout_samples: int = 16
    derivative_batch_size: int = 32
    derivative_epsilon: float = 1.e-8
    derivative_direction_factor: float = .5

    def __post_init__(self):
        if min(self.window_steps, self.epochs, self.batch_size) < 1:
            raise ValueError("task-value counts must be positive")
        if any(not math.isfinite(v) or v <= 0 for v in (self.lr, self.gradient_clip)):
            raise ValueError("task-value learning rate and gradient clip must be positive and finite")
        if not math.isfinite(self.target_tau) or not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must be in (0, 1]")
        if self.derivative_state_group not in DERIVATIVE_STATE_SCALES:
            raise ValueError('unsupported derivative state group')
        if min(self.derivative_samples, self.derivative_holdout_samples, self.derivative_batch_size) < 1:
            raise ValueError('derivative sample counts must be positive')
        if not math.isfinite(self.derivative_epsilon) or self.derivative_epsilon <= 0:
            raise ValueError('derivative epsilon must be positive and finite')
        if not math.isfinite(self.derivative_direction_factor) or self.derivative_direction_factor < 0:
            raise ValueError('derivative direction factor must be nonnegative and finite')
        if len(set(self.derivative_boundaries)) != len(self.derivative_boundaries) or any(
                t <= 0 or t % self.window_steps for t in self.derivative_boundaries):
            raise ValueError('derivative boundaries must be distinct positive window boundaries')

    def sampling_boundaries(self, horizon):
        if self.derivative_boundaries:
            if max(self.derivative_boundaries) >= horizon:
                raise ValueError('derivative boundaries must be before the horizon')
            return self.derivative_boundaries
        available = list(range(self.window_steps, horizon, self.window_steps))
        return tuple(sorted({available[0], available[len(available)//2], available[-1]})) if available else ()


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


@dataclass(frozen=True)
class DerivativeSamples:
    closed: ResponseClosedLoopState
    steps: torch.Tensor
    scene_ids: torch.Tensor
    gradients: torch.Tensor  # per-scene d(remaining task cost)/d(x/s), detached
    horizon: int
    state_group: str


def _select_closed(closed, indices):
    return type(closed)(*(type(state)(**{f.name: getattr(state, f.name)[indices].detach()
                         for f in fields(state)}) for state in (closed.physical, closed.policy)))


def select_derivative_samples(samples, indices):
    return replace(samples, closed=_select_closed(samples.closed, indices), steps=samples.steps[indices],
                   scene_ids=samples.scene_ids[indices], gradients=samples.gradients[indices])


def _state_leaf(closed, state_group):
    parent, field = state_group.split('.')
    leaf = getattr(getattr(closed, parent), field).detach().requires_grad_(True)
    return replace(closed, **{parent: replace(getattr(closed, parent), **{field: leaf})}), leaf


@contextmanager
def _frozen_actor(policy):
    parameters = list(policy.parameters())
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def collect_derivative_samples(policy, simulator, record, horizon, loss_config, config):
    """Small fresh TRAIN pools; held-out scenes never supply derivative loss.

    A per-scene value does not include batch CVaR weights. Actor applies those
    weights to terminal values later. No replay buffer or DEV calibration.
    """
    boundaries = config.sampling_boundaries(horizon)
    if not boundaries:
        return None, None  # a single terminal window has no bootstrap boundary
    batch = record.inputs.shape[1]
    if batch < 2:
        raise ValueError('derivative holdout requires at least two TRAIN scenes')
    scene_order = torch.randperm(batch, device=record.inputs.device)
    holdout_count = min(batch//2, max(1, math.ceil(config.derivative_holdout_samples/len(boundaries))))

    def collect(scene_ids, count):
        # Interleave independently shuffled scenes across boundaries, then cap
        # the pool. This covers all selected boundaries when count permits.
        scene_grid = torch.stack([scene_ids[torch.randperm(len(scene_ids), device=scene_ids.device)]
                                  for _ in boundaries], dim=1).flatten()[:count]
        step_grid = torch.tensor(boundaries, device=scene_ids.device).repeat(len(scene_ids))[:count]
        states, labels, steps, ids = [], [], [], []
        for step in boundaries:
            selected = scene_grid[step_grid == step]
            if not len(selected):
                continue
            closed = _select_closed(record.boundaries[step], selected)
            differentiable, leaf = _state_leaf(closed, config.derivative_state_group)
            continuation = rollout(policy, simulator, differentiable, horizon-step)
            costs = step_costs(continuation, loss_config, start=step, horizon=horizon).sum(0)
            gradient = torch.autograd.grad(costs.sum(), leaf)[0].flatten(1)
            gradient = gradient.detach() * DERIVATIVE_STATE_SCALES[config.derivative_state_group]
            if not _finite_state((costs, gradient)):
                raise FloatingPointError('nonfinite continuation derivative labels')
            states.append(closed); labels.append(gradient)
            steps.append(step_grid.new_full((len(selected),), step)); ids.append(selected)
            del continuation, costs, gradient, differentiable, leaf
        combined = type(states[0])(*(type(getattr(states[0], group))(**{
            f.name: torch.cat([getattr(getattr(s, group), f.name) for s in states])
            for f in fields(getattr(states[0], group))}) for group in ('physical', 'policy')))
        return DerivativeSamples(combined, torch.cat(steps), torch.cat(ids), torch.cat(labels),
                                 horizon, config.derivative_state_group)

    with _frozen_actor(policy), torch.enable_grad():
        return (collect(scene_order[holdout_count:], config.derivative_samples),
                collect(scene_order[:holdout_count], config.derivative_holdout_samples))


def predict_derivatives(critic, samples, *, create_graph=False):
    """Differentiate the same critic_features mapping used by terminal Actor V."""
    with torch.enable_grad():
        closed, leaf = _state_leaf(samples.closed, samples.state_group)
        features = critic_features(closed, 0, samples.horizon)
        # critic_features' final column is t/H; this minibatch mixes boundaries.
        features = torch.cat((features[:, :-1], samples.steps.to(features)[:, None]/samples.horizon), dim=-1)
        predicted = critic(features).sum()
        gradient = torch.autograd.grad(predicted, leaf, create_graph=create_graph)[0].flatten(1)
        return gradient * DERIVATIVE_STATE_SCALES[samples.state_group]


def derivative_losses(predicted, true, *, epsilon=1.e-8):
    """Per-sample direction and log magnitude; zero truth has no direction."""
    predicted, true = predicted.double(), true.detach().double()
    pn, tn = predicted.norm(dim=-1), true.norm(dim=-1)
    valid = tn > epsilon
    cosines = nn.functional.cosine_similarity(predicted, true, dim=-1, eps=epsilon)
    direction = ((1-cosines)[valid].mean() if bool(valid.any()) else predicted.sum()*0.)
    magnitude = nn.functional.smooth_l1_loss(torch.log(pn+epsilon)-torch.log(tn+epsilon), torch.zeros_like(tn))
    return direction, magnitude


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
            row.update(terminal_state_gradient(target, trace.end, end, horizon, mean_risk=False,
                                               weights=weights))
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
        config.sampling_boundaries(horizon)
        self.critic = TaskValueCritic(critic_features(example, 0, horizon).shape[-1]).to(next(policy.parameters()))
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr)
        self.completed_fits = 0
        self.derivative_balance = None
        self.derivative_sampling = None
        self.stage = 'idle'

    def state_dict(self):
        return {'objective': TASK_VALUE_OBJECTIVE, 'config': asdict(self.config),
                'critic': self.critic.state_dict(), 'target': self.target.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'completed_fits': self.completed_fits,
                'derivative_balance': copy.deepcopy(self.derivative_balance),
                'derivative_sampling': copy.deepcopy(self.derivative_sampling)}

    def load_state_dict(self, saved):
        if saved.get('objective') != TASK_VALUE_OBJECTIVE or saved.get('config') != asdict(self.config):
            raise ValueError("task-value objective/config changed; start a new experiment")
        if not _finite_state(saved):
            raise ValueError("nonfinite task-value checkpoint")
        self.critic.load_state_dict(saved['critic'])
        self.target.load_state_dict(saved['target'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.completed_fits = saved['completed_fits']
        self.derivative_balance = copy.deepcopy(saved['derivative_balance'])
        self.derivative_sampling = copy.deepcopy(saved['derivative_sampling'])

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

    def _balance_derivative_losses(self, value_loss, direction_loss, magnitude_loss):
        """One-time parameter-gradient ratios, BEFORE clipping and the 0.5 factor."""
        losses = {'value': value_loss, 'direction': direction_loss, 'magnitude': magnitude_loss}
        parameters = list(self.critic.parameters())
        norms = {name: float(_flat(torch.autograd.grad(loss, parameters, retain_graph=True,
                    allow_unused=True), parameters).norm(dtype=torch.float64)) for name, loss in losses.items()}
        epsilon = self.config.derivative_epsilon
        self.derivative_balance = {
            'direction': norms['value']/max(norms['direction'], epsilon),
            'magnitude': norms['value']/max(norms['magnitude'], epsilon),
            'direction_factor': self.config.derivative_direction_factor, 'initial_gradient_norms': norms,
            'denominator_floor': epsilon, 'calibrated_at_fit': self.completed_fits+1}
        if not _finite_state(self.derivative_balance):
            raise FloatingPointError('nonfinite derivative loss balance')

    def _derivative_metrics(self, network, samples):
        if samples is None:
            return None
        predicted = predict_derivatives(network, samples).detach()
        result = derivative_gradient_metrics(predicted, samples.gradients, self.config.derivative_epsilon)
        result['boundaries'] = {str(step): derivative_gradient_metrics(predicted[samples.steps == step],
            samples.gradients[samples.steps == step], self.config.derivative_epsilon)
            for step in sorted(set(samples.steps.tolist()))}
        return result

    def fit(self, record, derivatives=None, heldout=None):
        inputs = record.inputs.detach().flatten(0, 1)
        returns = record.returns.detach().flatten()
        before = self.prediction_metrics(inputs, returns)
        snapshot = copy.deepcopy(self.state_dict())
        gradient_norms = []
        derivative_losses_log = []
        try:
            target_before = self._derivative_metrics(self.target, heldout)
            if derivatives is not None:
                if heldout is None or not set(derivatives.scene_ids.tolist()).isdisjoint(heldout.scene_ids.tolist()):
                    raise ValueError('derivative holdout must use disjoint TRAIN scenes')
                self.derivative_sampling = {
                    'state_group': derivatives.state_group, 'horizon': derivatives.horizon,
                    'state_scale': DERIVATIVE_STATE_SCALES[derivatives.state_group],
                    'boundary_steps': sorted(set(derivatives.steps.tolist())),
                    'train_scene_ids': derivatives.scene_ids.tolist(), 'train_steps': derivatives.steps.tolist(),
                    'heldout_scene_ids': heldout.scene_ids.tolist(), 'heldout_steps': heldout.steps.tolist(),
                    'heldout_scope': 'unseen derivative labels on TRAIN scenes; value regression still uses all scenes'}
            for _ in range(self.config.epochs):
                for indices in torch.randperm(returns.numel(), device=returns.device).split(self.config.batch_size):
                    self.optimizer.zero_grad(set_to_none=True)
                    value_loss = nn.functional.smooth_l1_loss(self.critic(inputs[indices]).squeeze(-1), returns[indices])
                    loss = value_loss
                    if derivatives is not None:
                        selection = torch.randint(len(derivatives.steps), (self.config.derivative_batch_size,),
                                                  device=derivatives.steps.device)
                        samples = select_derivative_samples(derivatives, selection)
                        predicted = predict_derivatives(self.critic, samples, create_graph=True)
                        direction, magnitude = derivative_losses(predicted, samples.gradients,
                                                                  epsilon=self.config.derivative_epsilon)
                        if self.derivative_balance is None:
                            self._balance_derivative_losses(value_loss, direction, magnitude)
                        loss = (value_loss + self.config.derivative_direction_factor*self.derivative_balance['direction']*direction
                                + self.derivative_balance['magnitude']*magnitude)
                        derivative_losses_log.append((float(direction.detach()), float(magnitude.detach())))
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
            target_after = self._derivative_metrics(self.target, heldout)
            critic_after = self._derivative_metrics(self.critic, heldout)
            if not _finite_state(self.state_dict()):
                raise FloatingPointError("nonfinite fitted task-value state")
            self.completed_fits += 1
        except Exception:
            self.load_state_dict(snapshot)
            self.optimizer.zero_grad(set_to_none=True)
            raise
        return {'critic_loss_before': before['loss'], 'critic_loss_after': after['loss'],
                'critic_before': before, 'critic_after': after,
                'critic_gradient_norm_max': max(gradient_norms), 'critic_fits': self.completed_fits,
                'derivative_train_samples': 0 if derivatives is None else len(derivatives.steps),
                'derivative_minibatches': len(derivative_losses_log),
                'derivative_direction_loss': (sum(r[0] for r in derivative_losses_log)/len(derivative_losses_log)
                                              if derivative_losses_log else None),
                'derivative_magnitude_loss': (sum(r[1] for r in derivative_losses_log)/len(derivative_losses_log)
                                              if derivative_losses_log else None),
                'derivative_balance': copy.deepcopy(self.derivative_balance),
                'derivative_sampling': copy.deepcopy(self.derivative_sampling),
                'target_derivative_heldout_before': target_before,
                'target_derivative_heldout_after': target_after, 'critic_derivative_heldout_after': critic_after}

    def update(self, policy, optimizer, simulator, initial, horizon, loss_config, *, gradient_clip, probes=True):
        clock = time.monotonic()
        self.stage = 'forward'
        record = collect_task_trajectory(policy, simulator, initial, horizon, self.config.window_steps, loss_config)
        metrics = task_metrics(record.trajectory, loss_config)
        forward_seconds = time.monotonic() - clock
        self.stage = 'continuation_derivatives'
        derivatives, heldout = collect_derivative_samples(policy, simulator, record, horizon, loss_config, self.config)
        derivative_seconds = time.monotonic() - clock - forward_seconds
        self.stage = 'critic'
        fitted = self.fit(record, derivatives, heldout)
        fit_seconds = time.monotonic() - clock - forward_seconds - derivative_seconds
        self.stage = 'short_window_backward'
        actor = accumulate_task_gradients(policy, self.target, simulator, initial, horizon,
            self.config.window_steps, loss_config, record, probes=probes)
        norm = nn.utils.clip_grad_norm_(policy.parameters(), gradient_clip, error_if_nonfinite=True)
        after_clip = float(torch.cat([p.grad.flatten() for p in policy.parameters() if p.grad is not None]).norm(dtype=torch.float64))
        backward_seconds = time.monotonic() - clock - forward_seconds - derivative_seconds - fit_seconds
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
                'critic_seconds': fit_seconds, 'continuation_derivative_seconds': derivative_seconds,
                'backward_seconds': backward_seconds,
                'update_seconds': time.monotonic() - clock}

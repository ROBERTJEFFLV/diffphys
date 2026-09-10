"""Unified task-value Monte Carlo supervision and short-window Actor Adam.

Task costs always come from response_task.step_costs, including absolute time
weights. Critic readiness uses full feedback; there is no risk or EVAL veto.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import copy
import math
import time

import torch
from torch import nn

from response_critic import critic_features, detach_closed_state, _finite_state
from response_phase1 import (compare_boundary, task_loss_components, terminal_state_gradient,
                            derivative_gradient_metrics, dynamic_closed_state)
from response_adjoints import (collect_boundary_adjoints, linearized_terminal,
                               STATE_SCALES, gradient_coordinates, frozen_parameters)
from response_task import (
    HardRiskConfig, RiskConfig, TaskLossConfig, TaskTrajectory, ResponseClosedLoopState, initialize,
    hard_risk_metrics, physical_risk_metrics, risk_weights, rollout, step_costs, trajectory_metrics,
)

TASK_VALUE_OBJECTIVE = "task-value-v3-full-boundary-adjoints"


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
    derivative_state_groups: tuple[str, ...] = tuple(STATE_SCALES)
    derivative_boundaries: tuple[int, ...] = ()  # every nonterminal window
    derivative_holdout_scenes: int = 16
    derivative_batch_size: int = 32
    derivative_epsilon: float = 1.e-8
    derivative_direction_factor: float = .5
    derivative_balance_mode: str = 'minibatch'  # fixed is an explicit ablation
    terminal_mode: str = 'critic'
    warmup_max_fits: int = 8
    warmup_max_seconds: float = 120.
    ready_min_cosine: float = .9
    ready_max_relative_error: float = .5

    def __post_init__(self):
        if min(self.window_steps, self.epochs, self.batch_size) < 1:
            raise ValueError("task-value counts must be positive")
        if any(not math.isfinite(v) or v <= 0 for v in (self.lr, self.gradient_clip)):
            raise ValueError("task-value learning rate and gradient clip must be positive and finite")
        if not math.isfinite(self.target_tau) or not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must be in (0, 1]")
        if (not self.derivative_state_groups or len(set(self.derivative_state_groups)) != len(self.derivative_state_groups)
                or any(n not in STATE_SCALES for n in self.derivative_state_groups)):
            raise ValueError('unsupported or repeated derivative state groups')
        if min(self.derivative_holdout_scenes, self.derivative_batch_size) < 1:
            raise ValueError('derivative sample counts must be positive')
        if self.derivative_balance_mode not in ('minibatch', 'fixed'):
            raise ValueError('derivative balance mode must be minibatch or fixed')
        if self.terminal_mode not in ('critic', 'oracle_full_state', 'none'):
            raise ValueError('unsupported terminal mode')
        if self.warmup_max_fits < 1 or not math.isfinite(self.warmup_max_seconds) or self.warmup_max_seconds <= 0:
            raise ValueError('warmup budgets must be positive and finite')
        if not math.isfinite(self.ready_min_cosine) or not -1 <= self.ready_min_cosine <= 1:
            raise ValueError('readiness cosine must be in [-1,1]')
        if not math.isfinite(self.ready_max_relative_error) or self.ready_max_relative_error < 0:
            raise ValueError('readiness relative error must be nonnegative and finite')
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
        return tuple(range(self.window_steps, horizon, self.window_steps))


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
    boundaries = {0: detach_closed_state(initialize(policy, initial))}
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
    gradients: dict  # dimensionless covectors, rotation tangent covectors
    horizon: int
    state_groups: tuple
    metadata: dict


def _select_closed(closed, indices):
    return type(closed)(*(type(state)(**{f.name: getattr(state, f.name)[indices].detach()
                         for f in fields(state)}) for state in (closed.physical, closed.policy)))


def select_derivative_samples(samples, indices):
    return replace(samples, closed=_select_closed(samples.closed, indices), steps=samples.steps[indices],
                   scene_ids=samples.scene_ids[indices],
                   gradients={n:g[indices] for n,g in samples.gradients.items()})


def collect_derivative_samples(policy, simulator, record, horizon, loss_config, config, *, adjoints=None):
    """Every selected scene at every boundary; only derivative labels are held out."""
    boundaries = config.sampling_boundaries(horizon)
    if not boundaries:
        return None, None
    if adjoints is None:
        adjoints = collect_boundary_adjoints(policy, simulator, record, horizon, config.window_steps, loss_config)
    adjoints.validate(policy, horizon, config.window_steps, loss_config, record)
    batch = record.inputs.shape[1]
    if batch < 2:
        raise ValueError('derivative holdout requires at least two TRAIN scenes')
    order = torch.randperm(batch, device=record.inputs.device)
    holdout_count = min(config.derivative_holdout_scenes, max(1, batch//2))
    metadata = {'actor_sha256': adjoints.actor_sha256, 'objective': TASK_VALUE_OBJECTIVE,
        'adjoint_objective': adjoints.objective, 'horizon': horizon, 'loss_config': adjoints.loss_config,
        'initial_sha256': adjoints.initial_sha256,
        'state_scales': {n:STATE_SCALES[n] for n in config.derivative_state_groups},
        'rotation_coordinates': 'R exp(skew(delta)); radians; ambient adjoints retained for Actor VJP',
        'aliases': 'independent dynamic leaves; last_action and calls retained by oracle but absent/differentially inactive in critic_features'}

    def collect(ids):
        states = [_select_closed(record.boundaries[t], ids) for t in boundaries]
        combined = type(states[0])(*(type(getattr(states[0], group))(**{
            f.name: torch.cat([getattr(getattr(s, group), f.name) for s in states])
            for f in fields(getattr(states[0], group))}) for group in ('physical', 'policy')))
        gradients = {n:torch.cat([gradient_coordinates(n, adjoints.gradients[t][n][ids], state)
                                 for t,state in zip(boundaries, states)]).detach()
                     for n in config.derivative_state_groups}
        return DerivativeSamples(combined, torch.cat([ids.new_full((len(ids),), t) for t in boundaries]),
            ids.repeat(len(boundaries)), gradients, horizon, config.derivative_state_groups, metadata)
    return collect(order[holdout_count:]), collect(order[:holdout_count])


def predict_derivatives(critic, samples, *, create_graph=False):
    """Same complete state mapping as Actor terminal V; rotations use tangent coordinates."""
    with torch.enable_grad():
        closed, leaves, names = dynamic_closed_state(samples.closed)
        features = critic_features(closed, 0, samples.horizon)
        features = torch.cat((features[:, :-1], samples.steps.to(features)[:, None]/samples.horizon), dim=-1)
        predicted = critic(features).sum()
        raw = torch.autograd.grad(predicted, leaves, create_graph=create_graph, allow_unused=True)
        return {n:gradient_coordinates(n, torch.zeros_like(z) if g is None else g, closed)
                for n,z,g in zip(names, leaves, raw) if n in samples.state_groups}


def derivative_losses(predicted, true, *, epsilon=1.e-8):
    """Equal state-group losses, not one norm dominated by the largest field."""
    if isinstance(predicted, dict):
        if set(predicted) != set(true):
            raise ValueError('derivative prediction/label groups differ')
        losses = [derivative_losses(predicted[n], true[n], epsilon=epsilon) for n in predicted]
        return tuple(torch.stack([row[i] for row in losses]).mean() for i in (0,1))
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
                              config, record, *, probes=True, terminal_mode='critic', adjoints=None,
                              scene_indices=None):
    """Average ten task gradients; only the caller performs an optimizer step."""
    if horizon < 1 or window_steps < 1 or horizon % window_steps:
        raise ValueError("horizon must be a positive multiple of window_steps")
    if terminal_mode not in ('critic', 'oracle_full_state', 'none'):
        raise ValueError('unknown terminal mode')
    if terminal_mode == 'critic' and any(p.requires_grad for p in target.parameters()):
        raise ValueError("target parameters must be frozen, with input gradients enabled")
    if terminal_mode == 'oracle_full_state':
        if adjoints is None:
            raise ValueError('oracle terminal requires boundary adjoints')
        adjoints.validate(policy, horizon, window_steps, config, record)
    weights = record.weights.detach()
    weighted_scenarios = len(weights)
    if scene_indices is not None:
        ids = torch.as_tensor(scene_indices, device=weights.device, dtype=torch.long)
        if (ids.ndim != 1 or not ids.numel() or ids.unique().numel() != ids.numel()
                or bool((ids < 0).any()) or bool((ids >= len(weights)).any())):
            raise ValueError('scene_indices must be distinct valid scene IDs')
        # Keep the original CUDA forward batch: changing GEMM batch size can
        # perturb long-horizon gradients. Select held-out scenes by weights only.
        mask = torch.zeros_like(weights)
        mask[ids] = 1
        weights = weights * mask
        weighted_scenarios = ids.numel()
    policy.zero_grad(set_to_none=True)
    closed = initialize(policy, initial)
    parameters = list(policy.parameters())
    count = horizon // window_steps
    windows, surrogate = [], 0.
    for start in range(0, horizon, window_steps):
        end = start + window_steps
        trace = rollout(policy, simulator, closed, window_steps)
        boundary = compare_boundary(record.boundaries[end], trace.end, end)
        local = (weights * step_costs(trace, config, start=start, horizon=horizon).sum(0)).sum()
        terminal = local.new_zeros(())
        if end < horizon and terminal_mode == 'critic':
            # Network output ALREADY has cumulative task units. Do not multiply
            # by H-end, normalize it as risk, or detach the terminal state.
            terminal = (weights * target(critic_features(trace.end, end, horizon)).squeeze(-1)).sum()
        elif end < horizon and terminal_mode == 'oracle_full_state':
            terminal = (weights * linearized_terminal(trace.end,
                adjoints.gradients[end], record.returns[end])).sum()
        objective = local + terminal
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError("nonfinite window task surrogate")
        row = {"start": start, "end": end, "boundary": boundary,
               "local_task": float(local.detach()), "terminal_value": float(terminal.detach())}
        if probes and terminal_mode == 'critic':
            row.update(terminal_state_gradient(target, trace.end, end, horizon, mean_risk=False,
                                               weights=weights))
        if probes:
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
    return {"surrogate_loss": surrogate, "windows": windows, 'terminal_mode': terminal_mode,
            'forward_scenarios': len(weights), 'weighted_scenarios': weighted_scenarios}


def task_gradient_vector(policy, target, simulator, initial, horizon, window_steps,
                         config, record, *, terminal_mode, adjoints=None, scene_indices=None):
    """Read-only parameter-gradient probe, preserving the caller's .grad slots."""
    parameters = list(policy.parameters())
    saved = [p.grad for p in parameters]
    try:
        accumulate_task_gradients(policy, target, simulator, initial, horizon, window_steps,
            config, record, probes=False, terminal_mode=terminal_mode,
            adjoints=adjoints, scene_indices=scene_indices)
        return _flat([p.grad for p in parameters], parameters)
    finally:
        for p, grad in zip(parameters, saved):
            p.grad = grad


def audit_actor_gradient(policy, target, simulator, initial, horizon, window_steps,
                         config, record, adjoints, *, scene_indices=None, reference=None):
    from response_phase1 import parameter_gradient_metrics
    if reference is None:
        reference = task_gradient_vector(policy, target, simulator, initial, horizon, window_steps,
            config, record, terminal_mode='oracle_full_state', adjoints=adjoints, scene_indices=scene_indices)
    with frozen_parameters(target):
        candidate = task_gradient_vector(policy, target, simulator, initial, horizon, window_steps,
            config, record, terminal_mode='critic', scene_indices=scene_indices)
    return parameter_gradient_metrics(candidate, reference), reference


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
        self.balance_updates = 0
        self.derivative_sampling = None
        self.stage = 'idle'

    def state_dict(self):
        return {'objective': TASK_VALUE_OBJECTIVE, 'config': asdict(self.config),
                'critic': self.critic.state_dict(), 'target': self.target.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'completed_fits': self.completed_fits,
                'derivative_balance': copy.deepcopy(self.derivative_balance),
                'balance_updates': self.balance_updates,
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
        self.balance_updates = saved['balance_updates']
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
        """Measure every minibatch; fixed mode is only a registered ablation."""
        losses = {'value': value_loss, 'direction': direction_loss, 'magnitude': magnitude_loss}
        parameters = list(self.critic.parameters())
        norms = {name: float(_flat(torch.autograd.grad(loss, parameters, retain_graph=True,
                    allow_unused=True), parameters).norm(dtype=torch.float64)) if loss.requires_grad else 0.
                 for name, loss in losses.items()}
        epsilon = self.config.derivative_epsilon
        if self.derivative_balance is None or self.config.derivative_balance_mode == 'minibatch':
            self.balance_updates += 1
            self.derivative_balance = {n:norms['value']/norms[n] if norms[n] > epsilon else 0.
                                       for n in ('direction', 'magnitude')}
        if not _finite_state(self.derivative_balance):
            raise FloatingPointError('nonfinite derivative loss balance')
        weights = {n:self.derivative_balance[n] if norms[n] > epsilon else 0.
                   for n in ('direction', 'magnitude')}
        return {'raw_norms': norms, 'weights': weights, 'inactive': [n for n in weights if norms[n] <= epsilon],
                'weighted_norms': {'value': norms['value'],
                    'direction': norms['direction']*weights['direction']*self.config.derivative_direction_factor,
                    'magnitude': norms['magnitude']*weights['magnitude']}}

    def _derivative_metrics(self, network, samples):
        if samples is None:
            return None
        predicted = predict_derivatives(network, samples)
        def metrics(mask):
            return {n:derivative_gradient_metrics(predicted[n][mask], samples.gradients[n][mask],
                        self.config.derivative_epsilon) for n in samples.state_groups}
        return {'samples': len(samples.steps), 'state_groups': metrics(slice(None)),
                'boundaries': {str(step):metrics(samples.steps == step)
                               for step in sorted(set(samples.steps.tolist()))}}

    def fit(self, record, derivatives=None, heldout=None):
        inputs = record.inputs.detach().flatten(0, 1)
        returns = record.returns.detach().flatten()
        before = self.prediction_metrics(inputs, returns)
        snapshot = copy.deepcopy(self.state_dict())
        gradient_norms = []
        derivative_losses_log = []
        contributions = []
        try:
            target_before = self._derivative_metrics(self.target, heldout)
            if derivatives is not None:
                if heldout is None or not set(derivatives.scene_ids.tolist()).isdisjoint(heldout.scene_ids.tolist()):
                    raise ValueError('derivative holdout must use disjoint TRAIN scenes')
                self.derivative_sampling = {
                    **derivatives.metadata, 'state_groups': derivatives.state_groups,
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
                        contribution = self._balance_derivative_losses(value_loss, direction, magnitude)
                        contributions.append(contribution)
                        weights = contribution['weights']
                        loss = (value_loss + self.config.derivative_direction_factor*weights['direction']*direction
                                + weights['magnitude']*magnitude)
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
                'derivative_gradient_contributions': contributions,
                'derivative_sampling': copy.deepcopy(self.derivative_sampling),
                'target_derivative_heldout_before': target_before,
                'target_derivative_heldout_after': target_after, 'critic_derivative_heldout_after': critic_after}

    def warmup(self, policy, simulator, initial, horizon, loss_config, record, adjoints,
               derivatives, heldout, *, deadline):
        """Fixed Actor/data; stop at the registered fit/time budget, never alter Adam."""
        held_ids = heldout.scene_ids.unique() if heldout is not None else None
        references = {name:task_gradient_vector(policy, self.target, simulator, initial, horizon,
            self.config.window_steps, loss_config, record, terminal_mode='oracle_full_state',
            adjoints=adjoints, scene_indices=ids) for name,ids in (('all',None),('heldout',held_ids))}
        rounds, fitted = [], {}
        ready = False
        for _ in range(self.config.warmup_max_fits):
            if time.monotonic() >= deadline:
                break
            self.stage = 'critic'
            fitted = self.fit(record, derivatives, heldout)
            self.stage = 'critic_readiness'
            audit = {}
            for name, ids in (('all',None),('heldout',held_ids)):
                audit['target_'+name], _ = audit_actor_gradient(policy, self.target, simulator, initial,
                    horizon, self.config.window_steps, loss_config, record, adjoints,
                    scene_indices=ids, reference=references[name])
            audit['online_all'], _ = audit_actor_gradient(policy, self.critic, simulator, initial,
                horizon, self.config.window_steps, loss_config, record, adjoints, reference=references['all'])
            ready = all(audit[n]['cosine'] is not None and audit[n]['relative_error'] is not None
                and audit[n]['cosine'] >= self.config.ready_min_cosine
                and audit[n]['relative_error'] <= self.config.ready_max_relative_error
                for n in ('target_all','target_heldout'))
            rounds.append({'fit': self.completed_fits, 'audit': audit, 'ready': ready,
                           'value_loss': fitted['critic_loss_after'], 'critic_fit': fitted})
            if ready:
                break
        status = {'ready': ready, 'fits': len(rounds), 'rounds': rounds,
                  'last_audit': rounds[-1]['audit'] if rounds else None,
                  'stop_reason': 'ready' if ready else 'time_budget' if time.monotonic() >= deadline else 'fit_budget',
                  'min_cosine': self.config.ready_min_cosine,
                  'max_relative_error': self.config.ready_max_relative_error,
                  'scope': 'fixed TRAIN Actor/data; derivative holdout still participates in value regression'}
        return fitted, status, references['all']

    def update(self, policy, optimizer, simulator, initial, horizon, loss_config, *,
               gradient_clip, probes=True, critic_only=False):
        clock = time.monotonic()
        self.stage = 'forward'
        record = collect_task_trajectory(policy, simulator, initial, horizon, self.config.window_steps, loss_config)
        metrics = task_metrics(record.trajectory, loss_config)
        forward_seconds = time.monotonic() - clock
        self.stage = 'boundary_adjoints'
        adjoints = collect_boundary_adjoints(policy, simulator, record, horizon, self.config.window_steps, loss_config)
        derivatives, heldout = collect_derivative_samples(policy, simulator, record, horizon,
            loss_config, self.config, adjoints=adjoints)
        derivative_seconds = time.monotonic() - clock - forward_seconds
        self.stage = 'critic'
        reference = None
        if self.config.terminal_mode == 'critic' or critic_only:
            fitted, readiness, reference = self.warmup(policy, simulator, initial, horizon, loss_config,
                record, adjoints, derivatives, heldout, deadline=clock+self.config.warmup_max_seconds)
        else:
            fitted = self.fit(record, derivatives, heldout)
            readiness = {'ready': None, 'fits': 1, 'stop_reason': 'explicit_diagnostic_terminal'}
        fit_seconds = time.monotonic() - clock - forward_seconds - derivative_seconds
        base = {'train_before': metrics, **fitted, 'readiness': readiness,
                'boundary_continuity': adjoints.continuity,
                'forward_seconds': forward_seconds, 'boundary_adjoint_seconds': derivative_seconds,
                'critic_and_readiness_seconds': fit_seconds, 'finite': True}
        if critic_only or readiness['ready'] is False:
            self.stage = 'idle'
            return {**base, 'updated': False,
                    'status': 'critic_only_ready' if readiness['ready'] else 'critic_not_ready',
                    'windows': [], 'update_seconds': time.monotonic()-clock}
        self.stage = 'short_window_backward'
        actor = accumulate_task_gradients(policy, self.target, simulator, initial, horizon,
            self.config.window_steps, loss_config, record, probes=probes,
            terminal_mode=self.config.terminal_mode, adjoints=adjoints)
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
        delta = nn.utils.parameters_to_vector(policy.parameters()).detach() - before
        step_norm = float(delta.norm(dtype=torch.float64))
        dot = None if reference is None else float(torch.dot(reference.double(), delta.double())
                                                  * (horizon//self.config.window_steps))
        self.stage = 'idle'
        return {**base, **actor, 'updated': True, 'status': 'updated',
                'gradient_norm_before_clip': float(norm), 'gradient_norm_after_clip': after_clip,
                'parameter_step_norm': step_norm, 'oracle_gradient_dot_step': dot,
                'backward_seconds': backward_seconds, 'update_seconds': time.monotonic() - clock}

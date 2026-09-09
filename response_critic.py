"""Training-only Monte Carlo risk supervision and short-window physics gradients.

The deployable ResponseMotorPolicy never imports or calls this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import copy
import hashlib
import math

import torch
from torch import nn
from response_proposals import SubspaceConfig
from env_l2f import L2FParams, normalized_capability_target
from response_phase1 import Phase1ProbeError

from response_task import (
    ResponseClosedLoopState, RiskConfig, HardRiskConfig, TaskLossConfig, TaskTrajectory, concatenate, initialize,
    observation, physical_risk_metrics, risk_components, risk_weights, rollout, training_step_costs,
    suffix_risks, trajectory_metrics, hard_risk_metrics,
)


# Fixed task references; dynamics use the simulator's dimensionless capability
# coordinates. No raw CAD-field dictionary or data-dependent calibration.
RISK_OBJECTIVE = "component-mean-risk-v5-capability"


def critic_features(closed: ResponseClosedLoopState, step: int, horizon: int) -> torch.Tensor:
    """Task state, causal response history, six log capabilities and F/(mg).

    This is the registered physical-fit family (linear identical motors, Jx=Jy),
    not a sufficient representation of arbitrary nonlinear/asymmetric vehicles.
    Redundant last_action is omitted; previous_action is the actual command.
    """
    if horizon < 1 or not 0 <= step <= horizon:
        raise ValueError("Critic time must lie in the original positive horizon")
    physical, history = closed.physical, closed.policy
    batch = physical.position.shape[0]
    values = (
        physical.position / 5., physical.velocity / 5., physical.rotation.reshape(batch, 9),
        physical.omega / 10., physical.motor, physical.previous_action,
        history.memory, history.integral / .5,
        history.previous_velocity / 5., history.previous_omega / 10.,
        history.previous_rotation.reshape(batch, 9), history.older_action,
        (history.calls > 0).to(physical.position).reshape(batch, 1),
        normalized_capability_target(physical),
        physical.external_force / (physical.mass[:, None] * L2FParams().gravity),
        physical.position.new_full((batch, 1), step / horizon),
    )
    return torch.cat(values, dim=-1)


class RiskToGoCritic(nn.Module):
    """Predict four signed mean-per-remaining-step risks with a linear head."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def mean_future_risks(suffix_sums: torch.Tensor) -> torch.Tensor:
    """Convert exact cumulative labels at Z_0..Z_H to mean risks; Z_H is zero.

    Keep the original sums in the trajectory record for the unchanged full-flight
    CVaR and detached risk prefix used to construct Actor directions.
    """
    if suffix_sums.ndim != 3 or suffix_sums.shape[0] < 2 or suffix_sums.shape[-1] != 4:
        raise ValueError("risk labels must cover Z_0..Z_H with four components")
    horizon = suffix_sums.shape[0] - 1
    remaining = torch.arange(horizon, -1, -1, device=suffix_sums.device, dtype=suffix_sums.dtype)
    return torch.where(remaining[:, None, None] > 0,
                       suffix_sums / remaining.clamp_min(1)[:, None, None], torch.zeros_like(suffix_sums))


def detach_closed_state(closed: ResponseClosedLoopState) -> ResponseClosedLoopState:
    """Cut the graph while preserving every numerical physical/history value."""
    return ResponseClosedLoopState(*(
        type(state)(**{f.name: getattr(state, f.name).detach() for f in fields(state)})
        for state in (closed.physical, closed.policy)
    ))


@dataclass(frozen=True)
class DirectionSamples:
    """Reachable paired states with mean risks over their remaining continuations."""
    plus_inputs: torch.Tensor
    minus_inputs: torch.Tensor
    plus_returns: torch.Tensor
    minus_returns: torch.Tensor
    perturbations: torch.Tensor


@torch.no_grad()
def motor_perturbation_states(policy, simulator, closed, perturbation):
    """Execute a symmetric feasible command pair and advance all history once.

    Intersect amplitude and Actor slew limits. At a clamped control component
    the symmetric perturbation may be zero; tied continuation labels are ignored.
    """
    obs = observation(closed.physical, closed.policy.integral)
    base = policy(obs, closed.policy).action
    if perturbation.shape != base.shape or not bool(torch.isfinite(perturbation).all()):
        raise ValueError("motor perturbation must be finite and match the command shape")
    radius = policy.config.action_rate * policy.config.dt
    previous = closed.physical.previous_action
    lower, upper = (previous - radius).clamp(-1, 1), (previous + radius).clamp(-1, 1)
    headroom = torch.minimum(base - lower, upper - base).clamp_min(0)
    delta = perturbation.sign() * torch.minimum(perturbation.abs(), headroom)
    states = []
    for executed in (base + delta, base - delta):
        output = policy(obs, closed.policy, applied_action=executed)
        physical = simulator.step(closed.physical, executed, grad_decay=1.)
        states.append(ResponseClosedLoopState(physical, output.next_state))
    return tuple(states)


@torch.no_grad()
def collect_direction_samples(policy, simulator, boundaries, horizon, risk_config, epsilon):
    """A few reachable pairs, labeled only by real no-grad continuation to H."""
    if not boundaries:
        return None
    plus_inputs, minus_inputs, plus_returns, minus_returns, perturbations = [], [], [], [], []
    for step, closed in boundaries:
        if not 0 < step < horizon - 1:
            raise ValueError("direction boundary needs a nonempty remaining continuation")
        base = closed.physical.previous_action
        motor = torch.randint(4, (base.shape[0], 1), device=base.device)
        sign = torch.randint(2, (base.shape[0], 1), device=base.device).to(base) * 2 - 1
        delta = torch.zeros_like(base).scatter(1, motor, sign * epsilon)
        plus, minus = motor_perturbation_states(policy, simulator, closed, delta)
        # Labels at Z_(t+1) exclude the perturbed transition's immediate barrier.
        # Plus/minus share dynamics, disturbances, clock and the pre-action state.
        joined = ResponseClosedLoopState(*(
            type(a)(**{f.name: torch.cat((getattr(a, f.name), getattr(b, f.name))) for f in fields(a)})
            for a, b in ((plus.physical, minus.physical), (plus.policy, minus.policy))
        ))
        future = rollout(policy, simulator, joined, horizon - step - 1)
        risks = torch.stack(tuple(risk_components(future, risk_config).values()), -1).mean(0)
        count = base.shape[0]
        plus_inputs.append(critic_features(plus, step + 1, horizon))
        minus_inputs.append(critic_features(minus, step + 1, horizon))
        plus_returns.append(risks[:count])
        minus_returns.append(risks[count:])
        perturbations.append(delta)
    return DirectionSamples(*(torch.cat(values) for values in (
        plus_inputs, minus_inputs, plus_returns, minus_returns, perturbations)))


def direction_ranking_loss(plus, minus, true_plus, true_minus, *, temperature, min_gap=1.e-6):
    difference = (true_plus - true_minus).detach()
    valid = difference.abs() > min_gap
    losses = nn.functional.softplus(-difference.sign() * (plus - minus) / temperature)
    return (losses * valid).sum() / valid.sum().clamp_min(1)


@dataclass(frozen=True)
class MonteCarloTrajectory:
    """Keep exact suffix sums for Actor CVaR/prefixes; fit converts them to means."""
    trajectory: TaskTrajectory
    inputs: torch.Tensor
    returns: torch.Tensor
    weights: torch.Tensor
    directions: DirectionSamples | None = None


@torch.no_grad()
def collect_trajectory(policy, simulator, initial, horizon: int, loss_config: TaskLossConfig,
                       *, risk_config: RiskConfig = RiskConfig(), window_steps: int = 50,
                       direction_samples: int = 0, direction_epsilon: float = .02) -> MonteCarloTrajectory:
    """Continuous TRAIN flight, exact risk labels and a bounded set of branches.

    CVaR scene weights are selected once using full-flight performance, as before.
    Direction sampling never consults DEV and never changes the main trajectory.
    """
    closed = initialize(policy, initial)
    batch = closed.physical.position.shape[0]
    times = list(range(window_steps, horizon - 1, window_steps))
    choices = {}
    if direction_samples and times:
        selected = torch.randperm(len(times) * batch, device=initial.position.device)[:direction_samples].tolist()
        for index in selected:
            choices.setdefault(times[index // batch], []).append(index % batch)
    boundaries = []
    inputs = [critic_features(closed, 0, horizon)]
    parts = []
    for step in range(horizon):
        part = rollout(policy, simulator, closed, 1)
        closed = part.end
        parts.append(part)
        inputs.append(critic_features(closed, step + 1, horizon))
        if step + 1 in choices:
            indices = choices[step + 1]
            selected_state = ResponseClosedLoopState(*(
                type(state)(**{f.name: getattr(state, f.name)[indices] for f in fields(state)})
                for state in (closed.physical, closed.policy)
            ))
            boundaries.append((step + 1, selected_state))
    trace = concatenate(parts)
    returns = suffix_risks(torch.stack(tuple(risk_components(trace, risk_config).values()), -1))
    directions = collect_direction_samples(policy, simulator, boundaries, horizon, risk_config, direction_epsilon)
    return MonteCarloTrajectory(trace, torch.stack(inputs), returns,
                                risk_weights(training_step_costs(trace, loss_config).sum(0), loss_config), directions)


def accumulate_actor_gradients(
    policy, target, simulator, initial, horizon: int, window_steps: int,
    loss_config: TaskLossConfig, weights: torch.Tensor, *,
    risk_config: RiskConfig = RiskConfig(), risk_weight: float = 1.,
    baseline_returns: torch.Tensor, risk_smoothmax_beta: float = 10.,
    separate_objectives: bool = False,
    phase1_reference: dict | None = None,
) -> dict:
    """Backpropagate each window immediately; update no parameters here.

    The single caller-owned optimizer step comes after every window and every
    pooled scenario. Frozen target parameters still allow derivatives wrt Z.
    """
    if window_steps < 1 or horizon < 1 or horizon % window_steps:
        raise ValueError("the full horizon must be a positive multiple of window steps")
    if any(p.requires_grad for p in target.parameters()):
        raise ValueError("target Critic parameters must be frozen during Actor gradients")
    weights = weights.detach()
    baseline_returns = baseline_returns.detach()
    if baseline_returns.shape != (horizon + 1, weights.numel(), 4):
        raise ValueError("component baseline must cover Z_0 through Z_H and all pooled scenes")
    if not bool(torch.isfinite(baseline_returns).all()) or bool((baseline_returns < 0).any()):
        raise ValueError("component baseline must be finite and nonnegative")
    # Match the gate's separate full-flight CVaR for each component. Freeze
    # every tail for all windows; no window selects a new worst aircraft.
    component_weights = torch.stack([
        risk_weights(baseline_returns[0, :, j], loss_config) for j in range(4)
    ], -1)
    baseline = (component_weights * baseline_returns[0]).sum(0)
    policy.zero_grad(set_to_none=True)
    closed = detach_closed_state(initialize(policy, initial))
    windows = horizon // window_steps
    surrogate = weights.new_zeros(())
    parameters = list(policy.parameters())
    objective_gradients = weights.new_zeros((5, sum(p.numel() for p in parameters))) if separate_objectives else None
    phase1_windows = []
    for start in range(0, horizon, window_steps):
        trace = rollout(policy, simulator, closed, window_steps)
        costs = training_step_costs(trace, loss_config, start=start, horizon=horizon).sum(0)
        end = start + window_steps
        if phase1_reference is not None:
            from response_phase1 import compare_boundary, terminal_state_gradient
            window_probe = {"start": start, "end": end,
                            "boundary": compare_boundary(phase1_reference[end], trace.end, end),
                            **terminal_state_gradient(target, trace.end, end, horizon)}
            window_norms = []
        risk = torch.stack(tuple(risk_components(trace, risk_config).values()), -1).sum(0)
        # Include the detached true prefix so every window uses full-flight
        # units. Only the current window and learned suffix carry gradients.
        risk = risk + (baseline_returns[0] - baseline_returns[start])
        if end < horizon:
            # The network predicts a per-step mean. Restore the full suffix
            # and its state derivative; Z_H never calls the terminal Critic.
            risk = risk + (horizon - end) * target(critic_features(trace.end, end, horizon))
        component_totals = (component_weights * risk).sum(0)
        performance = (weights * costs).sum()
        if separate_objectives:
            objectives = torch.cat((performance.view(1), component_totals)) / windows
            if not bool(torch.isfinite(objectives).all()):
                raise FloatingPointError("nonfinite short-window objective components")
            for j in range(5):
                gradients = torch.autograd.grad(objectives[j], parameters, retain_graph=j < 4, allow_unused=True)
                flat_gradient = torch.cat([
                    torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                    for p, g in zip(parameters, gradients)
                ])
                objective_gradients[j] += flat_gradient
                if phase1_reference is not None:
                    # Report each raw window objective before the 1/windows average.
                    window_norms.append(float((flat_gradient * windows).norm(dtype=torch.float64)))
            if phase1_reference is not None:
                if not all(math.isfinite(value) for value in window_norms):
                    raise Phase1ProbeError("window_gradient_nonfinite", {"start": start, "end": end})
                window_probe.update(performance_gradient_norm=window_norms[0],
                    risk_gradient_norm=math.sqrt(sum(n*n for n in window_norms[1:])),
                    risk_component_gradient_norms=window_norms[1:])
                phase1_windows.append(window_probe)
            loss = objectives.detach().sum()
        else:
            risk_objective = normalized_risk_smoothmax(component_totals, baseline, beta=risk_smoothmax_beta)
            loss = (performance + risk_weight * risk_objective) / windows
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite short-window surrogate")
        if not separate_objectives:
            loss.backward()
        surrogate += loss.detach()
        closed = detach_closed_state(trace.end)
        del trace, costs, risk, loss
    return {"surrogate_loss": float(surrogate), "windows": windows, "end": closed,
            "risk_baseline": baseline.tolist(), "objective_gradients": objective_gradients,
            "phase1_windows": phase1_windows}


def normalized_risk_smoothmax(risks: torch.Tensor, baseline: torch.Tensor, *,
                              beta: float = 10., epsilon: float = 1.e-12) -> torch.Tensor:
    """Smooth maximum of full-flight component ratios; baseline has no gradient."""
    if not math.isfinite(beta) or beta <= 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("smooth-max beta and denominator epsilon must be positive and finite")
    ratios = risks / (baseline.detach() + epsilon)
    return torch.logsumexp(beta * ratios, dim=-1) / beta


@dataclass(frozen=True)
class CriticConfig:
    phase1_probes: bool = False
    acceptance_mode: str = "train-and-dev"
    window_steps: int = 50
    lr: float = 1.0e-3
    epochs: int = 1
    batch_size: int = 1024
    dev_relative_tolerance: float = 0.002
    risk: RiskConfig = RiskConfig()
    risk_weight: float = 1.0
    risk_smoothmax_beta: float = 10.
    hard_risk: HardRiskConfig = HardRiskConfig()
    direction_samples: int = 4
    direction_epsilon: float = .02
    direction_weight: float = .1
    direction_temperature: float = .1
    direction_min_gap: float = 1.e-6
    proposal: str = "physics-subspace"
    subspace: SubspaceConfig = SubspaceConfig()

    def __post_init__(self) -> None:
        if self.acceptance_mode not in ("train-and-dev", "train-objective"):
            raise ValueError("unknown Actor acceptance mode")
        if self.acceptance_mode == "train-objective" and self.proposal != "physics-subspace":
            raise ValueError("TRAIN-only Phase 1 requires physics-subspace search")
        if self.phase1_probes and self.proposal != "physics-subspace":
            raise ValueError("Phase 1 probes require the normal physics-subspace backend")
        if self.proposal not in ("physics-subspace", "smoothmax-adam"):
            raise ValueError("unknown short-window proposal backend")
        positive = (self.risk_weight, self.risk_smoothmax_beta,
                    self.direction_epsilon, self.direction_temperature)
        nonnegative = (self.direction_weight, self.direction_min_gap)
        if any(not math.isfinite(x) or x <= 0 for x in positive) or any(
            not math.isfinite(x) or x < 0 for x in nonnegative
        ) or self.direction_samples < 0 or self.direction_epsilon > 1:
            raise ValueError("invalid risk weight or direction supervision settings")
        if min(self.window_steps, self.epochs, self.batch_size) < 1:
            raise ValueError("Critic/window counts must be positive")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Critic learning rate must be finite and positive")
        if not math.isfinite(self.dev_relative_tolerance) or not 0 <= self.dev_relative_tolerance < 1:
            raise ValueError("DEV tolerance must lie in [0, 1)")


def _finite_state(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_state(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_state(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def hard_risk_deteriorated(before: dict, after: dict, config: HardRiskConfig) -> bool:
    """Only actual danger-zone exposure consumes a hard-risk budget."""
    return any(
        after["hard_risk_components"][name] > before["hard_risk_components"][name] * (
            1 + config.relative_tolerance) + config.absolute_tolerance
        for name in ("omega", "saturation")
    )


def acceptance_rejection(before, after, dev_before, dev_after, *, dev_relative_tolerance,
                         hard_risk_config: HardRiskConfig = HardRiskConfig()):
    """Real performance plus physical danger; soft risk sums do not veto an update."""
    if len(dev_before) != len(dev_after):
        raise ValueError("every DEV candidate needs its matching baseline")
    if not before.get("finite", False) or not _finite_state(before):
        raise FloatingPointError("nonfinite TRAIN baseline")
    if not after.get("finite", False) or not _finite_state(after):
        return "train_nonfinite"
    if after["hard_risk_bounds_violated"]:
        return "train_out_of_bounds"
    if after["task_objective"] >= before["task_objective"]:
        return "train_not_improved"
    if hard_risk_deteriorated(before, after, hard_risk_config):
        return "train_hard_risk_deteriorated"
    for old, new in zip(dev_before, dev_after):
        if not old.get("finite", False) or not _finite_state(old):
            raise FloatingPointError("nonfinite DEV baseline")
        if not new.get("finite", False) or not _finite_state(new):
            return "development_nonfinite"
        if new["hard_risk_bounds_violated"]:
            return "development_out_of_bounds"
        if new["task_objective"] > old["task_objective"] * (1 + dev_relative_tolerance):
            return "development_deteriorated"
        if hard_risk_deteriorated(old, new, hard_risk_config):
            return "development_hard_risk_deteriorated"
    return None


class CriticTrainer:
    """One current-trajectory supervised MLP, its fixed copy and its optimizer."""

    def __init__(self, policy, example: ResponseClosedLoopState, horizon: int,
                 config: CriticConfig = CriticConfig()) -> None:
        if horizon < 1 or horizon % config.window_steps:
            raise ValueError("horizon must be divisible by window_steps")
        self.config = config
        ref = next(policy.parameters())
        self.critic = RiskToGoCritic(critic_features(example, 0, horizon).shape[-1]).to(ref)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr)
        self.completed_fits = 0
        self._dev_cache = None

    def state_dict(self) -> dict:
        return {"objective": RISK_OBJECTIVE, "critic": self.critic.state_dict(), "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(), "completed_fits": self.completed_fits,
                "risk_config": asdict(self.config.risk)}

    def load_state_dict(self, saved: dict) -> None:
        if saved.get("objective") != RISK_OBJECTIVE:
            raise ValueError("risk objective changed; start a new weights-only Actor experiment")
        if saved.get("risk_config") != asdict(self.config.risk):
            raise ValueError("risk definition changed; start a new experiment")
        expected = self.critic.state_dict()
        for name in ("critic", "target"):
            values = saved[name]
            if values.keys() != expected.keys() or any(
                not isinstance(values[key], torch.Tensor) or values[key].shape != value.shape
                for key, value in expected.items()
            ):
                raise ValueError("Critic checkpoint does not match the mean-risk feature schema")
        if not _finite_state(saved):
            raise ValueError("nonfinite Critic training checkpoint")
        self.critic.load_state_dict(saved["critic"])
        self.target.load_state_dict(saved["target"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.completed_fits = saved["completed_fits"]

    def _value_loss(self, prediction, returns, *, reduction="mean"):
        return nn.functional.smooth_l1_loss(prediction, returns, reduction=reduction)

    @torch.no_grad()
    def _component_errors(self, inputs, returns):
        total = returns.new_zeros(4)
        for start in range(0, returns.shape[0], self.config.batch_size):
            end = start + self.config.batch_size
            prediction = self.critic(inputs[start:end])
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError("nonfinite physical-unit Critic prediction")
            total += (prediction - returns[start:end]).abs().sum(0)
        mae = total / returns.shape[0]
        if not bool(torch.isfinite(mae).all()):
            raise FloatingPointError("nonfinite physical-unit Critic error")
        return mae.tolist()

    @torch.no_grad()
    def _regression_loss(self, inputs, returns) -> float:
        total = inputs.new_zeros(())
        for start in range(0, returns.shape[0], self.config.batch_size):
            end = start + self.config.batch_size
            total += self._value_loss(self.critic(inputs[start:end]), returns[start:end], reduction="sum")
        return float(total / returns.numel())

    def _direction_loss(self, samples):
        if samples is None:
            return next(self.critic.parameters()).new_zeros(())
        return direction_ranking_loss(
            self.critic(samples.plus_inputs), self.critic(samples.minus_inputs),
            samples.plus_returns, samples.minus_returns,
            temperature=self.config.direction_temperature, min_gap=self.config.direction_min_gap,
        )

    @torch.no_grad()
    def _direction_metrics(self, samples):
        if samples is None:
            return {"pairs": 0, "valid_pairs": 0, "loss": 0., "accuracy": None}
        difference = samples.plus_returns - samples.minus_returns
        valid = difference.abs() > self.config.direction_min_gap
        predicted = self.critic(samples.plus_inputs) - self.critic(samples.minus_inputs)
        count = int(valid.sum())
        return {"pairs": difference.shape[0], "component_comparisons": difference.numel(), "valid_pairs": count,
                "loss": float(self._direction_loss(samples)),
                "accuracy": float(((predicted * difference > 0) & valid).sum() / count) if count else None,
                "component_valid_pairs": valid.sum(0).tolist(),
                "component_accuracy": [float(((predicted[:, j] * difference[:, j] > 0) & valid[:, j]).sum()
                                             / valid[:, j].sum()) if bool(valid[:, j].any()) else None
                                       for j in range(4)]}

    def fit(self, record: MonteCarloTrajectory) -> dict:
        inputs = record.inputs.detach().reshape(-1, record.inputs.shape[-1])
        returns = mean_future_risks(record.returns.detach()).reshape(-1, 4)
        if not _finite_state((inputs, record.returns, returns)) or (record.directions is not None and not _finite_state(
            [getattr(record.directions, f.name) for f in fields(record.directions)]
        )):
            raise FloatingPointError("nonfinite risk supervision")
        if bool((record.returns < 0).any()):
            raise ValueError("remaining risk labels must be nonnegative")
        before = self._regression_loss(inputs, returns)
        mae_before = self._component_errors(inputs, returns)
        direction_before = self._direction_metrics(record.directions)
        for _ in range(self.config.epochs):
            order = torch.randperm(returns.shape[0], device=returns.device)
            for start in range(0, returns.shape[0], self.config.batch_size):
                indices = order[start:start + self.config.batch_size]
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._value_loss(self.critic(inputs[indices]), returns[indices])
                loss = loss + self.config.direction_weight * self._direction_loss(record.directions)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite Monte Carlo Critic loss")
                loss.backward()
                if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.critic.parameters()):
                    raise FloatingPointError("nonfinite Critic gradient")
                self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        after = self._regression_loss(inputs, returns)
        direction_after = self._direction_metrics(record.directions)
        if not math.isfinite(after) or not _finite_state((self.state_dict(), direction_after)):
            raise FloatingPointError("nonfinite fitted Critic or optimizer")
        mae_after = self._component_errors(inputs, returns)
        if record.directions is not None:
            for sign in ("plus", "minus"):
                self._component_errors(getattr(record.directions, sign + "_inputs"),
                                       getattr(record.directions, sign + "_returns"))
        self.target.load_state_dict(self.critic.state_dict())
        self.completed_fits += 1
        report = {"critic_loss_before": before, "critic_loss_after": after,
                "critic_regression_coordinates": "mean_per_step_risk",
                "critic_component_mae_before": mae_before, "critic_component_mae_after": mae_after,
                "critic_direction_before": direction_before, "critic_direction_after": direction_after}
        if self.config.phase1_probes:
            from response_phase1 import critic_ranges
            report.update(critic_ranges(self.critic, inputs, returns, record.returns, self.config.batch_size))
        return report

    def _development_key(self, policy, simulator, initials, horizon, loss_config):
        from response_training import model_hash
        digest = hashlib.sha256(model_hash(policy).encode())
        digest.update(repr((asdict(policy.config), asdict(simulator.params), horizon,
                            asdict(loss_config), asdict(self.config.risk), asdict(self.config.hard_risk))).encode())
        for initial in initials:
            for field in fields(initial):
                original = getattr(initial, field.name)
                value = original.detach().cpu().contiguous()
                digest.update(repr((field.name, value.shape, value.dtype, original.device)).encode())
                digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def _metrics(self, trace, loss_config, *, candidate=False):
        row = {**trajectory_metrics(trace, loss_config),
               **physical_risk_metrics(trace, self.config.risk, loss_config),
               **hard_risk_metrics(trace, self.config.risk, loss_config, self.config.hard_risk)}
        if not row["finite"] or not _finite_state(row):
            if self.config.phase1_probes:
                raise Phase1ProbeError("candidate_nonfinite" if candidate else "baseline_nonfinite", "continuous trajectory metrics")
            if candidate:
                return {"finite": False}
            raise FloatingPointError("nonfinite continuous trajectory metrics")
        if self.config.phase1_probes:
            from response_phase1 import task_loss_components
            from response_task import training_loss_config
            row["task_loss_components"] = task_loss_components(trace, loss_config)
            row["training_loss_components"] = task_loss_components(trace, training_loss_config(loss_config))
        return row

    @torch.no_grad()
    def development_baseline(self, policy, simulator, initials, horizon, loss_config):
        """Cache only identical Actor, physical configuration and fixed DEV states."""
        key = self._development_key(policy, simulator, initials, horizon, loss_config)
        if self._dev_cache is not None and self._dev_cache[0] == key:
            return copy.deepcopy(self._dev_cache[1]), True
        rows = [self._metrics(rollout(policy, simulator, state, horizon), loss_config) for state in initials]
        self._dev_cache = (key, copy.deepcopy(rows))
        return rows, False

    def _propose_actor(self, policy, optimizer, gradients, train_before, evaluate_train,
                       gradient_clip, evidence):
        """Pick the lowest real TRAIN cost allowed by the configured acceptance mode."""
        from response_proposals import assign_parameters, search_candidates

        base = torch.nn.utils.parameters_to_vector(policy.parameters()).detach().clone()
        def accept_candidate(after):
            if self.config.acceptance_mode == "train-objective":
                if not after.get("finite", False) or not _finite_state(after):
                    return "train_nonfinite"
                return None if after["task_objective"] < train_before["task_objective"] else "train_not_improved"
            return acceptance_rejection(train_before, after, (), (),
                dev_relative_tolerance=self.config.dev_relative_tolerance,
                hard_risk_config=self.config.hard_risk)

        if self.config.proposal == "physics-subspace":
            rows = gradients["objective_gradients"]
            norm = rows.norm(dtype=torch.float64)
            if not bool(torch.isfinite(rows).all()) or not bool(torch.isfinite(norm)):
                raise FloatingPointError("nonfinite Actor objective gradient")
            evidence["gradient_norm"] = float(norm)
            selected = search_candidates(policy, rows, evaluate_train, self.config.subspace,
                                         baseline=train_before, accept_candidate=accept_candidate)
            evidence["search"] = selected.evidence
            if selected.parameters is None:
                evidence["rejection_reason"] = selected.reason
                return None
            assign_parameters(policy, selected.parameters)
            after = selected.metrics
        else:
            if optimizer is None:
                raise ValueError("legacy smooth-max proposals require an Actor optimizer")
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), gradient_clip)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("nonfinite Actor gradient")
            evidence["gradient_norm"] = float(norm)
            optimizer.step()
            if not _finite_state(policy.state_dict()) or not _finite_state(optimizer.state_dict()):
                raise FloatingPointError("nonfinite proposed network or optimizer")
            after = evaluate_train()
        reason = accept_candidate(after)
        evidence["rejection_reason"] = reason
        if after.get("finite", False):
            evidence.update(continuous_loss_after=after["task_objective"],
                continuous_risk_after=after["risk_objective"],
                continuous_risk_components_after=after["risk_components"],
                continuous_hard_risk_after=after["hard_risk_components"],
                continuous_bounds_after=after["hard_risk_bounds_violated"])
        if reason is not None:
            return None
        evidence["parameter_step_norm"] = float((
            torch.nn.utils.parameters_to_vector(policy.parameters()).detach() - base).norm())
        return after

    def guarded_step(
        self, policy, optimizer, simulator, initial, horizon: int,
        loss_config: TaskLossConfig, *, development_initials: tuple,
        gradient_clip: float,
    ) -> dict:
        """Fit Critic independently, then apply configured TRAIN-only or TRAIN/DEV acceptance.

        Failed Actor proposals restore Actor/optimizer and the post-fit RNG.
        Finite completed Critic fits and their learned weights always survive.
        TRAIN rejection skips DEV; DEV never selects a correction or step size.
        """
        from response_training import capture_rng, restore_rng

        train_only = self.config.acceptance_mode == "train-objective"
        if not development_initials and not train_only:
            raise ValueError("a fixed independent DEV bank is required")
        if train_only and development_initials:
            raise ValueError("TRAIN-only Phase 1 must not receive DEV banks")
        before_actor = copy.deepcopy(policy.state_dict())
        before_optimizer = None if optimizer is None else copy.deepcopy(optimizer.state_dict())
        before_critic = copy.deepcopy(self.state_dict())
        rollback_rng = capture_rng()
        critic_committed = False
        evidence = {"accepted": False, "proposal_finite": True, "proposal_backend": self.config.proposal,
                    "rejection_reason": None, "continuous_loss_after": None,
                    "development_evaluated": False, "acceptance_mode": self.config.acceptance_mode}
        stage = "forward"

        @torch.no_grad()
        def evaluate_train():
            return self._metrics(rollout(policy, simulator, initial, horizon), loss_config, candidate=True)

        try:
            record = collect_trajectory(
                policy, simulator, initial, horizon, loss_config, risk_config=self.config.risk,
                window_steps=self.config.window_steps, direction_samples=self.config.direction_samples,
                direction_epsilon=self.config.direction_epsilon,
            )
            metrics = self._metrics(record.trajectory, loss_config)
            if not _finite_state((record.inputs, record.returns)):
                raise FloatingPointError("nonfinite continuous TRAIN baseline")
            evidence.update(task_loss=metrics["task_objective"], continuous_loss_before=metrics["task_objective"],
                continuous_risk_before=metrics["risk_objective"], continuous_risk_components_before=metrics["risk_components"],
                continuous_hard_risk_before=metrics["hard_risk_components"],
                continuous_bounds_before=metrics["hard_risk_bounds_violated"],
                **{key: metrics[key] for key in ("position_rms", "velocity_rms", "omega_rms",
                                                "steady_success_rate", "motor_saturation_fraction")})
            reference = None
            if self.config.phase1_probes:
                from response_phase1 import boundary_reference
                evidence["phase1_loss_components"] = {
                    "evaluation": metrics["task_loss_components"],
                    "training": metrics["training_loss_components"],
                }
                reference = boundary_reference(policy, simulator, initial, horizon, self.config.window_steps)
            stage = "critic"
            evidence.update(self.fit(record))
            rollback_rng = capture_rng()
            critic_committed = True
            weights, baseline_returns = record.weights, record.returns
            del record
            stage = "short_window_backward"
            gradients = accumulate_actor_gradients(
                policy, self.target, simulator, initial, horizon, self.config.window_steps,
                loss_config, weights, risk_config=self.config.risk, risk_weight=self.config.risk_weight,
                baseline_returns=baseline_returns, risk_smoothmax_beta=self.config.risk_smoothmax_beta,
                separate_objectives=self.config.proposal == "physics-subspace",
                phase1_reference=reference,
            )
            evidence.update({key: gradients[key] for key in ("surrogate_loss", "windows", "risk_baseline")})
            if self.config.phase1_probes:
                evidence["phase1_windows"] = gradients["phase1_windows"]
            stage = "candidate_search"
            after = self._propose_actor(policy, optimizer, gradients, metrics, evaluate_train, gradient_clip, evidence)
            del gradients
            if train_only:
                evidence["accepted"] = after is not None
                retained = after if evidence["accepted"] else metrics
                evidence["phase1_retained"] = {key: retained[key] for key in (
                    "task_objective", "position_rms", "velocity_rms", "omega_rms", "steady_success_rate",
                    "risk_objective", "risk_components", "hard_risk_components", "motor_saturation_fraction",
                )}
                if self.config.phase1_probes:
                    evidence["phase1_retained"].update({key: retained[key] for key in (
                        "task_loss_components", "training_loss_components")})
            elif after is not None:
                stage = "development"
                candidate = copy.deepcopy(policy.state_dict())
                # The baseline belongs to the old Actor, never the candidate.
                policy.load_state_dict(before_actor)
                try:
                    dev_before, hit = self.development_baseline(policy, simulator, development_initials, horizon, loss_config)
                finally:
                    policy.load_state_dict(candidate)
                with torch.no_grad():
                    dev_after = [self._metrics(rollout(policy, simulator, state, horizon), loss_config, candidate=True)
                                 for state in development_initials]
                evidence.update(development_evaluated=True, development_baseline_cache_hit=hit,
                    development_loss_before=[row["task_objective"] for row in dev_before],
                    development_risk_before=[row["risk_objective"] for row in dev_before],
                    development_risk_components_before=[row["risk_components"] for row in dev_before],
                    development_loss_after=[row.get("task_objective") for row in dev_after],
                    development_risk_after=[row.get("risk_objective") for row in dev_after],
                    development_risk_components_after=[row.get("risk_components") for row in dev_after],
                    development_hard_risk_before=[row["hard_risk_components"] for row in dev_before],
                    development_hard_risk_after=[row.get("hard_risk_components") for row in dev_after],
                    development_bounds_after=[row.get("hard_risk_bounds_violated") for row in dev_after])
                evidence["rejection_reason"] = acceptance_rejection(
                    metrics, after, dev_before, dev_after, dev_relative_tolerance=self.config.dev_relative_tolerance,
                    hard_risk_config=self.config.hard_risk)
                evidence["accepted"] = evidence["rejection_reason"] is None
                if evidence["accepted"]:
                    self._dev_cache = (self._development_key(policy, simulator, development_initials, horizon, loss_config),
                                       copy.deepcopy(dev_after))
        except Phase1ProbeError as error:
            evidence.update(proposal_finite=False, rejection_reason=str(error),
                            phase1_failure={"stage": error.stage, "detail": error.detail})
        except FloatingPointError as error:
            evidence.update(proposal_finite=False, rejection_reason=str(error), failure_stage=stage)
        finally:
            if not evidence["accepted"]:
                policy.load_state_dict(before_actor)
                if optimizer is not None:
                    optimizer.load_state_dict(before_optimizer)
                if not critic_committed:
                    self.load_state_dict(before_critic)
                restore_rng(rollback_rng)
            policy.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)
        evidence["numerics_finite"] = evidence["proposal_finite"]
        evidence["critic_update_retained"] = critic_committed
        evidence["critic_fits"] = self.completed_fits
        if self.config.phase1_probes:
            from response_phase1 import proposal_summary
            evidence["phase1_proposal"] = proposal_summary(evidence)
        return evidence

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

from response_task import (
    ResponseClosedLoopState, RiskConfig, TaskLossConfig, TaskTrajectory, concatenate, initialize,
    observation, physical_risk_metrics, risk_components, risk_weights, rollout, step_costs,
    suffix_risks, trajectory_metrics,
)


# Fixed physical divisors; no batch statistics, clipping, or running RMS.
# Every truth/history field has an explicit scale so additions fail visibly.
PHYSICAL_SCALES = {
    "position": 5., "velocity": 5., "rotation": 1., "omega": 10.,
    "motor": 1., "previous_action": 1., "external_force": .5, "mass": .05,
    "thrust_coeff_c0": .1, "thrust_coeff_c1": .1, "thrust_coeff_c2": .1,
    "thrust_to_weight": 3., "torque_to_inertia": 500., "rotor_distance_factor": 1.,
    "inertia_factor": 1., "motor_time_rising": .1, "motor_time_falling": .1,
    "rotor_torque_constant": .01, "cbrt_mass": .4, "force_std": .5,
    "arm_length": .05, "inertia_x": 1.e-5, "inertia_y": 1.e-5, "inertia_z": 2.e-5,
    "alpha_roll_max": 500., "alpha_pitch_max": 500., "alpha_yaw_max": 100.,
    "eta_yaw": .2, "jz_over_jxy": 2., "dt_alpha_roll_max": 5., "dt_alpha_yaw_max": 1.,
}
POLICY_SCALES = {
    "memory": 1., "integral": .5, "previous_velocity": 5., "previous_omega": 10.,
    "previous_rotation": 1., "last_action": 1., "older_action": 1.,
}
RISK_OBJECTIVE = "component-risk-v3-warning-saturation-train-scaled"


def critic_features(closed: ResponseClosedLoopState, step: int, horizon: int) -> torch.Tensor:
    """Complete privileged state normalized by fixed physical divisors, plus t/H."""
    if horizon < 1 or not 0 <= step <= horizon:
        raise ValueError("Critic time must lie in the original positive horizon")
    batch = closed.physical.position.shape[0]
    values = [(getattr(closed.physical, f.name) / PHYSICAL_SCALES[f.name]).reshape(batch, -1)
              for f in fields(closed.physical)]
    values.extend((getattr(closed.policy, f.name) / (horizon if f.name == "calls"
                   else POLICY_SCALES[f.name])).reshape(batch, -1) for f in fields(closed.policy))
    values.append(closed.physical.position.new_full((batch, 1), step / horizon))
    return torch.cat(values, dim=-1)


class RiskToGoCritic(nn.Module):
    """One 256/256 SiLU MLP with four nonnegative remaining-risk components."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 4), nn.Softplus(),
        )

        # Set once from TRAIN labels. The MLP learns normalized targets while
        # callers receive physical remaining-risk units, including dR/dZ.
        self.register_buffer("output_scales", torch.ones(4))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs) * self.output_scales


def detach_closed_state(closed: ResponseClosedLoopState) -> ResponseClosedLoopState:
    """Cut the graph while preserving every numerical physical/history value."""
    return ResponseClosedLoopState(*(
        type(state)(**{f.name: getattr(state, f.name).detach() for f in fields(state)})
        for state in (closed.physical, closed.policy)
    ))


@dataclass(frozen=True)
class DirectionSamples:
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
        risks = torch.stack(tuple(risk_components(future, risk_config).values()), -1).sum(0)
        count = base.shape[0]
        plus_inputs.append(critic_features(plus, step + 1, horizon))
        minus_inputs.append(critic_features(minus, step + 1, horizon))
        plus_returns.append(risks[:count])
        minus_returns.append(risks[count:])
        perturbations.append(delta)
    return DirectionSamples(*(torch.cat(values) for values in (
        plus_inputs, minus_inputs, plus_returns, minus_returns, perturbations)))


def direction_ranking_loss(plus, minus, true_plus, true_minus, *, temperature, min_gap=1.e-6,
                           scales=None):
    scales = 1. if scales is None else scales.detach()
    difference = (true_plus - true_minus).detach() / scales
    valid = difference.abs() > min_gap
    losses = nn.functional.softplus(-difference.sign() * (plus - minus) / scales / temperature)
    return (losses * valid).sum() / valid.sum().clamp_min(1)


@dataclass(frozen=True)
class MonteCarloTrajectory:
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
                                risk_weights(step_costs(trace, loss_config).sum(0), loss_config), directions)


def accumulate_actor_gradients(
    policy, target, simulator, initial, horizon: int, window_steps: int,
    loss_config: TaskLossConfig, weights: torch.Tensor, *,
    risk_config: RiskConfig = RiskConfig(), risk_weight: float = 1.,
    baseline_returns: torch.Tensor, risk_smoothmax_beta: float = 10.,
    separate_objectives: bool = False,
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
    for start in range(0, horizon, window_steps):
        trace = rollout(policy, simulator, closed, window_steps)
        costs = step_costs(trace, loss_config, start=start, horizon=horizon).sum(0)
        end = start + window_steps
        risk = torch.stack(tuple(risk_components(trace, risk_config).values()), -1).sum(0)
        # Include the detached true prefix so every window uses full-flight
        # units. Only the current window and learned suffix carry gradients.
        risk = risk + (baseline_returns[0] - baseline_returns[start])
        if end < horizon:
            risk = risk + target(critic_features(trace.end, end, horizon))
        component_totals = (component_weights * risk).sum(0)
        performance = (weights * costs).sum()
        if separate_objectives:
            objectives = torch.cat((performance.view(1), component_totals)) / windows
            if not bool(torch.isfinite(objectives).all()):
                raise FloatingPointError("nonfinite short-window objective components")
            for j in range(5):
                gradients = torch.autograd.grad(objectives[j], parameters, retain_graph=j < 4, allow_unused=True)
                objective_gradients[j] += torch.cat([
                    torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                    for p, g in zip(parameters, gradients)
                ])
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
            "risk_baseline": baseline.tolist(), "objective_gradients": objective_gradients}


def normalized_risk_smoothmax(risks: torch.Tensor, baseline: torch.Tensor, *,
                              beta: float = 10., epsilon: float = 1.e-12) -> torch.Tensor:
    """Smooth maximum of full-flight component ratios; baseline has no gradient."""
    if not math.isfinite(beta) or beta <= 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("smooth-max beta and denominator epsilon must be positive and finite")
    ratios = risks / (baseline.detach() + epsilon)
    return torch.logsumexp(beta * ratios, dim=-1) / beta


@dataclass(frozen=True)
class CriticConfig:
    window_steps: int = 50
    lr: float = 1.0e-3
    epochs: int = 1
    batch_size: int = 1024
    dev_relative_tolerance: float = 0.002
    risk: RiskConfig = RiskConfig()
    risk_weight: float = 1.0
    risk_smoothmax_beta: float = 10.
    risk_relative_tolerance: float = 0.
    direction_samples: int = 4
    direction_epsilon: float = .02
    direction_weight: float = .1
    direction_temperature: float = .1
    direction_min_gap: float = 1.e-6
    return_scale_floor: float = 1.e-3
    proposal: str = "physics-subspace"
    subspace: SubspaceConfig = SubspaceConfig()

    def __post_init__(self) -> None:
        if self.proposal not in ("physics-subspace", "smoothmax-adam"):
            raise ValueError("unknown short-window proposal backend")
        positive = (self.risk_weight, self.risk_smoothmax_beta, self.return_scale_floor,
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
        if not math.isfinite(self.risk_relative_tolerance) or not 0 <= self.risk_relative_tolerance < 1:
            raise ValueError("risk relative tolerance must lie in [0, 1)")


def _finite_state(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_state(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_state(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def risk_deteriorated(before: dict, after: dict, *, relative_tolerance: float = 0.) -> bool:
    """No physical component may buy improvement by worsening another."""
    return after["risk_objective"] > before["risk_objective"] * (1 + relative_tolerance) or any(
        after["risk_components"][name] > value * (1 + relative_tolerance)
        for name, value in before["risk_components"].items()
    )


def acceptance_rejection(before, after, dev_before, dev_after, *, dev_relative_tolerance,
                         risk_relative_tolerance=0.):
    """Use real continuous trajectory measurements only, never Critic outputs."""
    if after["task_objective"] >= before["task_objective"]:
        return "train_not_improved"
    if risk_deteriorated(before, after, relative_tolerance=risk_relative_tolerance):
        return "train_risk_deteriorated"
    if any(new["task_objective"] > old["task_objective"] * (1 + dev_relative_tolerance)
           for old, new in zip(dev_before, dev_after)):
        return "development_deteriorated"
    if any(risk_deteriorated(old, new, relative_tolerance=risk_relative_tolerance)
           for old, new in zip(dev_before, dev_after)):
        return "development_risk_deteriorated"
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
        self.scales_calibrated = False
        self._dev_cache = None

    def state_dict(self) -> dict:
        return {"objective": RISK_OBJECTIVE, "critic": self.critic.state_dict(), "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(), "completed_fits": self.completed_fits,
                "risk_config": asdict(self.config.risk), "scales_calibrated": self.scales_calibrated,
                "return_scale_floor": self.config.return_scale_floor}

    def load_state_dict(self, saved: dict) -> None:
        if saved.get("objective") != RISK_OBJECTIVE:
            raise ValueError("risk objective changed; start a new weights-only Actor experiment")
        if saved.get("risk_config") != asdict(self.config.risk) or saved.get("return_scale_floor") != self.config.return_scale_floor:
            raise ValueError("risk definition/scales changed; start a new experiment")
        for name in ("critic", "target"):
            scales = saved[name]["output_scales"]
            if scales.shape != (4,) or not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
                raise ValueError("Critic checkpoint needs four positive finite TRAIN scales")
        if not torch.equal(saved['critic']['output_scales'], saved['target']['output_scales']):
            raise ValueError("Critic and target must use identical TRAIN units")
        self.critic.load_state_dict(saved["critic"])
        self.target.load_state_dict(saved["target"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.completed_fits = saved["completed_fits"]
        self.scales_calibrated = saved["scales_calibrated"]

    def initialize_from_scalar(self, saved: dict, proportions: torch.Tensor) -> None:
        """Reject old risk units; reuse only Actor weights in a new experiment."""
        raise ValueError("risk definition and supervision changed; start a new weights-only Actor experiment")

    def _value_loss(self, prediction, returns, *, reduction="mean"):
        scales = self.critic.output_scales.detach()
        return nn.functional.smooth_l1_loss(prediction / scales, returns / scales, reduction=reduction)

    @torch.no_grad()
    def _component_errors(self, inputs, returns):
        total = returns.new_zeros(4)
        for start in range(0, returns.shape[0], self.config.batch_size):
            end = start + self.config.batch_size
            total += (self.critic(inputs[start:end]) - returns[start:end]).abs().sum(0)
        mae = total / returns.shape[0]
        return mae.tolist(), (mae / self.critic.output_scales).tolist()

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
            scales=self.critic.output_scales,
        )

    @torch.no_grad()
    def _direction_metrics(self, samples):
        if samples is None:
            return {"pairs": 0, "valid_pairs": 0, "loss": 0., "accuracy": None}
        difference = (samples.plus_returns - samples.minus_returns) / self.critic.output_scales
        valid = difference.abs() > self.config.direction_min_gap
        predicted = (self.critic(samples.plus_inputs) - self.critic(samples.minus_inputs)) / self.critic.output_scales
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
        returns = record.returns.detach().reshape(-1, 4)
        if not _finite_state((inputs, returns)) or (record.directions is not None and not _finite_state(
            [getattr(record.directions, f.name) for f in fields(record.directions)]
        )):
            raise FloatingPointError("nonfinite risk supervision")
        if bool((returns < 0).any()):
            raise ValueError("remaining risk labels must be nonnegative")
        if not self.scales_calibrated:
            # Only fit(TRAIN) may set units. Never update them on later fits or DEV.
            with torch.no_grad():
                self.critic.output_scales.copy_(returns.mean(0).clamp_min(self.config.return_scale_floor))
                self.target.output_scales.copy_(self.critic.output_scales)
            self.scales_calibrated = True
        before = self._regression_loss(inputs, returns)
        mae_before, normalized_before = self._component_errors(inputs, returns)
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
        self.target.load_state_dict(self.critic.state_dict())
        after = self._regression_loss(inputs, returns)
        direction_after = self._direction_metrics(record.directions)
        if not math.isfinite(after) or not _finite_state((self.state_dict(), direction_after)):
            raise FloatingPointError("nonfinite fitted Critic or optimizer")
        self.completed_fits += 1
        mae_after, normalized_after = self._component_errors(inputs, returns)
        return {"critic_loss_before": before, "critic_loss_after": after,
                "critic_scale_source": "first_train_trajectory",
                "critic_component_scales": self.critic.output_scales.tolist(),
                "critic_component_mae_before": mae_before, "critic_component_mae_after": mae_after,
                "critic_component_normalized_mae_before": normalized_before,
                "critic_component_normalized_mae_after": normalized_after,
                "critic_direction_before": direction_before, "critic_direction_after": direction_after}

    def _development_key(self, policy, simulator, initials, horizon, loss_config):
        from response_training import model_hash
        digest = hashlib.sha256(model_hash(policy).encode())
        digest.update(repr((asdict(policy.config), asdict(simulator.params), horizon,
                            asdict(loss_config), asdict(self.config.risk))).encode())
        for initial in initials:
            for field in fields(initial):
                original = getattr(initial, field.name)
                value = original.detach().cpu().contiguous()
                digest.update(repr((field.name, value.shape, value.dtype, original.device)).encode())
                digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def _metrics(self, trace, loss_config):
        row = {**trajectory_metrics(trace, loss_config),
               **physical_risk_metrics(trace, self.config.risk, loss_config)}
        if not row["finite"] or not _finite_state(row):
            raise FloatingPointError("nonfinite continuous trajectory metrics")
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
        """Bounded TRAIN search; DEV is deliberately unavailable at this layer."""
        from response_proposals import assign_parameters, correct_direction

        direct = self.config.proposal == "physics-subspace"
        base = torch.nn.utils.parameters_to_vector(policy.parameters()).detach().clone()
        if direct:
            rows = gradients["objective_gradients"]
            if not bool(torch.isfinite(rows).all()):
                raise FloatingPointError("nonfinite Actor objective gradient")
            norm = rows.norm(dtype=torch.float64)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("nonfinite Actor gradient norm")
            evidence["gradient_norm"] = float(norm)
            def vector(metrics):
                # The total CVaR gate is an additional constraint row, even
                # though the gradient span has at most five directions.
                return torch.tensor([metrics["task_objective"], *metrics["risk_components"].values(),
                                     metrics["risk_objective"]], dtype=torch.float64)
            corrected = correct_direction(policy, rows, lambda: vector(evaluate_train()),
                                          self.config.subspace, baseline=vector(train_before))
            evidence["correction"] = corrected.evidence
            if corrected.direction is None:
                evidence.update(rejection_reason=corrected.reason)
                return None
            step = corrected.evidence["parameter_step"] * corrected.direction
            attempts = self.config.subspace.backtracks
        else:
            if optimizer is None:
                raise ValueError("legacy smooth-max proposals require an Actor optimizer")
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), gradient_clip)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("nonfinite Actor gradient")
            evidence["gradient_norm"] = float(norm)
            optimizer.step()
            attempts = 1
        evidence["train_line_search"] = []
        for index in range(attempts):
            if direct:
                assign_parameters(policy, base + (0.5 ** index) * step)
            if not _finite_state(policy.state_dict()) or (not direct and not _finite_state(optimizer.state_dict())):
                raise FloatingPointError("nonfinite proposed network or optimizer")
            after = evaluate_train()
            reason = acceptance_rejection(train_before, after, (), (),
                dev_relative_tolerance=self.config.dev_relative_tolerance,
                risk_relative_tolerance=self.config.risk_relative_tolerance)
            evidence["train_line_search"].append({"fraction": 0.5 ** index, "rejection_reason": reason,
                "performance": after["task_objective"], "risk_components": after["risk_components"],
                "risk_objective": after["risk_objective"]})
            evidence.update(continuous_loss_after=after["task_objective"],
                            continuous_risk_after=after["risk_objective"],
                            continuous_risk_components_after=after["risk_components"], rejection_reason=reason)
            if reason is None:
                evidence["parameter_step_norm"] = float((torch.nn.utils.parameters_to_vector(policy.parameters()).detach() - base).norm())
                return after
        if direct:
            evidence.update(rejection_reason="line_search_exhausted")
        return None

    def guarded_step(
        self, policy, optimizer, simulator, initial, horizon: int,
        loss_config: TaskLossConfig, *, development_initials: tuple,
        gradient_clip: float,
    ) -> dict:
        """Fit Critic independently, then accept a true TRAIN/DEV Actor proposal.

        Failed Actor proposals restore Actor/optimizer and the post-fit RNG.
        Finite completed Critic fits and their calibrated units always survive.
        TRAIN rejection skips DEV; DEV never selects a correction or step size.
        """
        from response_training import capture_rng, restore_rng

        if not development_initials:
            raise ValueError("a fixed independent DEV bank is required")
        before_actor = copy.deepcopy(policy.state_dict())
        before_optimizer = None if optimizer is None else copy.deepcopy(optimizer.state_dict())
        before_critic = copy.deepcopy(self.state_dict())
        rollback_rng = capture_rng()
        critic_committed = False
        evidence = {"accepted": False, "proposal_finite": True, "proposal_backend": self.config.proposal,
                    "rejection_reason": None, "continuous_loss_after": None,
                    "development_evaluated": False}

        @torch.no_grad()
        def evaluate_train():
            return self._metrics(rollout(policy, simulator, initial, horizon), loss_config)

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
                **{key: metrics[key] for key in ("position_rms", "velocity_rms", "omega_rms",
                                                "steady_success_rate", "motor_saturation_fraction")})
            evidence.update(self.fit(record))
            rollback_rng = capture_rng()
            critic_committed = True
            weights, baseline_returns = record.weights, record.returns
            del record
            gradients = accumulate_actor_gradients(
                policy, self.target, simulator, initial, horizon, self.config.window_steps,
                loss_config, weights, risk_config=self.config.risk, risk_weight=self.config.risk_weight,
                baseline_returns=baseline_returns, risk_smoothmax_beta=self.config.risk_smoothmax_beta,
                separate_objectives=self.config.proposal == "physics-subspace",
            )
            evidence.update({key: gradients[key] for key in ("surrogate_loss", "windows", "risk_baseline")})
            after = self._propose_actor(policy, optimizer, gradients, metrics, evaluate_train, gradient_clip, evidence)
            del gradients
            if after is not None:
                candidate = copy.deepcopy(policy.state_dict())
                # The baseline belongs to the old Actor, never the candidate.
                policy.load_state_dict(before_actor)
                try:
                    dev_before, hit = self.development_baseline(policy, simulator, development_initials, horizon, loss_config)
                finally:
                    policy.load_state_dict(candidate)
                with torch.no_grad():
                    dev_after = [self._metrics(rollout(policy, simulator, state, horizon), loss_config)
                                 for state in development_initials]
                evidence.update(development_evaluated=True, development_baseline_cache_hit=hit,
                    development_loss_before=[row["task_objective"] for row in dev_before],
                    development_risk_before=[row["risk_objective"] for row in dev_before],
                    development_risk_components_before=[row["risk_components"] for row in dev_before],
                    development_loss_after=[row["task_objective"] for row in dev_after],
                    development_risk_after=[row["risk_objective"] for row in dev_after],
                    development_risk_components_after=[row["risk_components"] for row in dev_after])
                evidence["rejection_reason"] = acceptance_rejection(
                    metrics, after, dev_before, dev_after, dev_relative_tolerance=self.config.dev_relative_tolerance,
                    risk_relative_tolerance=self.config.risk_relative_tolerance)
                evidence["accepted"] = evidence["rejection_reason"] is None
                if evidence["accepted"]:
                    self._dev_cache = (self._development_key(policy, simulator, development_initials, horizon, loss_config),
                                       copy.deepcopy(dev_after))
        except FloatingPointError as error:
            evidence.update(proposal_finite=False, rejection_reason=str(error))
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
        return evidence

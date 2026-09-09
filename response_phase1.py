"""Read-only Phase 1 diagnostics; only continuity/nonfinite failures stop a run."""
from __future__ import annotations

from dataclasses import fields
import math

import torch

from response_task import ResponseClosedLoopState, risk_weights, rollout, weighted_task_features


class Phase1ProbeError(RuntimeError):
    """Must escape candidate search's recoverable FloatingPointError handling."""

    def __init__(self, stage: str, detail):
        self.stage, self.detail = stage, detail
        super().__init__(f"Phase 1 {stage}: {detail}")


@torch.no_grad()
def boundary_reference(policy, simulator, initial, horizon: int, window_steps: int) -> dict:
    """One full no-grad call, snapshotting every complete boundary state."""
    snapshots = {}
    def observe(step, closed):
        if step % window_steps == 0:
            snapshots[step] = ResponseClosedLoopState(*(
                type(state)(**{f.name: getattr(state, f.name).detach().clone() for f in fields(state)})
                for state in (closed.physical, closed.policy)))
    rollout(policy, simulator, initial, horizon, boundary_observer=observe)
    return snapshots


@torch.no_grad()
def compare_boundary(reference, actual, step: int) -> dict:
    dtype = actual.physical.position.dtype
    atol, rtol = (1.e-6, 1.e-5) if dtype == torch.float32 else (1.e-12, 1.e-10)
    report = {"step": step, "atol": atol, "rtol": rtol, "passed": True, "exact": True, "fields": {}}
    for group in ("physical", "policy"):
        expected, observed = getattr(reference, group), getattr(actual, group)
        for f in fields(expected):
            a, b = getattr(expected, f.name), getattr(observed, f.name).detach()
            name = group + "." + f.name
            if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
                raise Phase1ProbeError("boundary_nonfinite", {"step": step, "field": name})
            error = float((a - b).abs().max())
            ok = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
            report["fields"][name] = error
            report["passed"] &= ok
            report["exact"] &= torch.equal(a, b)
            if not ok:
                raise Phase1ProbeError("boundary_continuity", {"step": step, "field": name, "max_error": error})
    return report


@torch.no_grad()
def task_loss_components(trace, config) -> dict:
    """Additive attribution with ONE common CVaR tail from total scene costs."""
    squared = weighted_task_features(trace, config).square()
    costs = squared.sum((0, 2))
    weights = risk_weights(costs, config)
    groups = {"position": slice(0, 3), "velocity": slice(3, 6),
              "omega": slice(6, 9), "regularization": slice(9, None)}
    result = {name: float((squared[:, :, subset].sum((0, 2)) * weights).sum())
              for name, subset in groups.items()}
    result.update(total=float((costs * weights).sum()), mean=float(costs.mean()))
    result['cvar_addition'] = result['total'] - result['mean']
    if not all(math.isfinite(value) for value in result.values()):
        raise Phase1ProbeError('loss_nonfinite', result)
    return result


def terminal_state_gradient(target, closed, step: int, horizon: int, *, mean_risk: bool = True) -> dict:
    """Frobenius norm of d(mean_scene cumulative risk_j)/d(dynamic closed state).

    Separate detached leaves avoid counting upstream graph paths twice. Truth
    parameters stay constants. This diagnostic does not touch Actor gradients.
    mean_risk=False accepts scalar cumulative task value with no H-t scaling.
    """
    if step == horizon:
        return {"terminal_state_gradient_norm": 0., "terminal_component_state_gradient_norms": [0.] * (4 if mean_risk else 1)}
    from response_critic import critic_features
    dynamic = {'position', 'velocity', 'rotation', 'omega', 'motor', 'previous_action'}
    leaves, states = [], []
    for group, state in (('physical', closed.physical), ('policy', closed.policy)):
        values = {}
        for f in fields(state):
            value = getattr(state, f.name).detach()
            if group == 'policy' or f.name in dynamic:
                if value.is_floating_point():
                    value = value.requires_grad_(True)
                    leaves.append(value)
            values[f.name] = value
        states.append(type(state)(**values))
    scale = horizon - step if mean_risk else 1.
    values = scale * target(critic_features(ResponseClosedLoopState(*states), step, horizon)).mean(0)
    norms = []
    for j in range(values.numel()):
        grads = torch.autograd.grad(values[j], leaves, retain_graph=j < values.numel() - 1, allow_unused=True)
        norm = sum(g.double().square().sum() for g in grads if g is not None).sqrt()
        norms.append(float(norm))
    if not all(math.isfinite(n) for n in norms):
        raise Phase1ProbeError('terminal_gradient_nonfinite', {'step': step})
    return {'terminal_state_gradient_norm': math.sqrt(sum(n*n for n in norms)),
            'terminal_component_state_gradient_norms': norms}


def proposal_summary(evidence) -> dict:
    baseline = evidence.get('continuous_loss_before')
    trials = evidence.get('search', {}).get('candidates', [])
    finite = [trial['performance'] for trial in trials if 'performance' in trial]
    acceptable = [trial['performance'] for trial in trials
                  if 'performance' in trial and trial['rejection_reason'] is None]
    best = min(finite) if finite else None
    return {'baseline_h500_loss': baseline, 'best_candidate_h500_loss': best,
            'candidate_count': evidence.get('search', {}).get('candidate_rollouts', 0),
            'basis_rank': evidence.get('search', {}).get('basis_rank', 0),
            'best_acceptable_h500_loss': min(acceptable) if acceptable else None,
            'best_improvement_fraction': None if best is None else (baseline-best)/max(abs(baseline), 1.e-12),
            'has_acceptable_train_candidate': bool(acceptable), 'acceptable_train_candidates': len(acceptable)}


@torch.no_grad()
def critic_ranges(critic, inputs, mean_returns, suffix_returns, batch_size: int) -> dict:
    """Report actual target/prediction ranges; no learned scales or extra gates."""
    predicted = torch.cat([critic(chunk) for chunk in inputs.split(batch_size)])
    if not bool(torch.isfinite(predicted).all()):
        raise Phase1ProbeError('critic_prediction_nonfinite', 'prediction range probe')
    def ranges(value):
        return {'min': value.amin(0).tolist(), 'max': value.amax(0).tolist(),
                'mean': value.mean(0).tolist()}
    batch = suffix_returns.shape[1]
    # Exclude the forced Z_H=0 row so it does not hide the nonterminal range.
    return {'critic_true_mean_risk_range': ranges(mean_returns[:-batch]),
            'critic_true_cumulative_risk_range': ranges(suffix_returns[:-1].reshape(-1, 4)),
            'critic_prediction_range': ranges(predicted[:-batch])}

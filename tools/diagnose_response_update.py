"""Offline derivative/optimizer autopsy; never commits a training update.

The full BPTT derivative is a numerical diagnostic, not a certificate that a
chaotic float32 rollout has a useful finite-step descent direction.
"""
from __future__ import annotations

from dataclasses import fields
import argparse
import copy
import json
import sys
from pathlib import Path
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from response_critic import critic_features, detach_closed_state
from response_phase1 import dynamic_closed_state
from response_task import risk_weights, rollout, step_costs


def flatten_gradients(gradients, parameters):
    return torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                      for p, g in zip(parameters, gradients)])


def parameter_gradient(policy):
    parameters = list(policy.parameters())
    return flatten_gradients([p.grad for p in parameters], parameters)


def compare_vectors(a, b):
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    if not finite:
        return {'finite': False, 'cosine': None, 'norm_ratio': None}
    na, nb = float(a.norm()), float(b.norm())
    return {'finite': True, 'norm_a': na, 'norm_b': nb,
            'norm_ratio': na / nb if nb else None,
            'cosine': float(torch.dot(a / na, b / nb)) if na and nb else None,
            'relative_error': float((a-b).norm()) / nb if nb else None}


def slice_state(state, start, end):
    return type(state)(**{f.name: getattr(state, f.name)[start:end] for f in fields(state)})


def exact_gradient(policy, simulator, initial, horizon, config, *, chunk_size=64):
    """Full H gradient, holding ONE pooled CVaR selection across scene chunks."""
    with torch.no_grad():
        costs = step_costs(rollout(policy, simulator, initial, horizon), config).sum(0)
        weights = risk_weights(costs, config).detach()
    parameters = list(policy.parameters())
    gradient = torch.zeros_like(torch.nn.utils.parameters_to_vector(parameters))
    max_scene_cost_error = 0.
    for start in range(0, len(weights), chunk_size):
        end = min(start + chunk_size, len(weights))
        trace = rollout(policy, simulator, slice_state(initial, start, end), horizon)
        scene_costs = step_costs(trace, config).sum(0)
        max_scene_cost_error = max(max_scene_cost_error, float((scene_costs.detach()-costs[start:end]).abs().max()))
        objective = (weights[start:end] * scene_costs).sum()
        grads = torch.autograd.grad(objective, parameters, allow_unused=True)
        gradient.add_(flatten_gradients(grads, parameters))
        del trace, objective, grads, scene_costs
    return gradient, {'finite': bool(torch.isfinite(gradient).all()),
                      'gradient_norm': float(gradient.double().norm()),
                      'objective': float((weights * costs).sum()),
                      'weights_sum': float(weights.sum()), 'chunk_size': chunk_size,
                      'max_scene_cost_error': max_scene_cost_error}


def boundary_comparison(policy, target, simulator, start_state, start, end, horizon, config, weights):
    """Compare partial dV/dZ and pull both through the SAME preceding window.

    Suffix parameters are numerically fixed. g_true excludes the suffix's direct
    parameter gradient: it contains only J_(preceding window)^T v_true.
    """
    trace = rollout(policy, simulator, detach_closed_state(start_state), end-start)
    independent, leaves, names = dynamic_closed_state(trace.end)
    predicted = (weights * target(critic_features(independent, end, horizon)).squeeze(-1)).sum()
    vc = torch.autograd.grad(predicted, leaves, allow_unused=True)
    continuation = rollout(policy, simulator, independent, horizon-end)
    true = (weights * step_costs(continuation, config, start=end, horizon=horizon).sum(0)).sum()
    vt = torch.autograd.grad(true, leaves, allow_unused=True)
    vc = [torch.zeros_like(x) if g is None else g.detach() for g, x in zip(vc, leaves)]
    vt = [torch.zeros_like(x) if g is None else g.detach() for g, x in zip(vt, leaves)]
    outputs, left, right = [], [], []
    for name, a, b in zip(names, vc, vt):
        group, field = name.split('.')
        output = getattr(getattr(trace.end, group), field)
        if output.requires_grad:
            outputs.append(output); left.append(a); right.append(b)
    parameters = list(policy.parameters())
    gc = flatten_gradients(torch.autograd.grad(outputs, parameters, grad_outputs=left,
                            retain_graph=True, allow_unused=True), parameters)
    gt = flatten_gradients(torch.autograd.grad(outputs, parameters, grad_outputs=right,
                            allow_unused=True), parameters)
    report = {'start': start, 'end': end, 'predicted_value': float(predicted.detach()),
              'true_value': float(true.detach()),
              'state': compare_vectors(torch.cat([x.flatten() for x in vc]), torch.cat([x.flatten() for x in vt])),
              'parameter': compare_vectors(gc, gt),
              'fields': {name: compare_vectors(a, b) for name, a, b in zip(names, vc, vt)}}
    return report, {'v_critic': [x.cpu() for x in vc], 'v_true': [x.cpu() for x in vt],
                    'names': names, 'g_critic': gc, 'g_true': gt}


def diagnose_update(before, after, reference, output, *, chunk_size=64):
    """Use before-Actor and after-fit target, exactly as in the recorded step."""
    import response_training as common
    from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
    from response_task import TaskLossConfig, initialize, sample_scenarios
    from response_value import (TaskValueCritic, collect_task_trajectory,
                                accumulate_task_gradients, task_metrics)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    binding = before['binding']
    horizon = binding['horizon']
    config = TaskLossConfig(**binding['training_loss'])
    policy = ResponseMotorPolicy(ResponsePolicyConfig(**before['policy_config'])).cuda()
    policy.load_state_dict(before['model'])
    parameters = list(policy.parameters())
    initial, seeds = common.sample_training_scenarios(binding['protocol']['scenarios_per_bank'],
        after['update']-1, batches=2, dt=policy.config.dt, device=torch.device('cuda'),
        dtype=torch.float32, scenario_mode='fixed-airframe')
    assert seeds == reference['progress']['history'][after['update']-1]['scenario_seeds']
    banks = [sample_scenarios(binding['protocol']['scenarios_per_bank'], seed=s,
             dt=policy.config.dt, device=torch.device('cuda'), scenario_mode='fixed-airframe')[0]
             for s in binding['evaluation_seeds']]
    eval_initial = type(initial)(**{f.name: torch.cat([getattr(b, f.name) for b in banks]) for f in fields(initial)})
    from env_l2f import L2FSimulator, L2FParams
    simulator = L2FSimulator(L2FParams(dt=policy.config.dt))
    window_steps = binding['critic']['window_steps']
    record = collect_task_trajectory(policy, simulator, initial, horizon, window_steps, config)
    # Historical diagnostic: read scalar MLP weights, never resume an old
    # Critic optimizer or translate its incompatible supervision schema.
    target = TaskValueCritic(record.inputs.shape[-1]).to(next(policy.parameters())).requires_grad_(False)
    target.load_state_dict(after['critic_training']['target'])
    short_report = accumulate_task_gradients(policy, target, simulator, initial, horizon,
        window_steps, config, record, probes=True)
    short = parameter_gradient(policy).clone()
    clip_norm = torch.nn.utils.clip_grad_norm_(parameters, binding['gradient_clip'], error_if_nonfinite=True)
    clipped = parameter_gradient(policy).clone()
    theta = torch.nn.utils.parameters_to_vector(parameters).detach().clone()
    next_theta = torch.cat([after['model'][name].flatten() for name, _ in policy.named_parameters()]).to(theta)
    delta = next_theta - theta

    # Replay just the numerical optimizer transformation, without another fit.
    shadow = copy.deepcopy(policy)
    optimizer = torch.optim.Adam(shadow.parameters(), lr=binding['lr'])
    optimizer.load_state_dict(before['optimizer'])
    for p, original in zip(shadow.parameters(), parameters):
        p.grad = None if original.grad is None else original.grad.detach().clone()
    optimizer.step()
    adam_exact = common.model_hash(shadow) == after['model_sha256']
    if not adam_exact:
        raise RuntimeError('reconstructed short gradient/clip/Adam does not reproduce the recorded update')
    del shadow, optimizer
    torch.cuda.reset_peak_memory_stats()
    exact, exact_evidence = exact_gradient(policy, simulator, initial, horizon, config, chunk_size=chunk_size)
    peak = torch.cuda.max_memory_allocated()
    result = {'update': after['update'], 'before_update': before['update'], 'seeds': seeds,
              'adam_replay_exact': adam_exact, 'exact': exact_evidence,
              'short_vs_exact': compare_vectors(short, exact),
              'ten_short_vs_exact': compare_vectors(short * (horizon // window_steps), exact),
              'clip_norm': float(clip_norm), 'clip_vs_raw': compare_vectors(clipped, short),
              'adam_vs_negative_short': compare_vectors(delta, -short),
              'adam_vs_negative_exact': compare_vectors(delta, -exact),
              'windows': short_report['windows'], 'boundaries': [], 'ray': [],
              'exact_peak_allocated_bytes': peak}

    def write():
        result['elapsed_seconds'] = time.monotonic() - started
        (output/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    write()
    print(json.dumps({'update': after['update'], 'short_vs_exact': result['short_vs_exact'],
                      'adam_vs_exact': result['adam_vs_negative_exact']}), flush=True)
    # A bad approximate gradient warrants the requested decomposition. Also
    # record the same boundaries for the preceding reference update.
    for end in (50, 250, 450):
        start = end - window_steps
        state = initialize(policy, initial) if start == 0 else record.boundaries[start]
        report, tensors = boundary_comparison(policy, target, simulator, state,
            start, end, horizon, config, record.weights)
        torch.save({k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in tensors.items()},
                   output/('boundary_%03d.pt' % end))
        result['boundaries'].append(report); write()
        print(json.dumps({'update': after['update'], 'boundary': end,
                          'state': report['state'], 'parameter': report['parameter']}), flush=True)
        del tensors
    with torch.no_grad():
        for alpha in (0., 1., .25, .0625, .015625):
            torch.nn.utils.vector_to_parameters(theta + alpha * delta, parameters)
            row = {'alpha': alpha}
            for name, state in (('train', initial), ('eval', eval_initial)):
                metrics = task_metrics(rollout(policy, simulator, state, horizon), config)
                metrics.pop('scenarios', None); row[name] = metrics
            result['ray'].append(row); write()
        torch.nn.utils.vector_to_parameters(theta, parameters)

    # A fresh-moments shadow answers whether the saved Adam transformation is
    # necessary for this outcome. It resets both moments, not momentum alone.
    shadow = copy.deepcopy(policy)
    old_group = before['optimizer']['param_groups'][0]
    optimizer = torch.optim.Adam(shadow.parameters(), lr=old_group['lr'],
                                betas=old_group['betas'], eps=old_group['eps'])
    for p, original in zip(shadow.parameters(), parameters):
        p.grad = None if original.grad is None else original.grad.detach().clone()
    optimizer.step()
    fresh_delta = torch.nn.utils.parameters_to_vector(shadow.parameters()).detach() - theta
    result['fresh_adam'] = {'vs_negative_short': compare_vectors(fresh_delta, -short),
                           'vs_negative_exact': compare_vectors(fresh_delta, -exact),
                           'vs_actual_adam': compare_vectors(fresh_delta, delta)}
    with torch.no_grad():
        for name, state in (('train', initial), ('eval', eval_initial)):
            metrics = task_metrics(rollout(shadow, simulator, state, horizon), config)
            metrics.pop('scenarios', None); result['fresh_adam'][name] = metrics
    torch.save({'exact': exact.cpu(), 'short': short.cpu(), 'clipped': clipped.cpu(),
                'actual_adam_delta': delta.cpu(), 'fresh_adam_delta': fresh_delta.cpu(),
                'weights': record.weights.cpu(), 'parameter_names': [n for n, _ in policy.named_parameters()]},
               output/'gradients.pt')
    write()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-dir', type=Path, required=True)
    parser.add_argument('--reference-checkpoint', type=Path, required=True)
    parser.add_argument('--updates', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    reference = torch.load(args.reference_checkpoint, map_location='cpu')
    for update in args.updates:
        before = torch.load(args.replay_dir/('%03d.pt' % (update-1)), map_location='cpu')
        after = torch.load(args.replay_dir/('%03d.pt' % update), map_location='cpu')
        if before['update'] != update-1 or after['update'] != update:
            raise ValueError('snapshots do not bracket requested update')
        if after['model_sha256'] != reference['progress']['history'][update-1]['model_sha256']:
            raise ValueError('snapshot is not the recorded Actor')
        diagnose_update(before, after, reference, args.output/('update_%03d' % update), chunk_size=args.chunk_size)


if __name__ == '__main__':
    main()

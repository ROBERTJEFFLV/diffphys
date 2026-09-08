from __future__ import annotations

import math
import pytest
import torch
import response_critic as critic


def test_smoothmax_normalizes_units_detaches_baseline_and_prioritizes_worsening():
    assert hasattr(critic, 'normalized_risk_smoothmax'), 'missing normalized risk scalarization'
    baseline = torch.tensor([100., 20., 5., .001], dtype=torch.float64, requires_grad=True)
    ratios = torch.tensor([.7, .9, 1.2, .8], dtype=torch.float64, requires_grad=True)
    risk = baseline.detach() * ratios
    value = critic.normalized_risk_smoothmax(risk, baseline, beta=10., epsilon=1.e-12)
    gradient = torch.autograd.grad(value, (ratios, baseline), allow_unused=True)
    assert gradient[1] is None
    assert gradient[0][2] > .9 and torch.all(gradient[0] > 0)
    expected = torch.logsumexp(10 * risk / (baseline.detach()+1.e-12), 0) / 10
    torch.testing.assert_close(value, expected)
    assert torch.isfinite(critic.normalized_risk_smoothmax(risk*1.e5, baseline, beta=10.))
    zero = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    critic.normalized_risk_smoothmax(zero, zero.detach(), beta=10.).backward()
    assert torch.isfinite(zero.grad).all()


def test_relative_risk_gate_uses_declared_tolerance_for_total_and_each_component():
    before = {'task_objective':10., 'risk_objective':20.,
              'risk_components':{'position':10., 'velocity':5., 'omega':4., 'saturation':1.}}
    after = {**before, 'task_objective':9., 'risk_objective':19.,
             'risk_components':{**before['risk_components'], 'saturation':1.+5.e-7}}
    assert 'risk_relative_tolerance' in critic.CriticConfig.__dataclass_fields__, 'gate lacks calibrated tolerance'
    assert critic.acceptance_rejection(before,after,[before],[after],dev_relative_tolerance=.002,
                                      risk_relative_tolerance=1.e-6) is None
    assert critic.acceptance_rejection(before,after,[before],[after],dev_relative_tolerance=.002,
                                      risk_relative_tolerance=0.) == 'train_risk_deteriorated'
    after['risk_components']['saturation'] = 1.+2.e-6
    assert critic.risk_deteriorated(before,after,relative_tolerance=1.e-6)
    for bad in (-1., float('nan'), 1.):
        with pytest.raises(ValueError):
            critic.CriticConfig(risk_relative_tolerance=bad)


def test_component_labels_match_real_suffixes_and_critic_has_four_outputs():
    import response_task as task
    from test_response_control import fixture
    policy, sim, initial = fixture()
    record = critic.collect_trajectory(policy, sim, initial, 6, task.TaskLossConfig())
    expected = torch.stack(tuple(task.risk_components(record.trajectory, task.RiskConfig()).values()), -1)
    assert record.returns.shape == (7, 2, 4), 'scalar labels cannot supervise component terminal risks'
    torch.testing.assert_close(record.returns[:-1], expected.flip(0).cumsum(0).flip(0))
    assert torch.equal(record.returns[-1], torch.zeros_like(record.returns[-1]))
    net = critic.RiskToGoCritic(record.inputs.shape[-1]).double()
    assert net(record.inputs[0]).shape == (2, 4)


def test_scalar_migration_rejects_old_risk_semantics_and_keeps_fresh_state():
    import copy
    import response_task as task
    from test_response_control import fixture
    from test_response_guarded_updates import assert_nested_equal
    policy, _, initial = fixture()
    state = critic.CriticTrainer(policy, task.initialize(policy, initial), 6,
                                 critic.CriticConfig(window_steps=2))
    before = copy.deepcopy(state.state_dict())
    with pytest.raises(ValueError, match="new weights-only Actor experiment"):
        state.initialize_from_scalar({'objective': 'risk-to-go-v1-fixed-physical-scales'},
                                     torch.tensor([.5, .3, .199, .001]))
    assert_nested_equal(state.state_dict(), before)


def test_short_window_component_objective_matches_independent_physics_oracle():
    import copy
    import response_task as task
    from test_response_control import fixture
    policy, sim, initial = fixture()
    config = task.TaskLossConfig(steady_steps=2)
    record = critic.collect_trajectory(policy, sim, initial, 6, config)
    target = critic.RiskToGoCritic(record.inputs.shape[-1]).double().requires_grad_(False)
    # Derive baseline independently from the real flight, never Critic predictions.
    components = torch.stack(tuple(task.risk_components(record.trajectory,task.RiskConfig()).values()),-1)
    returns = task.suffix_risks(components).detach()
    before = copy.deepcopy(policy.state_dict())
    result = critic.accumulate_actor_gradients(policy,target,sim,initial,6,2,config,record.weights,
                                               baseline_returns=returns, risk_smoothmax_beta=8.)
    actual = [None if p.grad is None else p.grad.clone() for p in policy.parameters()]
    policy.zero_grad(set_to_none=True)
    count = max(1,math.ceil(config.tail_fraction*initial.position.shape[0]))
    cw = torch.full_like(returns[0],1/initial.position.shape[0])
    for j in range(4):
        cw[returns[0,:,j].topk(count).indices,j] += config.tail_weight/count
    baseline = (cw*returns[0]).sum(0)
    closed = task.initialize(policy,initial)
    for start in (0,2,4):
        trace = task.rollout(policy,sim,critic.detach_closed_state(closed),2)
        risks = torch.stack(tuple(task.risk_components(trace,task.RiskConfig()).values()),-1).sum(0)
        risks = risks + returns[0]-returns[start]
        if start < 4:
            risks = risks + target(critic.critic_features(trace.end,start+2,6))
        ratios = (cw*risks).sum(0)/(baseline+1.e-12)
        value = torch.logsumexp(8*ratios,0)/8
        performance = (record.weights*task.step_costs(trace,config,start=start,horizon=6).sum(0)).sum()
        ((performance+value)/3).backward()
        closed = trace.end
    for p, expected in zip(policy.parameters(),actual):
        if expected is None: assert p.grad is None
        else: torch.testing.assert_close(p.grad,expected,rtol=1.e-10,atol=1.e-11)
    assert all(torch.equal(before[k],v) for k,v in policy.state_dict().items())
    assert torch.equal(result['end'].policy.memory,record.trajectory.end.policy.memory)
    assert all(p.grad is None for p in target.parameters())

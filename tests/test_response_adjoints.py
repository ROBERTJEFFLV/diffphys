from dataclasses import replace
import copy

import pytest
import torch

import response_value as value
from response_phase1 import dynamic_closed_state
from response_task import TaskLossConfig, initialize, rollout, step_costs, task_loss
from test_response_control import fixture
from test_response_guarded_updates import assert_nested_equal


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_complete_boundary_adjoint_matches_suffix_and_pooled_full_bptt(dtype):
    # Dropping a history leaf, multiplying CVaR twice, or restarting tail time
    # must disagree with these independently differentiated full trajectories.
    assert hasattr(value, 'collect_boundary_adjoints'), 'missing full boundary adjoints'
    policy, sim, initial = fixture(dtype)
    config = TaskLossConfig(steady_steps=2, tail_fraction=.5)
    record = value.collect_task_trajectory(policy, sim, initial, 6, 2, config)
    old = copy.deepcopy(policy.state_dict())
    labels = value.collect_boundary_adjoints(policy, sim, record, 6, 2, config)
    assert set(labels.gradients) == {0, 2, 4, 6}
    assert all(torch.count_nonzero(g) == 0 for g in labels.gradients[6].values())
    for step in (2, 4):
        state, leaves, names = dynamic_closed_state(record.boundaries[step])
        trace = rollout(policy, sim, state, 6-step,
                        parameters={n:p.detach() for n,p in policy.named_parameters()})
        direct = torch.autograd.grad(step_costs(trace, config, start=step, horizon=6).sum(),
                                     leaves, allow_unused=True)
        for name, leaf, gradient in zip(names, leaves, direct):
            expected = torch.zeros_like(leaf) if gradient is None else gradient
            torch.testing.assert_close(labels.gradients[step][name], expected,
                                       rtol=2e-5, atol=2e-7)
    assert all(p.requires_grad and p.grad is None for p in policy.parameters())
    target = value.TaskValueCritic(record.inputs.shape[-1]).to(dtype).requires_grad_(False)
    oracle = value.accumulate_task_gradients(policy, target, sim, initial, 6, 2, config, record,
                  terminal_mode='oracle_full_state', adjoints=labels, probes=True)
    actual = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten()
                        for p in policy.parameters()]) * 3
    policy.zero_grad(set_to_none=True)
    task_loss(rollout(policy, sim, initial, 6), config).backward()
    direct = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten()
                        for p in policy.parameters()])
    torch.testing.assert_close(actual, direct, rtol=3e-5, atol=3e-7)
    for row in oracle['windows']:
        expected = (record.weights*record.returns[row['end']]).sum()
        assert row['terminal_value'] == pytest.approx(float(expected), rel=1e-5)
        assert row['boundary']['passed']
    assert_nested_equal(old, policy.state_dict())


def test_oracle_rejects_labels_for_another_actor_and_restores_freeze_flags():
    assert hasattr(value, 'collect_boundary_adjoints'), 'missing full boundary adjoints'
    policy, sim, initial = fixture()
    config = TaskLossConfig(steady_steps=2)
    record = value.collect_task_trajectory(policy, sim, initial, 6, 2, config)
    labels = value.collect_boundary_adjoints(policy, sim, record, 6, 2, config)
    target = value.TaskValueCritic(record.inputs.shape[-1]).double().requires_grad_(False)
    with torch.no_grad():
        next(policy.parameters()).add_(.01)
    with pytest.raises(ValueError, match='Actor'):
        value.accumulate_task_gradients(policy, target, sim, initial, 6, 2, config, record,
              terminal_mode='oracle_full_state', adjoints=labels)


def test_rotation_covector_uses_tangent_directions():
    import importlib.util
    assert importlib.util.find_spec('response_adjoints'), 'missing state coordinate contract'
    from response_adjoints import gradient_coordinates
    policy, sim, initial = fixture()
    state = initialize(policy, initial)
    matrix = torch.tensor([[0., -3., 2.], [3., 0., -1.], [-2., 1., 0.]], dtype=torch.float64)
    # At I, G:[delta]_x gives [2,4,6]. A radial G=I has no SO(3) derivative.
    actual = gradient_coordinates('physical.rotation', matrix.expand(2,3,3), state)
    torch.testing.assert_close(actual, torch.tensor([[2.,4.,6.]]*2, dtype=torch.float64))
    torch.testing.assert_close(gradient_coordinates('physical.rotation', initial.rotation, state),
                               torch.zeros(2,3,dtype=torch.float64))


def test_adjoint_cannot_be_reused_on_other_initial_states_of_same_actor():
    policy, sim, initial = fixture()
    config = TaskLossConfig(steady_steps=2)
    original = value.collect_task_trajectory(policy,sim,initial,6,2,config)
    labels = value.collect_boundary_adjoints(policy,sim,original,6,2,config)
    changed = replace(initial, position=initial.position + .1)
    record = value.collect_task_trajectory(policy,sim,changed,6,2,config)
    target = value.TaskValueCritic(record.inputs.shape[-1]).double().requires_grad_(False)
    with pytest.raises(ValueError, match='initial'):
        value.accumulate_task_gradients(policy,target,sim,changed,6,2,config,record,
            terminal_mode='oracle_full_state',adjoints=labels)


def test_heldout_audit_keeps_original_forward_batch_and_masks_cvar_weights():
    policy,sim,initial = fixture(torch.float32)
    config = TaskLossConfig(steady_steps=2)
    record = value.collect_task_trajectory(policy,sim,initial,6,2,config)
    labels = value.collect_boundary_adjoints(policy,sim,record,6,2,config)
    target = value.TaskValueCritic(record.inputs.shape[-1]).requires_grad_(False)
    report = value.accumulate_task_gradients(policy,target,sim,initial,6,2,config,record,
        terminal_mode='oracle_full_state',adjoints=labels,scene_indices=torch.tensor([1]))
    assert report['forward_scenarios'] == 2 and report['weighted_scenarios'] == 1
    assert all(w['boundary']['exact'] for w in report['windows'])
    actual = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten() for p in policy.parameters()])*3
    policy.zero_grad(set_to_none=True)
    costs = step_costs(rollout(policy,sim,initial,6),config).sum(0)
    (costs[1]*record.weights[1]).backward()
    expected = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten() for p in policy.parameters()])
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-7)

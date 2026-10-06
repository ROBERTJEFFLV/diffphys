"""Native B readout, recurrent continuity and explicit checkpoint boundaries."""
import copy
from dataclasses import replace

import pytest
import torch

from env_raptor import RaptorSimulator
from loss_fixtures import test_loss
from response_adjoints import collect_rollout, backward_actor
from response_groups import GroupBalanceConfig
from response_policy import ARCHITECTURE, ResponseMotorPolicy, ResponsePolicyConfig
from response_task import initialize, rollout
from response_training import sample_training_scenarios
from test_reference_training import args_for, run


def test_readout_receives_only_current_hidden_without_hooks_or_masks():
    torch.manual_seed(7)
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).double()
    initial = RaptorSimulator().reset(3, seed=71, horizon=4, dtype=torch.float64)
    assert not policy.readout._forward_pre_hooks
    assert not policy.readout._forward_hooks
    received = []
    handle = policy.readout.register_forward_pre_hook(
        lambda module, inputs: received.append(inputs[0].detach().clone())
    )
    try:
        trace = rollout(policy, RaptorSimulator(), initial, 4)
    finally:
        handle.remove()
    assert policy.readout.weight.shape == (4, 8)
    assert all(row.shape == (3, 8) for row in received)
    assert torch.equal(received[-1], trace.end.policy.memory)
    assert not torch.equal(received[0], received[-1])
    assert torch.equal(trace.end.physical.previous_action, trace.actions[-1])
    # Statistical chunks pass the complete recurrent/history state onwards.
    first = rollout(policy, RaptorSimulator(), initialize(policy, initial), 2)
    second = rollout(policy, RaptorSimulator(), first.end, 2)
    assert torch.equal(trace.actions, torch.cat((first.actions, second.actions)))
    assert torch.equal(trace.end.policy.memory, second.end.policy.memory)


@pytest.mark.parametrize('decay', [0., 1.])
def test_native_b_all_parameter_group_gradients_match_vjp(decay):
    torch.manual_seed(17)
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    reference = copy.deepcopy(policy)
    initial, _ = sample_training_scenarios(
        32, 0, horizon=4, sampling='coverage128', dtype=torch.float64
    )
    groups = GroupBalanceConfig(max_groups=128, min_scenarios=1,
                               layout='coverage128', clip_norm=.08)
    actual = collect_rollout(policy, RaptorSimulator(), initial, test_loss(),
                            horizon=4, time_decay=decay, group_config=groups)
    expected = collect_rollout(reference, RaptorSimulator(), initial, test_loss(),
                              horizon=4, time_decay=decay,
                              group_config=replace(groups, backward_backend='vjp'))
    assert torch.equal(actual.costs, expected.costs)
    backward_actor(policy, RaptorSimulator(), actual, test_loss())
    backward_actor(reference, RaptorSimulator(), expected, test_loss())
    assert {name for name, _ in policy.named_parameters()} == {
        'response_memory.weight_ih', 'response_memory.weight_hh',
        'response_memory.bias_ih', 'response_memory.bias_hh',
        'readout.weight', 'readout.bias',
    }
    for (name, parameter), target in zip(policy.named_parameters(), reference.parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, target.grad, rtol=2e-10, atol=2e-11)


def test_historical_b_is_weights_only_and_cannot_resume_old_binding(tmp_path):
    run(args_for(tmp_path/'source', 1))
    saved = torch.load(tmp_path/'source/latest.pt', weights_only=True)
    assert saved['architecture'] == ARCHITECTURE == 'gru16-hidden-only-readout-absolute-motor-policy-v1'
    assert saved['optimizer']['state']
    saved['schema'] = 'wc-deletion-paired-seed7-v1-B'
    saved['binding']['source_sha256'] = 'historical-ablation-source'
    historical = tmp_path/'historical_b.pt'
    torch.save(saved, historical)
    original = historical.read_bytes()
    with pytest.raises(ValueError, match='contract|source|architecture|motor semantics'):
        run(args_for(tmp_path/'resume', 2, ('--resume', str(historical))))
    run(args_for(tmp_path/'import', 0, ('--init-checkpoint', str(historical))))
    imported = torch.load(tmp_path/'import/latest.pt', weights_only=True)
    assert imported['model_sha256'] == saved['model_sha256']
    assert imported['optimizer']['state'] == {}
    assert imported['progress']['updates'] == 0
    assert imported['progress']['initialization']['weights_only']
    assert historical.read_bytes() == original


def test_old_a_readout_is_not_silently_loaded_as_b(tmp_path):
    run(args_for(tmp_path/'source', 0))
    saved = torch.load(tmp_path/'source/latest.pt', weights_only=True)
    saved['architecture'] = 'gru16-direct-readout-absolute-motor-policy-v4'
    saved['model']['readout.weight'] = torch.zeros(4, 16 + saved['policy_config']['memory_dim'])
    old = tmp_path/'old_a.pt'
    torch.save(saved, old)
    with pytest.raises(ValueError, match='architecture|motor semantics'):
        run(args_for(tmp_path/'import', 0, ('--init-checkpoint', str(old))))

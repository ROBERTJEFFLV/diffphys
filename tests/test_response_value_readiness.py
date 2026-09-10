import copy
from dataclasses import replace

import pytest
import torch

import response_value as value
from response_task import TaskLossConfig, initialize
from test_response_control import fixture
from test_response_guarded_updates import assert_nested_equal


def test_unready_critic_keeps_completed_fits_without_touching_actor_or_adam():
    assert 'warmup_max_fits' in value.TaskValueConfig.__dataclass_fields__, 'missing bounded readiness'
    policy, sim, initial = fixture()
    config = value.TaskValueConfig(window_steps=2, batch_size=8, derivative_holdout_scenes=1,
        derivative_batch_size=2, warmup_max_fits=1, ready_min_cosine=1., ready_max_relative_error=0.)
    trainer = value.TaskValueTrainer(policy, initialize(policy, initial), 6, config)
    actor, critic = copy.deepcopy(policy.state_dict()), copy.deepcopy(trainer.critic.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    row = trainer.update(policy, optimizer, sim, initial, 6, TaskLossConfig(steady_steps=2), gradient_clip=10)
    assert not row['updated'] and row['status'] == 'critic_not_ready'
    assert row['readiness']['fits'] == 1
    assert trainer.completed_fits == 1
    assert optimizer.state_dict()['state'] == {}
    assert_nested_equal(actor, policy.state_dict())
    assert any(not torch.equal(v, trainer.critic.state_dict()[k]) for k,v in critic.items())
    assert all(p.grad is None for p in policy.parameters())
    assert 'target_all' in row['readiness']['last_audit']
    assert 'target_heldout' in row['readiness']['last_audit']
    assert len(row['boundary_continuity']) == 3
    assert all(b['exact'] for b in row['boundary_continuity'])
    fit_log = row['readiness']['rounds'][0]['critic_fit']
    assert len(fit_log['derivative_gradient_contributions']) == 2
    assert 'boundaries' in fit_log['target_derivative_heldout_after']


def test_critic_only_never_commits_even_if_registered_thresholds_pass():
    assert 'warmup_max_fits' in value.TaskValueConfig.__dataclass_fields__, 'missing bounded readiness'
    policy, sim, initial = fixture()
    config = value.TaskValueConfig(window_steps=2, batch_size=8, derivative_holdout_scenes=1,
        derivative_batch_size=2, warmup_max_fits=1, ready_min_cosine=-1., ready_max_relative_error=1e12)
    trainer = value.TaskValueTrainer(policy, initialize(policy, initial), 6, config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    actor = copy.deepcopy(policy.state_dict())
    row = trainer.update(policy, optimizer, sim, initial, 6, TaskLossConfig(steady_steps=2),
                         gradient_clip=10, critic_only=True)
    assert row['readiness']['ready'] and row['status'] == 'critic_only_ready'
    assert not row['updated'] and not optimizer.state
    assert_nested_equal(actor, policy.state_dict())


def test_actor_gradient_audit_preserves_existing_actor_gradients():
    assert hasattr(value, 'audit_actor_gradient'), 'missing actual Actor gradient audit'
    policy, sim, initial = fixture()
    config = TaskLossConfig(steady_steps=2)
    record = value.collect_task_trajectory(policy, sim, initial, 6, 2, config)
    labels = value.collect_boundary_adjoints(policy, sim, record, 6, 2, config)
    target = value.TaskValueCritic(record.inputs.shape[-1]).double().requires_grad_(False)
    for p in policy.parameters(): p.grad = torch.ones_like(p)
    saved = [p.grad.clone() for p in policy.parameters()]
    result, reference = value.audit_actor_gradient(policy, target, sim, initial, 6, 2,
                                    config, record, labels, scene_indices=torch.tensor([1]))
    assert result['finite'] and reference.norm() > 0
    for p,g in zip(policy.parameters(),saved): torch.testing.assert_close(p.grad,g,rtol=0,atol=0)


def test_readiness_cli_records_full_groups_budget_and_no_actor_mode(tmp_path):
    from tools.train_response_control import parse_args
    from response_training import critic_configuration
    # Parse errors before the new route is implemented are an intentional red test.
    args = parse_args(['--optimizer','task-adam','--scenario-mode','fixed-airframe',
        '--value-critic-only','--value-warmup-max-fits','2','--value-warmup-max-seconds','30',
        '--value-derivative-state-groups','policy.memory','physical.velocity'])
    cfg = critic_configuration(args)
    assert args.value_critic_only and cfg.warmup_max_fits == 2
    assert cfg.derivative_state_groups == ('policy.memory','physical.velocity')
    assert cfg.sampling_boundaries(500) == tuple(range(50,500,50))


def test_critic_only_runner_saves_business_stop_and_does_not_auto_resume(tmp_path, monkeypatch):
    from tools.train_response_control import parse_args
    from response_policy import ResponsePolicyConfig
    import response_training as training
    monkeypatch.setattr(training, 'FINAL_CLAIM', tmp_path/'unused-final')
    args = parse_args(['--optimizer','task-adam','--scenario-mode','fixed-airframe',
        '--value-critic-only','--value-warmup-max-fits','1',
        '--value-ready-min-cosine','1','--value-ready-max-relative-error','0',
        '--horizon','4','--window-steps','2','--scenarios','16','--updates','1',
        '--memory-dim','4','--hidden-dim','8','--work-dir',str(tmp_path/'run')])
    policy_config = ResponsePolicyConfig(memory_dim=4,hidden_dim=8)
    report = training.train(args,policy_config,TaskLossConfig(prediction_weight=0))
    assert report['status'] == 'critic_not_ready' and report['updates'] == 0
    saved = torch.load(args.work_dir/'latest.training.pt',map_location='cpu')
    assert saved['critic_training']['completed_fits'] == 1 and not saved['optimizer']['state']
    assert len(saved['progress']['evaluation']) == 1
    assert not (args.work_dir/'failure.pt').exists()
    with pytest.raises(RuntimeError, match='automatically resumed'):
        training.train(args,policy_config,TaskLossConfig(prediction_weight=0))

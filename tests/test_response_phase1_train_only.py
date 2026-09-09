from __future__ import annotations

import copy
import json

import pytest
import torch

import response_critic as critic
import response_task as task
import response_training as training
from response_policy import ResponsePolicyConfig
from test_response_control import fixture
from tools.train_response_control import parse_args


def test_train_objective_accepts_improvement_despite_risk_deterioration():
    policy, sim, initial = fixture()
    trainer = critic.CriticTrainer(policy, task.initialize(policy, initial), 4,
        critic.CriticConfig(window_steps=2, acceptance_mode='train-objective'))
    baseline = trainer._metrics(task.rollout(policy, sim, initial, 4), task.TaskLossConfig())
    baseline['task_objective'] = 10.
    worse_risk = copy.deepcopy(baseline)
    worse_risk.update(task_objective=8., hard_risk_components={'omega':1.e6,'saturation':1.e6},
                      hard_risk_bounds_violated=['position'])
    gradients = torch.ones(5, sum(p.numel() for p in policy.parameters()),dtype=next(policy.parameters()).dtype)
    evidence = {}
    after = trainer._propose_actor(policy, None, {'objective_gradients':gradients}, baseline,
                                   lambda:worse_risk, 10., evidence)
    assert after is not None and evidence['rejection_reason'] is None
    assert evidence['search']['candidate_rollouts'] > 0


def test_train_objective_still_rejects_nonimprovement():
    policy, sim, initial = fixture()
    trainer = critic.CriticTrainer(policy, task.initialize(policy, initial), 4,
        critic.CriticConfig(window_steps=2, acceptance_mode='train-objective'))
    baseline = trainer._metrics(task.rollout(policy, sim, initial, 4), task.TaskLossConfig())
    gradients = torch.ones(5, sum(p.numel() for p in policy.parameters()),dtype=next(policy.parameters()).dtype)
    evidence = {}
    after = trainer._propose_actor(policy, None, {'objective_gradients':gradients}, baseline,
                                   lambda:baseline, 10., evidence)
    assert after is None and evidence['rejection_reason'] == 'no_acceptable_candidate'


def test_phase1_train_never_samples_or_evaluates_dev_and_records_retained_metrics(tmp_path,monkeypatch):
    monkeypatch.setattr(training,'FINAL_CLAIM',tmp_path/'unused-final')
    seen=[]; sampler=training.sample_scenarios
    def sample(count, **kwargs):
        assert kwargs['scenario_mode']=='fixed-airframe'
        assert training.TRAIN_SEED_BASE <= kwargs['seed'] < min(training.DEVELOPMENT_SEEDS)
        seen.append(kwargs['seed'])
        return sampler(count, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('Phase 1 must not invoke DEV evaluation')
    monkeypatch.setattr(training,'sample_scenarios',sample)
    monkeypatch.setattr(training,'evaluate',forbidden)
    monkeypatch.setattr(critic.CriticTrainer,'development_baseline',forbidden)
    args=parse_args(['--phase1-train-only','--phase1-probes','--scenario-mode','fixed-airframe',
                    '--updates','2','--horizon','4','--window-steps','2','--scenarios','32',
                    '--memory-dim','4','--hidden-dim','8','--development-every','1',
                    '--work-dir',str(tmp_path/'run')])
    training.train(args,ResponsePolicyConfig(memory_dim=4,hidden_dim=8),task.TaskLossConfig(prediction_weight=0))
    saved=torch.load(tmp_path/'run/latest.training.pt',map_location='cpu')
    assert saved['progress']['development']==[]
    assert saved['progress']['attempts']==2
    assert saved['binding']['critic']['acceptance_mode']=='train-objective'
    assert set(seen)==set(range(training.TRAIN_SEED_BASE,training.TRAIN_SEED_BASE+4))
    for row in saved['progress']['history']:
        assert row['training_scenarios']==64
        assert not row['development_evaluated']
        assert row['phase1_retained']['task_objective'] <= row['continuous_loss_before']
        assert len(row['critic_true_mean_risk_range']['max'])==4
        assert row['phase1_proposal']['candidate_count'] > 0
        assert row['phase1_proposal']['basis_rank'] > 0
    assert not (tmp_path/'run/development').exists()


def test_phase1_contract_uses_nominal_scenarios(monkeypatch):
    import tools.check_response_training_contract as contract
    original = contract.sample_scenarios
    modes=[]
    def sample(*args, **kwargs):
        modes.append(kwargs['scenario_mode'])
        return original(*args, **kwargs)
    monkeypatch.setattr(contract,'sample_scenarios',sample)
    report=contract.run_contract(device='cpu',scenario_mode='fixed-airframe')
    assert report['passed'] and modes==['fixed-airframe']

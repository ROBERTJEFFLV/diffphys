"""Production defaults must optimize the verified settling-time objective."""
from __future__ import annotations

import copy
from dataclasses import replace
import json

import pytest
import torch

from diffphys import response_training as training
from diffphys.env_raptor import RaptorSimulator
from diffphys.hover_time_experiment import HoverTimeConfig, experiment
from diffphys.response_groups import GroupBalanceConfig
from diffphys.response_noise import DisturbanceConfig
from diffphys.response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from diffphys.response_task import TaskLossConfig
from test_reference_training import args_for, run
from tools.train_response_control import main


def test_standard_cli_adds_time_loss_and_checkpoint_evaluation_uses_saved_weight(tmp_path):
    work = tmp_path / 'default'
    main(['--device', 'cpu', '--dtype', 'float64', '--scenarios', '8',
          '--eval-scenarios', '4', '--horizon', '8', '--updates', '1',
          '--development-every', '1',
          '--memory-dim', '8', '--epsilon-p', '.5', '--epsilon-a', '.25',
          '--lambda-R', '.4', '--work-dir', str(work)])
    saved = torch.load(work / 'latest.pt', weights_only=True)
    cfg = saved['binding']['protocol']['hover_time']
    assert cfg == {'weight': 1., 'position_m': .1, 'speed_m_s': .1, 'omega_rad_s': .5}
    report = json.loads((work / 'evaluation.jsonl').read_text().splitlines()[-1])
    assert report['task_components']['hover_time'] > 0
    assert report['task_objective'] == pytest.approx(
        report['hover_objective'] + report['hover_time_proxy_seconds'] / .08)
    args = args_for(tmp_path / 'eval', extra=('--hover-time-weight', '0'))
    args.checkpoint = work / 'latest.pt'
    actual = training.evaluate_checkpoint(args)
    assert actual['hover_time_weight'] == 1.
    assert actual['task_objective'] == report['task_objective']


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('weight', [0., 1.])
def test_production_time_loss_matches_experiment_across_metric_chunks(dtype, weight):
    torch.manual_seed(111)
    actor = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).to(dtype)
    sim = RaptorSimulator()
    initial = sim.reset(32, seed=113, horizon=60, dtype=dtype,
                        disturbances=DisturbanceConfig.clean())
    initial = replace(initial, velocity=initial.velocity * .1)
    loss = TaskLossConfig(.01, .01, 25.)
    group = GroupBalanceConfig(max_groups=1, min_scenarios=32)
    results = []
    for production in (False, True):
        policy = copy.deepcopy(actor)
        kwargs = dict(horizon=60, time_decay=0., group_config=group)
        if production:
            record = training.collect_rollout(policy, sim, initial, loss,
                                               hover_time=HoverTimeConfig(weight=weight), **kwargs)
        else:
            with experiment(HoverTimeConfig(weight=weight)):
                record = training.collect_rollout(policy, sim, initial, loss, **kwargs)
        training.backward_actor(policy, sim, record, loss)
        results.append((record.costs.detach(), [p.grad.clone() for p in policy.parameters()],
                        record.metrics))
    a, b = results
    assert torch.equal(a[0], b[0])
    for x, y in zip(a[1], b[1]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert a[2] == b[2]


def test_zero_weight_is_bound_and_cannot_be_resumed_as_default(tmp_path):
    work = tmp_path / 'zero'
    run(args_for(work, 1, ('--hover-time-weight', '0')))
    report = json.loads((work / 'evaluation.jsonl').read_text().splitlines()[-1])
    assert report['task_components']['hover_time'] == 0
    assert report['task_objective'] == report['hover_objective']
    with pytest.raises(ValueError, match='configuration'):
        run(args_for(work, 2, ('--resume', str(work / 'latest.pt'))))


@pytest.mark.parametrize('weight', ['-1', 'nan', 'inf'])
def test_invalid_time_weight_is_rejected_at_cli(weight, tmp_path):
    with pytest.raises(SystemExit):
        args_for(tmp_path, extra=('--hover-time-weight', weight))


def test_explicit_ablation_still_has_only_one_configuration_difference(tmp_path):
    from tools.analyze_hover_time_ablation import verify_pair
    for name, weight in [('baseline', 0.), ('time', 1.)]:
        with experiment(HoverTimeConfig(weight=weight)):
            run(args_for(tmp_path / name, 1))
    saved, _ = verify_pair(tmp_path, 1)
    assert [v['binding']['protocol']['hover_time_experiment']['weight']
            for v in saved] == [0., 1.]

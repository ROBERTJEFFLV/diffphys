"""Bounded integration checks: checkpoint semantics, RNG and tiny Adam updates."""
from dataclasses import fields
from pathlib import Path
import copy
import json
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.train_response_control import parse_args
from response_policy import ResponsePolicyConfig, ResponseMotorPolicy
from response_task import TaskLossConfig
from response_training import train, evaluate_checkpoint, load_policy_checkpoint, require_reference_checkpoint


def args_for(path, profile='l2f', updates=2, extra=()):
    return parse_args(['--device','cpu','--dtype','float64','--scenarios','2',
                      '--eval-scenarios','3','--horizon','8','--window-steps','4',
                      '--development-every','1','--checkpoint-every','1','--updates',str(updates),
                      '--max-seconds','60','--hidden-dim','8','--memory-dim','8',
                      '--scenario-mode',profile,'--work-dir',str(path),*extra])


def run(args):
    pc=ResponsePolicyConfig(**{f.name:getattr(args,f.name) for f in fields(ResponsePolicyConfig)})
    lc=TaskLossConfig(**{f.name:getattr(args,f.name) for f in fields(TaskLossConfig)})
    return train(args,pc,lc)


@pytest.mark.parametrize('profile',['l2f','raptor'])
def test_checkpoint_resume_and_evaluate_match_uninterrupted(tmp_path,profile):
    a,b=tmp_path/'full',tmp_path/'resume'
    run(args_for(a,profile,3))
    run(args_for(b,profile,2))
    run(args_for(b,profile,3,('--resume',str(b/'latest.pt'))))
    sa=torch.load(a/'latest.pt',weights_only=True)
    sb=torch.load(b/'latest.pt',weights_only=True)
    assert sa['model_sha256']==sb['model_sha256']
    for k,v in sa['model'].items():
        torch.testing.assert_close(v,sb['model'][k],rtol=0,atol=0)
    for idx, state in sa['optimizer']['state'].items():
        for k,v in state.items():
            torch.testing.assert_close(v,sb['optimizer']['state'][idx][k],rtol=0,atol=0)
    assert torch.equal(sa['rng']['torch'],sb['rng']['torch'])
    assert sb['progress']['updates']==3
    args=args_for(tmp_path/'eval','raptor' if profile=='l2f' else 'l2f')
    args.checkpoint=b/'latest.pt'
    report=evaluate_checkpoint(args)
    assert report['reference_protocol']==profile  # Stored protocol wins over CLI defaults.
    assert report['scenario_count']==6  # independent EVAL count, not 4*2 or 2*2
    expected=json.loads((b/'evaluation.jsonl').read_text().splitlines()[-1])
    assert report['task_objective']==expected['task_objective']
    assert not any(k.startswith(('raptor_' if profile=='l2f' else 'l2f_')) for k in report)
    with pytest.raises(ValueError,match='configuration'):
        run(args_for(b,profile,4,('--resume',str(b/'latest.pt'),'--scenarios','3')))


@pytest.mark.parametrize('entry',['resume','init-checkpoint','evaluate'])
def test_old_checkpoint_cannot_silently_change_motor_meaning(tmp_path,entry):
    old=tmp_path/'old.pt'
    torch.save({'schema':'response-actor-only-exact-v2','architecture':'response-conditioned-motor-policy-v2-actor-only'},old)
    if entry=='evaluate':
        args=args_for(tmp_path/'eval');args.checkpoint=old
        with pytest.raises(ValueError,match='motor semantics'):
            evaluate_checkpoint(args)
    else:
        args=args_for(tmp_path/entry,extra=('--'+entry,str(old)))
        with pytest.raises(ValueError,match='motor semantics'):
            run(args)


def test_environment_contract_tampering_rejected(tmp_path):
    run(args_for(tmp_path/'run',updates=0))
    value=torch.load(tmp_path/'run/latest.pt',weights_only=True)
    value['binding']['protocol']['environment']['action_convention']='hover-delta'
    with pytest.raises(ValueError,match='contract mismatch'):
        require_reference_checkpoint(value)


def test_numeric_failure_rolls_back_model_and_adam(tmp_path,monkeypatch):
    run(args_for(tmp_path/'init',updates=0))
    baseline=torch.load(tmp_path/'init/latest.pt',weights_only=True)
    original=torch.optim.Adam.step
    def corrupt(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        self.param_groups[0]['params'][0].data.fill_(float('nan'))
        return result
    monkeypatch.setattr(torch.optim.Adam,'step',corrupt)
    with pytest.raises(FloatingPointError):
        run(args_for(tmp_path/'failed',updates=1))
    failure=torch.load(tmp_path/'failed/failure.pt',weights_only=True)
    assert failure['model_sha256']==baseline['model_sha256']
    assert failure['optimizer']['state']=={}
    assert failure['progress']['updates']==0
    assert failure['progress']['status']=='failed'


def test_deprecated_profiles_and_migration_fail_explicitly():
    for flag,value in [('--scenario-mode','physical-fit'),('--scenario-mode','fixed-airframe'),('--migrate-checkpoint','old.pt')]:
        with pytest.raises(SystemExit):
            parse_args([flag,value])

"""Evidence must contain the actual Actor/Adam update, and support read-only replay."""
from dataclasses import fields
import json
from pathlib import Path

import pytest
import torch

from tools.train_response_control import parse_args
from response_policy import ResponsePolicyConfig
from response_task import TaskLossConfig
from response_training import train
from response_audit import AuditConfig
from tools.replay_response_update import replay, compare_values


def args_for(path, updates=2, extra=()):
    return parse_args(['--device','cpu','--dtype','float64','--scenarios','32',
        '--train-sampling','coverage128','--group-max-groups','128','--group-min-scenarios','1',
        '--eval-scenarios','4','--horizon','4','--memory-dim','4','--group-clip-norm','.01',
        '--updates',str(updates),'--development-every','1','--checkpoint-every','1',
        '--work-dir',str(path),'--max-seconds','120',*extra])


def run(args):
    return train(args, ResponsePolicyConfig(**{f.name:getattr(args,f.name) for f in fields(ResponsePolicyConfig)}),
                 TaskLossConfig(**{f.name:getattr(args,f.name) for f in fields(TaskLossConfig)}))


def test_capsules_replay_rollout_gradient_and_adam_exactly(tmp_path):
    run(args_for(tmp_path))
    capsules = sorted((tmp_path/'audit/updates').glob('*.pt'))
    assert len(capsules) == 2
    assert len(list((tmp_path/'checkpoints').glob('*.pt'))) == 3
    for path in capsules:
        saved = path.read_bytes()
        value = torch.load(path,weights_only=True)
        assert value['groups']['layout']['group_ids'] == list(range(128))
        assert value['groups']['gradient']['vjp_calls'] == 8
        assert value['groups']['gradient']['values'][:,2].max() <= 1.
        for mode in ('adam','full'):
            result = replay(path, mode)
            assert result['passed'], result
            assert result['parameter_changes'] == value['parameter_changes']
        assert path.read_bytes() == saved
    value = torch.load(capsules[-1],weights_only=True)
    log = json.loads((tmp_path/'history.jsonl').read_text().splitlines()[-1])
    assert log['parameter_changes']['l2'] > 0
    assert log['parameter_changes'] == value['parameter_changes']


def test_same_source_resume_matches_continuous_after_audit(tmp_path):
    a,b = tmp_path/'a',tmp_path/'b'
    run(args_for(a,2));run(args_for(b,1));run(args_for(b,2,('--resume',str(b/'latest.pt'))))
    first,second = [torch.load(p/'latest.pt',weights_only=True) for p in (a,b)]
    for key in ('model','optimizer','rng'):
        assert compare_values(first[key],second[key])[0]


def test_nonfinite_attempt_is_saved_before_rollback(tmp_path,monkeypatch):
    original = torch.optim.Adam.step
    def corrupt(self, *args, **kwargs):
        value=original(self,*args,**kwargs)
        self.param_groups[0]['params'][0].data.fill_(float('nan'))
        return value
    monkeypatch.setattr(torch.optim.Adam, 'step', corrupt)
    with pytest.raises(FloatingPointError):run(args_for(tmp_path,1))
    paths=list((tmp_path/'audit/updates').glob('*.pt'));assert len(paths)==1
    value=torch.load(paths[0],weights_only=True)
    assert value['failure'].startswith('FloatingPointError')
    assert all(torch.isfinite(t).all() for t in value['before']['model'].values())
    assert any(not torch.isfinite(t).all() for t in value['after']['model'].values())
    failure=torch.load(tmp_path/'failure.pt',weights_only=True)
    assert compare_values(failure['model'],value['before']['model'])[0]
    assert compare_values(failure['optimizer'],value['before']['optimizer'])[0]
    assert compare_values(failure['rng'],value['before']['rng'])[0]
    assert list((tmp_path/'audit/events').glob('*/event.json'))


def test_evidence_retention_bounded_and_evaluation_pins_context(tmp_path):
    run(args_for(tmp_path,4,('--audit-history','2','--audit-max-events','1','--audit-raw-factor','1.01')))
    assert len(list((tmp_path/'audit/updates').glob('*.pt'))) == 2
    events=list((tmp_path/'audit/events').iterdir());assert len(events)==1
    assert len(list(events[0].glob('*.pt'))) <= 3  # pin before ring eviction
    manifest=json.loads((tmp_path/'audit/manifest.json').read_text())
    assert manifest['effective_history'] == 2
    entries=[json.loads(s) for s in (tmp_path/'audit/events.jsonl').read_text().splitlines()]
    assert any(e['action']=='retention_eviction' for e in entries)


@pytest.mark.parametrize('name,value',[('history',0),('max_events',0),('raw_factor',1),('step_factor',float('nan'))])
def test_invalid_audit_configs_fail(name,value):
    with pytest.raises(ValueError):AuditConfig(**{name:value})


def test_fixed_eval_regression_pins_all_recent_updates_without_veto(tmp_path):
    from response_audit import UpdateAudit
    from response_training import atomic_json,atomic_torch,SOURCE_FILES,ROOT
    run(args_for(tmp_path,2,('--audit-raw-factor','1000000000')))
    ckpt=torch.load(tmp_path/'latest.pt',weights_only=True)
    progress=ckpt['progress']
    observer=UpdateAudit(tmp_path,AuditConfig(),evaluation_interval=1,binding=ckpt['binding'],
                         source_root=ROOT,source_files=SOURCE_FILES,
                         torch_writer=atomic_torch,json_writer=atomic_json)
    before=(tmp_path/'latest.pt').read_bytes()
    observer.evaluation({'task_objective':3.},1.,progress)
    events=list((tmp_path/'audit/events').glob('*/event.json'))
    assert len(events)==1
    event=json.loads(events[0].read_text())
    assert event['reasons']==['fixed_eval_regression']
    assert len(event['capsules'])==2
    assert (tmp_path/'latest.pt').read_bytes()==before


def test_finite_large_actual_parameter_jump_is_archived_not_rejected(tmp_path,monkeypatch):
    original=torch.optim.Adam.step
    counter=[0]
    def jump(self,*a,**kw):
        value=original(self,*a,**kw)
        counter[0]+=1
        if counter[0]==2:
            self.param_groups[0]['params'][0].data.add_(.1)
        return value
    monkeypatch.setattr(torch.optim.Adam,'step',jump)
    run(args_for(tmp_path,2,('--audit-raw-factor','1000000000')))
    entries=[json.loads(x) for x in (tmp_path/'audit/events.jsonl').read_text().splitlines()]
    assert any('actual_adam_step_jump' in e.get('reasons',[]) for e in entries)
    saved=torch.load(tmp_path/'latest.pt',weights_only=True)
    assert saved['next_update']==2  # Evidence observer did not veto the finite step.

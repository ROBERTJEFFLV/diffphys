"""Small production updates exercise metric state, isolation, and exact resume."""
import copy
from dataclasses import fields
import json
from pathlib import Path
import sys
import torch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from response_policy import ResponsePolicyConfig
from response_task import TaskLossConfig
import response_training as training
from tools.train_response_control import parse_args


def args(path,updates=0,extra=()):
    return parse_args(['--device','cpu','--dtype','float64','--scenarios','1',
                       '--eval-scenarios','2','--horizon','8','--window-steps','4',
                       '--hidden-dim','8','--memory-dim','8','--updates',str(updates),
                       '--max-seconds','60','--work-dir',str(path),
                       '--contraction-weight','.1','--contraction-steps','3',
                       '--contraction-samples','2','--contraction-directions','2',
                       '--contraction-hidden-dim','8','--contraction-rank','2',*extra])


def run(options):
    pc=ResponsePolicyConfig(**{f.name:getattr(options,f.name) for f in fields(ResponsePolicyConfig)})
    lc=TaskLossConfig(**{f.name:getattr(options,f.name) for f in fields(TaskLossConfig)})
    return training.train(options,pc,lc)


def load(path):return torch.load(path,weights_only=True)


@pytest.mark.parametrize('guidance', [(), (
    '--contraction-context','physics','--contraction-fusion','film_gated',
    '--contraction-prefix-weight','.1','--contraction-tail-fraction','.5',
    '--contraction-actor-max-ratio','.5','--contraction-metric-gradient-scale','1',
    '--contraction-sampling','mass_quantiles',
)])
def test_both_models_train_resume_and_export_actor_only(tmp_path, guidance):
    paths=[tmp_path/n for n in ('init','a','b')]
    run(args(paths[0],extra=guidance));run(args(paths[1],updates=2,extra=guidance));run(args(paths[2],updates=1,extra=guidance))
    run(args(paths[2],updates=2,extra=(*guidance,'--resume',str(paths[2]/'latest.pt'))))
    initial,first,second=[load(p/'latest.pt') for p in paths]
    assert first['model_sha256']==second['model_sha256']!=initial['model_sha256']
    assert first['contraction']['model_sha256']==second['contraction']['model_sha256']!=initial['contraction']['model_sha256']
    for group in ('optimizer','contraction'):
        x=first[group] if group=='optimizer' else first[group]['optimizer']
        y=second[group] if group=='optimizer' else second[group]['optimizer']
        for idx,state in x['state'].items():
            for key,value in state.items():torch.testing.assert_close(value,y['state'][idx][key],rtol=0,atol=0)
    assert torch.equal(first['rng']['torch'],second['rng']['torch'])
    assert not any('metric' in k or 'contraction' in k for k in first['model'])
    policy,_=training.load_policy_checkpoint(paths[1]/'latest.pt','cpu',torch.float64)
    assert training.model_hash(policy)==first['model_sha256']
    rows=[json.loads(l) for l in (paths[1]/'history.jsonl').read_text().splitlines()]
    assert all(r['contraction']['actor_gradient_norm']>0 and r['contraction']['metric_gradient_norm']>0 for r in rows)
    ev=args(tmp_path/'eval');ev.checkpoint=paths[1]/'latest.pt'
    report=training.evaluate_checkpoint(ev)
    assert report['contraction']['certified'] is False
    altered=args(paths[1],updates=3,extra=(*guidance,'--resume',str(paths[1]/'latest.pt'),'--contraction-rate','.2'))
    with pytest.raises(ValueError,match='configuration'):run(altered)


def test_v1_metric_can_be_rescored_but_not_resumed(tmp_path):
    from response_contraction import LEGACY_CONTRACTION_VERSION
    path=tmp_path/'legacy';run(args(path))
    saved=load(path/'latest.pt')
    saved['binding']['contraction']={key:value for key,value in saved['binding']['contraction'].items()
        if key in ('version','weight','steps','samples','directions','rate','hidden_dim','rank','metric_min','metric_max')}
    saved['binding']['contraction']['version']=LEGACY_CONTRACTION_VERSION
    old_path=path/'v1.pt';torch.save(saved,old_path)
    ev=args(tmp_path/'rescore');ev.checkpoint=old_path
    report=training.evaluate_checkpoint(ev)
    assert report['contraction']['checkpoint_metric_version']==LEGACY_CONTRACTION_VERSION
    assert report['contraction']['context']=='legacy'
    assert report['contraction']['fusion']=='concat'
    with pytest.raises(ValueError,match='configuration'):
        run(args(path,updates=1,extra=('--resume',str(old_path))))


def test_zero_weight_preserves_initial_actor_and_training_stream(tmp_path):
    on,off=tmp_path/'on',tmp_path/'off'
    run(args(on));run(args(off,extra=('--contraction-weight','0')))
    a,b=load(on/'latest.pt'),load(off/'latest.pt')
    assert a['model_sha256']==b['model_sha256']
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    assert b['contraction'] is None
    base=tmp_path/'base';run(args(base,updates=1,extra=('--contraction-weight','0')))
    row=json.loads((base/'history.jsonl').read_text().splitlines()[0])
    assert row['contraction'] is None


def test_auxiliary_failure_rolls_back_both_models_and_optimizer_states(tmp_path,monkeypatch):
    path=tmp_path/'run'
    def bad_step(policy,sim,metric,*a,**kw):
        with torch.no_grad():
            next(policy.parameters()).add_(2)
            next(metric.parameters()).add_(3)
        torch.rand(5)
        raise FloatingPointError('injected contraction failure')
    monkeypatch.setattr(training,'_add_contraction_gradient',bad_step)
    with pytest.raises(FloatingPointError,match='injected'):run(args(path,updates=1))
    initial,failed=load(path/'latest.pt'),load(path/'failure.pt')
    assert initial['model_sha256']==failed['model_sha256']
    assert initial['contraction']['model_sha256']==failed['contraction']['model_sha256']
    assert initial['optimizer']==failed['optimizer']
    assert initial['contraction']['optimizer']==failed['contraction']['optimizer']
    assert torch.equal(initial['rng']['torch'],failed['rng']['torch'])
    assert failed['progress']['updates']==0


def test_full_and_reverse_windowed_joint_update_match(tmp_path):
    a,b=tmp_path/'full',tmp_path/'windows'
    run(args(a,updates=1));run(args(b,updates=1,extra=('--backprop-mode','windowed')))
    x,y=load(a/'latest.pt'),load(b/'latest.pt')
    for key in x['model']:torch.testing.assert_close(x['model'][key],y['model'][key],rtol=1e-10,atol=1e-10)
    for key in x['contraction']['model']:torch.testing.assert_close(x['contraction']['model'][key],y['contraction']['model'][key],rtol=0,atol=0)

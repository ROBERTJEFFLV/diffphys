"""Noise-aware exact resume, transactional Adam and strict checkpoint contracts."""
from dataclasses import fields
import json
import pytest
import torch

from tools.train_response_control import parse_args
from response_policy import ResponsePolicyConfig
from response_task import TaskLossConfig
from response_training import train,evaluate_checkpoint,require_reference_checkpoint


def args_for(path,updates=2,extra=()):
    return parse_args(['--device','cpu','--dtype','float64','--scenarios','8','--eval-scenarios','4',
                       '--horizon','8','--development-every','1','--checkpoint-every','1',
                       '--updates',str(updates),'--max-seconds','60','--memory-dim','8',
                       '--work-dir',str(path),*extra])


def run(args):
    pc=ResponsePolicyConfig(**{f.name:getattr(args,f.name) for f in fields(ResponsePolicyConfig)})
    lc=TaskLossConfig(**{f.name:getattr(args,f.name) for f in fields(TaskLossConfig)})
    return train(args,pc,lc)


def test_noise_checkpoint_resume_matches_uninterrupted_and_fixed_eval(tmp_path):
    a,b=tmp_path/'full',tmp_path/'resume'
    run(args_for(a,3));run(args_for(b,2));run(args_for(b,3,('--resume',str(b/'latest.pt'))))
    sa=torch.load(a/'latest.pt',weights_only=True);sb=torch.load(b/'latest.pt',weights_only=True)
    assert sa['model_sha256']==sb['model_sha256']
    for i,state in sa['optimizer']['state'].items():
        for k,v in state.items():torch.testing.assert_close(v,sb['optimizer']['state'][i][k],rtol=0,atol=0)
    assert torch.equal(sa['rng']['torch'],sb['rng']['torch'])
    args=args_for(tmp_path/'eval',extra=('--disturbance-budget','0'));args.checkpoint=b/'latest.pt'
    actual=evaluate_checkpoint(args)
    expected=json.loads((b/'evaluation.jsonl').read_text().splitlines()[-1])
    assert actual['task_objective']==expected['task_objective']
    assert actual['disturbances']==expected['disturbances']  # Stored settings win.
    assert actual['scenario_count']==8
    assert actual['disturbances']['maximum_total_fraction']<=.1
    for override in [('--disturbance-budget','.05'),('--disturbance-pool','0','0','0','0','1'),
                     ('--time-decay','0'),('--terminal-cost','100')]:
        with pytest.raises(ValueError,match='configuration'):
            run(args_for(b,4,('--resume',str(b/'latest.pt'),*override)))


@pytest.mark.parametrize('entry',['resume','init-checkpoint','evaluate'])
def test_old_environment_checkpoints_are_not_silently_reinterpreted(tmp_path,entry):
    path=tmp_path/'old.pt'
    torch.save({'schema':'response-actor-only-reference-v3','architecture':'gru16-direct-readout-absolute-motor-policy-v4'},path)
    if entry=='evaluate':
        args=args_for(tmp_path/'eval');args.checkpoint=path;call=lambda:evaluate_checkpoint(args)
    else:call=lambda:run(args_for(tmp_path/entry,extra=('--'+entry,str(path))))
    with pytest.raises(ValueError,match='motor semantics'):call()


def test_numeric_failure_rolls_back_model_adam_and_sampling_index(tmp_path,monkeypatch):
    run(args_for(tmp_path/'init',0));baseline=torch.load(tmp_path/'init/latest.pt',weights_only=True)
    original=torch.optim.Adam.step
    def corrupt(self,*args,**kwargs):
        output=original(self,*args,**kwargs);self.param_groups[0]['params'][0].data.fill_(float('nan'));return output
    monkeypatch.setattr(torch.optim.Adam,'step',corrupt)
    with pytest.raises(FloatingPointError):run(args_for(tmp_path/'failed',1))
    failure=torch.load(tmp_path/'failed/failure.pt',weights_only=True)
    assert failure['model_sha256']==baseline['model_sha256']
    assert failure['optimizer']['state']=={} and failure['progress']['updates']==0
    assert torch.equal(failure['rng']['torch'],baseline['rng']['torch'])


def test_noise_and_environment_contract_tampering_rejected(tmp_path):
    run(args_for(tmp_path/'init',0));saved=torch.load(tmp_path/'init/latest.pt',weights_only=True)
    saved['binding']['protocol']['environment']['action_convention']='hover-delta'
    with pytest.raises(ValueError,match='contract mismatch'):require_reference_checkpoint(saved)


@pytest.mark.parametrize('flags',[
    ['--scenario-mode','l2f'],['--backprop-mode','windowed'],['--window-steps','50'],
    ['--agc','.01'],['--action-rate','1'],['--group-gru-vmap-mode','native'],
    ['--no-group-balance'],['--mode','profile']])
def test_removed_paths_are_rejected_instead_of_silently_ignored(flags):
    with pytest.raises(SystemExit):parse_args(flags)


def test_checkpoint_rejects_changed_noise_implementation(tmp_path):
    run(args_for(tmp_path/'init',0))
    saved=torch.load(tmp_path/'init/latest.pt',weights_only=True)
    saved['binding']['protocol']['noise_source_sha256']='different-noise-code'
    with pytest.raises(ValueError,match='contract mismatch'):
        require_reference_checkpoint(saved)


def test_only_retained_config_matches_current_cli():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    configs=list((root/'configs').glob('*.args'))
    assert [p.name for p in configs]==['response_raptor_multi_airframe.args']
    args=parse_args(['@'+str(configs[0])])
    assert args.scenarios==128 and args.eval_scenarios==128 and args.horizon==500
    assert args.time_decay==1 and args.disturbance_budget==.1
    assert tuple(args.disturbance_pool)==(1,1,1,1,1)


def test_explicit_parent_actor_weight_import_uses_fresh_new_protocol(tmp_path):
    run(args_for(tmp_path/'source',1))
    saved=torch.load(tmp_path/'source/latest.pt',weights_only=True)
    saved['schema']='response-actor-only-reference-v3'
    saved['policy_config']['action_rate']=0.
    saved['binding']['protocol']['environment']['version']='old-environment'
    saved['binding']['protocol']['environment_source_sha256']='old-source'
    saved['binding']['protocol'].pop('noise_source_sha256')
    parent=tmp_path/'parent.pt';torch.save(saved,parent)
    run(args_for(tmp_path/'import',0,('--init-checkpoint',str(parent))))
    imported=torch.load(tmp_path/'import/latest.pt',weights_only=True)
    assert imported['model_sha256']==saved['model_sha256']
    assert imported['optimizer']['state']=={} and imported['progress']['updates']==0
    assert imported['schema']!='response-actor-only-reference-v3'
    assert imported['progress']['initialization']['weights_only']
    assert imported['binding']['protocol']['disturbances']['budget']==.1

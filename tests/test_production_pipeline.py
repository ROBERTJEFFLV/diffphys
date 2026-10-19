"""P must be the production path with exact ownership and rollback behavior."""
from __future__ import annotations

from dataclasses import fields
import gc
import json
from pathlib import Path

import pytest
import torch

from diffphys.response_policy import ResponsePolicyConfig
from diffphys.response_task import TaskLossConfig
from diffphys.response_training import train
from loss_fixtures import parse_loss_args


def args_for(path, updates=2, extra=()):
    return parse_loss_args(['--device','cpu','--scenarios','8','--eval-scenarios','4',
        '--horizon','3','--updates',str(updates),'--memory-dim','8',
        '--development-every','1','--checkpoint-every','1','--max-seconds','60',
        '--work-dir',str(path),*extra])


def run(args):
    return train(args,ResponsePolicyConfig(**{f.name:getattr(args,f.name) for f in fields(ResponsePolicyConfig)}),
                 TaskLossConfig(**{f.name:getattr(args,f.name) for f in fields(TaskLossConfig)}))


def test_production_uses_prefetch_gc_and_source_bound_sampler(tmp_path, monkeypatch):
    import diffphys.response_training as owner
    original=owner.collect_rollout
    flags=[]
    def observe(*a,**kw):
        flags.append(gc.isenabled())
        return original(*a,**kw)
    monkeypatch.setattr(owner,'collect_rollout',observe)
    run(args_for(tmp_path))
    assert flags==[False,False]
    saved=torch.load(tmp_path/'latest.pt',weights_only=True)
    assert saved['progress']['sampling_index']==saved['next_update']==2
    assert saved['binding']['execution']['cpu_sampling']=='spawn-one-batch'
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['cleanup']['worker_closed']
    assert summary['cleanup']['gc_enabled']==summary['cleanup']['entry_gc_enabled']
    assert not Path(f'/proc/{summary["cleanup"]["worker_pid"]}').exists()


@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('stage',['forward','backward','interrupt'])
def test_production_failure_restores_worker_gc_hooks_and_actor_adam_rng(tmp_path,monkeypatch,enabled,stage):
    import diffphys.response_training as owner
    from diffphys.response_grad_probe import GroupGradientProbe
    run(args_for(tmp_path/'initial',0))
    before=torch.load(tmp_path/'initial/latest.pt',weights_only=True)
    probes=[]
    constructor=GroupGradientProbe.__init__
    def observe(self,*a,**kw):
        constructor(self,*a,**kw);probes.append(self)
    monkeypatch.setattr(GroupGradientProbe,'__init__',observe)
    def fail(*a,**kw):
        assert not gc.isenabled()
        if stage=='interrupt':raise KeyboardInterrupt('injected')
        raise RuntimeError('injected')
    monkeypatch.setattr(owner,'backward_actor' if stage=='backward' else 'collect_rollout',fail)
    entry=gc.isenabled()
    try:
        (gc.enable if enabled else gc.disable)()
        with pytest.raises((RuntimeError,KeyboardInterrupt),match='injected'):
            run(args_for(tmp_path/'failed',1))
        assert gc.isenabled()==enabled
        assert all(p._closed and not p._tensor_handles and not p._payloads for p in probes)
        after=torch.load(tmp_path/'failed/failure.pt',weights_only=True)
        assert after['model_sha256']==before['model_sha256']
        assert after['optimizer']['state']==before['optimizer']['state']=={}
        assert torch.equal(after['rng']['torch'],before['rng']['torch'])
        assert after['next_update']==after['progress']['sampling_index']==0
        summary=json.loads((tmp_path/'failed/summary.json').read_text())
        assert summary['cleanup']['worker_closed']
        assert not Path(f'/proc/{summary["cleanup"]["worker_pid"]}').exists()
    finally:
        (gc.enable if entry else gc.disable)()


def test_removed_rotation_selector_is_rejected():
    with pytest.raises(SystemExit):parse_loss_args(['--rotation-backend','compile'])


def test_resume_in_a_fresh_process_matches_continuous_production(tmp_path):
    import subprocess
    import sys
    from tools.replay_response_update import compare_values
    full,split=tmp_path/'full',tmp_path/'split'
    def launch(work,updates,resume=False):
        command=[sys.executable,'tools/train_response_control.py','--device','cpu',
                 '--scenarios','8','--eval-scenarios','4','--horizon','3','--memory-dim','8',
                 '--development-every','1','--checkpoint-every','1','--updates',str(updates),
                 '--epsilon-p','.01','--epsilon-a','.01','--lambda-R','25',
                 '--work-dir',str(work)]
        if resume:command+=['--resume',str(work/'latest.pt')]
        result=subprocess.run(command,capture_output=True,text=True,timeout=45)
        assert result.returncode==0,result.stderr
    launch(full,3);launch(split,1);launch(split,3,True)
    a,b=[torch.load(p/'latest.pt',weights_only=True) for p in [full,split]]
    for field in ['model','optimizer','rng','next_update']:
        assert compare_values(a[field],b[field])[0],field
    assert b['progress']['sampling_index']==3
    rows=[json.loads(x) for x in (split/'history.jsonl').read_text().splitlines()]
    assert [r['update'] for r in rows]==[1,2,3]
    assert len({tuple(r['train_seeds']) for r in rows})==3


def test_sigint_waits_for_worker_and_stops_at_a_committed_update(tmp_path):
    import signal
    import subprocess
    import sys
    import time
    with (tmp_path/'process.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'tools/train_response_control.py','--device','cpu',
            '--scenarios','8','--eval-scenarios','4','--horizon','3','--memory-dim','8',
            '--updates','20','--epsilon-p','.01','--epsilon-a','.01','--lambda-R','25',
            '--work-dir',str(tmp_path/'run')],stdout=log,stderr=log)
        try:
            deadline=time.monotonic()+30
            while not (tmp_path/'run/history.jsonl').exists() and process.poll() is None:
                assert time.monotonic()<deadline
                time.sleep(.01)
            assert process.poll() is None
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=30)==0
        finally:
            if process.poll() is None:process.kill();process.wait()
    value=torch.load(tmp_path/'run/latest.pt',weights_only=True)
    summary=json.loads((tmp_path/'run/summary.json').read_text())
    assert value['progress']['status']=='interrupted'
    assert 0<value['next_update']<20
    assert value['next_update']==value['progress']['sampling_index']
    assert summary['cleanup']['worker_closed'] and summary['cleanup']['closed_probes']
    assert summary['cleanup']['gc_enabled']==summary['cleanup']['entry_gc_enabled']
    assert not Path(f'/proc/{summary["cleanup"]["worker_pid"]}').exists()


def test_replay_byte_gate_distinguishes_signed_zero():
    from tools.replay_response_update import compare_values
    a,b=torch.tensor([0.]),torch.tensor([-0.])
    assert torch.equal(a,b)
    assert not compare_values(a,b)[0]

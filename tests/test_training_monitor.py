"""Read-only monitor regressions. No rollout or trained checkpoint is needed."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from tools.response_monitor import (JsonlTail, TrainingLogs, discover_replays,
                                    envelope, launch_replay, metric_row, replay_entry)

ROOT = Path(__file__).resolve().parents[1]


def append(path, update, **extra):
    with path.open('ab') as out:
        out.write((json.dumps(dict(update=update, task_objective=10/(update+1), **extra))+'\n').encode())


def test_partial_record_waits_for_newline_and_unchanged_poll_does_no_read(tmp_path):
    path=tmp_path/'history.jsonl';tail=JsonlTail(path)
    assert not tail.poll() and not tail.rows
    path.write_bytes(b'{"update":1,"position_rms":0.')
    tail.poll();assert not tail.rows
    with path.open('ab') as out:out.write(b'25}\n')
    tail.poll();assert tail.latest['position_rms']==.25
    assert tail.latest['update']==1 and not tail.pending
    with patch.object(Path,'open',side_effect=AssertionError('unchanged file read')):
        assert not tail.poll()
    assert tail.last_read_bytes==0


def test_rotated_log_and_same_inode_rewrite_drop_stale_future(tmp_path):
    path=tmp_path/'history.jsonl'
    for i in range(3):append(path,i)
    tail=JsonlTail(path);tail.poll()
    other=tmp_path/'tmp';other.write_text('{"update":0}\n')
    other.replace(path);tail.poll();assert [r['update'] for r in tail.rows]==[0]
    path.write_text('{"update":1,"task_objective":123456789}\n')
    tail.poll();assert tail.latest['update']==1
    path.write_text('{"update":0,"task_objective":1234567890123456789}\n')
    tail.poll();assert [r['update'] for r in tail.rows]==[0]


def test_append_rollback_duplicates_and_invalid_rows(tmp_path):
    path=tmp_path/'history.jsonl';append(path,5)
    tail=JsonlTail(path);tail.poll()
    with path.open('ab') as out:out.write(b'{"update":5,"task_objective":9}\nnot-json\n[]\n{"update":true}\n')
    tail.poll();assert len(tail.rows)==1 and tail.latest['task_objective']==9
    assert tail.invalid_lines==3
    append(path,2);tail.poll();assert [r['update'] for r in tail.rows]==[2]
    row=metric_row({'update':1,'cuda_peak_bytes':float('nan'),'position_rms':float('inf'), 'huge':[0]*1000})
    assert row=={'update':1}


def test_bounded_tail_does_not_read_checkpoint_or_rewrite_log(tmp_path):
    path=tmp_path/'history.jsonl'
    for i in range(1000):append(path,i,position_rms=i/10)
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'latest.pt').write_bytes(b'not a checkpoint')
    tail=JsonlTail(path,max_rows=4,max_bytes=1024)
    tail.poll()
    assert len(tail.rows)==4 and tail.latest['update']==999
    assert tail.last_read_bytes<=1024 and tail.prefix_skipped
    assert before==hashlib.sha256(path.read_bytes()).hexdigest()
    assert (tmp_path/'latest.pt').read_bytes()==b'not a checkpoint'
    for i in range(1000,1030):append(path,i)
    for _ in range(4):tail.poll()
    assert tail.latest['update']==1029 and len(tail.rows)==4


def test_oversize_partial_line_memory_is_bounded(tmp_path):
    path=tmp_path/'history.jsonl';path.write_bytes(b'x'*200)
    tail=JsonlTail(path,max_bytes=64,max_line_bytes=32);tail.poll()
    assert len(tail.pending)<=32
    with path.open('ab') as out:out.write(b'\n{"update":7}\n')
    tail.poll();assert tail.latest['update']==7


def test_train_and_eval_updates_are_not_falsely_aligned_or_liveness_invented(tmp_path):
    append(tmp_path/'history.jsonl',12,update_seconds=2.5)
    append(tmp_path/'evaluation.jsonl',10,raptor_share_terminated=.2)
    (tmp_path/'summary.json').write_text(json.dumps(dict(updates=10,status='time_budget')))
    logs=TrainingLogs(tmp_path);logs.poll()
    assert logs.train.latest['update']==12 and logs.eval.latest['update']==10
    assert 'resumed' in logs.saved_status()
    assert logs.age(now=logs.train.mtime+120)==120
    assert 'status' not in logs.train.latest


def test_plot_envelope_preserves_isolated_gradient_spikes():
    points=[(i,1.) for i in range(2000)];points[511]=(511,1e10);points[1001]=(1001,-30.)
    reduced=envelope(points,100)
    assert points[511] in reduced and points[1001] in reduced
    assert len(reduced)<=202 and reduced[0]==points[0] and reduced[-1]==points[-1]


def make_replay(path):
    import numpy as np
    path.parent.mkdir(parents=True,exist_ok=True)
    scene=dict(scene_id=254,scene_index=0,mass_kg=.027,arm_length_m=.04,
               thrust_to_weight=2.,torque_to_inertia=100.,completed=True,
               failure_reason=None,failure_step=None)
    meta=dict(dt=.01,duration_seconds=.1,scenes=[scene],checkpoint_update=50,
              model_sha256='saved-model-not-live',arena_side=4.,target_margin=.5,
              force_period_seconds=.1,target_period_seconds=.1,force_fraction=.2,pulse_seconds=.1)
    path.write_text(json.dumps(meta))
    position=np.zeros((11,1,3));position[:,:,2]=2
    orientation=np.zeros((11,1,4));orientation[:,:,0]=1
    np.savez(path.with_suffix('.npz'),position=position,velocity=np.zeros_like(position),
        omega=np.zeros_like(position),orientation=orientation,targets=np.array([[0.,0.,2.]]),
        force_world=np.zeros((10,1,3)),points_body=np.zeros((1,1,3)),
        surface_half_extents=np.ones((1,3))*.025,
        rotor_positions=np.array([[[.028,-.028,0],[-.028,-.028,0],[-.028,.028,0],[.028,.028,0]]]))
    return path


def test_replay_scan_is_shallow_metadata_only_and_manual_launch_is_cpu(tmp_path):
    path=make_replay(tmp_path/'eval1/playback/playlist.json')
    ignored=make_replay(tmp_path/'nested/too/deep/playback/playlist.json')
    files,errors=discover_replays(tmp_path)
    assert [r.path for r in files]==[path.resolve()] and not errors
    assert replay_entry(path).update==50
    with patch('tools.response_monitor.subprocess.Popen') as popen:
        launch_replay(path)
        command=popen.call_args.args[0];kw=popen.call_args.kwargs
        assert '--replay' in command and '--run-dir' not in command
        assert '--checkpoint' not in command and not kw['shell']
        assert kw['env']['CUDA_VISIBLE_DEVICES']==''
        assert kw['env']['OPENBLAS_NUM_THREADS']=='1'
    path.with_suffix('.npz').unlink()
    with pytest.raises(ValueError):replay_entry(path)
    assert discover_replays(tmp_path)[0]==[]


def test_monitor_import_and_poll_never_import_torch_or_training(tmp_path):
    append(tmp_path/'history.jsonl',1)
    code='''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in ('torch','response_training','response_task','env_raptor'):
            raise AssertionError('forbidden viewer import: '+name)
sys.meta_path.insert(0,Block())
from tools.response_monitor import TrainingLogs
from tools.monitor_response_training import parse_args
from tools.play_response_long import Playback
logs=TrainingLogs(sys.argv[1]);logs.poll()
assert logs.train.latest['update']==1
assert 'torch' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_export_selection_tolerates_fewer_completed_flights():
    from tools.play_response_long import select_scenes
    failed=dict(scene_id=1,scene_index=0,mass_kg=.02,completed=False,failure_reason='arena_exit',failure_step=1)
    completed=dict(scene_id=2,scene_index=1,mass_kg=.04,completed=True,failure_reason=None,failure_step=None)
    assert select_scenes(dict(scenes=[failed]),allow_fewer=True)==[failed]
    assert select_scenes(dict(scenes=[failed,completed]),allow_fewer=True)==[failed,completed]
    with pytest.raises(ValueError):select_scenes(dict(scenes=[]),allow_fewer=True)


def test_replay_zero_failure_and_paging_are_safe():
    from tools.play_response_long import Playback, visible_scene_indices
    state=Playback(dict(dt=.01,duration_seconds=60.,scenes=[dict(failure_step=0)]))
    assert state.end_seconds==0 and state.frame==0
    assert visible_scene_indices(list(range(40)),1)==list(range(13,26))
    assert visible_scene_indices(list(range(40)),3)==[39]


@pytest.mark.parametrize('flag,value',[('--poll-seconds','0.1'),('--poll-seconds','nan'),
                                       ('--max-points','0'),('--max-frames','-1')])
def test_monitor_invalid_limits(flag,value):
    from tools.monitor_response_training import parse_args
    with pytest.raises(SystemExit):parse_args(['--run-dir','not-created',flag,value])


@pytest.mark.parametrize('window',['dashboard','replay'])
def test_headless_gui_renders_without_torch_and_does_not_change_run(tmp_path,window):
    try:
        __import__('pygame')
    except ImportError:
        if os.environ.get('REQUIRE_PYGAME')=='1':pytest.fail('CI must install GUI dependencies')
        pytest.skip('optional pygame not installed')
    path=make_replay(tmp_path/'eval/playback/playlist.json')
    train=tmp_path/'train';train.mkdir()
    for i in range(20):append(train/'history.jsonl',i,position_rms=.2,velocity_rms=.3,
                              omega_rms=.4,pre_global_clip_norm=.02,update_seconds=2.5)
    append(train/'evaluation.jsonl',10,position_rms=.22,raptor_share_terminated=.1)
    before={p.name:p.read_bytes() for p in train.iterdir()}
    screenshot=tmp_path/(window+'.png')
    code='''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in ('torch','response_training','env_raptor'):
            raise AssertionError('forbidden render import '+name)
sys.meta_path.insert(0,Block())
if sys.argv[1]=='dashboard':
    from tools.monitor_response_training import main
    main(['--run-dir',sys.argv[2],'--replay',sys.argv[3],'--screenshot',sys.argv[4]])
else:
    from tools.play_response_long import run_player
    run_player(sys.argv[3],screenshot=sys.argv[4])
assert 'torch' not in sys.modules
'''
    env=dict(os.environ,SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1')
    result=subprocess.run([sys.executable,'-c',code,window,str(train),str(path),str(screenshot)],
                          cwd=ROOT,env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert screenshot.stat().st_size>1000
    assert before=={p.name:p.read_bytes() for p in train.iterdir()}
    if os.environ.get('GUI_ARTIFACT_DIR'):
        import shutil
        out=Path(os.environ['GUI_ARTIFACT_DIR']);out.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(screenshot,out/(window+'.png'))

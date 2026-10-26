"""Typed replay must reproduce EVAL and show the commands actually issued."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from env_raptor import RaptorSimulator
from pass_stop import GoalConfig, GoalMotorPolicy, PassStopLossConfig, task_rollout
from response_noise import DisturbanceConfig
from response_policy import ResponsePolicyConfig
from response_training import (DEVELOPMENT_SEEDS, evaluate, load_policy_checkpoint,
                               sample_pool)
from tools.play_response_short import load_replay
from tools.short_eval_replay import capture_short_eval, export_checkpoint


def test_capture_retains_typed_birth_switches_and_failed_row():
    torch.manual_seed(99)
    horizon=12
    policy=GoalMotorPolicy(ResponsePolicyConfig(memory_dim=8)).double()
    class ReplayFixtureSimulator(RaptorSimulator):
        def step(self, state, action):
            result = super().step(state, action)
            # Exercise replay compaction/phase telemetry independently of a
            # trained policy: the new task always replaces initial speed by zero.
            point = result.position_limit[:, None] * result.position.new_tensor((0., .38, 0.))
            step = result.step_index
            offset = torch.where(step == 1, -3., torch.where(step == 2, 0., 3.))
            point = point + result.position.new_tensor((1., 0., 0.)) * (offset*2*result.arm_length)[:, None]
            failed = result.position_limit[:, None] * result.position.new_tensor((2., 0., 0.))
            return replace(result, position=torch.where((result.noise_row == 1)[:, None], failed, point))

    simulator=ReplayFixtureSimulator()
    raw=simulator.reset(2,seed=31,horizon=horizon,dtype=torch.float64,
                        disturbances=DisturbanceConfig.clean())
    mission=GoalConfig()
    config=PassStopLossConfig(.01,.01,25.,mission=asdict(mission))
    commands=[]
    def hook(module,args,out):
        state=out.next_state;stop=state.is_stop
        commands.append((torch.where(stop[:,None],state.stop,state.via).clone(),
                         torch.where(stop[:,None,None],state.stop_rotation,state.via_rotation).clone(),
                         stop.clone()))
    handle=policy.register_forward_hook(hook)
    with torch.no_grad():
        trace=task_rollout(policy,simulator,raw,horizon,config)
    handle.remove()
    expected=evaluate(policy,simulator,raw,horizon,config)
    report,a=capture_short_eval(policy,simulator,raw,horizon,config)
    assert report==expected
    np.testing.assert_array_equal(a['position'][0],trace.initial.position.numpy())
    assert not a['velocity'][0].any()
    np.testing.assert_array_equal(a['position'][1:],trace.positions.numpy())
    np.testing.assert_array_equal(a['valid'],trace.valid.numpy())
    assert a['valid'][:,1].sum()==1
    assert a['command_is_stop'][:,0].any()
    assert not a['command_is_stop'][:,1].any()
    for step,(p,R,stop) in enumerate(commands):
        active=a['valid'][step]
        np.testing.assert_array_equal(a['command_position'][step,active],p.numpy())
        np.testing.assert_array_equal(a['command_rotation'][step,active],R.numpy())
        np.testing.assert_array_equal(a['command_is_stop'][step,active],stop.numpy())
    assert not a['external_force_world'].any()
    assert a['pulse_force_world'].shape==(horizon,2,3)
    assert not a['pulse_force_world'].any()


@pytest.fixture
def typed_recording(tmp_path):
    from test_pass_stop_integration import cli
    folder=tmp_path/'run'
    cli(folder,0)
    value=torch.load(folder/'latest.pt',weights_only=True)
    policy,_=load_policy_checkpoint(folder/'latest.pt',torch.device('cpu'),torch.float32)
    cfg=PassStopLossConfig(**value['binding']['protocol']['loss'])
    horizon=value['binding']['horizon'];proto=value['binding']['protocol']
    initial=sample_pool(proto['eval_scenarios_per_bank'],DEVELOPMENT_SEEDS,horizon=horizon,
        dtype=torch.float32,disturbances=DisturbanceConfig(**proto['disturbances']))
    expected=json.loads(json.dumps(evaluate(policy,RaptorSimulator(),initial,horizon,cfg)))
    result=export_checkpoint(folder/'latest.pt',folder,device='cpu')
    return folder,result,expected


def test_checkpoint_export_preserves_task_and_exact_logged_metrics(typed_recording):
    folder,result,expected=typed_recording
    assert result['logged_metrics_match'] is True
    meta,a=load_replay(result['playlist'])
    assert meta['task']=='pass-stop' and meta['target_world_m'] is None
    assert meta['mission']==expected['loss_config']['mission']
    assert meta['metrics']==expected
    assert a['command_position'].shape==(8,4,3)
    assert a['command_rotation'].shape==(8,4,3,3)
    assert a['command_is_stop'].shape==(8,4)
    assert a['command_phase'].shape==(8,4)
    assert a['command_min_speed_m_s'].shape==(8,4)
    np.testing.assert_array_equal(a['command_is_stop'], a['command_phase']==2)
    np.testing.assert_allclose(a['region_radius_m'], 2*a['arm_length_m'])
    assert np.all(a['command_min_speed_m_s'][a['command_phase']!=2] == 2.)
    np.testing.assert_array_equal(a['position'][0],a['mission_points'][:,0])
    assert not a['external_force_world'].any() and not a['pulse_active'].any()


def test_region_player_shows_minimum_speed_and_separate_center_tolerance(typed_recording):
    from tools.play_response_short import goal_status
    meta, a = load_replay(typed_recording[1]['playlist'])
    a['command_phase'][0,0] = 1
    a['velocity'][0,0] = (0., 4., 0.)
    status = goal_status(meta, a, 0, 0)
    assert status['phase'] == 1
    assert status['region_radius_m'] == pytest.approx(2*a['arm_length_m'][0])
    assert status['center_tolerance_m'] == .1
    assert status['minimum_speed_m_s'] == 2.
    assert status['speed_shortfall_m_s'] == 0.
    a['velocity'][0,0] = (1.9, 0., 0.)
    assert goal_status(meta, a, 0, 0)['speed_shortfall_m_s'] == pytest.approx(.1)


def test_typed_player_is_torch_free(typed_recording,tmp_path):
    pytest.importorskip('pygame')
    result=typed_recording[1];png=tmp_path/'typed.png'
    script='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name in ('torch','pass_stop','env_raptor') or name.startswith('response_'):
            raise AssertionError('Player imported '+name)
sys.meta_path.insert(0,Block())
from tools.play_response_short import run_player
run_player(sys.argv[1],screenshot=sys.argv[2])
'''
    p=subprocess.run([sys.executable,'-c',script,result['playlist'],str(png)],
        cwd=Path(__file__).resolve().parents[1],env={**os.environ,'SDL_VIDEODRIVER':'dummy','CUDA_VISIBLE_DEVICES':''},
        capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr
    assert png.stat().st_size>10000


def test_typed_player_rejects_missing_command_telemetry(typed_recording):
    folder,result,_=typed_recording
    playlist=Path(result['playlist']);meta=json.loads(playlist.read_text())
    path=playlist.parent/meta['trajectory_file']
    with np.load(path,allow_pickle=False) as archive:
        a={name:archive[name] for name in archive.files if name!='command_is_stop'}
    np.savez_compressed(path,**a)
    meta['trajectory_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    playlist.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='PASS/STOP|typed'):
        load_replay(playlist)


@pytest.mark.parametrize('missing', ['command_phase', 'command_min_speed_m_s', 'region_radius_m'])
def test_minimum_speed_replay_requires_complete_region_telemetry(typed_recording, missing):
    playlist = Path(typed_recording[1]['playlist'])
    meta = json.loads(playlist.read_text())
    path = playlist.parent/meta['trajectory_file']
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != missing}
    np.savez_compressed(path, **arrays)
    meta['trajectory_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    playlist.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='region'):
        load_replay(playlist)


def test_historical_region_replay_without_new_telemetry_stays_readable(typed_recording):
    from tools.play_response_short import goal_status, REGION_FIELDS
    playlist = Path(typed_recording[1]['playlist'])
    meta = json.loads(playlist.read_text())
    meta['mission']['version'] = 'pass-region-pose-velocity-v2'
    path = playlist.parent/meta['trajectory_file']
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k not in REGION_FIELDS}
    np.savez_compressed(path, **arrays)
    meta['trajectory_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    playlist.write_text(json.dumps(meta))
    saved, arrays = load_replay(playlist)
    status = goal_status(saved, arrays, 0, 0)
    assert status['minimum_speed_m_s'] is None
    assert status['region_radius_m'] == pytest.approx(2*arrays['arm_length_m'][0])

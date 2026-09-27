"""Replay must preserve scene identity and stop at the actual failure frame."""
import importlib.util

import pytest


def api():
    assert importlib.util.find_spec('tools.play_response_long') is not None, 'switchable long EVAL player is missing'
    from tools import play_response_long
    return play_response_long


def scene(i, mass, completed=True, failure=None):
    return dict(scene_id=i+100, scene_index=i, mass_kg=mass, completed=completed,
                failure_reason=None if completed else 'arena_exit', failure_step=failure)


def test_selection_includes_all_exits_and_spans_completed_vehicle_sizes():
    rows=[scene(0,4),scene(1,.1,False,230),scene(2,1),scene(3,.02),
          scene(4,2),scene(5,5,False,310),scene(6,.5)]
    picked=api().select_scenes({'scenes':rows},completed_count=3)
    assert [r['scene_id'] for r in picked]==[101,105,103,102,100]
    assert len({r['scene_id'] for r in picked})==5
    with pytest.raises(ValueError): api().select_scenes({'scenes':rows},completed_count=6)


def test_switch_seek_and_autoplay_never_play_frozen_failure_padding():
    m=api(); data=dict(dt=.01,duration_seconds=60.,scenes=[scene(0,.1,False,230),scene(1,1)])
    player=m.Playback(data)
    player.seek(50)
    assert player.frame==230 and player.end_seconds==pytest.approx(2.3)
    player.paused=True;player.advance(10)
    assert player.index==0 and player.frame==230
    player.choose(1);assert player.frame==0 and player.end_seconds==60
    player.seek(55);assert player.frame==5500
    player.choose(0);player.paused=False;player.seek(player.end_seconds)
    player.advance(2.1)
    assert player.index==1 and player.frame==0
    player.choose(-1);assert player.index==1


def test_export_keeps_original_ids_and_actual_tensor_rows(tmp_path):
    import json
    import numpy as np
    import torch
    m=api();source=tmp_path/'run';source.mkdir()
    rows=[scene(0,4),scene(1,.1,False,2),scene(2,.02)]
    (source/'summary.json').write_text(json.dumps({'scenes':rows}))
    (source/'manifest.json').write_text(json.dumps({'config':dict(dt=.01,duration_seconds=.03,
        target_period_seconds=.03,force_period_seconds=.03,arena_side=4,target_margin=.5),
        'checkpoint_update':11150,'model_sha256':'frozen-actor'}))
    positions=torch.arange(36,dtype=torch.float32).reshape(4,3,3)
    torch.save(dict(position=positions,orientation=torch.zeros(4,3,4),velocity=positions,
        omega=positions,action=torch.zeros(3,3,4),valid=torch.ones(3,3,dtype=torch.bool),
        force_world=torch.zeros(3,3,3),torque_body=torch.zeros(3,3,3),memory_norm=torch.zeros(3,3)),source/'trajectory.pt')
    torch.save(dict(targets=torch.tensor([[1.,2.,3.]]),points_body=torch.zeros(1,3,3),
                    surface_half_extents=torch.ones(3,3)*.1),source/'schedule.pt')
    torch.save(dict(rotor_positions=torch.zeros(3,4,3),inertia=torch.ones(3,3)),source/'initial_state.pt')
    path=m.export_replay(source,completed_count=2)
    meta=json.loads(path.read_text())
    assert [r['scene_id'] for r in meta['scenes']]==[101,102,100]
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as out:
        assert np.array_equal(out['position'][:,0],positions[:,1].numpy())
        assert np.array_equal(out['position'][:,2],positions[:,0].numpy())
        assert out['position'].shape==(4,3,3)

"""EVAL contract: point-force physics, schedules and recurrent continuity."""
from dataclasses import fields, replace
import importlib.util
import math

import pytest
import torch

from env_raptor import RaptorSimulator
from response_noise import DisturbanceConfig, measured_observation
from response_policy import ResponseMotorPolicy


def api():
    assert importlib.util.find_spec('response_long_eval') is not None, 'long EVAL environment is missing'
    import response_long_eval
    return response_long_eval


def initial(count=3, dtype=torch.float64):
    return RaptorSimulator().reset(count, seed=7, horizon=6000, dtype=dtype,
                                   disturbances=DisturbanceConfig.clean())


def test_eval_retains_three_acquisition_times_and_replaces_training_pulses():
    m = api()
    s = RaptorSimulator().reset(2, seed=7, horizon=20, dtype=torch.float64)
    # Make the replaced training disturbance unmistakable at every transition.
    s = replace(s, pulse_tape=torch.ones(2, 21, 6, dtype=torch.float64),
                pulse_active_tape=torch.ones(2, 21, dtype=torch.bool))
    cfg = m.LongEvalConfig(duration_seconds=.04, target_period_seconds=.02,
        force_period_seconds=.02, pulse_seconds=.01, hover_tail_seconds=.01)
    env = m.LongHoverEval(ResponseMotorPolicy().double(), s, cfg)
    assert env.state.previous_velocity.shape == (2, 3, 3)
    assert not env.state.previous_velocity.any()
    assert not env.state.pulse_tape.any() and not env.state.pulse_active_tape.any()
    seen = [env.state.velocity.clone()]
    for _ in range(4):
        env.step()
        expected = torch.stack((seen[-1], seen[max(0, len(seen)-2)],
                                seen[max(0, len(seen)-3)]), 1)
        torch.testing.assert_close(env.state.previous_velocity, expected, rtol=0, atol=0)
        if env.step_index < cfg.steps:
            assert torch.isfinite(env.observation()).all()
        seen.append(env.state.velocity.clone())


def test_targets_change_at_10_seconds_and_pulses_stop_after_100ms():
    m = api(); cfg = m.LongEvalConfig(); schedule = m.make_schedule(initial(), cfg)
    assert schedule.targets.shape == (6, 3)
    assert schedule.forces_world.shape == (60, 3, 3)
    assert cfg.arena_side == 4 and cfg.arena_side-2*cfg.target_margin == 3
    center=schedule.targets.new_tensor([0.,0.,2.])
    assert torch.all((schedule.targets-center).abs() <= 1.5)
    assert cfg.target_index(999) == 0 and cfg.target_index(1000) == 1
    assert cfg.target_index(5999) == 5
    assert [cfg.pulse_active(k) for k in [0, 9, 10, 99, 100, 109, 110]] == [True, True, False, False, True, True, False]
    assert torch.all(schedule.targets[:, :2].abs() <= 1.5)
    assert torch.all((schedule.targets[:, 2] >= .5) & (schedule.targets[:, 2] <= 3.5))
    torch.testing.assert_close(schedule.forces_world.norm(dim=-1),
                               (.2 * 9.81 * initial().mass)[None].expand(60, -1))


def test_surface_sampler_is_area_uniform_not_equal_face_probability():
    m = api()
    half = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
    points, face = m.sample_box_surface(half, 60000, torch.Generator().manual_seed(17))
    assert points.shape == (60000, 1, 3)
    assert torch.all(points.abs() <= half + 1e-12)
    assert torch.all(torch.isclose(points.abs(), half, atol=1e-12).sum(-1) == 1)
    # Opposite faces have areas 24, 24, 12, 12, 8, 8, respectively.
    expected = torch.tensor([24, 24, 12, 12, 8, 8], dtype=torch.float64) / 88
    freq = torch.bincount(face.flatten(), minlength=6).double() / 60000
    assert torch.max(torch.abs(freq - expected)) < .008
    assert float(points.mean((0, 1)).abs().max()) < .035
    assert float(points.square().sum(-1).min()) > .99


def test_point_force_frame_conversion_and_yaw_torque():
    m = api()
    # Body x is world y after a +90 degree yaw.
    q = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]], dtype=torch.float64)
    r = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
    force = torch.tensor([[2., 0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(m.point_torque(q, r, force), torch.tensor([[0., 0., -2.]], dtype=torch.float64))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_center_force_matches_existing_rk4_exactly(dtype):
    m = api(); s = initial(dtype=dtype)
    force = s.position.new_tensor([[.1, -.3, .2]]).expand(3, -1).clone()
    s = replace(s, external_force=force, external_torque=torch.zeros_like(force))
    action = s.motor.new_full(s.motor.shape, .1)
    expected = RaptorSimulator().step(s, action)
    actual = m.PointForceSimulator().step(s, action, force, torch.zeros_like(force))
    for name in ['position', 'velocity', 'orientation', 'omega', 'motor', 'previous_action', 'previous_velocity', 'step_index']:
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


def test_rk4_updates_point_torque_at_intermediate_attitudes():
    m = api(); s = initial(1)
    s = replace(s, omega=torch.tensor([[3., -2., 4.]], dtype=torch.float64))
    force = torch.tensor([[.003, -.002, .001]], dtype=torch.float64)
    point = torch.tensor([[.01, -.005, .003]], dtype=torch.float64)
    action = s.motor.new_full(s.motor.shape, .1)
    expected = numpy_point_step(s, action, force, point)
    actual = m.PointForceSimulator().step(s, action, force, point)
    value = torch.cat([actual.position[0], actual.velocity[0], actual.orientation[0], actual.omega[0], actual.motor[0]])
    torch.testing.assert_close(value, expected, rtol=3e-12, atol=3e-12)
    frozen = replace(s, external_force=force, external_torque=m.point_torque(s.orientation, point, force))
    assert float((RaptorSimulator().step(frozen, action).omega - actual.omega).norm()) > 1e-7


def numpy_point_step(s, action, force, point):
    import numpy as np
    def a(name): return getattr(s, name)[0].numpy()
    target = (action[0].numpy()+1)/2
    z = np.concatenate([a('position'), a('velocity'), a('orientation'), a('omega'), a('motor')])
    def rhs(z):
        p,v,q,w,m = z[:3],z[3:6],z[6:10],z[10:13],z[13:17]
        def rotate(vector):
            t = 2*np.cross(q[1:], vector)
            return vector + q[0]*t + np.cross(q[1:], t)
        # R(q).T is the inverse rotation only for unit q. RK4 intermediate q
        # need not be unit; construct R's columns for the exact retained equations.
        rot = np.stack([rotate(np.eye(3)[i]) for i in range(3)], axis=1)
        th = (a('thrust_coefficients')*np.stack([np.ones(4),m,m*m],-1)).sum(-1)
        rotor_forces = np.stack([np.zeros(4),np.zeros(4),th],-1)
        torque = np.cross(a('rotor_positions'),rotor_forces).sum(0)
        torque[2] += (np.array([-1,1,-1,1])*a('rotor_torque_constant')*th).sum()
        torque += np.cross(point[0].numpy(),rot.T@force[0].numpy())
        acc = rot[:,2]*th.sum()/a('mass') + np.array([0,0,-9.81]) + force[0].numpy()/a('mass')
        qdot = .5*np.concatenate([[-q[1:]@w],q[0]*w+np.cross(q[1:],w)])
        wdot = (torque-np.cross(w,a('inertia')*w))/a('inertia')
        tau = np.where(target>=m,a('motor_time_rising'),a('motor_time_falling'))
        return np.concatenate([v,acc,qdot,wdot,(target-m)/tau])
    k1=rhs(z);k2=rhs(z+.005*k1);k3=rhs(z+.005*k2);k4=rhs(z+.01*k3)
    z += .01/6*(k1+2*k2+2*k3+k4);z[6:10]/=np.linalg.norm(z[6:10]);z[13:17]=np.clip(z[13:17],0,1)
    return torch.from_numpy(z)


def test_goal_switch_changes_only_position_observation_and_keeps_memory():
    m = api(); torch.manual_seed(8)
    policy = ResponseMotorPolicy().double()
    cfg = m.LongEvalConfig(duration_seconds=.04, target_period_seconds=.02,
        force_period_seconds=.02, pulse_seconds=.01, force_fraction=0., hover_tail_seconds=.01)
    env = m.LongHoverEval(policy, initial(1), cfg)
    env.step(); env.step()
    memory = env.memory.memory.clone(); before = env.state
    measured = measured_observation(before)
    expected_obs = torch.cat((measured[:, :3]-env.schedule.targets[1], measured[:, 3:]),-1)
    torch.testing.assert_close(env.observation(), expected_obs, rtol=0, atol=0)
    with torch.no_grad(): expected = policy(expected_obs, env.memory)
    event = env.step()
    torch.testing.assert_close(event['action'], expected.action, rtol=0, atol=0)
    torch.testing.assert_close(env.memory.memory, expected.next_state.memory, rtol=0, atol=0)
    assert memory.norm() > 0
    assert all(p.grad is None for p in policy.parameters())


def test_arena_termination_uses_com_world_position_not_target_error_or_body_extent():
    m = api(); cfg = m.LongEvalConfig()
    # At x=1.999 the shell/arms may cross the wall; only the COM matters.
    points = torch.tensor([[1.999,0.,2.], [2.01,0.,2.], [0.,0.,-.01],
                           [0.,0.,4.01], [2.,0.,4.], [-2.,-2.,0.], [-2.001,0.,2.]])
    assert m.arena_exited(points, cfg).tolist() == [False,True,True,True,False,False,True]


def test_long_schedule_repeats_exactly_and_has_all_torque_axes():
    m=api();s=initial(64);cfg=m.LongEvalConfig()
    a=m.make_schedule(s,cfg);b=m.make_schedule(s,cfg)
    for f in fields(a): assert torch.equal(getattr(a,f.name),getattr(b,f.name)), f.name
    unit=a.forces_world/(s.mass[None,:,None]*9.81*.2)
    assert unit.mean((0,1)).abs().max() < .045
    torque=torch.linalg.cross(a.points_body,a.forces_world,dim=-1)
    assert torch.all((torque>0).any((0,1)) & (torque<0).any((0,1)))


def test_failed_scene_is_frozen_and_not_counted_as_hover():
    m=api();torch.manual_seed(8)
    policy=ResponseMotorPolicy().double()
    cfg=m.LongEvalConfig(duration_seconds=.04,target_period_seconds=.02,
        force_period_seconds=.02,pulse_seconds=.01,force_fraction=0.,hover_tail_seconds=.01)
    env=m.LongHoverEval(policy,initial(1),cfg)
    env.state=replace(env.state, position=env.state.position.new_tensor([[1.9999,0.,2.]]),
                      velocity=env.state.position.new_tensor([[2.,0.,0.]]))
    first=env.step(); assert first['alive_before'].item() and not env.alive.item()
    frozen=env.state.position.clone();memory=env.memory.memory.clone()
    again=env.step()
    assert not again['valid'].item()
    assert torch.equal(env.state.position,frozen) and torch.equal(env.memory.memory,memory)
    result=env.run()
    assert result.summary['completed_count']==0
    assert result.summary['scenes'][0]['failure_reason']=='arena_exit'
    assert not any(x['hovered'] for x in result.summary['scenes'][0]['targets'])


@pytest.mark.parametrize('changes', [dict(pulse_seconds=1.1),dict(target_period_seconds=7.),
    dict(target_margin=2.),dict(force_fraction=float('nan')),dict(duration_seconds=.015)])
def test_invalid_eval_configuration_is_rejected(changes):
    with pytest.raises(ValueError): api().LongEvalConfig(**changes)


def test_cli_exports_reproducible_frozen_inputs_and_trajectory(tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    import json
    from test_reference_training import args_for, run
    root=Path(__file__).resolve().parents[1]
    entry=root/'tools/evaluate_response_long.py'
    assert entry.is_file(), 'packaged long EVAL entry is missing'
    run(args_for(tmp_path/'checkpoint',updates=0))
    common=[sys.executable,str(entry),'--checkpoint',str(tmp_path/'checkpoint/latest.pt'),
        '--scene-ids','0','1','--duration-seconds','.04','--target-period-seconds','.02',
        '--force-period-seconds','.02','--pulse-seconds','.01','--hover-tail-seconds','.01']
    for name in ('a','b'):
        result=subprocess.run([*common,'--work-dir',str(tmp_path/name)],cwd=root,text=True,capture_output=True)
        assert result.returncode==0,result.stderr
    a=torch.load(tmp_path/'a/trajectory.pt',weights_only=True)
    b=torch.load(tmp_path/'b/trajectory.pt',weights_only=True)
    for name in a: assert torch.equal(a[name],b[name]),name
    manifest=json.loads((tmp_path/'a/manifest.json').read_text())
    assert manifest['scene_ids']==[0,1]
    assert manifest['config']['pulse_seconds']==.01
    assert manifest['source_match']
    state=torch.load(tmp_path/'a/initial_state.pt',weights_only=True)
    assert not state['external_force'].any() and not state['external_torque'].any()
    assert not state['pulse_tape'].any() and not state['pulse_active_tape'].any()
    assert state['previous_velocity'].shape == (2, 3, 3)
    assert not manifest['sensor_command_noise']['pulse_enabled']
    assert not a['force_world'][1].any() and a['force_world'][2].norm()>0

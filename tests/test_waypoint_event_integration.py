from __future__ import annotations
from dataclasses import fields, replace
from pathlib import Path
import copy
import math
import json
import subprocess
import sys
import pytest
import torch

from diffphys.env_raptor import RaptorSimulator, ACTION_CONVENTION
from diffphys.response_noise import DisturbanceConfig, measured_observation
from diffphys.response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from diffphys.waypoint_config import WaypointConfig
from diffphys.waypoint_mission import build_mission, WaypointMission, goal_packet
from diffphys.waypoint_task import (rollout, evaluate_rollout, constraints_for_trace,
    midpoint_event, fractional_rk4, WaypointTrace, DYNAMIC, CONSTRAINTS)
from diffphys.waypoint_gradients import corrected_gradient
from diffphys.waypoint_checkpoint import (model_hash, import_weights, load_policy,
    atomic_save, checkpoint_payload, WaypointController)
from diffphys.waypoint_sampling import sample_bank
from diffphys.waypoint_training import one_update, parse_args, train, vector_of

torch.set_num_threads(1)


def cfg(**kwargs):
    return replace(WaypointConfig(),audit_substeps=1,event_directions=1,**kwargs)


def options(config,horizon=6):
    return dict(seed=71,horizon=horizon,dtype=torch.float64,sampling='random',
                disturbances=DisturbanceConfig(force_enabled=False),waypoint_config=config.snapshot())


def bank(config=None,horizon=6):
    config=cfg() if config is None else config
    return sample_bank(1,0,options(config,horizon))[1]


def actor(width=8):
    torch.manual_seed(37)
    return ResponseMotorPolicy(ResponsePolicyConfig(width,.01,True)).double()


def test_fractional_full_step_is_identical():
    b=bank();s=b.initial;a=s.previous_action
    expected=RaptorSimulator().step(s,a)
    actual=fractional_rk4(s,a,.01)
    for name in DYNAMIC:
        torch.testing.assert_close(getattr(actual,name),getattr(expected,name),atol=1e-14,rtol=1e-14)


def crossing_trace(speed):
    b=bank();s=replace(b.initial,**{f.name:getattr(b.initial,f.name)[:1] for f in fields(b.initial)
                                  if f.name not in ('noise_tape','rotation_tape','pulse_tape','pulse_active_tape')})
    zero=s.position.new_zeros(1,3)
    v=torch.stack((speed,speed*0,speed*0)).reshape(1,3)
    s=replace(s,position=zero+torch.tensor([-.004,0,0]),velocity=v,
              orientation=torch.tensor([[1.,0,0,0]],dtype=torch.float64),omega=zero,
              previous_action=2*s.motor-1)
    a=s.previous_action
    moved=RaptorSimulator().step(s,a)
    trace=WaypointTrace(s,moved,{name:torch.stack((getattr(s,name),getattr(moved,name))) for name in DYNAMIC},
                        a[None],torch.tensor([[1],[2]]),torch.ones(1,1,dtype=torch.bool),zero)
    m=WaypointMission(torch.tensor([[[-1.,0,0],[0,0,0],[1,0,0]]],dtype=torch.float64),
                      torch.eye(3,dtype=torch.float64)[None],torch.eye(3,dtype=torch.float64)[None],
                      torch.tensor([[1.,0,0]],dtype=torch.float64),torch.tensor([.4],dtype=torch.float64))
    return trace,m


def test_actual_rk4_event_root_gradient_and_fd():
    speed=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    tr,m=crossing_trace(speed)
    ev,exists,t,_=midpoint_event(tr,m,cfg())
    assert exists.item()
    ad=torch.autograd.grad(t.sum(),speed)[0]
    eps=1e-5
    with torch.no_grad():
        hi=midpoint_event(*crossing_trace(speed.detach()+eps),cfg())[2]
        lo=midpoint_event(*crossing_trace(speed.detach()-eps),cfg())[2]
    fd=(hi-lo)/(2*eps)
    torch.testing.assert_close(ad.reshape(1),fd,atol=3e-7,rtol=3e-5)
    torch.testing.assert_close(ad,torch.tensor(-.004,dtype=torch.float64),atol=2e-8,rtol=1e-6)
    assert ev['position'][0,0].abs()<1e-8


def test_hard_threshold_gradient_statistical_check():
    x=torch.tensor([0.],dtype=torch.float64)
    sigma=.2
    def function(v):
        return ((v-1).square()+10*(v>=0)).sum()
    grad,report=corrected_gradient(x,-2*torch.ones_like(x),function,sigma=sigma,directions=4000,seed=31)
    expected=-2+10/(sigma*math.sqrt(2*math.pi))
    assert abs(grad.item()-expected)<5*report['standard_error_l2']
    # The raw function itself still has a jump; this is not its point derivative.
    assert (function(x+1e-8)-function(x-1e-8)).item()>9.99


def test_no_event_still_owes_pose_and_final_stop():
    b=bank();p=actor();c=cfg()
    trace=rollout(p,RaptorSimulator(),b.initial,b.mission,c,horizon=6)
    _,g,_=constraints_for_trace(trace,b.mission,c)
    assert (g[:,CONSTRAINTS.index('midpoint_plane_crossing')]>0).all()
    assert (g[:,CONSTRAINTS.index('midpoint_attitude')]>0).all()
    assert (g[:,CONSTRAINTS.index('final_speed')]>0).all()
    # Changing only the command index cannot erase final velocity/attitude duties.
    shifted=replace(trace,target_index=torch.full_like(trace.target_index,2))
    _,g2,_=constraints_for_trace(shifted,b.mission,c)
    for name in ('final_position','final_speed','final_attitude','final_motor_rate'):
        torch.testing.assert_close(g[:,CONSTRAINTS.index(name)],g2[:,CONSTRAINTS.index(name)])


def test_final_command_keeps_pending_midpoint_information():
    b=bank();obs=measured_observation(b.initial)
    obs[:,:3]=b.mission.points[:,1]-.05*b.mission.direction
    packet=goal_packet(obs,b.mission,torch.full_like(b.initial.step_index,2),2.)
    torch.testing.assert_close(packet[:,:3],b.mission.points[:,2])
    torch.testing.assert_close(packet[:,9:18],b.mission.rotation.flatten(1))
    assert (packet[:,-1]==1).all()
    obs[:,:3]=b.mission.points[:,2]
    packet=goal_packet(obs,b.mission,torch.full_like(b.initial.step_index,2),2.)
    torch.testing.assert_close(packet[:,9:18],b.mission.level_rotation.flatten(1))
    assert (packet[:,-1]==0).all()


def test_controller_and_rollout_issue_identical_commands():
    c=cfg();b=bank(c);p=actor();sim=RaptorSimulator()
    trace=rollout(p,sim,b.initial,b.mission,c,horizon=6)
    ctrl=WaypointController(p,b.mission,c)
    s=b.initial
    for k in range(6):
        a=ctrl.step(measured_observation(s))
        torch.testing.assert_close(a,trace.actions[k],rtol=0,atol=0)
        s=sim.step(s,a)
    torch.testing.assert_close(ctrl.memory.memory,trace.final_memory,rtol=0,atol=0)


def test_partial_constraint_gradients_through_original_physics():
    c=cfg();b=bank(c);p=actor();sim=RaptorSimulator()
    r=evaluate_rollout(p,sim,b.initial,b.mission,c,horizon=6)
    gradients=torch.autograd.grad(r.costs.mean(),list(p.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    vector=torch.cat([g.flatten() for g in gradients]);theta=vector_of(p)
    z=torch.randn(theta.shape,generator=torch.Generator().manual_seed(16),dtype=torch.float64)
    z=z/z.norm();eps=1e-5
    from diffphys.waypoint_training import assign_vector
    with torch.no_grad():
        assign_vector(p,theta+eps*z)
        hi=evaluate_rollout(p,sim,b.initial,b.mission,c,horizon=6).costs.mean()
        assign_vector(p,theta-eps*z)
        lo=evaluate_rollout(p,sim,b.initial,b.mission,c,horizon=6).costs.mean()
        assign_vector(p,theta)
    fd=(hi-lo)/(2*eps);ad=torch.dot(vector,z)
    assert abs((fd-ad).item())/max(1,abs(fd.item()),abs(ad.item()))<2e-5


def test_one_update_restores_perturbations_and_changes_weights():
    c=cfg();b=bank(c);p=actor();before=vector_of(p)
    optim=torch.optim.Adam(p.parameters(),lr=3e-4)
    mu=torch.zeros(4,len(CONSTRAINTS),dtype=torch.float64)
    report=one_update(p,optim,RaptorSimulator(),b,c,mu,horizon=6,seed=71,gradient_clip=10.)
    assert report['gradient']['extra_forward_rollouts']==2
    assert (vector_of(p)-before).norm()>0
    assert mu.count_nonzero()==0
    assert all(v['step']==1 for v in optim.state.values())


def test_exception_restores_parameter_and_adam(monkeypatch):
    import diffphys.waypoint_training as module
    c=cfg();b=bank(c);p=actor();before=vector_of(p)
    optim=torch.optim.Adam(p.parameters(),lr=3e-4)
    def broken(theta,gp,evaluate,**kwargs):
        evaluate(theta+.001)
        raise RuntimeError('injected paired failure')
    monkeypatch.setattr(module,'corrected_gradient',broken)
    with pytest.raises(RuntimeError,match='injected'):
        module.one_update(p,optim,RaptorSimulator(),b,c,torch.zeros(4,len(CONSTRAINTS)),
                          horizon=6,seed=71,gradient_clip=10.)
    torch.testing.assert_close(vector_of(p),before,rtol=0,atol=0)
    assert len(optim.state)==0


def save_hover(path):
    torch.manual_seed(41)
    p=ResponseMotorPolicy(ResponsePolicyConfig(8,.01,False)).double()
    atomic_save(path,{'schema':'raptor-multi-airframe-gaussian-v6','architecture':p.architecture,
                     'policy_config':{'memory_dim':8,'dt':.01,'goal_conditioning':False},
                     'model':p.state_dict(),'model_sha256':model_hash(p),
                     'binding':{'protocol':{'environment':{'action_convention':ACTION_CONVENTION}}}})
    return p


def test_hover_import_is_full_finetuning_and_independent(tmp_path):
    src=tmp_path/'hover.pt';old=save_hover(src)
    p,report=import_weights(src,dtype=torch.float64)
    assert report['zero_padded_goal_columns']==20
    assert (p.response_memory.weight_ih[:,16:]==0).all()
    b=bank();obs=measured_observation(b.initial)
    # At zero target and initial zero extra weights, the old policy's output is identical.
    goals=obs.new_zeros(4,23);goals[:,9:18]=torch.eye(3,dtype=obs.dtype).flatten()
    torch.testing.assert_close(old(obs).action,p(obs,goals=goals).action,rtol=0,atol=0)
    opt=torch.optim.Adam(p.parameters(),lr=3e-4)
    destination=tmp_path/'new.pt'
    atomic_save(destination,checkpoint_payload(p,opt,cfg(),{'next_update':0},{},None,-1,None))
    src.unlink()
    loaded,_,_=load_policy(destination)
    torch.testing.assert_close(vector_of(p),vector_of(loaded),rtol=0,atol=0)
    code=f"from diffphys.waypoint_checkpoint import load_policy; p,c,v=load_policy({str(destination)!r}); assert len(p.state_dict())==6"
    subprocess.run([sys.executable,'-c',code],check=True,cwd=Path(__file__).resolve().parents[1])


def test_serial_exact_resume_matches_two_updates(tmp_path):
    config=cfg(dual_interval=1,dual_updates_per_bank=3)
    conf=tmp_path/'task.json';conf.write_text(json.dumps(config.snapshot()))
    hover=tmp_path/'hover.pt';save_hover(hover)
    def arguments(folder,updates,extra):
        return parse_args(['--device','cpu','--dtype','float64','--threads','1','--scenarios','1',
                           '--eval-scenarios','1','--train-sampling','random','--no-prefetch',
                           '--horizon','6','--updates',str(updates),'--checkpoint-every','1',
                           '--agile-config',str(conf),'--work-dir',str(folder)]+extra)
    full=tmp_path/'full';split=tmp_path/'split'
    train(arguments(full,2,['--init-checkpoint',str(hover)]),config)
    train(arguments(split,1,['--init-checkpoint',str(hover)]),config)
    train(arguments(split,2,['--resume',str(split/'checkpoint.pt')]),config)
    a=torch.load(full/'checkpoint.pt',weights_only=True);b=torch.load(split/'checkpoint.pt',weights_only=True)
    def same(x,y):
        if torch.is_tensor(x):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:same(x[k],y[k])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for u,v in zip(x,y):same(u,v)
        else: assert x==y
    for name in ('model','optimizer','multipliers','rng','progress','bank_key'):
        same(a[name],b[name])


def test_no_deceleration_detects_braking_above_speed_floor():
    # A valid native one-step transition: >2m/s forward but backward thrust.
    trace,mission=crossing_trace(torch.tensor(3.,dtype=torch.float64))
    pitch=-math.radians(10.)
    q=torch.tensor([[math.cos(pitch/2),0.,math.sin(pitch/2),0.]],dtype=torch.float64)
    initial=replace(trace.initial,orientation=q)
    final=RaptorSimulator().step(initial,trace.actions[0])
    trace=replace(trace,initial=initial,final=final,state={name:torch.stack((getattr(initial,name),getattr(final,name)))
                                                         for name in DYNAMIC})
    _,g,_=constraints_for_trace(trace,mission,cfg())
    assert g[0,CONSTRAINTS.index('passage_forward_speed')]<0
    assert g[0,CONSTRAINTS.index('passage_no_deceleration')]>0


def test_grad_and_no_grad_use_identical_event_forward():
    trace,m=crossing_trace(torch.tensor(1.,dtype=torch.float64,requires_grad=True))
    yes=midpoint_event(trace,m,cfg())
    with torch.no_grad():
        no=midpoint_event(trace,m,cfg())
    for name in yes[0]:
        torch.testing.assert_close(yes[0][name],no[0][name],rtol=0,atol=0)
    torch.testing.assert_close(yes[2],no[2],rtol=0,atol=0)


def test_force_free_supervision_rejects_hidden_pulses():
    c=cfg();b=bank(c)
    tape=b.initial.pulse_tape.clone();tape[0,0,0]=.1
    bad=replace(b.initial,pulse_tape=tape)
    with pytest.raises(ValueError,match='force-free'):
        rollout(actor(),RaptorSimulator(),bad,b.mission,c,horizon=6)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_h500_original_physics_forward_and_backward(dtype):
    # Finite H500 gradient test, not a claim of successful agile flight.
    c=cfg();b=bank(c,horizon=500).to('cpu',dtype);p=actor().to(dtype=dtype)
    result=evaluate_rollout(p,RaptorSimulator(),b.initial,b.mission,c,horizon=500,record_trace=True)
    assert result.trace.actions.shape==(500,4,4)
    assert result.trace.target_index.shape==(501,4)
    assert torch.isfinite(result.costs).all()
    gradients=torch.autograd.grad(result.costs.mean(),list(p.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)


def test_spawn_prefetch_matches_serial_bank():
    from diffphys.waypoint_sampling import WaypointPrefetch
    c=cfg();opt=options(c)
    _,expected,_,_,_=sample_bank(1,0,opt)
    pre=WaypointPrefetch(1,options=opt,threads=1)
    try:
        pre.submit(0)
        actual,_,_,metadata=pre.take(0)
        assert actual.key==expected.key
        assert not metadata['worker_cuda_initialized']
        for f in fields(actual.initial):
            torch.testing.assert_close(getattr(actual.initial,f.name),getattr(expected.initial,f.name),rtol=0,atol=0)
        pre.submit(1)
        next_bank,_,_,_=pre.take(1)
        assert next_bank.key!=actual.key
    finally:
        pre.close()


def test_multipliers_reset_on_bank_change_and_resume(tmp_path):
    # interval1 x one dual update means each new update changes banks.
    config=cfg(dual_interval=1,dual_updates_per_bank=1)
    conf=tmp_path/'task.json';conf.write_text(json.dumps(config.snapshot()))
    hover=tmp_path/'hover.pt';save_hover(hover)
    def args(folder,updates,extra):
        return parse_args(['--device','cpu','--dtype','float64','--threads','1','--scenarios','1',
                           '--eval-scenarios','1','--train-sampling','random','--no-prefetch',
                           '--horizon','6','--updates',str(updates),'--agile-config',str(conf),
                           '--work-dir',str(folder)]+extra)
    full=tmp_path/'full';split=tmp_path/'split'
    train(args(full,2,['--init-checkpoint',str(hover)]),config)
    train(args(split,1,['--init-checkpoint',str(hover)]),config)
    old=torch.load(split/'checkpoint.pt',weights_only=True)
    train(args(split,2,['--resume',str(split/'checkpoint.pt')]),config)
    a=torch.load(full/'checkpoint.pt',weights_only=True);b=torch.load(split/'checkpoint.pt',weights_only=True)
    assert old['bank_key']!=b['bank_key']
    assert a['multipliers_bank_index']==b['multipliers_bank_index']==1
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    torch.testing.assert_close(a['multipliers'],b['multipliers'],rtol=0,atol=0)
    # New-bank duals equal rho*g+, not the previous bank's accumulated duals.
    p,_,_=load_policy(split/'checkpoint.pt')
    current=sample_bank(1,1,{**options(config), 'seed':7})[1]
    with torch.no_grad():
        r=evaluate_rollout(p,RaptorSimulator(),current.initial,current.mission,config,horizon=6)
    torch.testing.assert_close(b['multipliers'],(config.rho*r.constraints).clamp_min(0),rtol=0,atol=0)


def test_completion_time_is_last_departure_not_first_entry():
    from diffphys.waypoint_task import completion_time
    at_goal=torch.tensor([[False],[True],[False],[True],[True]],dtype=torch.bool)
    indices=torch.full((5,1),2,dtype=torch.long)
    clock,good=completion_time(at_goal,indices,torch.tensor([True]),torch.tensor([0.]),.01)
    torch.testing.assert_close(clock,torch.tensor([.75]))
    assert good[-1].item()
    clock,good=completion_time(at_goal,indices,torch.tensor([False]),torch.tensor([0.]),.01)
    torch.testing.assert_close(clock,torch.tensor([1.]))
    assert not good.any()


def test_main_entry_trains_evaluates_and_replays_new_protocol(tmp_path):
    root=Path(__file__).resolve().parents[1]
    config=cfg();conf=tmp_path/'task.json';conf.write_text(json.dumps(config.snapshot()))
    hover=tmp_path/'hover.pt';save_hover(hover);work=tmp_path/'run'
    command=[sys.executable,'tools/train_response_control.py','--task','agile','--device','cpu',
             '--threads','1','--dtype','float64','--scenarios','1','--eval-scenarios','1',
             '--train-sampling','random','--no-prefetch','--horizon','6','--updates','1',
             '--agile-config',str(conf),'--init-checkpoint',str(hover),'--work-dir',str(work)]
    run=subprocess.run(command,cwd=root,capture_output=True,text=True,check=True)
    summary=json.loads(run.stdout)
    assert summary['updates']==1
    hover.unlink()
    replay=tmp_path/'replay.npz'
    # No --task: the main entry must recognize the saved NEW checkpoint schema.
    evaluation=subprocess.run([sys.executable,'tools/train_response_control.py','--mode','evaluate',
                               '--device','cpu','--threads','1','--eval-scenarios','1',
                               '--checkpoint',str(work/'checkpoint.pt'),'--verify-audit-substeps','2',
                               '--replay-out',str(replay)],cwd=root,capture_output=True,text=True,check=True)
    report=json.loads(evaluation.stdout)
    assert report['checkpoint_update']==1
    assert report['constraint_sample_interval_s']==.005
    import numpy as np
    with np.load(replay) as r:
        assert r['actions'].shape==(6,4,4)
        assert r['target_index'].shape==(7,4)
        assert len(r['constraint_names'])==len(CONSTRAINTS)


def test_default_goal_angle_is_not_downgraded_by_airframe():
    c=cfg();b=bank(c)
    body_up=b.mission.rotation[:,:,2]
    angle=torch.acos(body_up[:,2])
    torch.testing.assert_close(angle,angle.new_full(angle.shape,math.radians(105.)),rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(b.initial.position,b.mission.points[:,0])
    assert not (b.initial.velocity.norm(dim=-1)==0).any()

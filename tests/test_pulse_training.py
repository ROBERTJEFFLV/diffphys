"""Pulse causality, live-row isolation, chunked full BPTT and checkpoint binding."""
from dataclasses import fields, replace
import copy
import json
import pytest
import torch

from env_raptor import RaptorSimulator, IMMUTABLE_TAPES, environment_contract, RaptorParams
from response_noise import DisturbanceConfig, measured_observation, pulse_at
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import rollout, step_costs, TaskLossConfig, _select_rows
from response_adjoints import collect_rollout, backward_actor
from response_groups import normalize_group_rows
from response_training import sample_pool, sample_training_scenarios, DEVELOPMENT_SEEDS
from test_episode_termination import flat_grads
from test_reference_training import args_for, run


def fixture(n=64, horizon=70):
    s = RaptorSimulator().reset(n, seed=72, horizon=horizon, dtype=torch.float64)
    # A bounded verification fixture, not a claim that a random Actor can recover.
    tape = torch.zeros_like(s.pulse_tape); active = torch.zeros_like(s.pulse_active_tape)
    tape[:, 49:59, :3] = (s.mass[:, None]*s.mass.new_tensor([.1, -.2, .3]))[:, None]
    tape[:, 49:59, 3:] = .4*s.rotor_positions[:, None, 0]
    active[:, 49:59] = True
    return replace(s, position=s.position*0, velocity=s.velocity*0, omega=s.omega*0,
                   previous_velocity=s.previous_velocity*0,
                   orientation=s.orientation.new_tensor([1.,0.,0.,0.]).expand(n,4).clone(),
                   position_limit=torch.full_like(s.position_limit,1000), external_force=s.external_force*0,
                   pulse_tape=tape, pulse_active_tape=active)


def test_pulse_cannot_leak_current_or_future_force_to_actor():
    s = fixture(n=8); torch.manual_seed(28)
    p = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).double()
    altered = replace(s, pulse_tape=s.pulse_tape+10, pulse_active_tape=~s.pulse_active_tape,
                      force_std=s.force_std+4, external_force=s.external_force+7)
    a,b = p(measured_observation(s)),p(measured_observation(altered))
    assert torch.equal(a.action,b.action) and torch.equal(a.next_state.memory,b.next_state.memory)
    no_pulse = replace(s,pulse_tape=torch.zeros_like(s.pulse_tape))
    x,y = rollout(p,RaptorSimulator(),s,60),rollout(p,RaptorSimulator(),no_pulse,60)
    # Force first acts over transition 49->50; command at t=49 cannot anticipate it.
    torch.testing.assert_close(x.actions[:50],y.actions[:50],rtol=0,atol=0)
    torch.testing.assert_close(x.positions[:49],y.positions[:49],rtol=0,atol=0)
    assert not torch.equal(x.positions[49:],y.positions[49:])


def test_pulse_tapes_follow_original_scene_id_not_compacted_rank():
    s = fixture(n=32); ids=torch.tensor([17,2,29,5])
    s=replace(s,step_index=torch.full_like(s.step_index,52))
    small=_select_rows(s,ids)
    for name in IMMUTABLE_TAPES:
        assert getattr(small,name).data_ptr()==getattr(s,name).data_ptr()
    for a,b in zip(pulse_at(small),pulse_at(s)):
        torch.testing.assert_close(a,b[ids],rtol=0,atol=0)
    action=torch.zeros_like(s.motor)
    full=RaptorSimulator().step(s,action)
    part=RaptorSimulator().step(small,action[ids])
    for name in ('position','velocity','orientation','omega','motor','previous_velocity'):
        torch.testing.assert_close(getattr(part,name),getattr(full,name)[ids],rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('alpha',[0.,1.])
def test_off_center_pulse_crosses_metric_boundary_without_gradient_detach(alpha):
    s=fixture();sim=RaptorSimulator();loss=TaskLossConfig();torch.manual_seed(28)
    p=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).double();other=copy.deepcopy(p)
    direct=rollout(other,sim,s,70,time_decay=alpha)
    costs=step_costs(direct,loss).sum(0)
    record=collect_rollout(p,sim,s,loss,horizon=70,time_decay=alpha)
    assert direct.valid.all()
    torch.testing.assert_close(record.costs,costs,rtol=1e-11,atol=1e-11)
    rows=[]
    for seed in record.group_coefficients:
        g=torch.autograd.grad(costs,list(other.parameters()),grad_outputs=.1*seed,retain_graph=True)
        rows.append(torch.cat([x.flatten() for x in g]))
    expected,_=normalize_group_rows(torch.stack(rows),record.group_config.gradient_epsilon)
    backward_actor(p,sim,record,loss)
    torch.testing.assert_close(flat_grads(p),expected,rtol=1e-9,atol=1e-9)
    assert torch.isfinite(flat_grads(p)).all() and flat_grads(p).norm()>0


def test_failed_episode_does_not_advance_or_apply_scheduled_future_pulses():
    s=fixture(n=3);pos=s.position.clone();vel=s.velocity.clone()
    pos[0,0]=s.position_limit[0]-.000001;vel[0,0]=1
    s=replace(s,position=pos,velocity=vel)
    p=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).double()
    a=rollout(p,RaptorSimulator(),s,65)
    assert a.valid[:,0].sum()==1 and a.end.physical.step_index[0]==1
    tape=s.pulse_tape.clone();tape[0,2:]=1e6
    b=rollout(p,RaptorSimulator(),replace(s,pulse_tape=tape),65)
    assert torch.equal(a.actions,b.actions) and torch.equal(a.valid,b.valid)
    for name in IMMUTABLE_TAPES:
        assert getattr(a.end.physical,name).data_ptr()==getattr(s,name).data_ptr()


def test_train_pulses_resample_and_eval_pulses_repeat():
    a,_=sample_training_scenarios(8,0,horizon=140)
    b,_=sample_training_scenarios(8,1,horizon=140)
    x=sample_pool(4,DEVELOPMENT_SEEDS,horizon=140)
    y=sample_pool(4,DEVELOPMENT_SEEDS,horizon=140)
    assert not torch.equal(a.pulse_tape,b.pulse_tape)
    assert not torch.equal(a.pulse_active_tape,b.pulse_active_tape)
    for f in fields(x):assert torch.equal(getattr(x,f.name),getattr(y,f.name))
    assert x.pulse_active_tape.any()


def test_pulse_checkpoint_resume_and_stored_eval_settings(tmp_path):
    from response_training import evaluate_checkpoint, require_reference_checkpoint
    def args(path,updates,extra=()):
        return args_for(path,updates,('--horizon','120',*extra))
    full,split=tmp_path/'full',tmp_path/'split'
    run(args(full,2));run(args(split,1));run(args(split,2,('--resume',str(split/'latest.pt'))))
    a=torch.load(full/'latest.pt',weights_only=True);b=torch.load(split/'latest.pt',weights_only=True)
    assert a['model_sha256']==b['model_sha256']
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    for i,state in a['optimizer']['state'].items():
        for key,value in state.items():
            torch.testing.assert_close(value,b['optimizer']['state'][i][key],rtol=0,atol=0)
    ev=args(tmp_path/'eval',0,('--disable-pulses',));ev.checkpoint=split/'latest.pt'
    result=evaluate_checkpoint(ev)
    expected=json.loads((split/'evaluation.jsonl').read_text().splitlines()[-1])
    assert result['task_objective']==expected['task_objective']
    assert result['disturbances']==expected['disturbances']
    assert result['disturbances']['transient_force_schedule']
    assert b['binding']['protocol']['disturbances']['pulse_enabled']
    with pytest.raises(ValueError,match='configuration'):
        run(args(split,3,('--resume',str(split/'latest.pt'),'--disable-pulses')))
    b['binding']['protocol']['environment']['version']='raptor-multi-airframe-gaussian-v4'
    with pytest.raises(ValueError,match='contract mismatch'):require_reference_checkpoint(b)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_off_center_pulse_rollout_and_backward():
    s=fixture(n=32).to('cuda',torch.float32)
    p=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).cuda()
    tr=rollout(p,RaptorSimulator(),s,70,time_decay=1.)
    step_costs(tr,TaskLossConfig()).sum().backward()
    assert torch.isfinite(flat_grads(p)).all() and flat_grads(p).norm()>0

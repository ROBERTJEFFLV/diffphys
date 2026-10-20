"""Native group backward, original hover and checkpoint integration."""
from dataclasses import asdict, replace
from pathlib import Path
import copy
import importlib.util
import json
import os
import subprocess
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from env_raptor import RaptorSimulator
from response_noise import DisturbanceConfig, measured_observation
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig, rollout, scenario_costs
from response_adjoints import collect_rollout, backward_actor
from response_groups import GroupBalanceConfig
from response_training import load_policy_checkpoint, model_hash
from pass_stop import GoalMotorPolicy, GoalConfig, PassStopLossConfig
from test_pass_stop import cfg

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('task',['hover','pass'])
def test_group_probe_matches_vjp(dtype,task):
    torch.manual_seed(71)
    sim=RaptorSimulator();ini=sim.reset(64,seed=73,horizon=9,dtype=dtype,disturbances=DisturbanceConfig.clean())
    policy=(GoalMotorPolicy() if task=='pass' else ResponseMotorPolicy()).to(dtype)
    loss=cfg() if task=='pass' else TaskLossConfig(.01,.01,25.)
    results=[]
    for backend in ('probe','vjp'):
        group=GroupBalanceConfig(max_groups=2,min_scenarios=32,backward_backend=backend,vjp_chunk_size=1)
        record=collect_rollout(policy,sim,ini,loss,horizon=9,time_decay=0,group_config=group)
        report=backward_actor(policy,sim,record,loss,gradient_scale=.1)
        results.append(torch.cat([p.grad.flatten() for p in policy.parameters()]))
    tol=3e-6 if dtype==torch.float32 else 2e-12
    torch.testing.assert_close(results[0],results[1],rtol=tol,atol=tol)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_hover_collector_objective_and_unclipped_gradient_unchanged(dtype):
    torch.manual_seed(31)
    policy=ResponseMotorPolicy().to(dtype);sim=RaptorSimulator()
    ini=sim.reset(32,seed=75,horizon=55,dtype=dtype,disturbances=DisturbanceConfig.clean())
    loss=TaskLossConfig(.01,.01,25.)
    group=GroupBalanceConfig(max_groups=1,min_scenarios=32,clip_norm=1e30)
    trace=rollout(policy,sim,ini,55)
    direct=scenario_costs(trace,loss)
    g=torch.autograd.grad(direct.mean()*.1,list(policy.parameters()))
    record=collect_rollout(policy,sim,ini,loss,horizon=55,time_decay=0,group_config=group)
    torch.testing.assert_close(record.costs,direct,rtol=1e-6,atol=1e-6)
    backward_actor(policy,sim,record,loss,gradient_scale=.1)
    tol=3e-5 if dtype==torch.float32 else 5e-12
    for p,d in zip(policy.parameters(),g):
        torch.testing.assert_close(p.grad,d,rtol=tol,atol=tol)

def cli(work,updates,*,mission=True,extra=()):
    cmd=[sys.executable,'tools/train_response_control.py','--mode','train','--device','cpu',
         '--rotation-backend','eager','--scenarios','8','--eval-scenarios','2','--horizon','8',
         '--time-decay','0','--epsilon-p','.01','--epsilon-a','.01','--lambda-R','25',
         '--updates',str(updates),'--max-seconds','120','--development-every','1','--checkpoint-every','1',
         '--work-dir',str(work),'--threads','1','--seed','11']
    if mission:cmd+=['--mission-config','configs/pass_stop.json']
    result=subprocess.run(cmd+list(extra),cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          timeout=90,env={**os.environ,'OMP_NUM_THREADS':'1'})
    assert result.returncode==0,(result.stdout[-2000:],result.stderr[-6000:])
    return result

def equal_tree(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b

def test_cli_init_standalone_resume_evaluate_replay(tmp_path):
    # Create a real hover-protocol checkpoint with the unchanged native path (zero updates).
    hover=tmp_path/'hover';cli(hover,0,mission=False)
    base=hover/'latest.pt'
    first=tmp_path/'first';continuous=tmp_path/'continuous'
    cli(first,1,extra=['--init-checkpoint',str(base)])
    cli(continuous,2,extra=['--init-checkpoint',str(base)])
    base.unlink() # Output must not refer to a required base file.
    cli(first,2,extra=['--resume',str(first/'latest.pt')])
    a=torch.load(first/'latest.pt',weights_only=True)
    b=torch.load(continuous/'latest.pt',weights_only=True)
    assert equal_tree(a['model'],b['model'])
    assert equal_tree(a['optimizer'],b['optimizer'])
    assert equal_tree(a['rng'],b['rng'])
    policy,loaded=load_policy_checkpoint(first/'latest.pt',torch.device('cpu'),torch.float32)
    assert isinstance(policy,GoalMotorPolicy) and model_hash(policy)==loaded['model_sha256']
    cmd=[sys.executable,'tools/train_response_control.py','--mode','evaluate','--device','cpu',
         '--checkpoint',str(first/'latest.pt'),'--work-dir',str(tmp_path/'evaluate'),
         '--epsilon-p','.01','--epsilon-a','.01','--lambda-R','25']
    run=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=60)
    assert run.returncode==0,run.stderr
    result=json.loads((tmp_path/'evaluate'/'evaluation.json').read_text())
    assert result['source_match'] and result['time_proxy_seconds']>=0
    capsule=sorted((first/'audit'/'updates').glob('*.pt'))[-1]
    run=subprocess.run([sys.executable,'tools/replay_response_update.py',str(capsule)],cwd=ROOT,
                       text=True,capture_output=True,timeout=60)
    assert run.returncode==0,(run.stdout,run.stderr)
    report=json.loads(run.stdout)
    assert report['passed'] and report['gradient_equal'] and report['adam_equal']


def test_default_configuration_and_no_time_decay():
    spec=importlib.util.spec_from_file_location('entry',ROOT/'tools/train_response_control.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cwd=os.getcwd()
    try:
        os.chdir(ROOT)
        args=module.parse_args(['@configs/agile_flight.args'])
    finally:os.chdir(cwd)
    assert args.horizon==500 and args.time_decay==0 and args.scenarios==512
    assert args.group_backward=='probe' and args.rotation_backend=='compile'
    assert args.mission.via_quaternion==GoalConfig().via_quaternion

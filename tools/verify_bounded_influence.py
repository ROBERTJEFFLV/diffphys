#!/usr/bin/env python3
"""Bounded CPU/CUDA probes, not long-run convergence or a failure-checkpoint replay."""
from __future__ import annotations
import argparse
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from response_policy import ResponseMotorPolicy,ResponsePolicyConfig
from response_task import TaskLossConfig,tensors_finite
from response_adjoints import collect_boundary_rollout,backward_actor
from response_training import sample_training_scenarios,safe_global_clip
from env_l2f import L2FSimulator,L2FParams


def assert_same(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: assert_same(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_same(x,y)
    else:
        assert a==b


def parent_equivalence(baseline_root):
    # Compare only scientific/runtime state, not source bindings or wall-clock times.
    with tempfile.TemporaryDirectory() as tmp:
        saved=[]
        for name,root in [('parent',baseline_root),('new_disabled',ROOT)]:
            work=Path(tmp)/name
            cmd=[sys.executable,str(root/'tools/train_response_control.py'),
                 '--device','cpu','--dtype','float64','--scenarios','2',
                 '--eval-scenarios','2','--horizon','8','--window-steps','4',
                 '--hidden-dim','8','--memory-dim','8','--updates','2',
                 '--max-seconds','60','--work-dir',str(work)]
            result=subprocess.run(cmd,cwd=root,text=True,capture_output=True,timeout=90)
            if result.returncode:
                raise RuntimeError(result.stderr+result.stdout)
            saved.append(torch.load(work/'latest.pt',weights_only=True))
        for key in ('model','optimizer','rng','model_sha256'):
            assert_same(saved[0][key],saved[1][key])
        return {'updates':2,'model_adam_rng_bitwise_equal':True,
                'model_sha256':saved[0]['model_sha256']}


def probe(count,unit_size,device,stress=False,equilibrium=False):
    torch.manual_seed(7)
    policy=ResponseMotorPolicy(ResponsePolicyConfig()).to(device)
    optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
    simulator=L2FSimulator(L2FParams(protocol='raptor'))
    config=TaskLossConfig()
    initial,seeds=sample_training_scenarios(count//4,0,device=torch.device(device),horizon=500)
    if equilibrium:
        # Separate deterministic long-flight fixture, NOT the randomized benchmark.
        policy=policy.double()
        initial=initial.to(device,torch.float64)
        initial=type(initial)(**{f.name:getattr(initial,f.name)[:1].expand(
            count,*getattr(initial,f.name).shape[1:]).clone() for f in fields(initial)})
        c0,c1,c2=initial.thrust_coefficients[0,0]
        motor=(-c1+(c1*c1-4*c2*(c0-initial.mass[0]*9.81/4)).sqrt())/(2*c2)
        action=2*motor-1
        with torch.no_grad():
            policy.controller[-1].weight.zero_()
            policy.controller[-1].bias.fill_(float(torch.atanh(action)))
        orientation=torch.zeros_like(initial.orientation)
        orientation[:,0]=1
        initial=replace(initial,position=torch.zeros_like(initial.position),
            velocity=torch.zeros_like(initial.velocity),omega=torch.zeros_like(initial.omega),
            orientation=orientation,external_force=torch.zeros_like(initial.external_force),
            external_torque=torch.zeros_like(initial.external_torque),
            motor=torch.full_like(initial.motor,float(motor)),
            previous_action=torch.full_like(initial.previous_action,float(action)))
        optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
    def sync():
        if device=='cuda': torch.cuda.synchronize()
    sync(); start=time.monotonic()
    record=collect_boundary_rollout(policy,simulator,initial,config,horizon=500,
        window_steps=50,backprop_mode='full',time_decay=1.)
    sync(); forward_seconds=time.monotonic()-start
    costs_before=record.costs.detach().clone()
    weights_before=record.weights.clone()
    valid_before=record.valid.clone()
    rng_before=torch.get_rng_state().clone()
    if stress:
        # Controlled backward-only fault, NOT the real scene 323 or a physical fault.
        multiplier=torch.ones_like(record.costs)
        multiplier[0]=1e8
        record.costs.register_hook(lambda gradient: gradient*multiplier)
    # One graph, no repeated forward, no changes to batch compaction or failure masks.
    start=time.monotonic()
    reference=torch.autograd.grad(.1*(record.weights*record.costs).sum(),
                                 list(policy.parameters()),retain_graph=True)
    sync(); pooled_backward_seconds=time.monotonic()-start
    reference_vector=torch.cat([g.reshape(-1) for g in reference]).double()
    # Fixed experimental cap in de-averaged-vote units, not the total-gradient cap.
    clip=.1 if stress else 5.
    start=time.monotonic()
    info=backward_actor(policy,simulator,record,config,contribution_clip=clip,
                        contribution_unit_size=unit_size)
    sync(); clipped_backward_seconds=time.monotonic()-start
    aggregated=torch.cat([p.grad.reshape(-1) for p in policy.parameters()]).double()
    assert torch.equal(costs_before,record.costs.detach())
    assert torch.equal(weights_before,record.weights)
    assert torch.equal(valid_before,record.valid)
    assert torch.equal(rng_before,torch.get_rng_state())
    if not stress:
        assert info['aggregation']['clipped_units']==0
        torch.testing.assert_close(aggregated,reference_vector,atol=2e-6,rtol=2e-4)
    else:
        assert info['aggregation']['clipped_units']>=1
    safe_global_clip(policy.parameters(),10.)
    optimizer.step()
    assert tensors_finite(list(policy.parameters())+[
        v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)])
    if equilibrium:
        assert bool(record.valid.all()), 'long-flight fixture unexpectedly terminated'
    return {'scenario_count':count,'horizon_cap':500,'equilibrium_fixture':equilibrium,
        'actual_transitions':int(record.valid.sum()),
        'survived_all_500':int(record.valid.all(0).sum()),'time_decay':1.,'train_seeds':seeds,
        'stress':stress,'stress_kind':'synthetic backward multiplier' if stress else None,
        'forward_seconds':forward_seconds,'pooled_backward_seconds':pooled_backward_seconds,
        'clipped_backward_seconds':clipped_backward_seconds,
        'backward_slowdown':clipped_backward_seconds/pooled_backward_seconds,
        'forward_costs_weights_masks_rng_unchanged':True,'shadow_adam_finite':True,
        'original_pooled_norm':float(reference_vector.norm()),
        'gradient_max_abs_difference':float((aggregated-reference_vector).abs().max()),
        'aggregation':info['aggregation']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('verification-results.json'))
    parser.add_argument('--baseline-root',type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA was explicitly requested but is unavailable')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    result={'torch':torch.__version__,'device':args.device,'cpu_threads':1,
        'baseline_commit':'1564d25a210c139a94431fc6ae59d10cc0430ffe',
        'long_training_run':False,'real_17065_checkpoint_replayed':False,
        'production_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
           ['response_adjoints.py','response_training.py','tools/train_response_control.py',
            'response_policy.py','response_task.py','env_l2f.py','response_execution.py']}}
    if args.baseline_root:
        result['disabled_parent_equivalence']=parent_equivalence(args.baseline_root.resolve())
    result['probes']=[probe(8,1,args.device,False),probe(8,1,args.device,True),
                      probe(512,64,args.device,False),probe(512,64,args.device,True),
                      probe(4,1,args.device,False,True),probe(4,1,args.device,True,True)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()

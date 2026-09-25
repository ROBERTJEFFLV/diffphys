#!/usr/bin/env python3
"""Strict, same-graph CUDA audit. Never commit a candidate optimizer state.

--checkpoint may be repeated for an ordinary and a high-sensitivity checkpoint.
Use saved Adam moments; no weights-only substitution. No training/long rollout
is started by this script beyond the bounded cases explicitly selected here.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import time
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from env_l2f import L2FSimulator, L2FParams
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_groups import GroupBalanceConfig, _batched_group_vjp, normalize_group_rows
from response_task import TaskLossConfig
from response_adjoints import collect_boundary_rollout
from response_training import (load_policy_checkpoint, migrate_named_adam, sample_training_scenarios,
                               safe_global_clip, capture_rng, SOURCE_FILES)
from tools.verify_group_balance import same


def byte_equal(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and
            torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),
                        b.detach().contiguous().reshape(-1).view(torch.uint8)))


def strict_same(a, b):
    if isinstance(a, torch.Tensor):
        assert byte_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: strict_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x,y in zip(a,b): strict_same(x,y)
    else: assert a == b


def digest_boundaries(record):
    digest = hashlib.sha256()
    for t, closed in sorted(record.boundaries.items()):
        digest.update(str(t).encode())
        for state in (closed.physical, closed.policy):
            for field in fields(state):
                digest.update(getattr(state, field.name).detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def combine(gradients, parameters, groups, epsilon):
    rows = torch.cat([p.new_zeros(groups, p.numel()) if g is None else
                      g.detach().reshape(groups, -1) for p,g in zip(parameters,gradients)], 1)
    merged, report = normalize_group_rows(rows, epsilon)
    result, offset = [], 0
    for p,g in zip(parameters,gradients):
        result.append(None if g is None else merged[offset:offset+p.numel()].reshape_as(p))
        offset += p.numel()
    return rows, result, report


def shadow(policy, gradients, saved, limit):
    clone = copy.deepcopy(policy)
    opt = torch.optim.Adam(clone.parameters(), lr=3e-4)
    opt.load_state_dict(copy.deepcopy(saved))
    for p,g in zip(clone.parameters(),gradients):
        p.grad = None if g is None else g.detach().clone()
    safe_global_clip(clone.parameters(), limit)
    opt.step()
    return copy.deepcopy(clone.state_dict()), copy.deepcopy(opt.state_dict())


def enabled_parent_equivalence(baseline):
    with tempfile.TemporaryDirectory() as tmp:
        values=[]
        for name,root in [('parent',baseline),('candidate',ROOT)]:
            work=Path(tmp)/name
            command=[sys.executable,str(root/'tools/train_response_control.py'),
                '--device','cpu','--dtype','float32','--scenarios','16','--eval-scenarios','16',
                '--horizon','20','--window-steps','10','--group-balance','--group-max-groups','2',
                '--updates','2','--max-seconds','120','--work-dir',str(work)]
            done=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=150)
            if done.returncode: raise RuntimeError(done.stdout+done.stderr)
            values.append(torch.load(work/'latest.pt',weights_only=True))
        for key in ('model','optimizer','rng','model_sha256'):
            strict_same(values[0][key],values[1][key])
        return {'group_balance':True,'scenarios':64,'groups':2,'horizon':20,'updates':2,
                'Actor_Adam_RNG_byte_equal':True,'model_sha256':values[0]['model_sha256']}


def audit_case(checkpoint, args):
    device=torch.device(args.device)
    torch.manual_seed(7)
    saved=None
    if checkpoint:
        raw=torch.load(checkpoint,map_location='cpu',weights_only=True)
        dtype=getattr(torch,raw['binding']['dtype'])
        policy,saved=load_policy_checkpoint(checkpoint,device,dtype)
        binding=saved['binding']; protocol=binding['protocol']
        horizon=int(binding['horizon']); window=int(binding['window_steps'])
        scenarios=int(protocol['scenarios_per_bank']); mode=protocol['scenario_mode']
        loss=TaskLossConfig(**protocol['loss']); decay=float(binding['time_decay'])
        scale=float(binding['gradient_scale']); limit=float(binding['gradient_clip'])
        options={k:v for k,v in binding.get('group_balance',{}).items()
                 if k in {f.name for f in fields(GroupBalanceConfig)}}
        options.update(enabled=True,gru_vmap_mode='fallback')
        group=GroupBalanceConfig(**options)
        attempt=int(saved['next_update']) if args.attempt_index is None else args.attempt_index
    else:
        dtype=torch.float32
        policy=ResponseMotorPolicy(ResponsePolicyConfig()).to(device=device,dtype=dtype)
        horizon=args.horizon; window=50 if horizon%50==0 else horizon
        scenarios=args.scenarios//4; mode='raptor'; loss=TaskLossConfig()
        decay=1.; scale=.1; limit=10.; group=GroupBalanceConfig(enabled=True)
        attempt=0 if args.attempt_index is None else args.attempt_index
    optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
    if saved:
        migrate_named_adam(optimizer,policy,saved['optimizer'],saved['optimizer_parameter_names'])
    optimizer_state=copy.deepcopy(optimizer.state_dict())
    initial,seeds=sample_training_scenarios(scenarios,attempt,device=device,dtype=dtype,
                                          scenario_mode=mode,horizon=horizon)
    def sync():
        if device.type=='cuda': torch.cuda.synchronize()
    sync(); start=time.perf_counter()
    record=collect_boundary_rollout(policy,L2FSimulator(L2FParams(protocol=mode)),initial,loss,
        horizon=horizon,window_steps=window,backprop_mode='full',time_decay=decay,group_config=group)
    sync(); forward_seconds=time.perf_counter()-start
    fingerprint=digest_boundaries(record); before_rng=capture_rng()
    old_cost=record.costs.detach().clone(); old_mask=record.valid.clone(); old_weight=record.weights.clone()
    parameters=list(policy.parameters()); groups=record.group_coefficients.shape[0]
    cotangents=scale*record.group_coefficients
    sync(); start=time.perf_counter()
    expected=_batched_group_vjp(record.costs,parameters,cotangents,retain_graph=True)
    sync(); baseline_seconds=time.perf_counter()-start
    rows,merged,report=combine(expected,parameters,groups,group.gradient_epsilon)
    expected_shadow=shadow(policy,merged,optimizer_state,limit)
    result={'device':str(device),'torch':str(torch.__version__),'checkpoint_used':saved is not None,
        'checkpoint_hash':None if checkpoint is None else hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'historical_adam_restored':saved is not None,'attempt_index':attempt,'train_seeds':seeds,
        'scenarios':initial.position.shape[0],'horizon':horizon,'groups':groups,
        'valid_transitions':int(record.valid.sum()),'forward_seconds':forward_seconds,
        'reference_backward_seconds':baseline_seconds,'cases':[]}
    if device.type!='cuda':
        result.update(cuda_backend_exercised=False,
                      note='CPU uses unfused GRU; this is not a validation of either CUDA candidate.')
        return result
    result['cuda_backend_exercised']=True
    for mode in ('native','sparse'):
        # Verification is untimed; it compares every GRU operator invocation.
        checked=_batched_group_vjp(record.costs,parameters,cotangents,retain_graph=True,
                                  gru_vmap_mode='verify' if mode=='native' else 'sparse-verify')
        checked_rows,_,_=combine(checked,parameters,groups,group.gradient_epsilon)
        sync(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        stats={}
        actual=_batched_group_vjp(record.costs,parameters,cotangents,retain_graph=mode!='sparse',
                                 gru_vmap_mode=mode, gru_vmap_stats=stats)
        sync(); seconds=time.perf_counter()-start
        candidate,gradient,norm_report=combine(actual,parameters,groups,group.gradient_epsilon)
        equal=byte_equal(candidate,rows) and byte_equal(checked_rows,rows)
        equal=equal and all((a is None)==(b is None) for a,b in zip(actual,expected))
        snapshot=shadow(policy,gradient,optimizer_state,limit)
        try:
            strict_same(snapshot,expected_shadow)
            adam_equal=True
        except AssertionError:
            adam_equal=False
        same(before_rng,capture_rng())
        boundary_equal=fingerprint==digest_boundaries(record)
        assert torch.equal(old_cost,record.costs.detach()) and torch.equal(old_mask,record.valid)
        assert torch.equal(old_weight,record.weights)
        row={'mode':mode,'raw_group_gradients_equal':equal,'norm_table_equal':byte_equal(report['values'],norm_report['values']),
             'boundary_equal':boundary_equal,'adam_parameters_and_moments_equal':adam_equal,
             'backward_seconds':seconds,'forward_plus_backward_seconds':forward_seconds+seconds,
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'max_abs_gradient_error':float((rows-candidate).abs().max()),'native_gru_stats':stats}
        row['accepted']=bool(equal and adam_equal and boundary_equal and row['norm_table_equal']
                             and stats.get('calls',0)>0)
        result['cases'].append(row)
        print(json.dumps(row),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--checkpoint',action='append',type=Path,default=[])
    parser.add_argument('--baseline-root',type=Path)
    parser.add_argument('--attempt-index',type=int)
    parser.add_argument('--scenarios',type=int,default=512)
    parser.add_argument('--horizon',type=int,default=500)
    parser.add_argument('--output',type=Path,default=Path('vjp-acceleration-audit.json'))
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available(): parser.error('CUDA unavailable; cannot validate CUDA speed/equality')
    if args.scenarios<32 or args.scenarios%4 or args.horizon<1: parser.error('invalid bounded probe size')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    result={'base_commit':'0417ad3bd5b546e7e9ed3bd813e19fdf19236f6d',
            'source_hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_FILES},
            'long_training':False,'results':[]}
    if args.baseline_root:
        result['enabled_parent_equivalence']=enabled_parent_equivalence(args.baseline_root.resolve())
    for checkpoint in args.checkpoint or [None]:
        try:
            result['results'].append(audit_case(checkpoint,args))
        except Exception as error:
            result['error']=type(error).__name__ + ': ' + str(error)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            raise
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if any(not case['accepted'] for report in result['results'] for case in report['cases']):
        raise SystemExit('Equivalence failed: do not enable the accelerated candidate.')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()

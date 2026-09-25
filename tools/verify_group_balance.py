#!/usr/bin/env python3
"""Bounded CPU/CUDA checks of physical-group gradient normalization and cost."""
from __future__ import annotations
import argparse
import ast
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env_l2f import L2FParams, L2FSimulator
from response_policy import ARCHITECTURE, ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig, tensors_finite
from response_groups import GroupBalanceConfig, group_gradient_coefficients, normalize_group_rows
from response_adjoints import collect_boundary_rollout, backward_actor
from response_training import sample_training_scenarios, safe_global_clip, SOURCE_FILES


def sync(device):
    if device == 'cuda': torch.cuda.synchronize()


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    else:
        assert a == b


def require_same_actor(baseline):
    """Cross-version bitwise audits require the SAME Actor, not an architecture ablation."""
    source = ast.parse((baseline / 'response_policy.py').read_text())
    tag = next((ast.literal_eval(node.value) for node in source.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'ARCHITECTURE' for t in node.targets)), None)
    if tag != ARCHITECTURE:
        raise ValueError('parent equivalence requires the same Actor architecture; '
                         'the pre-GRU16 model is not an equivalence baseline')


def disabled_parent_equivalence(baseline):
    require_same_actor(baseline)
    with tempfile.TemporaryDirectory() as tmp:
        saved = []
        for name, root in [('parent', baseline), ('new', ROOT)]:
            work = Path(tmp)/name
            cmd = [sys.executable, str(root/'tools/train_response_control.py'),
                '--device','cpu','--dtype','float64','--scenarios','2','--eval-scenarios','2',
                '--horizon','8','--window-steps','4','--memory-dim','8',
                '--updates','2','--max-seconds','60','--work-dir',str(work)]
            r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=90)
            if r.returncode: raise RuntimeError(r.stdout+r.stderr)
            saved.append(torch.load(work/'latest.pt', weights_only=True))
        for key in ('model','optimizer','rng'): same(saved[0][key], saved[1][key])
        return {'updates':2, 'actor_adam_rng_bitwise_identical':True}


def fixture(device, hover):
    torch.manual_seed(7)
    policy = ResponseMotorPolicy(ResponsePolicyConfig()).to(device)
    z, _ = sample_training_scenarios(128, 0, horizon=500, device=torch.device(device))
    if hover:
        # Constructed full-length fixture, NOT randomized learned performance.
        # Same TWR/mass/thrust in all rows, different TTI and rising/falling lag.
        coefficients = z.thrust_coefficients[:1].expand_as(z.thrust_coefficients).clone()
        mass = z.mass[:1].expand_as(z.mass).clone()
        twr = z.thrust_to_weight[:1].expand_as(z.thrust_to_weight).clone()
        inertia = z.inertia[:1]*(z.torque_to_inertia[:1]/z.torque_to_inertia)[:,None]
        c0,c1,c2 = coefficients[0,0]
        rotor = (-c1+(c1*c1-4*c2*(c0-mass[0]*9.81/4)).sqrt())/(2*c2)
        action = 2*rotor-1
        with torch.no_grad():
            policy.readout.weight.zero_()
            policy.readout.bias.fill_(float(torch.atanh(action)))
        q = torch.zeros_like(z.orientation); q[:,0] = 1
        z = replace(z, mass=mass, thrust_to_weight=twr, inertia=inertia,
            thrust_coefficients=coefficients,
            rotor_positions=z.rotor_positions[:1].expand_as(z.rotor_positions).clone(),
            rotor_torque_constant=z.rotor_torque_constant[:1].expand_as(z.rotor_torque_constant).clone(),
            position=torch.zeros_like(z.position), velocity=torch.zeros_like(z.velocity),
            omega=torch.zeros_like(z.omega), orientation=q,
            external_force=torch.zeros_like(z.external_force), external_torque=torch.zeros_like(z.external_torque),
            motor=torch.full_like(z.motor,float(rotor)), previous_action=torch.full_like(z.previous_action,float(action)))
    return policy, z


def timed_update(device, hover, grouped, chunk_size):
    gc.collect()
    policy, initial = fixture(device, hover)
    simulator, config = L2FSimulator(L2FParams()), TaskLossConfig()
    optimizer = torch.optim.Adam(policy.parameters(),lr=3e-4)
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    sync(device); t = time.perf_counter()
    record = collect_boundary_rollout(policy,simulator,initial,config,horizon=500,
        window_steps=50,backprop_mode='full',time_decay=1.,group_config=GroupBalanceConfig(enabled=grouped, vjp_chunk_size=chunk_size))
    sync(device); forward = time.perf_counter()-t
    original = torch.autograd.grad
    t = time.perf_counter()
    with patch('torch.autograd.grad',wraps=original) as grad:
        info = backward_actor(policy,simulator,record,config)
        expected_calls = ((16 + chunk_size - 1)//chunk_size) if grouped else 1
        assert grad.call_count == expected_calls
    sync(device); backward = time.perf_counter()-t
    if grouped:
        assert record.group_balance['group_count'] == 16
        assert bool((record.group_balance['values'][:,0] == 32).all())
    if hover: assert bool(record.valid.all()), 'constructed fixture did not fly H500'
    if grouped:
        values = info['group_gradient']['values']
        eligible = values[:,0] >= 1e-12
        torch.testing.assert_close(values[eligible,1],values[eligible,3],rtol=1e-6,atol=1e-10)
    safe_global_clip(policy.parameters(),10.)
    optimizer.step()
    assert tensors_finite(list(policy.parameters())+[v for s in optimizer.state.values()
        for v in s.values() if torch.is_tensor(v)])
    report = {'grouped':grouped,'constructed_hover':hover,'batch':512,'horizon_cap':500,
        'actual_transitions':int(record.valid.sum()),'full_h500_scenes':int(record.valid.all(0).sum()),
        'group_count':16 if grouped else None,'scenes_per_group':32 if grouped else None,
        'forward_including_group_stats_seconds':forward,'backward_seconds':backward,
        'backward_calls':expected_calls,'group_vjp_count':16 if grouped else 1,'shadow_adam_finite':True,'raw_task_objective':record.metrics['task_objective'],
        'group_gradient':(None if not grouped else {**info['group_gradient'],
            'values':info['group_gradient']['values'].cpu().tolist()}),
        'cuda_peak_tensor_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None}
    return report, (record.weights.detach().cpu(),record.costs.detach().cpu(),record.valid.cpu())


def grouping_microbenchmark(device):
    _, initial = fixture(device,False)
    weights = torch.full((512,),1/512,device=device)
    config = GroupBalanceConfig(enabled=True)
    for _ in range(3): group_gradient_coefficients(weights,initial,config)
    times = []
    for _ in range(20):
        sync(device); t = time.perf_counter()
        group_gradient_coefficients(weights,initial,config)
        sync(device); times.append(time.perf_counter()-t)
    return {'batch':512,'groups':16,'repetitions':20,
            'median_seconds':statistics.median(times),'max_seconds':max(times)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--output',type=Path,default=Path('group-gradient-verification.json'))
    p.add_argument('--baseline-root',type=Path)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--group-vjp-chunk-size',type=int,default=16)
    args = p.parse_args()
    GroupBalanceConfig(vjp_chunk_size=args.group_vjp_chunk_size)
    if not 1 <= args.repeats <= 5: p.error('bounded verification permits 1-5 repetitions')
    if args.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    result = {'torch':str(torch.__version__),'device':args.device,'python':sys.version.split()[0],
        'cpu_threads':1,'long_training':False,'real_incident_checkpoint_replayed':False,'group_vjp_chunk_size':args.group_vjp_chunk_size,
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}}
    if args.baseline_root: result['disabled_parent_equivalence'] = disabled_parent_equivalence(args.baseline_root.resolve())
    result['grouping_only'] = grouping_microbenchmark(args.device)
    result['cases'] = []
    for hover in (False,True):
        reference = None
        for repeat in range(args.repeats):
            for grouped in (False,True):
                report, tensors = timed_update(args.device,hover,grouped,args.group_vjp_chunk_size)
                if reference is None: reference = tensors
                else: same(reference,tensors)
                report['repeat'] = repeat
                result['cases'].append(report)
                print(json.dumps(report),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()

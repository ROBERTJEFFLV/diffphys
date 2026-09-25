#!/usr/bin/env python3
"""Research-only single-adjoint reconstruction; NOT a production backend.

This probe is expected to expose floating-point differences in some H500 cases.
It never changes repository source, global rollout functions, checkpoints or
optimizer state outside its process. Faster but non-identical is NOT acceptance.
"""
from __future__ import annotations
import argparse
import copy
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import response_task
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_groups import (GroupBalanceConfig, group_gradient_coefficients,
                             _batched_group_vjp, normalize_group_rows)
from env_l2f import L2FSimulator, L2FParams

class _Capture:
    def __init__(self, policy):
        self.parameters=list(policy.parameters())
        self.names={id(p):n for n,p in policy.named_parameters()}
        self.ptr={p.data_ptr():i for i,p in enumerate(self.parameters)}
        self.rows=None; self.handles=[]; self.nodes=[]; self.records=[]; self.active=False
        for module in policy.modules():
            if isinstance(module,(torch.nn.Linear,torch.nn.GRUCell)):
                self.handles.append(module.register_forward_hook(self.hook))
    def hook(self,module,args,out):
        # Only known native affine nodes, bounded by this module's inputs.
        # Do not copy/decompose GRU forward or traverse previous time steps.
        stops={a.grad_fn for a in args if torch.is_tensor(a) and a.grad_fn is not None}
        seen=set(); stack=[out.grad_fn]; found=[]
        while stack:
            fn=stack.pop()
            if fn is None or fn in seen or fn in stops:continue
            seen.add(fn)
            if type(fn).__name__ in ('MmBackward0','AddmmBackward0'):
                edge=fn.next_functions[-1][0]
                if edge is not None and type(edge).__name__ in ('TBackward0','TransposeBackward0'):
                    edge=edge.next_functions[0][0]
                w=edge.variable if hasattr(edge,'variable') else None
                if w is not None and w.data_ptr() in self.ptr:
                    idx=self.ptr[w.data_ptr()]; bidx=None
                    if type(fn).__name__=='AddmmBackward0':
                        b=fn.next_functions[0][0]
                        if hasattr(b,'variable') and id(b.variable) in self.names:bidx=self.ptr[b.variable.data_ptr()]
                    elif isinstance(module,torch.nn.GRUCell):
                        name=self.names[id(self.parameters[idx])]
                        bparam=module.bias_ih if name.endswith('weight_ih') else module.bias_hh
                        if bparam is not None:bidx=self.ptr[bparam.data_ptr()]
                    rowids=self.rows
                    self.nodes.append(fn)
                    found.append(self.names[id(self.parameters[idx])])
                    self.handles.append(fn.register_prehook(lambda g,f=fn,j=idx,b=bidx,r=rowids:self.capture(f,j,b,r,g)))
                    continue
            stack.extend(n for n,_ in fn.next_functions)
        exp=2 if isinstance(module,torch.nn.GRUCell) else 1
        if len(found)!=exp: raise RuntimeError((module,found))
    def capture(self,fn,j,b,rows,g):
        if self.active:
            self.steps+=1
            with torch.no_grad():
                delta=g[0].detach()
                masked=torch.where(self.membership[:,rows,None],delta[None],0)
                def local(d):
                    vals=fn(d)
                    return vals[-1].t()
                gw=torch.vmap(local)(masked)
                if self.accum[j] is None:self.accum[j]=gw.clone()
                else:self.accum[j].add_(gw)
                if b is not None:
                    gb=torch.vmap(lambda d:d.sum(0))(masked)
                    if self.accum[b] is None:self.accum[b]=gb.clone()
                    else:self.accum[b].add_(gb)
    def close(self):
        for handle in self.handles:handle.remove()
        self.handles.clear()
        self.nodes.clear()
    def backward(self,costs,seeds):
        self.active=True;self.membership=seeds!=0;self.accum=[None]*len(self.parameters);self.steps=0
        try:
            ref=torch.autograd.grad(costs,self.parameters,grad_outputs=seeds.sum(0),allow_unused=True,retain_graph=False)
        finally:self.active=False
        return self.accum


def run_probe(*, scenes=512, horizon=500, device='cpu', dtype=torch.float32):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = ResponseMotorPolicy(ResponsePolicyConfig()).to(device=device, dtype=dtype)
    initial = response_task.sample_scenarios(scenes, seed=31000007, horizon=horizon,
                                           device=torch.device(device), dtype=dtype)
    simulator = L2FSimulator(L2FParams())
    cfg = response_task.TaskLossConfig()
    with torch.no_grad():
        original = response_task.rollout(policy, simulator, initial, horizon, time_decay=1.)
    cap = _Capture(policy)
    # Isolated copy, with one metadata assignment; never mutate response_task.rollout.
    source = inspect.getsource(response_task.rollout)
    old = 'output = policy(current_observation.index_select(0, indices), live.policy)'
    if source.count(old) != 1:
        raise RuntimeError('rollout changed; review this diagnostic before use')
    source = source.replace(old, 'capture.rows = indices\n            ' + old)
    namespace = dict(vars(response_task), capture=cap)
    exec(compile(source, '<isolated-row-metadata-probe>', 'exec'), namespace)
    def sync():
        if device == 'cuda': torch.cuda.synchronize()
    try:
        trace = namespace['rollout'](policy, simulator, initial, horizon, time_decay=1.)
        unchanged = all(torch.equal(getattr(original, field), getattr(trace, field))
                        for field in ('actions', 'positions', 'velocities', 'omegas', 'valid'))
        if not unchanged:
            raise RuntimeError('probe changed forward values')
        costs = response_task.scenario_costs(trace, cfg)
        weights = response_task.risk_weights(costs, cfg)
        coef, _ = group_gradient_coefficients(weights, initial, GroupBalanceConfig(enabled=True))
        seeds = .1 * coef
        sync(); start = time.perf_counter()
        reference = _batched_group_vjp(costs, list(policy.parameters()), seeds, retain_graph=True)
        sync(); reference_seconds = time.perf_counter()-start
        sync(); start = time.perf_counter()
        candidate = cap.backward(costs, seeds)
        sync(); capture_seconds = time.perf_counter()-start
        groups = coef.shape[0]
        def rows(values):
            return torch.cat([v.reshape(groups,-1) for v in values],1)
        a, b = rows(reference), rows(candidate)
        expected, _ = normalize_group_rows(a, 1e-12)
        actual, _ = normalize_group_rows(b, 1e-12)
        original_state = copy.deepcopy(policy.state_dict())
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        saved_optim = copy.deepcopy(optimizer.state_dict())
        after = []
        for combined in (expected, actual):
            policy.load_state_dict(original_state)
            optimizer.load_state_dict(copy.deepcopy(saved_optim))
            offset = 0
            for p in policy.parameters():
                p.grad = combined[offset:offset+p.numel()].reshape_as(p).clone()
                offset += p.numel()
            optimizer.step()
            after.append((copy.deepcopy(policy.state_dict()),copy.deepcopy(optimizer.state_dict())))
        weights_equal = all(torch.equal(after[0][0][key], after[1][0][key]) for key in original_state)
        moments_equal = all(torch.equal(value, after[1][1]['state'][pid][key])
                            for pid, state in after[0][1]['state'].items() for key,value in state.items())
        return {'device':device,'dtype':str(dtype),'scenarios':scenes,'horizon_cap':horizon,
                'valid_transitions':int(trace.valid.sum()),'group_count':groups,
                'forward_equal':unchanged,'group_gradients_equal':torch.equal(a,b),
                'normalized_gradient_equal':torch.equal(expected,actual),
                'adam_weights_equal':weights_equal,'adam_moments_equal':moments_equal,
                'max_abs_group_error':float((a-b).abs().max()),
                'relative_l2_group_error':float((a.double()-b.double()).norm()/a.double().norm().clamp_min(1e-30)),
                'reference_backward_seconds':reference_seconds,'capture_backward_seconds':capture_seconds,
                'native_affine_replays':cap.steps,'production_enabled':False,
                'decision':'research-only; any nonidentity disqualifies strict-equivalence adoption'}
    finally:
        cap.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--scenarios',type=int,default=512)
    parser.add_argument('--horizon',type=int,default=500)
    parser.add_argument('--dtype',choices=('float32','float64'),default='float32')
    parser.add_argument('--output',type=Path,default=Path('capture-probe.json'))
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available(): parser.error('CUDA unavailable')
    if args.scenarios<32 or args.horizon<1: parser.error('need >=32 scenes and a positive horizon')
    report=run_probe(scenes=args.scenarios,horizon=args.horizon,device=args.device,dtype=getattr(torch,args.dtype))
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':
    main()

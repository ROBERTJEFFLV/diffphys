"""The production compiler must keep original physics and gradient semantics."""
from __future__ import annotations

import copy
from dataclasses import fields, replace

import pytest
import torch

from diffphys.compiled_physics import CompiledRaptorSimulator, make_simulator, physics_backend_contract
from diffphys.env_raptor import IMMUTABLE_TAPES, RaptorParams, RaptorSimulator
from tools.replay_response_update import compare_values


def exact(a,b):
    assert compare_values(a,b)[0]


def test_device_selection_and_strict_backend_binding():
    assert type(make_simulator(RaptorParams(),'cpu')) is RaptorSimulator
    assert physics_backend_contract('cpu')=={'physics':'native-cpu-reference'}
    contract=physics_backend_contract('cuda')
    assert contract['physics']=='compiled' and contract['torch']=='2.10'
    assert len(contract['native_boundaries'])==4
    assert not contract['options']['triton.cudagraphs']


def test_compiler_wraps_original_bound_step_fullgraph_and_keeps_history(monkeypatch):
    calls=[]
    compile_original=torch.compile
    def capture(function,**options):
        calls.append((function.__func__,options))
        return compile_original(function,backend='aot_eager',fullgraph=True,dynamic=True)
    monkeypatch.setattr(torch,'compile',capture)
    native=RaptorSimulator();compiled=CompiledRaptorSimulator()
    assert calls[0][0] is RaptorSimulator.step
    assert calls[0][1]['fullgraph'] and calls[0][1]['dynamic']
    assert calls[0][1]['backend'].__name__=='strict_inductor'
    state=native.reset(4,seed=73,horizon=3,dtype=torch.float64)
    names=('position','velocity','orientation','omega','motor','previous_velocity')
    state=replace(state,**{name:getattr(state,name).requires_grad_() for name in names})
    action=torch.full_like(state.motor,.13,requires_grad=True)
    inputs=[getattr(state,name) for name in names]+[action]
    expected,actual=native.step(state,action),compiled.step(state,action)
    for f in fields(state):exact(getattr(expected,f.name),getattr(actual,f.name))
    for name in IMMUTABLE_TAPES:assert getattr(actual,name) is getattr(state,name)
    grads=[torch.autograd.grad(sum(getattr(result,name).square().sum() for name in names),inputs)
           for result in (expected,actual)]
    for a,b in zip(*grads):torch.testing.assert_close(a,b,rtol=3e-9,atol=2e-10)
    shrunk=replace(state,**{f.name:getattr(state,f.name)[:2] for f in fields(state) if f.name not in IMMUTABLE_TAPES})
    expected,actual=native.step(shrunk,action[:2]),compiled.step(shrunk,action[:2])
    for f in fields(state):exact(getattr(expected,f.name),getattr(actual,f.name))
    with pytest.raises(ValueError,match='action must match'):compiled.step(state,action[:,:3])
    torch._dynamo.reset()


def test_compiler_failure_propagates_without_fallback(monkeypatch):
    def fail(*args,**kw):raise RuntimeError('deliberate compiler failure')
    monkeypatch.setattr(torch,'compile',fail)
    with pytest.raises(RuntimeError,match='deliberate compiler failure'):CompiledRaptorSimulator()
    from diffphys.strict_inductor import strict_inductor
    monkeypatch.setattr(torch,'__version__','2.11.0')
    with pytest.raises(RuntimeError,match='verified PyTorch 2.10'):strict_inductor(None,None)


def test_strict_backend_does_not_mutate_global_compiler_tables():
    from torch._inductor.decomposition import decompositions
    from torch._inductor.lowering import lowerings,fallbacks
    from diffphys.strict_inductor import strict_inductor
    before=dict(lowerings),dict(decompositions),set(fallbacks)
    def arithmetic(x,y):
        return ((x*y).sum(-1),torch.linalg.cross(x[:,:3],y[:,:3],dim=-1),
                torch.linalg.vector_norm(x,dim=-1),x[:,:3,None]@y[:,None,:3])
    x=torch.tensor([[1e8,1.,-1e8,1.],[.07,-.13,.71,.91]],requires_grad=True)
    y=torch.tensor([[1.,1.,1.,1.],[.19,.23,-.31,.89]],requires_grad=True)
    expected=arithmetic(x,y)
    actual=torch.compile(arithmetic,backend=strict_inductor,fullgraph=True)(x,y)
    exact(expected,actual)
    gradients=[torch.autograd.grad(sum(v.square().sum() for v in values),(x,y)) for values in (expected,actual)]
    for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
    assert lowerings==before[0] and decompositions==before[1] and fallbacks==before[2]
    torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_h500_compiled_trajectory_raw_group_gradients_and_adam_match_native():
    from diffphys.response_adjoints import collect_rollout,backward_actor
    from diffphys.response_execution import critical_gc
    from diffphys.response_groups import GroupBalanceConfig
    from diffphys.response_policy import ResponseMotorPolicy
    from diffphys.response_task import TaskLossConfig,rollout
    from diffphys.response_training import sample_training_scenarios,safe_global_clip
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(31)
    actor=ResponseMotorPolicy().cuda();other=copy.deepcopy(actor)
    initial,_=sample_training_scenarios(32,49,horizon=500,sampling='coverage128',device='cuda')
    native=RaptorSimulator();compiled=make_simulator(RaptorParams(),'cuda')
    with torch.no_grad():
        traces=[rollout(policy,sim,initial,500,time_decay=1.) for policy,sim in [(actor,native),(other,compiled)]]
    for f in fields(traces[0]):
        a,b=getattr(traces[0],f.name),getattr(traces[1],f.name)
        if torch.is_tensor(a):exact(a,b)
    for f in fields(initial):exact(getattr(traces[0].end.physical,f.name),getattr(traces[1].end.physical,f.name))
    exact(traces[0].end.policy.memory,traces[1].end.policy.memory)
    del traces
    results=[]
    for policy,sim in [(actor,native),(other,compiled)]:
        optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
        loss=TaskLossConfig(.01,.01,25.)
        with critical_gc():
            record=collect_rollout(policy,sim,initial,loss,horizon=500,time_decay=1.,
                group_config=GroupBalanceConfig(max_groups=128,min_scenarios=1,layout='coverage128'))
            backward=backward_actor(policy,sim,record,loss,gradient_scale=.1)
            record=replace(record,costs=record.costs.detach())
        safe_global_clip(policy.parameters(),10.)
        raw=record.probe.rows.detach().cpu().clone()
        report=backward['group_gradient']
        gradients={n:p.grad.detach().cpu().clone() for n,p in policy.named_parameters()}
        optimizer.step()  # Disposable equality check; never persisted or used for learning.
        results.append((raw,report,gradients,{n:p.detach().cpu().clone() for n,p in policy.named_parameters()},
                        optimizer.state_dict()))
        del record,backward
    exact(results[0],results[1])
    torch._dynamo.reset()

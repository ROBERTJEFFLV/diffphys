"""Pre-merge norm bounds; no inference that clipped gradients improve flight."""
from dataclasses import fields
from pathlib import Path
import copy
import json
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from response_adjoints import _bounded_contribution_gradients, backward_actor, collect_boundary_rollout
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig, sample_scenarios
from env_l2f import L2FSimulator, L2FParams
from response_training import train
from tools.train_response_control import parse_args


def linear_problem(vectors, dtype=torch.float64):
    matrix = torch.tensor(vectors, dtype=dtype)
    parameter = torch.nn.Parameter(torch.zeros(matrix.shape[1], dtype=dtype))
    return parameter, matrix @ parameter + 1.0, matrix


def bounded(parameter, costs, weights, limit, unit_size=1, scale=1.0):
    return _bounded_contribution_gradients(
        costs, weights, [parameter], gradient_scale=scale,
        max_norm=limit, unit_size=unit_size,
    )


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('unit_size', [1, 2, 5])
def test_inactive_clipping_preserves_weighted_sum_not_double_mean(dtype, unit_size):
    p, costs, matrix = linear_problem([[1.,2.],[-2.,3.],[4.,-1.],[2.,5.],[-3.,1.]], dtype)
    # Includes an upper-tail increment: weights deliberately sum to 1.5, not 1.
    weights = torch.tensor([.2,.2,.7,.2,.2], dtype=dtype)
    grads, report = bounded(p, costs, weights, 1e10, unit_size, .1)
    expected = .1 * (weights[:,None] * matrix).sum(0)
    torch.testing.assert_close(grads[0], expected)
    assert report['clipped_units'] == 0
    assert report['unit_count'] == (5 + unit_size - 1)//unit_size


def test_outlier_is_clipped_before_aggregation():
    p, costs, _ = linear_problem([[1.,0.],[1.,0.],[1.,0.],[-1e9,0.]])
    grads, report = bounded(p, costs, torch.full((4,), .25, dtype=p.dtype), 2.0)
    torch.testing.assert_close(grads[0], torch.tensor([.25,0.], dtype=p.dtype))
    assert report['clipped_units'] == 1
    assert report['max_unit_contribution_bound'] == .5
    assert report['pooled_gradient_norm'] > 1e8
    assert report['aggregated_gradient_norm'] == pytest.approx(.25)


@pytest.mark.parametrize('outlier', [1e3, 1e12, 1e30])
def test_huge_finite_float32_votes_have_finite_bounded_result(outlier):
    p, costs, _ = linear_problem([[outlier, -outlier],[0.,0.]], torch.float32)
    grads, report = bounded(p, costs, torch.full((2,), .5), 3.0)
    assert torch.isfinite(grads[0]).all()
    assert float(grads[0].double().norm()) <= 1.5*(1+1e-6)
    assert report['clipped_units'] == 1


def test_fixed_weight_single_replacement_bound():
    results = []
    for outlier in ([1e6,2e6],[-2e9,3e9]):
        p, costs, _ = linear_problem([[1.,0.],[0.,1.],outlier])
        grads, _ = bounded(p, costs, torch.full((3,),1/3,dtype=p.dtype), 2.)
        results.append(grads[0])
    assert float((results[0]-results[1]).norm()) <= 4/3 + 1e-12


def test_zero_gradients_and_unused_parameter_preserved():
    p, costs, _ = linear_problem([[0.,0.],[0.,0.]])
    unused = torch.nn.Parameter(torch.ones(1,dtype=p.dtype))
    grads, report = _bounded_contribution_gradients(
        costs, torch.tensor([.5,.5],dtype=p.dtype), [p,unused],
        gradient_scale=.1, max_norm=2., unit_size=1)
    assert torch.equal(grads[0],torch.zeros_like(p))
    assert grads[1] is None
    assert report['clipped_units'] == 0
    assert p.grad is None and unused.grad is None


def test_nonfinite_derivative_never_published():
    p = torch.nn.Parameter(torch.tensor([0.],dtype=torch.float64))
    p.grad = torch.tensor([7.],dtype=p.dtype)
    costs = p.sqrt()  # finite forward, infinite derivative
    with pytest.raises(FloatingPointError,match='contribution'):
        bounded(p,costs,torch.ones_like(costs),1.)
    assert p.grad.item() == 7.


def test_last_partial_block_uses_its_actual_size():
    p,costs,_ = linear_problem([[10.,0.],[10.,0.],[-10.,0.]])
    grads,report = bounded(p,costs,torch.full((3,),1/3,dtype=p.dtype),1.,2)
    # (2/3)*(+1) + (1/3)*(-1), NOT equal weighting of the two blocks.
    torch.testing.assert_close(grads[0],torch.tensor([1/3,0.],dtype=p.dtype))
    assert report['unit_sizes'] == [2,1]
    assert report['unit_kind'] == 'block'


def test_block_clipping_is_not_a_per_scene_bound():
    p,costs,_ = linear_problem([[1e9,0.],[-1e9+2,0.]])
    grads,report = bounded(p,costs,torch.full((2,),.5,dtype=p.dtype),5.,2)
    assert report['clipped_units'] == 0  # cancellation inside a block is invisible
    assert report['unit_kind'] == 'block'
    torch.testing.assert_close(grads[0],torch.tensor([1.,0.],dtype=p.dtype))


@pytest.mark.parametrize('decay',[0.,1.])
@pytest.mark.parametrize('unit_size',[1,3])
def test_real_graph_forward_rng_and_no_clip_gradient_equivalence(decay,unit_size):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    pc=ResponsePolicyConfig(hidden_dim=8,memory_dim=8)
    policy=ResponseMotorPolicy(pc).double()
    sim=L2FSimulator(L2FParams(protocol='raptor'))
    initial=sample_scenarios(5,seed=31000007,dtype=torch.float64,horizon=8)
    lc=TaskLossConfig()
    reference=collect_boundary_rollout(policy,sim,initial,lc,horizon=8,window_steps=4,
                                      backprop_mode='full',time_decay=decay)
    backward_actor(policy,sim,reference,lc)
    expected=[None if p.grad is None else p.grad.clone() for p in policy.parameters()]
    policy.zero_grad(set_to_none=True)
    record=collect_boundary_rollout(policy,sim,initial,lc,horizon=8,window_steps=4,
                                   backprop_mode='full',time_decay=decay)
    torch.testing.assert_close(reference.costs,record.costs,atol=0,rtol=0)
    assert torch.equal(reference.weights,record.weights)
    assert torch.equal(reference.valid,record.valid)
    before=copy.deepcopy(record.boundaries)
    rng=torch.get_rng_state().clone()
    info=backward_actor(policy,sim,record,lc,contribution_clip=1e12,
                        contribution_unit_size=unit_size)
    assert torch.equal(rng,torch.get_rng_state())
    for t,z in record.boundaries.items():
        for part in ('physical','policy'):
            for f in fields(getattr(z,part)):
                assert torch.equal(getattr(getattr(z,part),f.name),getattr(getattr(before[t],part),f.name))
    for p,g in zip(policy.parameters(),expected):
        if g is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad,g,rtol=1e-10,atol=1e-11)
    assert info['aggregation']['clipped_units'] == 0


@pytest.mark.parametrize('options',[
    ['--contribution-clip','-1'], ['--contribution-clip','nan'],
    ['--contribution-clip','inf'], ['--contribution-unit-size','0'],
    ['--contribution-clip','1','--backprop-mode','windowed'],
    ['--contribution-clip','1','--agc','.01'],
])
def test_invalid_or_conflicting_cli_rejected(options):
    with pytest.raises(SystemExit):
        parse_args(options)


def training_args(path,updates=2,extra=()):
    return parse_args(['--device','cpu','--dtype','float64','--scenarios','2',
        '--eval-scenarios','2','--horizon','8','--window-steps','4','--hidden-dim','8',
        '--memory-dim','8','--updates',str(updates),'--max-seconds','60',
        '--development-every','1','--checkpoint-every','1','--contribution-clip','.01',
        '--work-dir',str(path),*extra])


def run_training(args):
    pc=ResponsePolicyConfig(**{f.name:getattr(args,f.name) for f in fields(ResponsePolicyConfig)})
    lc=TaskLossConfig(**{f.name:getattr(args,f.name) for f in fields(TaskLossConfig)})
    return train(args,pc,lc)


def test_clipped_training_exact_resume_and_binding(tmp_path):
    full, resumed=tmp_path/'full',tmp_path/'resumed'
    run_training(training_args(full,2))
    run_training(training_args(resumed,1))
    run_training(training_args(resumed,2,('--resume',str(resumed/'latest.pt'))))
    a=torch.load(full/'latest.pt',weights_only=True)
    b=torch.load(resumed/'latest.pt',weights_only=True)
    assert a['model_sha256']==b['model_sha256']
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    for pid,state in a['optimizer']['state'].items():
        for k,v in state.items():
            torch.testing.assert_close(v,b['optimizer']['state'][pid][k],atol=0,rtol=0)
    assert a['binding']['contribution_clip']==.01
    assert a['binding']['contribution_unit_size']==1
    assert '+bounded-contribution' in a['binding']['algorithm']
    row=json.loads((full/'history.jsonl').read_text().splitlines()[-1])
    assert row['gradient_aggregation']['unit_kind']=='scene'
    assert row['gradient_aggregation']['clipped_units']>0
    with pytest.raises(ValueError,match='configuration'):
        run_training(training_args(resumed,3,('--resume',str(resumed/'latest.pt'),
                     '--contribution-clip','.02')))


def test_failed_contribution_does_not_call_adam_or_publish_partial_step(tmp_path,monkeypatch):
    import response_training as training
    run_training(training_args(tmp_path/'init',0))
    before=torch.load(tmp_path/'init/latest.pt',weights_only=True)
    calls=[]
    original=training.backward_actor
    def fail(*args,**kwargs):
        original(*args,**kwargs)
        raise FloatingPointError('injected contribution failure')
    monkeypatch.setattr(training,'backward_actor',fail)
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**kw: calls.append(True))
    with pytest.raises(FloatingPointError):
        run_training(training_args(tmp_path/'failed',1))
    saved=torch.load(tmp_path/'failed/failure.pt',weights_only=True)
    assert not calls
    assert saved['model_sha256']==before['model_sha256']
    assert saved['optimizer']['state']=={}
    assert saved['progress']['updates']==0


def test_bounded_h500_forward_can_have_explosive_time_decayed_gradient():
    from response_task import _GradientDecay
    import math
    p=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64))
    x=torch.zeros_like(p)
    for _ in range(500):
        x=torch.tanh(1.1*_GradientDecay.apply(x,math.exp(-.01))+p)
        assert x.item()==0.0  # never leaves its state boundary
    costs=torch.cat([1+p,1+p,1+p,1-x])
    grads,report=bounded(p,costs,torch.full((4,),.25,dtype=p.dtype),2.)
    assert report['pooled_gradient_norm']>1e15
    assert report['clipped_units']==1
    torch.testing.assert_close(grads[0],torch.tensor([.25],dtype=p.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_cuda_retained_graph_matches_pooled_without_active_clipping(dtype):
    torch.manual_seed(7)
    policy=ResponseMotorPolicy(ResponsePolicyConfig()).to(device='cuda',dtype=dtype)
    sim=L2FSimulator(L2FParams())
    initial=sample_scenarios(8,seed=31000007,horizon=20,device='cuda',dtype=dtype)
    cfg=TaskLossConfig()
    record=collect_boundary_rollout(policy,sim,initial,cfg,horizon=20,window_steps=10,
                                   backprop_mode='full',time_decay=1.)
    objective=.1*(record.costs*record.weights).sum()
    expected=torch.autograd.grad(objective,list(policy.parameters()),retain_graph=True)
    backward_actor(policy,sim,record,cfg,contribution_clip=1e10,contribution_unit_size=1)
    for p,g in zip(policy.parameters(),expected):
        torch.testing.assert_close(p.grad,g,rtol=2e-4,atol=2e-6)

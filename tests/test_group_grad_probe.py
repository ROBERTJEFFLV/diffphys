"""Single traversal group probes must reproduce independent full group VJPs."""

from loss_fixtures import test_loss
from dataclasses import replace
import copy
from unittest.mock import patch

import pytest
import torch

from response_groups import GroupBalanceConfig, normalize_group_rows
from response_adjoints import collect_rollout, backward_actor
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig, rollout
from response_training import sample_training_scenarios


def cfg(backend="probe", cap=1., chunk=16):
    return GroupBalanceConfig(max_groups=128, min_scenarios=1, layout="coverage128",
                              clip_norm=cap, vjp_chunk_size=chunk, backward_backend=backend)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("decay", [0., 1.])
def test_one_graph_traversal_matches_every_group_and_leaves_actor_forward_unchanged(dtype, decay):
    from response_grad_probe import GroupGradientProbe
    torch.manual_seed(31)
    initial, _ = sample_training_scenarios(32, 0, horizon=12, sampling="coverage128", dtype=dtype)
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).to(dtype=dtype)
    # Native forward uses the original whole pool, including deliberately early failures.
    limit = initial.position_limit.clone()
    limit[::11] = .001
    initial = replace(initial, position_limit=limit)
    original = copy.deepcopy(policy)
    from env_raptor import RaptorSimulator
    simulator = RaptorSimulator()
    loss = test_loss()
    reference = collect_rollout(original, simulator, initial, loss, horizon=12,
                                 time_decay=decay, group_config=cfg("vjp", .08))
    record = collect_rollout(policy, simulator, initial, loss, horizon=12,
                              time_decay=decay, group_config=cfg("probe", .08))
    assert torch.equal(record.costs, reference.costs)
    assert record.metrics == reference.metrics
    with patch("torch.autograd.grad", wraps=torch.autograd.grad) as calls:
        result = backward_actor(policy, simulator, record, loss)
    assert calls.call_count == 1
    assert result["group_gradient"]["graph_traversals"] == 1
    assert result["group_gradient"]["backend"] == "probe"
    assert not policy._forward_hooks and not policy.response_memory._forward_hooks
    expected = backward_actor(original, simulator, reference, loss)["group_gradient"]
    tol = dict(rtol=3e-4, atol=2e-6) if dtype == torch.float32 else dict(rtol=3e-10, atol=1e-11)
    for actual, target in zip(policy.parameters(), original.parameters()):
        torch.testing.assert_close(actual.grad, target.grad, **tol)
    torch.testing.assert_close(result["group_gradient"]["values"], expected["values"], **tol)
    # Nothing installed on the deployment Actor or its weights/state_dict.
    for name,value in policy.state_dict().items():
        assert torch.equal(value, original.state_dict()[name])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gru_gate_derivative_matches_native_gru_including_recurrent_bias(dtype):
    from response_grad_probe import gru_gate_deltas
    torch.manual_seed(79)
    gru = torch.nn.GRUCell(5, 7).to(dtype=dtype)
    x, h, d = [torch.randn(9, n, dtype=dtype) for n in (5, 7, 7)]
    y = gru(x,h)
    expected = torch.autograd.grad(y, tuple(gru.parameters()), d)
    di,dh = gru_gate_deltas(gru, x, h, d)
    actual = [di.T@x,dh.T@h,di.sum(0),dh.sum(0)]
    tol = dict(rtol=2e-5,atol=2e-6) if dtype == torch.float32 else dict(rtol=1e-12,atol=1e-12)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,**tol)


def test_probe_rejects_mutated_weights_before_replay_and_cleans_up():
    from env_raptor import RaptorSimulator
    initial,_=sample_training_scenarios(32,0,horizon=4,sampling="coverage128",dtype=torch.float64)
    policy=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    rec=collect_rollout(policy,RaptorSimulator(),initial,test_loss(),horizon=4,group_config=cfg())
    with torch.no_grad():
        policy.readout.weight.add_(.1)
    with pytest.raises((ValueError,RuntimeError),match="changed|modified|version"):
        backward_actor(policy,RaptorSimulator(),rec,test_loss())
    assert not policy.response_memory._forward_hooks
    assert not policy.readout._forward_hooks



def test_probe_raw_group_vectors_match_serial_vjp_with_recurrent_parameter_reuse():
    from response_grad_probe import GroupGradientProbe
    from response_groups import physics_group_layout
    from env_raptor import RaptorSimulator
    torch.manual_seed(77)
    # Unequal adaptive groups verify the N/n_g seed normalization as well.
    initial=RaptorSimulator().reset(65,seed=37,horizon=6,dtype=torch.float64)
    config=GroupBalanceConfig(clip_norm=.07)
    a=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    b=copy.deepcopy(a)
    loss=test_loss()
    r=collect_rollout(a,RaptorSimulator(),initial,loss,horizon=6,group_config=config)
    s=collect_rollout(b,RaptorSimulator(),initial,loss,horizon=6,
                      group_config=replace(config,backward_backend="vjp"))
    rows=[]
    for seed in s.group_coefficients:
        grad=torch.autograd.grad(s.costs,tuple(b.parameters()),.37*seed,retain_graph=True)
        rows.append(torch.cat([g.flatten() for g in grad]))
    expected=torch.stack(rows)
    backward_actor(a,RaptorSimulator(),r,loss,gradient_scale=.37)
    torch.testing.assert_close(r.probe.rows,expected,rtol=2e-11,atol=2e-11)
    combined,_=normalize_group_rows(expected,config.gradient_epsilon,max_norm=config.clip_norm)
    torch.testing.assert_close(torch.cat([p.grad.flatten() for p in a.parameters()]),
                               combined,rtol=2e-11,atol=2e-11)
    with pytest.raises(RuntimeError,match="single-use"):
        backward_actor(a,RaptorSimulator(),r,loss)


@pytest.mark.parametrize("decay", [0., 1.])
def test_long_horizon_probe_handles_delay_pulses_and_terminations_across_metric_boundary(decay):
    from env_raptor import RaptorSimulator
    torch.manual_seed(6)
    initial,_=sample_training_scenarios(32,8,horizon=80,sampling="coverage128",dtype=torch.float64)
    a=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    b=copy.deepcopy(a)
    loss=test_loss()
    r=collect_rollout(a,RaptorSimulator(),initial,loss,horizon=80,time_decay=decay,group_config=cfg(cap=.1))
    s=collect_rollout(b,RaptorSimulator(),initial,loss,horizon=80,time_decay=decay,group_config=cfg("vjp",cap=.1))
    assert torch.equal(r.costs,s.costs)
    assert r.metrics == s.metrics
    backward_actor(a,RaptorSimulator(),r,loss)
    backward_actor(b,RaptorSimulator(),s,loss)
    torch.testing.assert_close(torch.cat([p.grad.flatten() for p in a.parameters()]),
                               torch.cat([p.grad.flatten() for p in b.parameters()]),
                               rtol=2e-9,atol=2e-10)
    assert r.probe.observed["gru"] == r.probe.observed["linear"] == 80


def test_failing_probe_does_not_publish_partial_parameter_gradients(monkeypatch):
    from response_grad_probe import GroupGradientProbe
    from env_raptor import RaptorSimulator
    initial,_=sample_training_scenarios(32,1,horizon=3,sampling="coverage128",dtype=torch.float64)
    p=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    for param in p.parameters():
        param.grad=torch.ones_like(param)*7
    r=collect_rollout(p,RaptorSimulator(),initial,test_loss(),horizon=3,group_config=cfg())
    original=GroupGradientProbe._linear_partial
    def poison(self,*args):
        original(self,*args)
        self.rows[0,0]=float("nan")
    monkeypatch.setattr(GroupGradientProbe,"_linear_partial",poison)
    with pytest.raises(FloatingPointError):
        backward_actor(p,RaptorSimulator(),r,test_loss())
    assert all(torch.equal(param.grad,torch.ones_like(param)*7) for param in p.parameters())
    assert not r.probe._tensor_handles and not r.probe._payloads
    assert not p.response_memory._forward_hooks and not p.readout._forward_hooks


def test_forward_failure_removes_hooks_without_changing_actor(monkeypatch):
    from env_raptor import RaptorSimulator
    initial,_=sample_training_scenarios(32,1,horizon=3,sampling="coverage128",dtype=torch.float64)
    p=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    state=copy.deepcopy(p.state_dict())
    def fail(*args):
        raise RuntimeError("injected physics failure")
    sim=RaptorSimulator()
    monkeypatch.setattr(sim,"step",fail)
    with pytest.raises(RuntimeError,match="injected"):
        collect_rollout(p,sim,initial,test_loss(),horizon=3,group_config=cfg())
    assert not p.response_memory._forward_hooks and not p.readout._forward_hooks
    assert all(torch.equal(state[k],v) for k,v in p.state_dict().items())


def test_actor_forward_raw_tensors_remain_bitwise_equal_with_probe():
    from env_raptor import RaptorSimulator
    from response_grad_probe import GroupGradientProbe
    from response_groups import physics_group_layout
    initial,_=sample_training_scenarios(32,4,horizon=53,sampling="coverage128",dtype=torch.float64)
    p=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    probe=GroupGradientProbe(p,*physics_group_layout(initial,cfg()))
    with probe.capture():
        actual=rollout(p,RaptorSimulator(),initial,53,time_decay=1.,actor_probe=probe)
    plain=rollout(p,RaptorSimulator(),initial,53,time_decay=1.)
    for key in ("actions","observations","positions","velocities","omegas","valid"):
        assert torch.equal(getattr(actual,key),getattr(plain,key))
    probe.close()


def test_repeated_capture_backward_does_not_leave_live_hooks_or_saved_inputs():
    from env_raptor import RaptorSimulator
    initial,_=sample_training_scenarios(32,7,horizon=3,sampling="coverage128",dtype=torch.float64)
    p=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    for _ in range(3):
        r=collect_rollout(p,RaptorSimulator(),initial,test_loss(),horizon=3,group_config=cfg())
        backward_actor(p,RaptorSimulator(),r,test_loss())
        assert not r.probe._tensor_handles and not r.probe._payloads
        assert not p.response_memory._forward_hooks and not p.readout._forward_hooks


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_cuda_fused_gru_single_backward_matches_reference(dtype):
    from env_raptor import RaptorSimulator
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    initial,_=sample_training_scenarios(32,3,horizon=65,sampling="coverage128",device="cuda",dtype=dtype)
    torch.manual_seed(7)
    a=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).to(device="cuda",dtype=dtype)
    b=copy.deepcopy(a)
    loss=test_loss()
    r=collect_rollout(a,RaptorSimulator(),initial,loss,horizon=65,group_config=cfg(cap=.1))
    s=collect_rollout(b,RaptorSimulator(),initial,loss,horizon=65,group_config=cfg("vjp",cap=.1))
    assert torch.equal(r.costs,s.costs)
    backward_actor(a,RaptorSimulator(),r,loss)
    backward_actor(b,RaptorSimulator(),s,loss)
    tol=dict(rtol=5e-4,atol=3e-5) if dtype==torch.float32 else dict(rtol=3e-9,atol=2e-10)
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa.grad,pb.grad,**tol)
    # Same-backend replay must be bitwise identical on the same hardware.
    c=collect_rollout(a,RaptorSimulator(),initial,loss,horizon=65,group_config=cfg(cap=.1))
    old=[pa.grad.clone() for pa in a.parameters()]
    backward_actor(a,RaptorSimulator(),c,loss)
    for pa,old_grad in zip(a.parameters(),old):
        assert torch.equal(pa.grad,old_grad)



def test_cli_defaults_to_probe_reference_is_explicit_and_checkpoint_bound(tmp_path):
    from loss_fixtures import parse_loss_args as parse_args
    from response_training import binding, SOURCE_FILES
    default=parse_args(["--device","cpu"])
    ref=parse_args(["--device","cpu","--group-backward","vjp"])
    assert GroupBalanceConfig.from_args(default).backward_backend == "probe"
    assert GroupBalanceConfig.from_args(ref).backward_backend == "vjp"
    assert "response_grad_probe.py" in SOURCE_FILES
    a=binding(default,ResponsePolicyConfig(),test_loss())
    b=binding(ref,ResponsePolicyConfig(),test_loss())
    assert a["group_balance"]["backward_backend"] != b["group_balance"]["backward_backend"]


def test_recorded_vjp_backend_replays_and_cannot_resume_as_probe(tmp_path):
    from test_update_audit import args_for,run
    from tools.replay_response_update import replay
    run(args_for(tmp_path,1,("--group-backward","vjp")))
    capsule=next((tmp_path/"audit/updates").glob("*.pt"))
    assert replay(capsule)["passed"]
    with pytest.raises(ValueError,match="configuration"):
        run(args_for(tmp_path,2,("--resume",str(tmp_path/"latest.pt"))))


def test_probe_does_not_call_reference_or_depend_on_chunk_size(monkeypatch):
    from env_raptor import RaptorSimulator
    import response_adjoints
    initial,_=sample_training_scenarios(32,7,horizon=4,sampling="coverage128",dtype=torch.float64)
    torch.manual_seed(6)
    p=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4)).double()
    def forbidden(*a,**kw):
        raise AssertionError("expensive reference called by production probe")
    monkeypatch.setattr(response_adjoints,"backward_group_gradients",forbidden)
    grads=[]
    for chunk in (1,16,64):
        r=collect_rollout(p,RaptorSimulator(),initial,test_loss(),horizon=4,group_config=cfg(chunk=chunk))
        report=backward_actor(p,RaptorSimulator(),r,test_loss())["group_gradient"]
        assert report["vjp_chunk_size"] is None
        grads.append(torch.cat([x.grad.flatten() for x in p.parameters()]).clone())
    assert all(torch.equal(grads[0],g) for g in grads[1:])



def test_probe_refuses_autocast_instead_of_silently_changing_group_derivatives():
    from env_raptor import RaptorSimulator
    initial,_=sample_training_scenarios(32,0,horizon=2,sampling="coverage128")
    p=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4))
    with torch.autocast("cpu",dtype=torch.bfloat16):
        with pytest.raises(ValueError,match="autocast"):
            collect_rollout(p,RaptorSimulator(),initial,test_loss(),horizon=2,group_config=cfg())

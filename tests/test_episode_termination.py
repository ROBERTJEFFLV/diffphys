"""Independent prefix/termination regressions; no obsolete recomputation backend."""
from dataclasses import replace
import copy
import pytest
import torch

from env_raptor import RaptorSimulator
from response_noise import DisturbanceConfig
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import ResponseClosedLoopState, TaskTrajectory, TaskLossConfig, initialize, observation, rollout, step_costs, risk_weights, reference_episode_metrics, _select_rows


def select(state,index):
    return _select_rows(state,torch.arange(len(state.mass),device=state.mass.device)[index])


def actor(dtype=torch.float64):
    torch.manual_seed(17)
    return ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).to(dtype=dtype)


def flat_grads(policy):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                      for p in policy.parameters()])


class CountingSimulator(RaptorSimulator):
    def __init__(self):
        super().__init__();self.ids=[]
    def step(self,state,action):
        assert not self.terminated(state).any(), 'physics called after termination'
        self.ids.extend(state.mass.tolist())
        position=torch.cat((state.position[:,:1]+1,state.position[:,1:]),-1)
        return replace(state,position=position,velocity=state.velocity+.02*action[:,:3],
                       omega=.95*state.omega+.1*action[:,:3],motor=.9*state.motor+.1*(action+1)/2,
                       previous_action=action,previous_velocity=state.velocity,step_index=state.step_index+1)


def scheduled(lengths,horizon,dtype=torch.float64):
    s=RaptorSimulator().reset(len(lengths),seed=7,dtype=dtype,horizon=horizon,
                             disturbances=DisturbanceConfig(budget=0))
    return replace(s,position=torch.zeros_like(s.position),velocity=torch.zeros_like(s.velocity),
                   previous_velocity=torch.zeros_like(s.velocity),omega=torch.zeros_like(s.omega),
                   position_limit=s.mass.new_tensor(lengths)-.5,
                   mass=torch.arange(1,len(lengths)+1,dtype=dtype))


def manual_prefix(policy,simulator,initial,horizon):
    closed=initialize(policy,initial);obs=[observation(initial)]
    actions,positions,velocities,omegas,da,dw=[],[],[],[],[],[]
    for _ in range(horizon):
        before=closed.physical;output=policy(obs[-1],closed.policy)
        after=simulator.step(before,output.action)
        closed=ResponseClosedLoopState(after,output.next_state)
        actions.append(output.action);positions.append(after.position)
        velocities.append(after.velocity);omegas.append(after.omega)
        da.append(output.action-before.previous_action);dw.append(after.omega-before.omega)
        obs.append(observation(after))
        if bool(simulator.terminated(after).all()):break
    return TaskTrajectory(closed,torch.stack(obs),torch.stack(actions),torch.stack(positions),
                          torch.stack(velocities),torch.stack(omegas),torch.stack(da),torch.stack(dw),
                          initial,torch.ones(len(actions),1,dtype=torch.bool,device=initial.mass.device))


@torch.no_grad()
def test_h500_first_crossing_and_no_post_terminal_calls():
    horizon=500;initial=scheduled([1,19,50,500,501],horizon)
    policy=actor();sim=CountingSimulator();trace=rollout(policy,sim,initial,horizon)
    lengths=torch.tensor([1,19,50,500,500])
    assert torch.equal(trace.valid.sum(0),lengths)
    assert torch.equal(trace.end.physical.step_index,lengths)
    assert len(sim.ids)==int(lengths.sum())
    for i,length in enumerate(lengths):
        single=manual_prefix(policy,CountingSimulator(),select(initial,slice(i,i+1)),horizon)
        torch.testing.assert_close(trace.end.policy.memory[i:i+1],single.end.policy.memory,rtol=1e-11,atol=1e-12)
        assert trace.valid[length-1,i] and not trace.valid[length:,i].any()
    assert reference_episode_metrics(trace)['raptor_share_terminated']==.8


@pytest.mark.parametrize('lengths',[[1,3,4,7,8,9],[1,1],[2,3],[9,9]])
def test_full_bptt_and_adam_equal_independent_prefixes(lengths):
    h=8;initial=scheduled(lengths,h);loss=TaskLossConfig()
    batched=actor();serial=copy.deepcopy(batched)
    trace=rollout(batched,CountingSimulator(),initial,h)
    cb=step_costs(trace,loss).sum(0)
    cs=torch.cat([step_costs(manual_prefix(serial,CountingSimulator(),select(initial,slice(i,i+1)),h),
                             loss,horizon=h).sum(0) for i in range(len(lengths))])
    for model,costs in [(batched,cb),(serial,cs)]:
        (.1*(risk_weights(costs,loss)*costs).sum()).backward()
    torch.testing.assert_close(cb,cs,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(flat_grads(batched),flat_grads(serial),rtol=1e-9,atol=1e-10)
    for model in (batched,serial):torch.optim.Adam(model.parameters(),lr=3e-4).step()
    for a,b in zip(batched.parameters(),serial.parameters()):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_noisy_compaction_preserves_other_aircraft_and_terminal_tape(dtype):
    sim=RaptorSimulator();initial=sim.reset(3,seed=7,dtype=dtype,horizon=20)
    p=initial.position.clone();v=initial.velocity.clone()
    p[0]=0;p[0,0]=initial.position_limit[0]-.00001;v[0,0]=1
    initial=replace(initial,position=p,velocity=v)
    model=actor(dtype);trace=rollout(model,sim,initial,20)
    assert trace.valid[:,0].sum()==1
    assert trace.end.physical.noise_tape.data_ptr()==initial.noise_tape.data_ptr()
    tol=2e-5 if dtype==torch.float32 else 1e-10
    for i in (1,2):
        single=manual_prefix(model,sim,select(initial,slice(i,i+1)),20)
        for key in ('actions','positions','velocities','omegas'):
            torch.testing.assert_close(getattr(trace,key)[:len(single.actions),i:i+1],getattr(single,key),rtol=tol,atol=tol)
    tape=initial.noise_tape.clone();tape[0,2:]+=100
    other=rollout(model,sim,replace(initial,noise_tape=tape),20)
    assert torch.equal(trace.actions,other.actions)
    assert torch.equal(trace.valid,other.valid)


def test_terminal_gradient_kept_padding_gradient_zero():
    initial=scheduled([1,3],8);model=actor();trace=rollout(model,CountingSimulator(),initial,8)
    trace.velocities.retain_grad()
    step_costs(trace,TaskLossConfig()).sum().backward()
    assert trace.velocities.grad[0,0].abs().sum()>0
    assert not trace.velocities.grad[1:,0].any()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_terminal_batch():
    initial=scheduled([1,3,9],8,torch.float32).to('cuda',torch.float32)
    p=actor(torch.float32).cuda();trace=rollout(p,CountingSimulator(),initial,8)
    assert trace.valid.sum(0).tolist()==[1,3,8]
    step_costs(trace,TaskLossConfig()).sum().backward()
    assert torch.isfinite(flat_grads(p)).all()

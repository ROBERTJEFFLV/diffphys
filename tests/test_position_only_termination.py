from dataclasses import replace
import torch
from env_raptor import RaptorSimulator
from response_task import rollout,TaskLossConfig,step_costs,reference_episode_metrics
from test_episode_termination import actor,flat_grads


def test_position_remains_the_only_episode_boundary():
    sim=RaptorSimulator();s=sim.reset(2,horizon=8,dtype=torch.float64)
    s=replace(s,position=torch.zeros_like(s.position),velocity=torch.full_like(s.velocity,2.1),
              omega=torch.full_like(s.omega,35.1),position_limit=torch.full_like(s.position_limit,100))
    p=actor();trace=rollout(p,sim,s,8)
    assert trace.valid.all() and reference_episode_metrics(trace)['raptor_share_terminated']==0
    step_costs(trace,TaskLossConfig()).sum().backward()
    assert torch.isfinite(flat_grads(p)).all() and flat_grads(p).norm()>0
    pos=s.position.clone();pos[0,0]=s.position_limit[0];pos[1,0]=s.position_limit[1]+1e-5
    assert sim.terminated(replace(s,position=pos)).tolist()==[False,True]

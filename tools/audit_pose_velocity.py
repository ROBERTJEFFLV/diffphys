#!/usr/bin/env python3
"""Small CPU gradient audit; no planner, policy training, or reference rollout."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env_raptor import RaptorSimulator
from response_noise import DisturbanceConfig
from pass_stop import GoalConfig, GoalMotorPolicy, PassStopLossConfig, prepare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    sim=RaptorSimulator()
    policy=GoalMotorPolicy().double()
    config=PassStopLossConfig(.01,.01,25,mission=asdict(GoalConfig()))
    initial=sim.reset(2,seed=17,horizon=8,dtype=torch.float64,
                      disturbances=DisturbanceConfig.clean())
    initial=prepare(policy,initial,config).physical
    target=initial.velocity.new_tensor((2.,0.,0.))
    records=[]
    for angle_deg in (0.,45.,90.,105.,170.):
        z=torch.cat((torch.full((2,),math.radians(angle_deg),dtype=torch.float64),
                     torch.linspace(-.1,.2,8,dtype=torch.float64))).requires_grad_()
        def objective(values):
            angle=values[:2]
            quat=torch.stack((torch.cos(angle/2),torch.sin(angle/2),
                              torch.zeros_like(angle),torch.zeros_like(angle)),-1)
            state=replace(initial,orientation=quat)
            action=values[2:].reshape(2,4)
            for _ in range(8):
                state=sim.step(state,action)
            return .5*(state.velocity-target).square().sum()+.2*state.position.square().sum()
        value=objective(z)
        gradient=torch.autograd.grad(value,z)[0]
        for index in range(len(z)):
            h=1e-6
            direction=torch.zeros_like(z);direction[index]=h
            difference=(objective(z.detach()+direction)-objective(z.detach()-direction))/(2*h)
            analytic=gradient[index]
            absolute=float((analytic-difference).abs())
            relative=absolute/max(1.,float(analytic.abs()),float(difference.abs()))
            records.append(dict(angle_deg=angle_deg,variable=index,
                                derivative=float(analytic),finite_difference=float(difference),
                                absolute_error=absolute,normalized_error=relative))
    maximum=max(r['normalized_error'] for r in records)
    if not maximum < 2e-5:
        raise AssertionError(f'gradient finite difference failed: {maximum}')
    report={
        'base_commit':'42e5f1704ba1128877c8cab738085403e2a8d04a',
        'torch':str(torch.__version__),'device':'cpu','dtype':'float64',
        'native_physics_unchanged':True,'independent_checks':len(records),
        'airframes':2,'control_steps_per_case':8,
        'maximum_normalized_error':maximum,
        'maximum_absolute_error':max(r['absolute_error'] for r in records),
        'checks':records,
        'scope':'Native velocity/position objective -> initial attitude and four motor commands; fixed smooth branches.',
        'not_verified':['CUDA','3072-scene throughput','learned acrobatics','hard-switch derivatives','real-flight safety'],
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))

if __name__=='__main__':main()

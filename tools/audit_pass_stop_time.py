#!/usr/bin/env python3
"""Bounded time-surrogate derivative and counterexample audit; no Actor training."""
from pathlib import Path
import sys
import argparse
import json
import math
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pass_stop import temporal_proxy

def run():
    torch.set_num_threads(1)
    rows=[]
    t=torch.arange(501,dtype=torch.float64)*.01
    # Delaying both PASS and STOP is a physical-time shift, not a per-step constant.
    for pass_t in (.4,.8,1.2,1.6):
        for duration in (.6,1.,1.4):
            for width in (.06,.12):
                d=torch.tensor(.0037,dtype=torch.float64,requires_grad=True)
                def value(delay):
                    p=((t-pass_t-delay)/.06).square()[:,None]
                    s=(torch.nn.functional.softplus((pass_t+duration+delay-t)/width)/math.log(2))[:,None]
                    return temporal_proxy(p,s)[0].sum()
                objective=value(d)
                grad=torch.autograd.grad(objective,d)[0]
                eps=1e-6
                fd=(value(d.detach()+eps)-value(d.detach()-eps))/(2*eps)
                # Ten bounded descent steps on a common temporal shift. No dynamics learning claim.
                delay=d.detach().clone();start=float(value(delay))
                for _ in range(10):
                    delay.requires_grad_();g=torch.autograd.grad(value(delay),delay)[0]
                    delay=(delay-.01*g).detach().clamp_min(-.2)
                end=float(value(delay))
                assert grad>0 and abs(float(grad-fd))<1e-6 and end<start
                rows.append(dict(pass_time=pass_t,stop_time=pass_t+duration,width=width,
                                 gradient=float(grad),finite_difference=float(fd),
                                 proxy_before=start,proxy_after=end,shift_after=float(delay)))
    monotonic=[]
    for dtype in (torch.float32,torch.float64):
        generator=torch.Generator().manual_seed(171)
        p=(1.01+torch.rand(501,128,dtype=dtype,generator=generator)*40).requires_grad_()
        s=(1.01+torch.rand(501,128,dtype=dtype,generator=generator)*40).requires_grad_()
        gp,gs=torch.autograd.grad(temporal_proxy(p,s)[0].sum(),(p,s))
        negative=int((gp<0).sum()+(gs<0).sum())
        assert negative==0
        monotonic.append(dict(dtype=str(dtype),partial_derivatives=gp.numel()+gs.numel(),
                              negative=negative,nonzero=int((gp!=0).sum()+(gs!=0).sum()),
                              finite=bool(torch.isfinite(gp).all()&torch.isfinite(gs).all())))
    # The hard radius condition is discontinuous. Do not pretend its index has a derivative.
    thresholds=[]
    for distance in (.100001,.1,.099999):
        thresholds.append(dict(distance_m=distance,position_switch=distance**2<=.1**2))
    # In the exact 0/1 quality limit the clock equals last departure, not first touch.
    p=torch.full((501,1),torch.inf,dtype=torch.float64);p[80]=1
    s=torch.full_like(p,torch.inf);s[200:]=1
    early=float(temporal_proxy(p,s)[0]);s[400]=torch.inf
    leave=float(temporal_proxy(p,s)[0]);assert early==2 and leave==4.01
    return dict(version='pass-stop-time-audit-v1',torch_version=str(torch.__version__),
                temporal_shift_trials=rows,monotonicity=monotonic,threshold_samples=thresholds,
                settled_at_2_seconds=early,leaves_again_at_4_seconds=leave,
                caveats=['Synthetic residual/time-shift tests, not dynamically optimized flight.',
                         'Hard control switches and first-hit selection remain nondifferentiable.',
                         'The surrogate is not an exact ranking of arbitrary incomplete flights.',
                         'No CUDA execution or trained-hover checkpoint tested.'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();report=run()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({"temporal_shift_trials":len(report['temporal_shift_trials']),
                      "negative_partials":sum(x['negative'] for x in report['monotonicity']),
                      "max_fd_error":max(abs(x['gradient']-x['finite_difference']) for x in report['temporal_shift_trials'])}))

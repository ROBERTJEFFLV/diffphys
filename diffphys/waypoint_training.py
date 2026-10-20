"""Integrate event-constrained fixed-point learning into the accelerated repository.

One scalar PHR objective, exact branch BPTT, plus antithetic hard-program probes.
Primal/dual iterations reuse a fixed scene bank. No old multiplier is applied to
new scenes. All finite optimizer steps commit; success metrics never gate training.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import torch

from .compiled_physics import make_simulator, physics_backend_contract
from .env_raptor import RaptorParams
from .response_execution import critical_gc
from .response_noise import DisturbanceConfig
from .response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from .response_sampling import sampling_contract, validate_sampling
from .waypoint_config import WaypointConfig, GOAL_CONVENTION
from .waypoint_sampling import sample_bank, WaypointPrefetch
from .waypoint_task import evaluate_rollout, CONSTRAINTS
from .waypoint_gradients import corrected_gradient, update_duals
from .waypoint_checkpoint import (ROOT, atomic_save, checkpoint_payload, source_manifest,
                                  import_weights, load_policy, read_checkpoint, model_hash)


@torch.no_grad()
def assign_vector(policy, vector):
    offset = 0
    for parameter in policy.parameters():
        parameter.copy_(vector[offset:offset+parameter.numel()].reshape_as(parameter))
        offset += parameter.numel()
    if offset != vector.numel():
        raise ValueError('parameter vector size mismatch')


def vector_of(policy):
    return torch.cat([p.detach().flatten() for p in policy.parameters()]).clone()


def one_update(policy, optimizer, simulator, bank, config, multipliers, *, horizon, seed,
               gradient_clip):
    """Transactional update. Paired probes change no optimizer, duals or tapes."""
    theta = vector_of(policy)
    adam_before = copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    try:
        with critical_gc():
            center = evaluate_rollout(policy,simulator,bank.initial,bank.mission,config,multipliers,
                                      horizon=horizon)
            branch_grad = torch.autograd.grad(center.costs.mean(),list(policy.parameters()),allow_unused=True)
            gp = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                            for p,g in zip(policy.parameters(),branch_grad)])
            center_report = center.report
            del center,branch_grad
            def evaluate(parameters):
                assign_vector(policy,parameters)
                result = evaluate_rollout(policy,simulator,bank.initial,bank.mission,config,multipliers,
                                          horizon=horizon,make_report=False)
                return result.costs.mean()
            try:
                gradient,probe_report = corrected_gradient(theta,gp,evaluate,sigma=config.event_sigma,
                                                           directions=config.event_directions,seed=seed)
            finally:
                assign_vector(policy,theta)
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError('nonfinite hard-event gradient')
        offset = 0
        for parameter in policy.parameters():
            parameter.grad = gradient[offset:offset+parameter.numel()].reshape_as(parameter).clone()
            offset += parameter.numel()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),gradient_clip,error_if_nonfinite=True)
        optimizer.step()
        state_tensors = [x for state in optimizer.state.values() for x in state.values() if torch.is_tensor(x)]
        if not all(bool(torch.isfinite(x).all()) for x in [*policy.parameters(),*state_tensors]):
            raise FloatingPointError('nonfinite Adam update')
        report = {**center_report,'gradient':probe_report,
                  'pre_clip_gradient_norm':float(norm),'gradient_clip':gradient_clip,
                  'raw_estimator_postprocessing':'global norm clipping then Adam; not an unbiased parameter step',
                  'parameter_delta_l2':float((vector_of(policy)-theta).norm())}
        return report
    except BaseException:
        assign_vector(policy,theta)
        optimizer.load_state_dict(adam_before)
        optimizer.zero_grad(set_to_none=True)
        raise


def _options(args, config, *, split='train'):
    return dict(horizon=args.horizon,dtype=getattr(torch,args.dtype),seed=args.seed,
                sampling=args.train_sampling if split == 'train' else 'random',split=split,
                disturbances=DisturbanceConfig(enabled=not args.disable_disturbances,force_enabled=False),
                waypoint_config=config.snapshot())


def binding_for(args,config):
    return {
        'task':'fixed-point-event-constraints','goal_convention':GOAL_CONVENTION,
        'source':source_manifest(),'torch':str(torch.__version__),
        'physics_backend':physics_backend_contract(args.device),
        'cuda_runtime':torch.version.cuda,'tf32':False,
        'run':{name:getattr(args,name) for name in (
            'horizon','dtype','device','threads','seed','scenarios','train_sampling',
            'lr','gradient_clip','disable_disturbances','eval_scenarios')},
        'sampling':sampling_contract(args.train_sampling),'config':config.snapshot(),
        'gradient':'uniform scalar AL -> native BPTT control variate -> Gaussian hard probes -> global clip -> Adam',
        'physical_group_reweighting':False,
    }


def _recover_log(path,next_update):
    if not path.exists():
        return
    lines = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if row.get('update',0) > next_update:
            break
        lines.append(line)
    path.write_text(''.join(line+'\n' for line in lines))


def train(args,config):
    dtype = getattr(torch,args.dtype)
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    binding = binding_for(args,config)
    start = 0; saved_mu = None; saved_bank = -1; saved_key = None
    init_report = {'from_scratch':True}
    if args.resume:
        policy,saved_config,value = load_policy(args.resume,device=device,dtype=dtype)
        if saved_config != config or value['binding'] != binding:
            raise ValueError('exact resume requires identical source, task, runtime and numerical settings; use --init-checkpoint for a fork')
        if args.memory_dim is not None and args.memory_dim != policy.config.memory_dim:
            raise ValueError('resume hidden width mismatch')
        start = value['progress']['next_update']
        saved_mu,saved_bank,saved_key = value['multipliers'],value['multipliers_bank_index'],value['bank_key']
        init_report = value['progress'].get('initialization',{})
    elif args.init_checkpoint:
        policy,init_report = import_weights(args.init_checkpoint,device=device,dtype=dtype,memory_dim=args.memory_dim)
    else:
        policy = ResponseMotorPolicy(ResponsePolicyConfig(args.memory_dim or 64,.01,True)).to(device=device,dtype=dtype)
    optimizer = torch.optim.Adam(policy.parameters(),lr=args.lr)
    if args.resume:
        if value['optimizer_parameter_names'] != [n for n,_ in policy.named_parameters()]:
            raise ValueError('named optimizer parameter order changed')
        optimizer.load_state_dict(value['optimizer'])
        torch.set_rng_state(value['rng']['torch'].cpu())
        if value['rng']['cuda']:
            torch.cuda.set_rng_state_all([x.cpu() for x in value['rng']['cuda']])
    work = Path(args.work_dir); work.mkdir(parents=True,exist_ok=True)
    if (work/'checkpoint.pt').exists() and not args.resume:
        raise ValueError('run directory already has a checkpoint; use a new directory or --resume')
    atomic_save(work/'binding.json',binding,json_file=True)
    _recover_log(work/'train.jsonl',start)
    simulator = make_simulator(RaptorParams(),device)
    options = _options(args,config)
    per_bank = config.dual_interval*config.dual_updates_per_bank
    desired_bank = start//per_bank
    worker = None; bank = None; bank_index = saved_bank; bank_key = saved_key
    mu = None if saved_mu is None else saved_mu.to(device=device,dtype=dtype)
    progress = {'next_update':start,'initialization':init_report}
    def save():
        atomic_save(work/'checkpoint.pt',checkpoint_payload(policy,optimizer,config,progress,binding,
                                                           mu,bank_index,bank_key))
    save()
    began = time.perf_counter(); status = 'update_budget'; last = None
    try:
        if not args.no_prefetch and start < args.updates:
            worker = WaypointPrefetch(args.scenarios,options=options,threads=args.threads)
            worker.submit(desired_bank)
        for update in range(start,args.updates):
            if time.perf_counter()-began >= args.max_seconds:
                status = 'time_budget'; break
            wanted = update//per_bank
            sampling_stats = {}
            if bank is None or wanted != bank_index:
                if worker is not None:
                    cpu_bank,_,_,sampling_stats = worker.take(wanted)
                else:
                    _,cpu_bank,_,_,sampling_stats = sample_bank(args.scenarios,wanted,options)
                if mu is not None and wanted == saved_bank:
                    if cpu_bank.key != saved_key or mu.shape != (4*args.scenarios,len(CONSTRAINTS)):
                        raise ValueError('multiplier bank fingerprint/shape mismatch')
                else:
                    mu = torch.zeros(4*args.scenarios,len(CONSTRAINTS),device=device,dtype=dtype)
                bank = cpu_bank.to(device,dtype)
                bank_index,bank_key = wanted,cpu_bank.key
                del cpu_bank
                if worker is not None and (wanted+1)*per_bank < args.updates:
                    worker.submit(wanted+1)
            before_mu = mu.clone()
            # The primal transaction also rolls back if a post-update dual evaluation fails.
            before_theta, before_adam = vector_of(policy), copy.deepcopy(optimizer.state_dict())
            tick = time.perf_counter()
            try:
                last = one_update(policy,optimizer,simulator,bank,config,mu,horizon=args.horizon,
                                  seed=(args.seed+1000003*update+0xEA1701)&((1<<63)-1),
                                  gradient_clip=args.gradient_clip)
                if (update+1)%config.dual_interval == 0:
                    with torch.no_grad():
                        post = evaluate_rollout(policy,simulator,bank.initial,bank.mission,config,mu,
                                                horizon=args.horizon,make_report=False)
                        mu = update_duals(post.constraints,mu,config.rho)
                        del post
                    last['dual_updated'] = True
                else:
                    last['dual_updated'] = False
            except BaseException:
                assign_vector(policy,before_theta); optimizer.load_state_dict(before_adam); mu = before_mu
                raise
            progress['next_update'] = update+1
            last.update(update=update+1,bank_index=bank_index,bank_key=bank_key,
                        seconds=time.perf_counter()-tick,multiplier_max=float(mu.max()))
            with (work/'train.jsonl').open('a') as stream:
                stream.write(json.dumps(last,allow_nan=False)+'\n');stream.flush()
            if (update+1)%args.eval_every == 0:
                with torch.no_grad():
                    _,evaluation_bank,_,_,_ = sample_bank(args.eval_scenarios,0,_options(args,config,split='validation'))
                    evaluation_bank = evaluation_bank.to(device,dtype)
                    result = evaluate_rollout(policy,simulator,evaluation_bank.initial,evaluation_bank.mission,
                                              config,horizon=args.horizon)
                atomic_save(work/'validation.json',{'update':update+1,**result.report},json_file=True)
                del result,evaluation_bank
            if (update+1)%args.checkpoint_every == 0:
                save()
    except KeyboardInterrupt:
        status = 'interrupted'
    except BaseException:
        save()
        raise
    finally:
        if worker is not None:
            worker.close()
        save()
    summary = {'status':status,'updates':progress['next_update'],'wall_seconds':time.perf_counter()-began,
               'checkpoint':str(work/'checkpoint.pt'),'model_sha256':model_hash(policy),
               'initialization':init_report,'last_update':last,'deployment_authorized':False}
    atomic_save(work/'summary.json',summary,json_file=True)
    return summary


@torch.no_grad()
def evaluate_checkpoint(args):
    policy,config,value = load_policy(args.checkpoint,device=args.device)
    if getattr(args,'verify_audit_substeps',None) is not None:
        from dataclasses import replace
        config = replace(config,audit_substeps=args.verify_audit_substeps)
    torch.set_num_threads(args.threads)
    saved = value['binding']['run']
    options = dict(horizon=saved['horizon'],dtype=next(policy.parameters()).dtype,seed=saved['seed'],
                   sampling='random',split=args.eval_split,
                   disturbances=DisturbanceConfig(enabled=not saved['disable_disturbances'],force_enabled=False),
                   waypoint_config=config.snapshot())
    _,bank,_,_,_ = sample_bank(args.eval_scenarios,0,options)
    bank = bank.to(args.device,next(policy.parameters()).dtype)
    simulator = make_simulator(RaptorParams(),args.device)
    result = evaluate_rollout(policy,simulator,bank.initial,bank.mission,config,horizon=saved['horizon'],
                              record_trace=args.replay_out is not None)
    if args.replay_out:
        import numpy as np
        trace = result.trace
        data = {k:v.cpu().numpy() for k,v in trace.state.items()}
        data.update(actions=trace.actions.cpu().numpy(),target_index=trace.target_index.cpu().numpy(),
                    valid=trace.valid.cpu().numpy(),points=bank.mission.points.cpu().numpy(),
                    target_rotation=bank.mission.rotation.cpu().numpy(),direction=bank.mission.direction.cpu().numpy(),
                    constraints=result.constraints.cpu().numpy(),constraint_names=np.array(CONSTRAINTS))
        data.update({'event_'+k:v.cpu().numpy() for k,v in result.event_data.items()})
        Path(args.replay_out).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.replay_out,**data)
    return {'checkpoint_update':value['progress']['next_update'],'split':args.eval_split,
            'bank_key':bank.key,'evaluation_runtime':physics_backend_contract(args.device),**result.report}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,fromfile_prefix_chars='@')
    p.convert_arg_line_to_args = lambda line: line.split() if not line.lstrip().startswith('#') else []
    p.add_argument('--task',choices=['agile'],default='agile')
    p.add_argument('--mode',choices=['train','evaluate'],default='train')
    p.add_argument('--agile-config','--waypoint-config',dest='config',type=Path,default=ROOT/'configs/waypoint_flight.json')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--dtype',choices=['float32','float64'],default='float32')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--scenarios',type=int,default=128,help='four physical TRAIN banks are pooled')
    p.add_argument('--eval-scenarios',type=int,default=32,help='four fixed EVAL sub-banks')
    p.add_argument('--train-sampling',choices=['random','coverage128'],default='coverage128')
    p.add_argument('--horizon',type=int,default=500)
    p.add_argument('--time-decay',type=float,default=0.,help='must remain zero for the declared branch gradient')
    p.add_argument('--updates',type=int,default=50)
    p.add_argument('--max-seconds',type=float,default=1800.)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--gradient-clip',type=float,default=10.)
    p.add_argument('--memory-dim',type=int)
    p.add_argument('--no-prefetch',action='store_true',help='serial numerical comparison only')
    p.add_argument('--disable-disturbances',action='store_true',help='zero measurement noise/latency test fixture')
    p.add_argument('--disable-external-forces',action='store_true',help='already mandatory for this waypoint protocol')
    p.add_argument('--verify-audit-substeps',type=int,help='evaluation-only denser native-RK4 constraint sampling, 1..10')
    p.add_argument('--checkpoint-every',type=int,default=10)
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--eval-split',choices=['validation','test'],default='validation')
    p.add_argument('--work-dir',type=Path,default=Path('runs/waypoints/seed7'))
    group = p.add_mutually_exclusive_group()
    group.add_argument('--init-checkpoint',type=Path)
    group.add_argument('--resume',type=Path)
    group.add_argument('--from-scratch',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--replay-out',type=Path)
    args = p.parse_args(argv)
    for name in ('threads','scenarios','eval_scenarios','horizon','checkpoint_every','eval_every'):
        if getattr(args,name) < 1:
            p.error(name+' must be positive')
    if args.memory_dim is not None and args.memory_dim < 1:
        p.error('memory-dim must be positive')
    if args.updates < 0 or args.time_decay != 0:
        p.error('updates must be nonnegative; time-decay must be zero')
    for name in ('max_seconds','lr','gradient_clip'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name) <= 0:
            p.error(name+' must be finite and positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable; use --device cpu explicitly for reference tests')
    if args.mode == 'train' and not (args.init_checkpoint or args.resume or args.from_scratch):
        p.error('provide --init-checkpoint hover.pt (or explicitly --from-scratch)')
    if args.mode == 'evaluate' and not args.checkpoint:
        p.error('evaluate requires --checkpoint')
    if args.verify_audit_substeps is not None and (args.mode != 'evaluate' or not 1 <= args.verify_audit_substeps <= 10):
        p.error('verify-audit-substeps is evaluation-only and must lie in 1..10')
    validate_sampling(args.train_sampling,4*args.scenarios)
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.mode == 'evaluate':
        result = evaluate_checkpoint(args)
    else:
        result = train(args,WaypointConfig.load(args.config))
    print(json.dumps(result,indent=2,allow_nan=False))
    return result

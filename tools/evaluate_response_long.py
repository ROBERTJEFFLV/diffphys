#!/usr/bin/env python3
"""Frozen-Actor 60 s moving-goal EVAL with off-center 20%-weight pulses."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch

from response_long_eval import LongEvalConfig, LongHoverEval, make_schedule, EvalSchedule, LONG_EVAL_VERSION
from response_noise import DisturbanceConfig
from response_task import _select_rows
from response_training import (DEVELOPMENT_SEEDS, SOURCE_FILES, atomic_json, atomic_torch,
                               load_policy_checkpoint, model_hash, sample_pool, source_hash)


class ArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self,line):
        return line.split('#',1)[0].split()


def parse_args(argv=None):
    parser=ArgumentParser(description=__doc__,fromfile_prefix_chars='@')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--dtype',choices=('float32','float64'),default='float32')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--scene-ids',type=int,nargs='+',help='zero-based IDs in the checkpoint fixed EVAL pool; default: all')
    for f in fields(LongEvalConfig):
        parser.add_argument('--'+f.name.replace('_','-'),default=f.default,type=type(f.default))
    args=parser.parse_args(argv)
    if args.threads<1: parser.error('threads must be positive')
    try: LongEvalConfig(**{f.name:getattr(args,f.name) for f in fields(LongEvalConfig)})
    except ValueError as exc: parser.error(str(exc))
    return args


def main(argv=None):
    args=parse_args(argv)
    config=LongEvalConfig(**{f.name:getattr(args,f.name) for f in fields(LongEvalConfig)})
    torch.set_num_threads(args.threads)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA is unavailable')
    out=args.work_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('use a new output directory; existing EVAL evidence is not overwritten')
    out.mkdir(parents=True,exist_ok=True)
    frozen=out/'checkpoint.pt'
    shutil.copyfile(args.checkpoint,frozen)
    device=torch.device(args.device);dtype=getattr(torch,args.dtype)
    policy,saved=load_policy_checkpoint(frozen,device,dtype)
    policy.eval()
    before_hash=model_hash(policy)
    cfg=saved['binding']['protocol']
    sensor_config=replace(DisturbanceConfig(**cfg['disturbances']),pulse_enabled=False)
    initial=sample_pool(cfg['eval_scenarios_per_bank'],DEVELOPMENT_SEEDS,
        dt=policy.config.dt,device=args.device,dtype=dtype,horizon=config.steps,
        disturbances=sensor_config)
    schedule=make_schedule(initial,config)
    total=len(initial.mass)
    ids=list(range(total)) if args.scene_ids is None else args.scene_ids
    if len(set(ids))!=len(ids) or any(i<0 or i>=total for i in ids):
        raise ValueError(f'scene IDs must be unique and inside [0,{total-1}]')
    if ids!=list(range(total)):
        # Explicit indexing also supports a permutation of the entire pool.
        # The production compactor assumes a full-length index vector is identity.
        if len(ids)==total:
            index=torch.tensor(ids,device=device)
            from env_raptor import IMMUTABLE_TAPES
            initial=replace(initial,**{f.name:getattr(initial,f.name).index_select(0,index)
                           for f in fields(initial) if f.name not in IMMUTABLE_TAPES})
        else:
            initial=_select_rows(initial,torch.tensor(ids,device=device))
        schedule=EvalSchedule(schedule.targets,schedule.forces_world[:,ids],schedule.points_body[:,ids],
                              schedule.face_ids[:,ids],schedule.surface_half_extents[ids])
    env=LongHoverEval(policy,initial,config,schedule)
    evaluator_files=list(SOURCE_FILES)+['response_long_eval.py','tools/evaluate_response_long.py']
    digest=hashlib.sha256()
    for name in evaluator_files:
        content=(ROOT/name).read_bytes();digest.update(name.encode());digest.update(content)
        destination=out/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(content)
    manifest=dict(protocol=LONG_EVAL_VERSION,created_at=datetime.now(timezone.utc).isoformat(),
        config=asdict(config),checkpoint_input=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
        model_sha256=saved['model_sha256'],checkpoint_update=saved['progress']['updates'],
        checkpoint_source_sha256=saved['binding']['source_sha256'],
        training_source_sha256=source_hash(),evaluator_source_sha256=digest.hexdigest(),
        source_match=saved['binding']['source_sha256']==source_hash(),
        torch_version=torch.__version__,python_version=sys.version,device=str(device),dtype=args.dtype,
        fixed_airframe_seeds=list(DEVELOPMENT_SEEDS),full_pool_size=total,scene_ids=ids,
        checkpoint_disturbances=cfg['disturbances'],sensor_command_noise=asdict(sensor_config),
        replaced_noise_components=['external_force','external_torque','training_force_at_point_pulses'],
        surface='area-uniform cuboid envelope; x/y half extent = max(abs(rotor coordinate)) + 0.12*arm; z half extent = 0.10*arm; exact ratios in config',
        surface_half_extents_m=schedule.surface_half_extents.cpu().tolist(),
        initial_condition='arena center, level attitude, zero velocities, static hover motors, zero GRU memory',
        arena_coordinates='x/y in [-side/2,side/2], z in [0,side]; COM-only exit, no collision/contact model',
        force_semantics='world direction held for 0.1-second pulse, body-fixed surface point; actual durations from config',
        reproducibility='initial_state.pt and schedule.pt store all noise tapes and random draws; source/ stores the exact evaluator',
        deployment_authorized=False)
    atomic_json(out/'manifest.json',manifest)
    atomic_torch(out/'initial_state.pt',{f.name:getattr(env.initial,f.name).cpu() for f in fields(env.initial)})
    atomic_torch(out/'schedule.pt',{f.name:getattr(schedule,f.name).cpu() for f in fields(schedule)})
    started=time.monotonic()
    def progress(value):
        value['wall_seconds']=round(time.monotonic()-started,3)
        print(json.dumps(value),flush=True)
    try:
        result=env.run(progress)
        if model_hash(policy)!=before_hash or any(p.grad is not None for p in policy.parameters()):
            raise RuntimeError('EVAL unexpectedly modified the Actor or accumulated gradients')
        summary=result.summary
        for row,scene_id in zip(summary['scenes'],ids):
            j=row['scene_index'];row['scene_id']=scene_id
            row['mass_kg']=float(initial.mass[j]);row['arm_length_m']=float(initial.arm_length[j])
            row['thrust_to_weight']=float(initial.thrust_to_weight[j])
            row['torque_to_inertia']=float(initial.torque_to_inertia[j])
        summary.update(checkpoint_update=manifest['checkpoint_update'],wall_seconds=time.monotonic()-started,
                       actor_unchanged=True)
        atomic_torch(out/'trajectory.pt',result.trajectory)
        atomic_json(out/'summary.json',summary)
        atomic_json(out/'status.json',{'status':'complete','wall_seconds':summary['wall_seconds']})
        print(json.dumps({k:v for k,v in summary.items() if k!='scenes'},indent=2,allow_nan=False),flush=True)
    except Exception as exc:
        atomic_json(out/'status.json',{'status':'error','error':str(exc),'wall_seconds':time.monotonic()-started})
        raise


if __name__=='__main__':
    main()

"""Complete independent Actor checkpoints and strict, named optimizer resume."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch

from .env_raptor import ACTION_CONVENTION
from .response_policy import ResponseMotorPolicy, ResponsePolicyConfig, ARCHITECTURE, AGILE_ARCHITECTURE
from .waypoint_config import VERSION, GOAL_CONVENTION, WaypointConfig

SCHEMA = 'event-waypoint-checkpoint-v1'
ROOT = Path(__file__).resolve().parents[1]


def model_hash(state):
    state = state.state_dict() if hasattr(state,'state_dict') else state
    digest = hashlib.sha256()
    for name,value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        for metadata in (name,str(value.dtype),str(tuple(value.shape))):
            digest.update(metadata.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_manifest():
    files = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'diffphys').glob('waypoint_*.py'))
    files += ['diffphys/'+name+'.py' for name in (
        'env_raptor','response_noise','response_policy','compiled_physics','strict_inductor',
        'response_sampling','scene_prefetch','response_execution')]
    files += ['configs/physics_coverage.json','tools/train_response_control.py']
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}


def atomic_save(path, value, *, json_file=False):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary = tempfile.mkstemp(dir=path.parent,prefix=path.name+'.')
    try:
        with os.fdopen(fd,'wb') as stream:
            if json_file:
                stream.write((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
            else:
                torch.save(value,stream)
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def read_checkpoint(path):
    # This repository's saved RNG is lists/tensors, so unrestricted pickle is unnecessary.
    value = torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(value,dict) or not isinstance(value.get('model'),dict):
        raise ValueError('expected a complete repository checkpoint, not an unknown state_dict')
    if value.get('model_sha256') != model_hash(value['model']):
        raise ValueError('checkpoint model checksum mismatch')
    return value


def import_weights(path, *, device='cpu', dtype=torch.float32, memory_dim=None):
    value = read_checkpoint(path)
    cfg = ResponsePolicyConfig(**value['policy_config'])
    if cfg.dt != .01 or value.get('architecture') not in (ARCHITECTURE,AGILE_ARCHITECTURE):
        raise ValueError('incompatible recurrent Actor or timestep')
    if memory_dim is not None and cfg.memory_dim != memory_dim:
        raise ValueError('requested hidden width differs from the source checkpoint')
    if value.get('schema') == SCHEMA:
        convention = value.get('action_convention')
        if value.get('goal_convention') != GOAL_CONVENTION:
            raise ValueError('unknown source waypoint goal semantics')
    else:
        convention = value.get('binding',{}).get('protocol',{}).get('environment',{}).get('action_convention')
        if value.get('schema') != 'raptor-multi-airframe-gaussian-v6':
            raise ValueError('unsupported legacy checkpoint schema')
    if convention != ACTION_CONVENTION:
        raise ValueError('motor ordering/frame/action convention mismatch')
    source_dim = 36 if cfg.goal_conditioning else 16
    if value['architecture'] != (AGILE_ARCHITECTURE if cfg.goal_conditioning else ARCHITECTURE):
        raise ValueError('source Actor metadata inconsistent')
    policy = ResponseMotorPolicy(ResponsePolicyConfig(cfg.memory_dim,.01,True)).to(device=device,dtype=dtype)
    expected = policy.state_dict()
    if set(value['model']) != set(expected):
        raise ValueError('checkpoint must contain all six native Actor parameter tensors')
    with torch.no_grad():
        for name,target in expected.items():
            old = value['model'][name]
            shape = (target.shape[0],source_dim) if name == 'response_memory.weight_ih' else target.shape
            if (tuple(old.shape) != tuple(shape) or old.dtype not in (torch.float32,torch.float64)
                    or not bool(torch.isfinite(old).all())):
                raise ValueError('invalid source tensor: '+name)
            if name == 'response_memory.weight_ih' and source_dim == 16:
                target.zero_(); target[:,:16].copy_(old.to(target))
            else:
                target.copy_(old.to(target))
    policy.load_state_dict(expected,strict=True)
    return policy, {'source_model_sha256':value['model_sha256'],'source_architecture':value['architecture'],
                    'zero_padded_goal_columns':20 if source_dim == 16 else 0,
                    'fresh_optimizer':True,'all_parameters_trainable':True,
                    'old_trajectory_goal_semantics_replaced':source_dim == 36 and value.get('schema') != SCHEMA}


def checkpoint_payload(policy,optimizer,config,progress,binding,multipliers,bank_index,bank_key):
    return cpu_tree({
        'schema':SCHEMA,'architecture':policy.architecture,'action_convention':ACTION_CONVENTION,
        'goal_convention':GOAL_CONVENTION,'policy_config':asdict(policy.config),
        'model':policy.state_dict(),'model_sha256':model_hash(policy),
        'optimizer':optimizer.state_dict(),'optimizer_parameter_names':[n for n,_ in policy.named_parameters()],
        'config':config.snapshot(),'progress':progress,'binding':binding,
        'multipliers':multipliers,'multipliers_bank_index':bank_index,'bank_key':bank_key,
        'rng':{'torch':torch.get_rng_state(),
               'cuda':torch.cuda.get_rng_state_all() if next(policy.parameters()).is_cuda else []},
        'deployment_authorized':False,
    })


def load_policy(path, *, device='cpu', dtype=None):
    """Deployment weights/config are all in THIS file; no base checkpoint lookup."""
    value = read_checkpoint(path)
    if (value.get('schema') != SCHEMA or value.get('goal_convention') != GOAL_CONVENTION
            or value.get('action_convention') != ACTION_CONVENTION):
        raise ValueError('not an event-waypoint standalone checkpoint')
    config = WaypointConfig(**value['config'])
    cfg = ResponsePolicyConfig(**value['policy_config'])
    if (not cfg.goal_conditioning or cfg.dt != .01
            or value.get('architecture') != AGILE_ARCHITECTURE):
        raise ValueError('checkpoint needs the native goal-conditioned 100 Hz Actor')
    if any(not torch.is_tensor(x) or x.dtype not in (torch.float32,torch.float64)
           or not bool(torch.isfinite(x).all()) for x in value['model'].values()):
        raise ValueError('invalid standalone model tensor')
    dtype = next(iter(value['model'].values())).dtype if dtype is None else dtype
    if dtype not in (torch.float32,torch.float64):
        raise ValueError('model requires float32 or float64')
    policy = ResponseMotorPolicy(cfg).to(device=device,dtype=dtype)
    policy.load_state_dict(value['model'],strict=True)
    return policy,config,value


class WaypointController:
    """Deployment/reference harness: one complete model, measured state, fixed points.

    Not a hardware safety supervisor. Never resets GRU merely because a target changed.
    """
    def __init__(self,policy,mission,config):
        self.policy,self.mission,self.config = policy,mission,config
        self.memory = None
        self.index = torch.zeros(len(mission.points),dtype=torch.long,device=mission.points.device)

    @torch.no_grad()
    def step(self,observation22):
        from .waypoint_mission import advance_target,goal_packet
        self.index = advance_target(observation22[:,:3],self.mission,self.index,self.config.arrival_radius_m)
        goals = goal_packet(observation22,self.mission,self.index,self.config.passage_speed_m_s)
        output = self.policy(observation22,self.memory,goals=goals)
        self.memory = output.next_state
        return output.action

"""Deterministic finite scene banks; multipliers never migrate to unrelated rows."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
import os
import time
import torch

from .env_raptor import RaptorSimulator, RaptorState
from .response_noise import DisturbanceConfig, attach_disturbances
from .response_sampling import sample_coverage, validate_sampling
from .scene_prefetch import ScenePrefetch
from .waypoint_mission import build_mission
from .waypoint_config import WaypointConfig


@dataclass(frozen=True)
class WaypointBank:
    initial: RaptorState
    mission: object
    key: str
    report: dict

    def to(self, device, dtype):
        return replace(self,initial=self.initial.to(device,dtype),mission=self.mission.to(device,dtype))


def bank_seeds(seed, bank_index, split):
    if split not in ('train','validation','test'):
        raise ValueError("unknown bank split")
    raw = json.dumps(['waypoint-bank-v1',int(seed),int(bank_index),split]).encode()
    value = int.from_bytes(hashlib.sha256(raw).digest()[:8],'little') & ((1<<60)-1)
    return [value+j for j in range(4)]


@torch.no_grad()
def sample_bank(scenarios, index, options):
    started = time.perf_counter()
    count = 4*scenarios
    seeds = bank_seeds(options['seed'],index,options.get('split','train'))
    dtype = options['dtype']
    mode = options['sampling']
    validate_sampling(mode,count)
    if mode == 'coverage128':
        state, report = sample_coverage(count,seeds,dtype=dtype)
    else:
        parts = [RaptorSimulator().reset(scenarios,seed=s,dtype=dtype,horizon=1,
                                        disturbances=DisturbanceConfig.clean()) for s in seeds]
        state = RaptorState(**{f.name:torch.cat([getattr(s,f.name) for s in parts]) for f in fields(parts[0])})
        state = replace(state,noise_row=torch.arange(count))
        report = {'mode':'random','total_train':count}
    state = attach_disturbances(state,options['disturbances'],seed=seeds[0]^0xFACE731,
                                horizon=options['horizon'])
    config = WaypointConfig(**options['waypoint_config'])
    state,mission = build_mission(state,config,seed=seeds[0]^0x1764BC1)
    digest = hashlib.sha256()
    for obj in (state,mission):
        for f in fields(obj):
            v = getattr(obj,f.name).contiguous()
            digest.update(f.name.encode()); digest.update(str(v.dtype).encode())
            digest.update(str(tuple(v.shape)).encode()); digest.update(v.numpy().tobytes())
    key = digest.hexdigest()
    report = {**report,'bank_index':index,'split':options.get('split','train'),
              'seeds':seeds,'bank_key':key,'requested_tilt_degrees':config.tilt_degrees,
              'band_half_width_m':[float(mission.band_half_width.min()),float(mission.band_half_width.max())],
              'initial_speed_range_m_s':[float(state.velocity.norm(dim=-1).min()),float(state.velocity.norm(dim=-1).max())],
              'external_force_enabled':options['disturbances'].force_enabled}
    bank = WaypointBank(state,mission,key,report)
    stats = {'sampling_cpu_seconds':time.perf_counter()-started,'worker_finished_at':time.perf_counter(),
             'worker_pid':os.getpid(),'worker_threads':torch.get_num_threads(),
             'worker_interop_threads':torch.get_num_interop_threads(),
             'worker_cuda_initialized':torch.cuda.is_initialized()}
    return index,bank,seeds,report,stats


class WaypointPrefetch(ScenePrefetch):
    """Reuse the accelerated one-worker/one-pending lifecycle, with a new payload."""
    def __init__(self,scenarios,*,options,threads):
        super().__init__(scenarios,horizon=options['horizon'],threads=threads,dtype=options['dtype'],
                         sampling=options['sampling'],disturbances=options['disturbances'])
        self.options.update(options)

    def submit(self,index):
        if self.closed or self.future is not None:
            raise RuntimeError("sampler closed or one bank already pending")
        self.future = self.pool.submit(sample_bank,self.scenarios,index,self.options)
        self.index = index

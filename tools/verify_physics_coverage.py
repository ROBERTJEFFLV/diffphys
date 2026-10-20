#!/usr/bin/env python3
"""Bounded CPU sampling audit; no rollout, optimization, or EVAL-driven tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from env_raptor import RaptorSimulator
from response_noise import DisturbanceConfig
from response_sampling import physics_cell_ids, sampling_contract, yaw_authority, TTI_EDGES
from response_training import sample_training_scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.batches <= 64:
        parser.error('batches must be in [1,64]')
    torch.set_num_threads(1)
    definition = sampling_contract('coverage128')
    output = {'definition': definition, 'batches': args.batches, 'total_train_per_batch': 2048,
              'torch_version': str(torch.__version__), 'device': 'cpu',
              'scope': 'parameter coverage and sampler wall time, NOT learned flight performance'}
    for mode in ('random','coverage128'):
        counts, times, rounds = [], [], []
        for i in range(args.batches):
            report = {}
            start = time.perf_counter()
            state, _ = sample_training_scenarios(512, 17000+i, horizon=5, sampling=mode, sampling_report=report)
            times.append(time.perf_counter()-start)
            counts.append(torch.bincount(physics_cell_ids(state,definition),minlength=128))
            if 'candidate_rounds' in report:
                rounds.append(report['candidate_rounds'])
        values = torch.stack(counts)
        output[mode] = {'min_cell_count': int(values.min()), 'max_cell_count': int(values.max()),
                        'missing_cells_total': int((values==0).sum()),
                        'cell_count_std': float(values.double().std(unbiased=False)),
                        'median_sampling_seconds': statistics.median(times),
                        'max_sampling_seconds': max(times), 'candidate_rounds': rounds,
                        'cell_counts_by_batch': values.tolist()}
    # Independent prior check: no labels or policy; do not retune the definition.
    count, low = torch.zeros(4,dtype=torch.long),torch.zeros(4,dtype=torch.long)
    for i in range(32):
        state=RaptorSimulator().reset(4096,seed=7_500_000_000+i,dtype=torch.float64,
                                      horizon=1,disturbances=DisturbanceConfig.clean())
        ids=torch.bucketize(state.torque_to_inertia,torch.tensor(TTI_EDGES,dtype=torch.float64),right=True)
        is_low=yaw_authority(state)<torch.tensor(definition['yaw_medians_by_tti_bin'],dtype=torch.float64)[ids]
        count += torch.bincount(ids,minlength=4)
        low += torch.bincount(ids[is_low],minlength=4)
    output['independent_prior_check']={'seed_base':7500000000, 'scenes':int(count.sum()),
                                      'low_yaw_fraction_by_tti_bin':(low/count).tolist(),
                                      'interpretation':'Empirical thresholds approximate, not exactly equal, prior probability.'}
    print(json.dumps(output,indent=2,allow_nan=False))


if __name__=='__main__':
    with torch.no_grad():
        main()

#!/usr/bin/env python3
"""Reproduce the fixed yaw splits from independent clean parameter draws only.

Print JSON; never overwrite a definition, train an Actor, or read EVAL outcomes.
The released definition is used during training, not recalibrated per batch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from env_raptor import RaptorSimulator
from response_noise import DisturbanceConfig
from response_sampling import SAMPLING_VERSION, TTI_EDGES, RISING_EDGES, FALLING_EDGES, yaw_authority


def calibrate() -> dict:
    batch_size, batches, seed_base = 4096, 64, 6_500_000_000
    groups = [[] for _ in range(4)]
    for i in range(batches):
        state = RaptorSimulator().reset(batch_size, seed=seed_base+i, dtype=torch.float64,
                                        horizon=1, disturbances=DisturbanceConfig.clean())
        ids = torch.bucketize(state.torque_to_inertia, torch.tensor(TTI_EDGES, dtype=torch.float64), right=True)
        authority = yaw_authority(state)
        for j in range(4):
            groups[j].append(authority[ids == j])
    values = [torch.cat(group) for group in groups]
    return {"version": SAMPLING_VERSION,
            "environment_source_sha256": hashlib.sha256((ROOT / 'env_raptor.py').read_bytes()).hexdigest(),
            "tti_edges": list(TTI_EDGES), "rising_edges": list(RISING_EDGES),
            "falling_edges": list(FALLING_EDGES),
            "yaw_authority": "single_rotor_km_times_maximum_thrust_over_Jz",
            "yaw_medians_by_tti_bin": [float(torch.quantile(v, .5)) for v in values],
            "calibration": {"source": "independent_original_clean_reset_no_actor_or_eval",
                            "seed_base": seed_base, "batches": batches, "batch_size": batch_size,
                            "dtype": "float64", "quantile": "linear_0.5", "torch_version": str(torch.__version__),
                            "counts_by_tti_bin": [v.numel() for v in values]}}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_num_threads(1)
    with torch.no_grad():
        print(json.dumps(calibrate(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()

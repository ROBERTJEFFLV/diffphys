"""The prefetch boundary must preserve data, RNG and bounded process ownership."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path

import pytest
import torch

from diffphys.response_training import capture_rng, sample_training_scenarios
from tools.replay_response_update import compare_values

def exact(a,b):
    assert compare_values(a,b)[0]


@pytest.mark.parametrize('sampling,scenarios', [('coverage128',32),('random',8)])
def test_spawned_cpu_batches_match_original_all_fields_report_and_main_rng(sampling,scenarios):
    from diffphys.scene_prefetch import ScenePrefetch
    torch.set_num_threads(1)
    before = capture_rng()
    worker = ScenePrefetch(scenarios, horizon=3, threads=1, sampling=sampling)
    try:
        for index in [0, 1, 49]:
            worker.submit(index)
            with pytest.raises(RuntimeError, match='pending'):
                worker.submit(index+1)
            with pytest.raises(ValueError, match='index'):
                worker.take(index+1)
            state, seeds, report, stats = worker.take(index)
            reference_report = {}
            reference, expected_seeds = sample_training_scenarios(
                scenarios, index, horizon=3, sampling=sampling, sampling_report=reference_report)
            exact(asdict(reference), asdict(state))
            exact(seeds, expected_seeds)
            exact(report, reference_report)
            assert stats['worker_pid'] != os.getpid()
            assert not stats['worker_cuda_initialized']
            assert stats['worker_threads'] == 1
            assert all(t.device.type == 'cpu' for t in asdict(state).values())
        exact(before, capture_rng())
        pid = stats['worker_pid']
    finally:
        worker.close()
    assert not Path(f'/proc/{pid}').exists()
    with pytest.raises(RuntimeError, match='closed'):
        worker.submit(50)


def test_worker_failure_propagates_and_close_joins_running_unused_batch():
    from diffphys.scene_prefetch import ScenePrefetch
    worker = ScenePrefetch(32, horizon=3, threads=1)
    worker.submit(-1)
    with pytest.raises(ValueError, match='sampling index'):
        worker.take(-1)
    worker.submit(0)
    _, _, _, stats = worker.take(0)
    worker.submit(1)
    worker.close()
    assert not Path(f'/proc/{stats["worker_pid"]}').exists()
    worker.close()

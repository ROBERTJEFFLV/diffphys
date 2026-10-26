"""One spawn CPU sampler and one pending batch; no training or CUDA in worker."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import time

import torch

from response_noise import DisturbanceConfig


def _initialize(threads: int, interop_threads: int):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(interop_threads)
    # A terminal interrupt is handled cooperatively by the owning training process.
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _sample(scenarios: int, index: int, options: dict):
    from response_training import sample_training_scenarios
    started = time.perf_counter()
    report = {}
    state, seeds = sample_training_scenarios(scenarios, index, device='cpu',
                                            sampling_report=report, **options)
    finished = time.perf_counter()
    stats = {'sampling_cpu_seconds':finished-started, 'worker_finished_at':finished,
             'worker_pid':os.getpid(), 'worker_threads':torch.get_num_threads(),
             'worker_interop_threads':torch.get_num_interop_threads(),
             'worker_cuda_initialized':torch.cuda.is_initialized()}
    return index, state, seeds, report, stats


class ScenePrefetch:
    """Capacity counts the one future whether queued, running or ready."""
    def __init__(self, scenarios: int, *, horizon: int, threads: int = 1,
                 dtype=torch.float32, sampling: str = 'coverage128', dt: float = .01,
                 disturbances: DisturbanceConfig = DisturbanceConfig()):
        self.scenarios, self.horizon, self.dtype = scenarios, horizon, dtype
        self.options = dict(horizon=horizon, dtype=dtype, sampling=sampling, dt=dt,
                            disturbances=disturbances)
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                                        initializer=_initialize,
                                        initargs=(threads, torch.get_num_interop_threads()))
        self.future = None
        self.index = None
        self.pid = None
        self.closed = False
        self.close_seconds = 0.

    def submit(self, index: int):
        if self.closed:
            raise RuntimeError('sampler closed')
        if self.future is not None:
            raise RuntimeError('one batch already pending')
        self.future = self.pool.submit(_sample, self.scenarios, index, self.options)
        self.index = index

    def take(self, index: int):
        if self.future is None or index != self.index:
            raise ValueError('requested batch index does not match pending index')
        begin = time.perf_counter()
        future, self.future = self.future, None
        self.index = None
        actual, state, seeds, report, stats = future.result()
        delivered = time.perf_counter()
        if actual != index:
            raise ValueError('worker returned wrong batch index')
        self.pid = stats['worker_pid']
        if stats['worker_cuda_initialized']:
            raise RuntimeError('CPU sampler initialized CUDA')
        stats['batch_wait_seconds'] = delivered-begin
        # Includes the intentional overlap with training, not a pure IPC latency.
        stats['batch_age_seconds'] = delivered-stats.pop('worker_finished_at')
        return state, seeds, report, stats

    def close(self):
        if not self.closed:
            begin = time.perf_counter()
            # Running work cannot be cancelled: its completion and process join are charged.
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.future = None
            self.index = None
            self.closed = True
            self.close_seconds = time.perf_counter()-begin


class PrefetchedSampler:
    """Upload the current batch before submitting exactly one next CPU task."""
    def __init__(self, scenes: int, horizon: int, next_index: int, updates: int, *, dtype,
                 threads: int = 1, sampling: str = 'coverage128', dt: float = .01,
                 disturbances: DisturbanceConfig = DisturbanceConfig()):
        self.worker = ScenePrefetch(scenes//4, horizon=horizon, threads=threads, dtype=dtype,
                                    sampling=sampling, dt=dt, disturbances=disturbances)
        self.updates = updates
        self.last = {}
        try:
            self.worker.submit(next_index)
        except BaseException:
            self.worker.close()
            raise

    def __call__(self, scenarios, index, *, horizon, sampling, device, dtype,
                 sampling_report, dt=.01, disturbances=DisturbanceConfig()):
        if (scenarios != self.worker.scenarios or horizon != self.worker.horizon
                or sampling != self.worker.options['sampling'] or dtype != self.worker.dtype
                or dt != self.worker.options['dt'] or disturbances != self.worker.options['disturbances']):
            raise ValueError('prefetch sampling configuration mismatch')
        begin = time.perf_counter()
        state, seeds, report, stats = self.worker.take(index)
        delivered = time.perf_counter()
        sampling_report.update(report)
        initial = state.to(device, dtype)
        uploaded = time.perf_counter()
        del state
        if index+1 < self.updates:
            self.worker.submit(index+1)
        self.last = {**stats, 'sampling_blocking_seconds':time.perf_counter()-begin,
                     'sampling_to_cuda_seconds':uploaded-delivered if torch.device(device).type=='cuda' else 0.,
                     'batch_delivery_and_upload_seconds':uploaded-delivered}
        return initial, seeds

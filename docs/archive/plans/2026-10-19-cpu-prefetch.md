> Historical implementation plan; see [archive notes](../README.md).

# Single CPU sampler prefetch implementation plan

**Goal:** Compare fixed C against P, with one persistent spawn CPU sampler and one pending/ready batch.
**Architecture:** Reuse the original deterministic sampler on CPU. P consumes by update index, uses original blocking state upload, then submits the next index. C and all GPU/GC/optimizer operations remain unchanged.
**Tech stack:** Python multiprocessing spawn via ProcessPoolExecutor, PyTorch 2.10, existing experiment runner/audit.
**Spec:** User phase-three request, 300 actual updates: C100, P100, P50 + separate-process resume50.

Project authorization overrides skill suggestions to delegate, ask for a second approval or commit. Execute in this thread; preserve production defaults and archived source bindings.

## Constraints and review focus

- One CPU worker, intra-op threads 1 matching C; no CUDA context or policy in worker.
- One pending/ready next batch; no seed offsets, task reordering, pinning, stream, custom shared-memory pool or GPU algorithm changes.
- Sampling completion does not advance progress. Shutdown waits for running work inside total timing.
- Test wrong index, duplicate submit, worker errors, interrupted update, GC restoration and abandoned prefetch cleanup.
- Fresh source/config-bound run directories. Never edit old checkpoint hashes. Main RNG must remain byte identical.

## Work

- [x] Add failing CPU tests for real spawned sampling, exact fields/reports/RNG, capacity/index rejection, and close/error lifecycle.
- [x] Implement `diffphys/experimental/scene_prefetch.py`; integrate P only into independent candidate runner/CLI and source binding.
- [x] Verify CPU continuous C/P and independent-process P resume, interruption and fault cleanup.
- [x] Compare actual N2048 H500 CPU batches at multiple indices, all fields/tapes/report byte exact.
- [x] Freeze experiment sources; execute exactly 300 actual GPU updates in four processes. Record generation, parent wait, delivery/upload, total wall including startup/close, compiler counts and combined RSS/PSS.
- [x] Offline verify every audit chain and final checkpoint, plus key H500 trajectories/group gradients, outside performance timing.
- [x] Review actual changes, run CPU suite and focused CUDA checks, publish measured gains/limits and provenance. Keep production default unchanged.


Result: 300 actual updates completed; all state chains and key H500 replays are raw-byte equal. Full process wall C244.440s -> P216.016s (-11.63%); all ten matched blocks improve. Training totals include worker start/close, first batch, audits/checkpoints and final cleanup. No new graphs, no observed sustained live-memory growth in the 100-update window. CPU530 passed/24 skipped; CUDA-environment focused45 passed. Production remains unchanged; see docs/cpu_prefetch.md and reports/cpu_prefetch_20261019/.

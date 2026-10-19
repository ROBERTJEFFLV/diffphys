> Archived pre-promotion experiment. Runtime commands require the source in
> [optimization_sources_20261019.zip](optimization_sources_20261019.zip).
> See [current production execution](../gpu_efficiency.md) for the adopted P path.

# Single-process CPU prefetch acceptance, 2026-10-19

This experiment holds [candidate C](compiled_continuous_acceptance.md) fixed:
strict Compiled physics, original dynamic compaction, and GC deferral over
forward/backward. P adds one persistent CPU sampling subprocess, started with
`spawn`, with capacity for exactly one pending or ready next batch. Production
defaults remain unchanged.

## Measured result

**P passes this bounded numerical/resume acceptance.** Full process wall falls
from **244.440 s to 216.016 s**,
a **11.63% elapsed-time reduction**
(1.132x throughput),
including imports/compiler warmup, worker startup/first batch and shutdown,
normal audit/checkpoints and final cleanup. Production remains unchanged.

| Chain, 100 actual updates each | Full process wall | Training incl. worker start/close + final I/O/GC | Frozen compiler startup | Cycle median | p90 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C serial | 244.440 s | 222.419 s | 19.201 s | 2.2135 s | 2.2710 s | 2.2910 s | 2.3586 s |
| P prefetch | 216.016 s | 194.167 s | 18.961 s | 1.9049 s | 1.9540 s | 2.0780 s | 3.7418 s |
| P 50 + fresh resume50 | 243.733 s | 199.865 s | 38.279 s | 1.9418 s | 1.9840 s | 3.7015 s | 3.8102 s |

The training-only total decreases **12.70%**.
Worker creation is inside that timer, after frozen compiler stabilization.
P's first update is the slowest because it includes first-batch wait; it is
not discarded. P's charged worker shutdown takes **0.354957 s**;
the split chain's two shutdowns total **0.712999 s**.
The split chain incurs two full process/compiler starts and first batches,
so it is a resume-correctness comparison rather than an additional paired
steady-state speed estimate. Its entire two-process cost is retained above.

| Stage | C median | P median | C sum /100 | P sum /100 |
| --- | ---: | ---: | ---: | ---: |
| CPU generation | 0.253221 s | 0.251315 s | 25.441599 s | 25.144286 s |
| Main wait, including first batch | 0.253221 s | 0.000005 s | 25.441599 s | 1.732446 s |
| Delivery + unchanged blocking upload | 0.013626 s | 0.015482 s | 1.377100 s | 1.550428 s |

P first-batch wait is **1.731948 s**. The other
99 batch waits total **0.000497965 s**;
median **4.59 microseconds**,
maximum **15.33 microseconds**.
The worker still spends roughly 0.25 s generating each batch, but its result
is ready before the GPU training process needs it. Removing serial generation
wait saves a measured **23.709 s**, after
charging the first batch. Delivery/upload is slightly slower and is retained
in the measured total. Background-thread IPC work is also charged to wall time.

All ten corresponding ten-update blocks are faster with P; block reductions
range **3.08%–15.48%**,
including first-batch cost in block 1. These are correlated blocks within one
C-then-P process pair, not independent repetitions or a confidence interval.
The approved budget did not include repeated pairs or reversed run ordering.

The earlier 11.87% ideal bound used the previous 200-update workload and assumed
other stage timings stayed fixed. Here forward+backward totals also differ:
C **187.559 s**, P
**181.349 s**. Thus the measured
12.70% training reduction is not all attributed to removed sampling wait.
This experiment does not separate GPU clock effects, host submission timing
and other run-to-run variation. It establishes overlapped preparation and lower
actual elapsed time in this bounded run, not a revised theoretical ceiling.

### Exact learning and memory

All 300 actual update capsules pass raw-byte checks of every input/output
Actor/Adam/main RNG state, sampling seed/report, group/clipping report, gradient
entering Adam and parameter-change metric. Each chain has 100 nonzero parameter
updates and exactly sampling indices 0..99. Named Adam advances from donor
step 78,950 to 79,050. The saved split midpoint and fresh-process final result
match the uninterrupted chain. Final Actor SHA256:
`02ac8eb9fe4bdeac0ab8d272d86f02df22959dd148a0143b80a5242c655900f5`.

CPU batches at indices 0/1/49/50/99 are identical in every dataclass tensor field,
all noise/rotation/pulse tapes, order, seeds and report, with unchanged main RNG.
Frozen H500 replays at updates **1/50/51/100** pass byte equality of initial state,
all captured step/trajectory/recurrent/history/active/termination values, raw
128-group gradients, clipping and final gradients; final gradients equal the
actual training audits too. No thresholds are relaxed. Numerical comparisons
run outside the timing process and call no Adam.

There are exactly four Dynamo graphs after each frozen startup and **zero new
graph compilations during all 300 learning updates**. GC remains deferred over
the same forward/backward region for both C and P; neither has GC callbacks
inside forward/backward. Normal/end-of-run collection costs remain charged.

Post-release CUDA live allocation is **19,339,264 bytes** at every boundary in
all three chains. P's combined parent/worker PSS ten-update medians, in MiB, are
`[2235.55, 2250.82, 2245.29, 2313.93, 2297.73, 2296.96, 2300.79, 2288.64, 2310.46, 2267.68]`;
C's are `[1731.11, 1788.61, 1797.77, 1813.25, 1809.73, 1811.39, 1811.39, 1819.0, 1807.78, 1814.67]`.
P worker RSS medians are
`[775.6, 790.37, 784.71, 853.29, 817.25, 816.44, 820.23, 807.95, 829.72, 786.93]` MiB.
This is a bounded additional CPU runtime/cache/batch footprint, with allocator
warming and fluctuations rather than a sustained rising sequence. Combined
RSS ranges 2610.5–2817.3 MiB;
combined PSS ranges 2129.2–2355.5 MiB.
There is no claim of constant RSS or unlimited-run memory stability.

Each uninterrupted P run uses one worker PID; the resumed chain uses a new
worker PID after restart. All delivered batches report CUDA uninitialized,
intra-op threads=1 and inter-op threads=20 matching the parent. All actual
workers are joined and gone at exit. Tests additionally cover wrong consumption
index, duplicate outstanding submission, sampler exception and continued use,
unused-work shutdown, real SIGINT, forward/backward exceptions and direct
KeyboardInterrupt with GC initially enabled/disabled. GC restores its entry
state; Probe hooks/payloads and failed-update learning state are cleaned up.

Fresh verification: **530 CPU tests passed, 24 skipped**; **45 CUDA-environment
focused tests passed**. The real performance driver and offline verifier both
exit zero. Production source SHA256 remains
`0965252f0b69a6d9f2a755d9ad2dafae2f2ac20e9503b58736f903414fe8af61`;
the original `best.pt` and `tests/core_contract.json` are unchanged.

## Implementation and numerical boundary

`diffphys/experimental/scene_prefetch.py` uses the standard-library
`ProcessPoolExecutor(max_workers=1, mp_context=...spawn...)`. It has one future,
including when the batch is still queued, running or already ready; a second
submission and consumption by an incorrect update number are rejected.
The worker calls the original `sample_training_scenarios(..., device='cpu')`.
It does not construct an Actor or optimizer, execute a rollout, or initialize
CUDA. Intra-op threads are fixed at 1, matching the parent; inter-op threads
match the parent's setting. There is no worker seed offset or changed random
call order. Sampling still uses the original update-derived local generators.

The main process consumes k, checks its number, calls the original blocking
`RaptorState.to(device, dtype)`, releases the CPU batch, and only then submits
k+1. Forward/backward/Adam still use the current, sequentially updated model.
Only the existing successful update commit advances both `updates` and
`sampling_index`. There is no trajectory precomputation, pinning, new CUDA
stream, async audit, custom shared-memory pool, boolean-fill change, or
compaction/RK4/GRU/gradient/optimizer change.

C retains its original serial sampler and transfer. P's CPU boundary performs
one additional no-op CPU `.to` inside the unmodified sampler before the original
GPU upload; the complete CPU data boundary is checked at the byte level.
IPC uses PyTorch's standard CPU tensor transport, including any storage-sharing
copy and serialization cost. Communication is not assumed free.

## Initialization, budget and resume

Authorized actual CUDA updates: C100, P100, and P50 followed by a fresh process
restoring the checkpoint for another 50, **300 total**. Each process has a
frozen full H500 forward/backward compiler warmup with no Adam. Fresh learning
runs import the same local `best.pt` Actor and named Adam as an explicitly
identified experiment fork, starting sampling index zero. This is not exact
resume of the donor's old source. Its hash and original contents are retained.
Scenes=2048, H500, FP32, coverage128/groups128, loss `.01/.01/25`, Time Decay=1,
gradient scale=0.1, group cap=1, global cap=10; boolean-fill remains disabled.

New candidate checkpoints use schema `continuous-compiled-candidate-v2` with
strict production/experimental source, actual config and numerical-runtime
bindings. Existing v1 experiment evidence remains in its archived source
snapshot; the current CLI does not pretend that v1 is a v2 exact resume.
No checkpoint binding/hash is edited. The new source/config requires new run
directories. The v2 target is 100 even for the first 50-update segment.

The process prefetch buffer is not checkpoint state. On save/exit, any unused
batch is discarded; on restore the saved next update index is regenerated.
`shutdown(wait=True, cancel_futures=True)` cancels work only if still pending;
already running work is joined before the total timer ends. Worker state is
not used to infer successful training progress.

Run the four commands below only under a new approved 300-update budget:

```bash
python tools/train_compiled_candidate.py --config configs/compiled_prefetch_acceptance.json \
  --scheme C --donor best.pt --work-dir /tmp/prefetch-C
python tools/train_compiled_candidate.py --config configs/compiled_prefetch_acceptance.json \
  --scheme P --donor best.pt --work-dir /tmp/prefetch-P
python tools/train_compiled_candidate.py --config configs/compiled_prefetch_acceptance.json \
  --scheme P --donor best.pt --work-dir /tmp/prefetch-split --stop-after 50
python tools/train_compiled_candidate.py --config configs/compiled_prefetch_acceptance.json \
  --scheme P --resume /tmp/prefetch-split/latest.pt --work-dir /tmp/prefetch-split
```

## Measurement definitions

- CPU generation: worker time inside the original sampler, before result IPC.
  C uses original sampler elapsed minus observed original upload elapsed.
- Main batch wait: P time blocked in `future.result()`, including result
  readiness/transport. C's corresponding time is its serial CPU preparation.
- Delivery/upload: P's report handoff plus original blocking state upload;
  C's observed original transfer. IPC unmarshalling in the executor management
  thread can also consume host time outside this small handoff interval: its
  cost remains in actual cycle/whole-process wall time, not subtracted.
- Training total: starts before pool creation/first submit, includes first
  wait, all sampling/IPC/upload, model/Adam backup, full forward/backward, clipping,
  Adam, audit/history fsync, checkpoints every 50 updates, telemetry, worker
  shutdown/join, final GC, and final checkpoint/log I/O.
- Full process wall: driver time from process launch to exit, additionally
  includes imports/setup/source audit and frozen compiler startup. Both this
  total and the training total are reported; startup is not a free operation.

CPU generation overlaps GPU execution. These stage durations cannot be added
to reconstruct elapsed time. `batch_age_seconds` includes intentional overlap
between worker completion and main receipt; it is not a pure IPC latency.
Timing and offline correctness replay are separate. No EVAL is scheduled for
this bounded update/resume study. Newly triggered compilation during learning
remains inside the measured total.

Memory records parent + sampling-worker RSS sums and PSS sums at update
boundaries. RSS counts shared pages twice; PSS apportions shared pages. The
standard multiprocessing resource-tracker helper is not a second sampler;
its small fixed memory is outside this named parent/worker sum, and its work
is inside process wall time. A bounded extra batch/runtime footprint is allowed;
a rising live-storage trend is not interpreted as an optimization success.

## Evidence and scope

Local ignored artifacts are under `reports/cpu_prefetch_20261019/`, with CPU
field/tape hashes, all actual update audit capsules, original midpoint save,
process commands/times, segment/timing/GC/memory logs, offline comparison
reports, source archive, test logs and an evidence manifest. The offline gate is:

```bash
python tools/verify_cpu_prefetch.py --root reports/cpu_prefetch_20261019
```

It checks every learning-state transition across C/P/P_split, then replays
updates 1/50/51/100 using their actual audited input weights, with P data
prepared by the spawn worker. It compares full captured H500 inputs/outputs,
active/termination trajectories, recurrent/history state, raw group gradients,
clipping and gradients entering Adam; these gradients must also equal the
actual training audit. These replays do not call Adam or count toward timing.

Numerical and speed conclusions apply only to this RTX 4060 Ti, PyTorch
2.10.0+cu126, CUDA12.6 and matching configuration. They do not establish learned
control quality, deployment safety, other hardware/software behavior, or
indefinite memory stability. The production chain remains unchanged.

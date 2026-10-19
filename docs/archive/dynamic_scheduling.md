> Archived pre-promotion experiment. Runtime commands require the source in
> [optimization_sources_20261019.zip](optimization_sources_20261019.zip).
> See [current production execution](../gpu_efficiency.md) for the adopted P path.

# Dynamic compaction: bool writes and GC, 2026-10-19

Follow-up: [C continuous training and exact-resume acceptance](compiled_continuous_acceptance.md)
checks 200 genuine changing-weight updates and a fresh-process 100+100 resume.
It keeps original bool writes and advances the sampling index every update.

All four schemes retain dynamic compaction and the strict Compiled physics
backend. The bool-write replacement is numerically exact but has no demonstrated
net update speedup by itself. Deferring automatic cyclic GC through forward and
backward reduces the measured cumulative time, including cleanup, on this fixture.
No production module or default was changed.

## Schemes and controls

| Scheme | Change from current Compiled + dynamic |
| --- | --- |
| A | Original bool assignment and automatic GC |
| B | Only `active[indices] = True` becomes `active.index_fill_(0, indices, True)` |
| C | Only critical-region GC scheduling and completed-graph release before restoring GC |
| D | B and C together |

The experiment substitutes exactly one statement in an isolated copy of the
original rollout. The original collector bytecode gets a private rollout binding.
`nonzero`, compacted batch sizes, stable scene IDs, Actor inputs, RK4 order,
Time Decay, native probe, group accumulation/clipping and Adam remain unchanged.
There is no CPU prefetch, CUDA Graph capture or additional compiler fusion.

Runtime/fixture match the [previous H500 scheduling study](rollout_scheduling.md):
RTX 4060 Ti 8 GiB; Python 3.11; PyTorch 2.10.0+cu126; FP32, TF32 disabled,
one CPU thread; 2,048 coverage128 scenes, 128 fixed groups, H500. There are
**1,018,203 valid transitions**. Loss `.01/.01/25`, Time Decay=1, gradient scale
0.1, group cap=1 and global cap=10 are unchanged.

Each scheme has **one captured warmup update and 20 timed updates: 84 updates
total**. The timed rounds alternate in balanced orders `ABDC`, `BCAD`, `CDBA`,
`DACB`, repeated five times. Every update restores the same local `best.pt`
weights, named Adam moments/steps, RNG and sampling index 0; this is not exact
training continuation. The checkpoint remains at update 78,950; every experiment
Adam advances from 78,950 to 78,951, in a separate audit capsule.

Checkpoint SHA256:
`5ddbef550e1bd5a15701caa32d0244e1512b38b6a212b13cc5c3665cd2764851`.
Initial state and immutable tapes SHA256:
`f24f69dc63db49974b95cc97eaf37bc54183e27a8931c2516c0c90fa6ec0673d`.
Production source SHA256 remains
`0965252f0b69a6d9f2a755d9ad2dafae2f2ac20e9503b58736f903414fe8af61`.
Compiler counters are stable throughout all 80 timed updates: four unique
graphs before and after; no timed recompilation occurred.

## Full-update and release-inclusive measurements

Times are seconds. Update includes backup, sampling, statistics, native backward,
group/global clipping, persistent Adam, finite checks, real audit capsule and
history fsync. Setup/reset, verification and release are also timed in the
enclosing cycle. Periodic EVAL and checkpoint saving are excluded.

| Scheme | Update median | Update + release median | Update + release p90 | Update + release max |
| --- | ---: | ---: | ---: | ---: |
| A | 2.2054 | 2.2351 | 2.2866 | 2.3626 |
| B | 2.1853 | 2.2151 | 2.3115 | 2.3266 |
| C | 2.0599 | 2.0607 | 2.1071 | 2.1465 |
| D | 2.0414 | 2.0422 | 2.1050 | 2.1287 |

The enclosing cycle is the stricter basis for the GC claim: it additionally
charges restored-checkpoint reset, exact numerical verification, graph/temporary
release, gradient clearing, CUDA completion, memory telemetry and progress output.
It catches collections that happen between the core update and the next cycle.

| Scheme | Cycle median | Cycle p90 | Cycle max | 20 cycles summed | Sum + entire terminal drain |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 2.3071 | 2.3595 | 2.4357 | 45.3564 | 45.4978 |
| B | 2.2876 | 2.3867 | 2.4009 | 45.6405 | 45.7818 |
| C | 2.1328 | 2.1792 | 2.2182 | 42.8094 | 42.9507 |
| D | 2.1155 | 2.1804 | 2.2058 | 42.6162 | 42.7576 |

The campaign has **176.5645 s** elapsed: summed cycles 176.4226 s, final debt
drain **0.14134 s**, and approximately 0.00057 s of remaining coordinator time.
There is no forced collection between updates. One setup collection, before any
timed round, removes compiler/capture objects. One terminal full collection is
inside the campaign clock, after all update objects and hooks are released.
The last column conservatively charges its **entire** cost to each scheme
separately; those columns are alternative bounds, not values to sum together.

Even charging the entire terminal drain only to the candidate, C's cumulative
time is **5.30% below A's undrained sum**, and D's is **5.73% below**. The observed
benefit therefore survives accounting for cleanup rather than just moving it
out of the forward timer. The four schemes share an interleaved process and GC
debt; these are cumulative treatment times in that campaign, not four isolated
20-update uninterrupted training runs.

Paired differences use corresponding rounds; intervals below are exploratory
95% bootstrap intervals for the paired mean, from 5,000 fixed-seed resamples:

| Cycle comparison | Mean difference | Interval | Faster rounds |
| --- | ---: | ---: | ---: |
| B minus A | +14.20 ms | -44.41 to +68.13 ms | 10/20 |
| C minus A | -127.35 ms | -167.01 to -83.84 ms | 17/20 |
| D minus A | -137.01 ms | -173.46 to -98.32 ms | 19/20 |
| D minus C | -9.66 ms | approximately -24.27 to +2.83 ms | 14/20 |

B has a slightly lower median but a higher cumulative time; its interval spans
zero. The additional D-over-C benefit also spans zero. This campaign supports
GC scheduling as the meaningful improvement, and does **not** establish that
bool-fill savings are a stable net gain or that the two benefits add. These
intervals describe one fixed-fixture campaign, not independent machine/seed
replications or a universal throughput guarantee.

## Bool-write timeline: preserve nonzero, remove the extra transfer

Detailed CUDA timelines were collected **after** performance testing, in separate
frozen A/B processes with no optimizer calls. The only B computation change is
the bool assignment. The [PyTorch 2.10 index_fill_ contract](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.index_fill_.html)
describes scalar filling at specified indices; its actual transfer behavior here
is established by CUPTI rather than inferred from that API contract.

| Measured inside the original 500 step bodies | A | B |
| --- | ---: | ---: |
| `nonzero` calls | 500 | **500** |
| Synchronization calls inside those nonzero scopes | 500 | **500** |
| One-byte H2D copies associated with step runtime calls | 500 | **0** |
| Other synchronization calls | 500 | **0** |
| Total synchronization calls | 1,000 | **500** |
| Total synchronization API duration in this capture | 9.404 ms | 14.263 ms |

There are another ten chunk-entry nonzero calls in both schemes: 510 each in
the full forward pass. Whole-forward sync counts are 1,086 versus 586; backward
has seven each. CUPTI copyKind=1 identifies H2D; correlation IDs locate the
runtime launch inside a step. A has 5,739 step-associated copy activities and B
5,239, accounting for exactly those 500 removed transfers.

Forward kernel launch API counts are **88,199 in both**; backward counts are
**156,423 in both**. Thus B removes the identified copy/barrier without adding
masked rows or extra backward launches. C/D retain these same rollout operators;
their GC behavior is measured by callbacks during the real timed campaign, not
inferred from these A/B-only timelines.

The remaining nonzero waits do not shrink in proportion to the count: their
inside-step API duration is 5.657 ms in A and 14.263 ms in B. Waiting can move to
a subsequent synchronization, and independent captures vary. Removed wait count
is not a guarantee of removed end-to-end time. That distinction agrees with B's
inconclusive 20-update result.

Forward GPU activity unions are 161.36/161.78 ms in 1090.68/1100.80 ms of traced
wall time for A/B. No-traced-work fractions are 85.21%/85.30%; backward fractions
are 71.52%/71.89%. Those are this process's kernel/copy/memset interval unions,
**not SM occupancy or whole-device utilization**. They show substantial remaining
submission gaps. Nsight inflates frozen wall time by 21.5%/25.9% versus the bare
runs, so timeline durations are not substituted for the unprofiled update results.

## GC policy, frequency and cleanup correctness

C/D disable **automatic cyclic GC only** through forward and backward. Ordinary
reference counting continues; the distinction is documented by
[CPython 3.11](https://docs.python.org/3.11/library/gc.html). After the original
native backward completes, Probe's existing `finally: close()` has removed its
module/tensor hooks and cleared payload references. C/D then replace the record's
costs with a detached view of the already-completed result to release the finished
autograd graph before automatic GC resumes. Nothing is detached during H500 or
between chunks, and no additional backward or optimizer operation is performed.
Remaining detached record/probe values are released at the measured cycle end.

The GC context restores the entry enabled state in `finally`, including exception
exit, without changing thresholds `(700, 10, 10)`. Automatic collection resumes
normally; no fixed per-update `gc.collect()` policy is introduced. Callbacks
record collection start/end, stage, generation, collected/uncollectable objects
and trial ID for all schemes.

| Observed GC in 20 timed updates | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| Generation-2 events during forward | 13 | 17 | 0 | 0 |
| Their summed duration | 2199.94 ms | 2781.75 ms | 0 | 0 |
| Generation-0/1 GC at resume boundary | — | — | 3.47 ms | 3.50 ms |

A's generation-2 pauses range from about **154 to 183 ms**, occurring in 13/20
updates rather than every update. Their collected-object counts are all zero:
an expensive scan does not itself prove a leak. C/D avoid scanning the live H500
graph during submission. Releasing the finished graph before collection is
consistent with fewer transient objects being promoted, but promotion rates
were not independently measured.
The terminal full collection takes 141.34 ms and collects 517 objects, already
charged above. Those are measured observations; the full GC scan is not assumed
to be a reusable 132 ms saving per update.

Core forward/backward medians help explain the change, but are not the acceptance
metric:

| Phase | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| Forward/statistics | 0.9775 | 0.9625 | 0.7973 | 0.7838 |
| Backward/group clipping | 1.0138 | 1.0104 | 1.0158 | 1.0132 |
| Finished-graph release / GC-resume boundary | 0.000005 | 0.000005 | 0.0176 | 0.0179 |
| End-of-cycle release/telemetry | 0.0301 | 0.0295 | 0.0009 | 0.0009 |

Graph release and audit costs move between phases. Both tables above charge them
to the candidate; forward-only comparisons would exaggerate the benefit.

Tests exercise normal backward, parameter-version failure during backward and
forward failure through the real collector/probe. They check that hooks/payloads
are cleared and weak probe references die after graph release. The scheduled
update's injected forward failure also checks Actor/Adam/RNG rollback and GC
restoration. Every timed update asserts both the probe and record are immediately
released after dropping the experiment result, without a forced GC.

## Memory and exactness

All 80 timed cycles end with **19,339,264 bytes (18.443 MiB)** of allocated and
active CUDA memory: min=max for A/B/C/D. Peak allocation is 4,334,490,112 bytes
(4.037 GiB) in every cycle. Reserved allocator memory is also constant at
4,439,670,784 bytes (4.135 GiB); reserved cache is not live tensor storage.

RSS drops from 6.126 GB before the first timed cycle to approximately 5.278 GB,
then remains stable. Per-scheme last-five minus first-five median RSS changes
are only 0.217–0.254 MB, including retained bounded GC-event/report metadata.
The terminal drain ends at 5.278 GB and the same 18.443 MiB CUDA allocation.
No sustained tensor-memory growth is observed in this 80-update campaign; this
bounded experiment does not prove indefinite training memory stability.

All four warmup updates compare every H500 active mask, per-scene termination
step, actual trajectory field, step input/output, delayed history, Actor memory
and chunk-end state, plus unaggregated 128-group gradients, clipping norms and
multipliers, final gradients, Actor and persistent Adam tensors. **B/C/D are
bitwise identical to A for every compared field: maximum difference zero.**
All 80 timed updates also reproduce A's initial-state/tape hash, costs, raw group
rows, clipping, final gradients and Adam output bitwise. The timers contain no
trajectory capture hooks. Full per-step capture belongs to qualification rather
than performance measurement. No numerical threshold was relaxed.

## Reproduction and artifacts

```bash
python tools/benchmark_dynamic_scheduling.py --repeats 20 \
  --output reports/dynamic_scheduling_20261019/updates
```

The output directory must be empty. This command executes the bounded 84 updates;
rerunning training needs a new budget. The measurements used the existing local
`/tmp/diffphys-cuda-venv/bin/python` environment. `best.pt` is read-only input.

The reusable experimental code is `diffphys/experimental/dynamic_scheduling.py`
and `scheduled_update.py`. Production and previous benchmark tools/sources are
unchanged. `tools/profile_dynamic_scheduling.py` performs separate frozen A/B
Nsight probes; it never calls Adam. The prior interval-union analyzer is reused.

For frozen timeline reproduction, use a fresh prefix (omit `--index-fill` for A):

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=false --output=/tmp/dynamic-bool \
  python tools/profile_dynamic_scheduling.py --index-fill --report /tmp/bool-wall.json
nsys export --type sqlite --output=/tmp/dynamic-bool.sqlite /tmp/dynamic-bool.nsys-rep
python tools/profile_rollout_scheduling.py --analyze /tmp/dynamic-bool.sqlite \
  --report /tmp/bool-analysis.json
```

The local Nsight Systems 2024.6.2 binary is at `/usr/local/cuda-12.8/bin/nsys`.

Local ignored evidence is preserved under `reports/dynamic_scheduling_20261019/`:
`updates/report.json`, `updates/gc_events.json`, raw reference update tensors,
84 real audit capsules with before/after Adam and source archives, derived timing
and GC analyses, Nsight reports/SQLite files, and test logs. The final source
snapshot and evidence manifest bind runtime, original/generated rollout sources,
checkpoint/source hashes and all artifacts. No checkpoints were overwritten,
and no production promotion, long evaluation or deployment validation occurred.

Fresh checks: complete CPU suite **510 passed, 24 skipped**; focused CUDA
experiment suite **25 passed**. Protected kernel contract is byte-identical to
HEAD; previous experimental source hashes and the checkpoint are unchanged.

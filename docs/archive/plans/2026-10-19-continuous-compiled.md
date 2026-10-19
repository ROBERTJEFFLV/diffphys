> Historical implementation plan; see [archive notes](../README.md).

# Continuous Compiled + dynamic + GC candidate acceptance

**Goal:** Accept C under changing weights, persistent Adam, advancing scene
sampling and fresh-process exact resume before considering CPU prefetch.
Single-agent execution; production/default source is untouched.

**Budget:** A 200 real updates; C 200 real updates; independent C 100 updates,
exit/save, then 100 updates in a new process: 600 real GPU updates total.
Startup frozen compiler warmups and five selected-round frozen numerical replays
do not call Adam. Small deterministic CPU/unit checks are separate.

**Files:** New `diffphys/experimental/continuous_training.py`, independent CLI
`tools/train_compiled_candidate.py`, deterministic acceptance tests, run argument
file and final report. Reuse the original equations/collector/probe/clipping/Adam
and audit operations; do not reuse the benchmark's hardcoded sampling index 0.

- [x] Test real state continuity, advancing per-update sample seeds, named Adam,
  strict checkpoint/config/source validation and fresh-process resume. Reject
  nonempty new runs and incompatible donor interfaces; never rewrite old hashes.
- [x] Implement C-only GC deferral with exact entry-state restoration in finally;
  release finished graphs only after native probe close. Keep original bool
  writes by default; preserve bool-fill as a separate disabled option.
- [x] Test normal/disabled-GC entry, KeyboardInterrupt, forward/backward errors,
  real SIGINT boundary stop and Hook/temporary cleanup with rollback semantics.
- [x] Execute independent A200 and C200 continuous training, each update carrying
  the previous Actor/Adam state, using sampling indices 0..199. Preserve every
  audit capsule and periodic/final checkpoint. Record CPU sampling/transfer wait,
  recompilation counters/times, memory, GC and total time including cleanup/I/O.
- [x] Execute C100, exit, resume its new bound checkpoint in a fresh process for
  another 100 updates. Compare every update state/RNG/sampling chain and final
  checkpoint against uninterrupted C; elapsed wall clocks are not state equality.
- [x] Offline compare A/C every-update Actor/Adam/RNG and sampling, and separately
  replay selected updates 1/50/100/150/200 for exact H500 trajectories, active/
  termination, group rows and actual audit gradients. Exclude correctness probe
  time from performance claims; charge normal audit/checkpoint/final GC to training.
- [x] Run full CPU and targeted CUDA tests, review changes, archive source/runtime
  and artifact hashes, document speed/memory/resume limits. Report CPU wait before
  deciding whether a one-batch CPU sampler would be worth a later experiment.

**Initialization:** An explicitly named experiment donor imports the mature B
Actor and named Adam into a new experiment with sampling progress 0 and provenance
binding. It is neither production `--init-checkpoint` (weights only) nor an exact
resume of the old source. Only new experiment checkpoints support `--resume`,
requiring identical source, config and runtime bindings.

No nonzero restructuring, Mask advancement, expanded four-way sweep, CPU prefetch,
new CUDA streams, async audit, new RK4 fusion or relaxed numerical gates.

**Outcome:** All 600 actual GPU updates completed. Every update's Actor/Adam/RNG,
sampling/group reports and gradients match byte-for-byte across A/C/C_split;
all 200 steps change parameters and advance Adam once. Selected actual-input
H500 replays 1/50/100/150/200 have identical trajectory and raw group-gradient
bytes, and reproduce actual audited gradients. Fresh-process 100+100 final
algorithmic state equals uninterrupted 200 (wall clocks excluded).

Training incl. final I/O/GC: A 470.247 s, C 432.711 s (7.98% reduction), split C
439.565 s; startups are separate and preserved. New training compilations: 0
in every process. Live CUDA allocation remains 18.443 MiB at all 600 boundaries;
C RSS warms then fluctuates without sustained growth. CPU preparation totals
51.357 s (11.87% of C), supporting a later bounded prefetch experiment but no
prefetch speedup claim. CPU suite 520 passed/24 skipped; focused CUDA 35 passed.
The original midpoint checkpoint and all source/artifact hashes are preserved.
See [report](../compiled_continuous_acceptance.md).

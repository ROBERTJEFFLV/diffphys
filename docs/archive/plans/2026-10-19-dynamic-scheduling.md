> Historical implementation plan; see [archive notes](../README.md).

# Dynamic compaction: bool write and GC experiment plan

**Goal:** Isolate the extra bool-write transfer from nonzero synchronization,
then measure whether deferring automatic cyclic GC reduces total work time.
Execute in this single agent thread; production remains unchanged.

**Budget:** A/B/C/D each one captured H500 warmup update and 20 timed updates;
84 total restored-checkpoint updates. Additional frozen timeline probes never
call Adam. Compiler stabilization, if needed, uses frozen probes. No EVAL.

**Architecture:** A is the original Compiled/dynamic computation; B changes only
`active[indices] = True` to `active.index_fill_(0, indices, True)` in an isolated
copy of the original rollout. C defers automatic cyclic GC around forward and
backward, releases the completed graph after probe close, then restores the
previous GC enabled state without forcing a full collection each update. D
combines B and C. Keep nonzero, batches, Actor, RK4, gradients and Adam unchanged.

**Files:** New experimental scheduling/GC helpers, independent update measurement
helper, `tools/benchmark_dynamic_scheduling.py`, focused tests, report doc. Reuse
the existing H500 capture/comparison and interval-union profile analyzer. Preserve
the previous benchmark sources and evidence unchanged.

- [x] Verify Probe.close on backward success, failure and forward exceptions;
  weak references must die after releasing the graph. Test GC state restoration
  for enabled/disabled entry and exceptions, with no threshold changes.
- [x] Write and run deterministic red tests for bool-fill active masks,
  first-failure trajectories, group gradients and real Adam equality. Implement
  the isolated one-statement substitution and GC scheduling helpers.
- [x] Qualify all four actual warmup updates against A: all H500 state fields,
  active flags, termination steps, unaggregated group rows, clipped gradients,
  Actor and saved persistent Adam must remain bitwise equal. Stop on divergence.
- [x] Measure 20 interleaved updates per scheme, balanced execution order;
  restore identical weights/Adam/RNG and sample index 0 for every update. Record
  median/p90/max, paired differences and variability, per-scheme cumulative
  time, outer cycle time including reset/verification/release/GC and job total.
- [x] Record every GC event's stage/generation/time/collected count. Do not move
  collections outside measured cycles. Do one final debt drain for the campaign,
  charge it to total time, and report conservative per-scheme bounds including
  its entire cost. Interleaving shares GC debt: disclose this limitation.
- [x] Track RSS and CUDA allocated/active/reserved bytes after release, peak
  allocation, closed probe references and final debt drain. Do not equate the
  caching allocator's reserved bytes with live tensors or claim a leak from RSS.
- [x] Separately capture A/B H500 Nsight timelines. Prove one-byte H2D/wait counts
  change while nonzero remains. Recheck production/checkpoint hashes, run CPU
  suite and focused CUDA checks, review actual new files, archive evidence.

**Acceptance:** Exact equality required, with no tolerance relaxation. Do not
promote an optimization based on forward time alone or exclude deferred GC from
the overall cost. No CPU prefetch, CUDA Graphs, new RK4 fusion or group redesign.

**Outcome:** All 84 updates completed, all exact gates passed, no timed
recompilation. A/B/C/D inclusive cycle medians: 2.3071/2.2876/2.1328/2.1155 s;
20-cycle sums: 45.3564/45.6405/42.8094/42.6162 s. Terminal full drain 0.14134 s
is included in campaign time and conservative per-scheme bounds. B removes
500 one-byte H2D copies and 500 synchronization APIs while retaining all 500
inner nonzero calls; no stable net B-only or D-over-C speedup is established.
C/D show a cumulative-time reduction. CUDA allocated/active bytes are constant,
RSS settles, all probes/records are released. CPU 510 passed/24 skipped; focused
CUDA 25 passed. Production and previous experimental sources are unchanged.
See [report](../dynamic_scheduling.md) and
`reports/dynamic_scheduling_20261019/evidence_manifest.json`.

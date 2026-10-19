# Promote P and remove superseded runtime paths

**Goal:** Make verified P the single production training path and remove duplicate experiment implementations.
**Architecture:** Promote strict compiled physics and bounded spawn sampling into the core package. Modify the original production trainer to use them with the verified GC context, preserving its evaluation, audit, checkpoint and rollback behavior. Retain native physics as CPU/numerical reference and grouped VJP as explicit oracle.
**Stack:** Python 3.11, PyTorch 2.10, standard-library spawn ProcessPoolExecutor.
**Spec:** User authorizes adopting P and applying Occam's razor repository-wide; no new long training budget.

Execution stays in this thread. No delegation, extra approval, commits or long training. The full pre-cleanup source and historical experiments are preserved before deletion; generated runs/checkpoints/licenses/audit evidence/local agents are retained.

## Review focus

- CPU-only worker, one future, exact index/config/local RNG; support existing random/coverage samplers and disturbance settings.
- GC restored in finally, Probe closed, failed Actor/Adam/RNG rolled back, running worker joined even on exception.
- GPU strict compiler has no relaxed fallback; CPU uses native numerical reference. Remove old rotation-only and experimental backend selectors.
- Resume binds new sources/config/runtime; old checkpoint hashes remain unchanged. Weight initialization retains its original semantics.
- Source kernel AST contracts stay byte-identical; calibration thresholds remain unchanged when recording retired rotation-wrapper source changes.

## Tasks

- [x] Add failing production-entry tests for P defaults, spawn lifecycle, GC/rollback, continuous state/resume, unsupported backend rejection.
- [x] Promote compiler and sampler; integrate original trainer, evaluator, replay and read-only profiler; bind exact new sources/runtime.
- [x] Archive then remove duplicate training/benchmark/Mask/selector code and their obsolete tests/configs. Retain useful numerical and integration regressions using production APIs.
- [x] Run focused checks, full CPU suite and bounded CUDA numerical/H500 checks without long training. Debug failures before proceeding.
- [x] Review full retained tree, update README/AGENTS/current architecture and archive reproduction notes, check all imports/CLI/config links/source contracts and preserved artifacts.


Completed: production CUDA defaults to P through the original single CLI; CPU is the native physics reference with the same bounded sampler/GC lifecycle. Net active Python source: 83 -> 64 files, 13928 -> 10941 lines (2987 fewer). Archived 25 retired active paths and historical reports/plans; kept the original 340 KiB source archive and every old report/checkpoint artifact. Fresh CPU497 passed/19 skipped; CUDA36 passed, including H500 raw-byte trajectory/group/final-gradient and disposable Adam checks. Full simulator method ASTs, Actor/task/groups/probe/sampling/noise, strict compiler, protected contract and calibration values retain the qualified implementations. No new long training or throughput campaign was launched.

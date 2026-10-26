# AFT mainline simplification implementation plan

**Execution:** Native single-agent execution in this session, as required by AGENTS.md. The user has authorized cleanup and publication to the AFT branch.

**Goal:** Remove unused independent workflows and duplicate configuration while keeping the trained AFT program byte-identical.

**Architecture:** Retain the Goal69 three-point mission, original RK4/sensing, complete H500 BPTT, physical group probe, strict CUDA compiler, CPU prefetch/storage, audit, fixed EVAL and short replay. Delete the separate long-hover simulator/evaluator/player and old standalone diagnostics superseded by the actual-task profiler. Flatten the sole recommended AFT argument file without changing effective arguments.

**Tech stack:** Python3.11, PyTorch2.10, NumPy, optional Pygame.

**Spec:** User instruction to remove inactive and unnecessary features; current task contract in docs/pass_stop.md and frozen/20261026-u1456/manifest.json.

## Constraints

- Do not alter any byte of response_training.SOURCE_FILES, protected kernel hashes or physics coverage calibration.
- Preserve frozen/, all generated runs/checkpoints/reports, notices, local configuration and the frozen tag.
- Keep independent numerical/gradient references and hover-weight import; they are needed to validate and initialize the retained Actor.
- No training or long evaluation; use existing deterministic regressions and a zero-update exact-resume check.
- Preserve native single-pass group backward, current loss and all current default argument values.

## Review focus

- The AFT preset must parse with only itself and its mission JSON; no deleted hover preset dependency.
- Short-replay discovery/launch must work and must reject the removed long schema before launching a process.
- Removing the long workflow must not remove shared sensor/physics references or provenance.
- Frozen checkpoint source binding must remain identical after cleanup, with its model/Adam/RNG retained on a zero-update resume.
- Historical frozen manifests apply to their frozen tag, not to a file-deleted working branch; current documentation must say so.

## Task 1: Bound removal and add regression checks

- [x] Trace imports and config references from the active entry and short monitor.
- [x] Add a failing self-contained AFT preset test and unsupported-replay schema test.
- [x] Run and record the expected failures.

## Task 2: Remove the independent legacy workflow

- [x] Flatten configs/agile_flight.args; remove configs/response_raptor_multi_airframe.args and configs/response_long_eval.args.
- [x] Remove response_long_eval.py, tools/evaluate_response_long.py, tools/play_response_long.py and their exclusive tests/document.
- [x] Remove tools/benchmark_group_backward.py and tools/audit_pass_stop_time.py after checking they are not dependencies of the active path.
- [x] Restrict tools/response_monitor.py to the retained short replay schema and remove dead long-player selection.
- [x] Update relevant config/monitor regressions; retain all shared physics and gradient tests.

## Task 3: Document, verify and publish

- [x] Rewrite README and update AGENTS.md/docs/training_gui.md around the sole AFT preset, while preserving execution boundaries.
- [x] Record deleted paths/reasons and measured size reduction in docs/aft_cleanup.md.
- [x] Run the remaining complete CPU suite and retained GUI smoke checks.
- [x] Verify all source-bound files and frozen artifacts against baseline; run zero-update exact resume without rewriting the checkpoint.
- [x] Review the complete diff and prepare the verified tree for publication.

Publication evidence is recorded in the local cleanup report and the resulting Git commit; the frozen tag is preserved.

# PASS/STOP compatibility and replay fixes

Goal: fix the seven current full-suite failures and make PASS/STOP checkpoint
export/rendering faithful to the real EVAL, then run CPU and bounded CUDA checks.
Current thread only; no long training or deployment.

- [x] Trace failures to checkpoint loss parsing/error diagnostics, obsolete config inventory, and hover-only replay construction.
- [x] Add failing replay checks for retasked initial state, exact native flight/metrics, command targets/attitudes, pulse-free telemetry, and Torch-free rendering.
- [x] Re-run the existing failing checkpoint/config tests before fixes.
- [x] Fix invalid-checkpoint rejection without weakening source/schema checks; update the config inventory regression to cover the new mission entry.
- [x] Retain normal EVAL once; record its actual typed commands and prepared initial state, then load/render the typed metadata without affecting hover replay.
- [x] Run focused checks, full CPU suite, protected kernel hashes and bounded CUDA forward/backward/CLI export tests as supported.
- [x] Review diff and document actual results and remaining limits. Keep user models/notes/archives untouched.

Do not change dynamics, sensor acquisition, Actor/mission semantics, task losses,
BPTT/group clipping or protected contract hashes. Source-bound existing runs keep
their old source; no checkpoint hash rewrite is authorized.

Results (2026-10-21): full CPU suite 550 passed / 19 skipped; bounded CUDA
suite 21 passed; H500 CUDA smoke from the user's unchanged best.pt completed
one accepted update (32 TRAIN / 4 fixed EVAL), then exact gradient/model/Adam/RNG
replay and matching fixed-EVAL trajectory export. Pygame screenshot inspected.
Fourteen protected files and all four preserved user artifacts match their hashes.
No continuous training or learned-success claim: this tiny EVAL has zero PASS/STOP
success. Evidence: reports/pass_stop_bugfix_20261021_145241/.

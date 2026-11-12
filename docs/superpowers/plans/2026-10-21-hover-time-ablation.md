# Hover settling-time ablation implementation plan

**Goal:** Compare the unchanged hover objective with the same objective plus a
settling-time surrogate, with exactly 3000 accepted updates per arm.

**Architecture:** An explicit experimental adapter wraps the existing production
trainer and observes its real chunked rollout. No physics, Actor, sampler,
gradient-group or optimizer kernel changes. One native thread, no delegation.

**Spec:** User confirmed retaining hover and adding only the time term.
**Stack:** Existing Python 3.11 / PyTorch 2.10 CUDA environment.

## Fixed comparison

- Both arms: from zero, identical seed-7 random Actor initialization, fresh Adam;
  no checkpoint or `best.pt` initialization. Use
  seed 7, 3072 coverage128 TRAIN scenes, H500, 256 fixed EVAL scenes, original
  external forces/pulses/noise/delay, Time Decay 1, clipping and learning rate.
- Tolerances for a common hover diagnostic: |p|<=0.1 m, |v|<=0.1 m/s,
  |omega|<=0.5 rad/s. No absolute attitude requirement: steady external-force
  rejection can require a tilted aircraft. No PASS/STOP mission is imported.
- r=max(|p|/.1, |v|/.1, |omega|/.5), q=1/(1+relu(r-1)); future quality is
  reverse-cummin(q), masked to zero after boundary failure. T=dt*sum(1-quality)
  on samples 0..H-1. This specializes the current PASS/STOP proxy to an already
  satisfied PASS. Compare weights 0 and 1 for the dimensionless extra term T/(H*dt).
- Retain original Time Decay 1 in BOTH arms. Its backward decay and group/global
  clipping remain surrogate gradient transformations; no exact-gradient claim.
- Same immutable TRAIN seeds by update. Sequential runs avoid GPU contention.
  Save and temporarily stop the current agile run; restore it after experiments.
- Compare final update 3000, using the original hover objective, settling proxy,
  finite-horizon settling time/success, RMS and boundary failures on the same
  512 held-out scenes. Do not compare two differently defined total losses.
- One seed pair cannot establish a multi-seed advantage. Retain failures and
  censored 5 s times; never report early exits as quick recovery.

## Tasks and review checks

- [x] Add failing deterministic tests for final departure, failure masking,
  temporal-shift finite differences, weight-zero numerical continuity,
  physical BPTT gradients, and checkpoint objective isolation.
- [x] Implement `diffphys/hover_time_experiment.py` and
  `tools/run_hover_time_experiment.py`; bind/archive experimental source/config
  in new checkpoints. Ordinary training remains unchanged.
- [x] Run focused/full CPU checks, bounded compiled-CUDA checks, then freeze code.
- [x] Implement the sequential 3000+3000 runner and automatic common evaluation,
  comparison JSON/plots, progress metadata, and restoration of the paused run.
- [ ] Verify accepted update counts, matched initial weights/seeds/config,
  unchanged protected hashes and final evaluation before reporting a result.

Execution: source frozen in
`runs/hover_time_ablation/from_zero_seed7_20261021_155714/source`.
Full CPU: 509 passed, 20 skipped; subsequent failure-gradient regression and
package checks: 15 passed, 1 skipped. CUDA H500 native/compiled augmented loss,
gradients and Adam were byte-equal. A disposable full-3072 2+2-update pipeline
generated paired 512-scene held-out JSON/CSV/plots. Formal 3000+3000 execution is
running separately; no final learning-effect conclusion is claimed here.

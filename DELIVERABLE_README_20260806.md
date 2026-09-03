# DiffPhys continuity/cadence follow-up — 2026-08-06

## Outcome

Arm C remains experimental and default-off. The supplied three-seed screen is
internally consistent, but C failed its preregistered performance gates:

| Comparison | Horizon | Difference | 95% interval |
|---|---:|---:|---:|
| C - A | H500 | -0.553 pp | crossed [-1.595,+0.260] pp |
| C - B | H500 | -0.293 pp | scenario-block [-0.684,+0.098] pp |
| C - A | H10000 | +0.098 pp | crossed [-0.293,+0.521] pp |
| C - B | H10000 | +0.098 pp | scenario-block [-0.098,+0.293] pp |

All A/B/C training-safety gates passed. This follow-up does not claim a new
controller-performance improvement and does not change the Q2/T0 default GRU,
integral, damping, action mapping, motor model, or historical `physical`
sampler.

## Corrected mechanism interpretation

C did not isolate state continuity from supervision or optimizer cadence. In
each H1000 episode, B already has H250-early and H1000-final CVaR events. C adds
an H500-final event, a midpoint optimizer commit, and an H750-early event, and
changes which gradients reach the final commit. The added H500-final signal is
material; only the H750-early recovery signal is very small. Therefore the old
pooled statement that C's extra supervision was uniformly weak is rejected,
while the formal decision against promoting C remains unchanged.

Velocity is intentionally absent from independent threshold CVaR even though
the formal joint success rule contains velocity. There are no velocity-only
H500 failures in the current screen, so adding velocity CVaR is deferred until
a component-gradient and unique-tail-coverage audit; it is not ruled out.

## Default-off Arm D

`configs/continuity_cadence_D.args` implements a staged follow-up experiment:

- use B's compressed reset sequence and 67 optimizer commits;
- emit independent position/omega CVaR event packs every H500 (75 blocks);
- accumulate both H500 blocks in an H1000 episode into one commit;
- avoid C's midpoint update and old-weight-hidden/new-weight continuation.

D is not a pure cadence intervention: in H1000 episodes it changes both CVaR
event timing and configured CVaR weight mass from B's 1.25 to 2.5. It therefore
tests a denser/larger CVaR event package, not timing alone.

The CPU semantic and full-schedule smokes verify implementation semantics only.
They show 150 H250 segments, 59 H500 + 8 H1000 reset episodes, 67 commits, 75
CVaR blocks, no midpoint H1000 commit, and finite accepted updates. The tiny
batch-1 schedule smoke clipped all 67 updates and is not performance or formal
numeric-safety evidence. Formal D performance is unknown until the registered
CUDA preflight and fresh B'/D training are run.

The formal workflow is now closed end-to-end: runners accept D, manifests
freeze exact config/code/initial-checkpoint hashes, validators reject stale or
incomplete provenance, MATLAB inputs bind to the trained checkpoint hashes,
and the staged analyzer computes D-B before it can request A or evaluate D-A.

## Physical coverage and retain-bank findings

- The current `physical-fit` source sampler passed 22 listed necessary and
  source-consistency gates over 65,536 samples, with zero inertia or fall/rise
  violations and exact 4^4 root-cell balance.
- That is not real-fleet validation: no sampled rise time is <=35 ms; `Jx=Jy`,
  zero quadratic thrust coefficient, identical motors, constant mass-scaled
  force, and missing time-varying force/torque disturbances remain important
  limitations.
- A constructive H250 scale-equivalent counterexample spans 0.02 to 5 kg with
  maximum tracked-state difference `1.914e-10`. The current normalized model
  therefore does not force identification of absolute size; this is not a
  general impossibility theorem.
- Historical T0 failure rates are nearly flat across mass quartiles but strongly
  degrade with high thrust-to-weight ratio, force scale, and required tilt.
  These correlated stratifications motivate a separate challenge bank; they do
  not establish causality.
- The legacy retain bank contains 172/281 impossible inertia samples and also
  violates the rise/fall ordering. At a 25% retain fraction, invalid inertia
  alone would occupy about 15.30% of all reset slots. A strict provenance and
  feasibility guard now prevents that bank from entering `physical-fit` while
  preserving historical `physical` reproducibility.

`physical-fit` remains opt-in. No sampler has been promoted to a default.

## Main reports

- `reports/EXPERIMENT_DECISIONS_20260806_ZH.md`: complete Chinese decision and
  next-step report.
- `reports/EXPERIMENT_DECISIONS_20260806.md`: compact English decision report.
- `reports/PREREGISTRATION_ARM_D_20260806.md`: frozen staged B'/D then A'
  protocol, gates, uncertainty method, and forbidden simultaneous changes.
- `reports/continuity_cadence_mechanism_audit_20260806/`: event-level mechanism
  and failure-channel correction.
- `reports/arm_d_cpu_semantic_smoke_20260806/` and
  `reports/arm_d_cpu_schedule_smoke_20260806/`: implementation-only smokes and
  source/output provenance.
- `reports/physical_fit_sampler_audit_20260806/`,
  `reports/size_causality_audit_20260806/`,
  `reports/physical_failure_axes_audit_20260806/`, and
  `reports/retain_bank_compatibility_audit_20260806/`: physical coverage,
  scale-equivalence, failure-axis, and retain-bank evidence.
- The supplied formal A/B/C checkpoints, MATLAB results, internal evaluations,
  and original decision tables remain under their dated report directories.

## Verification

From the extracted bundle root:

```bash
python -m pytest -q
python tools/validate_continuity_cadence_screen.py
python tools/validate_continuity_cadence_label_parity.py
```

The final available CPU regression was `136 passed, 4 skipped`; all four skips
are CUDA-only tests. The formal D CUDA preflight and three-seed performance
screen have deliberately not been represented as completed.

`BUNDLE_INDEX_20260806.csv` records the size and SHA-256 of every payload file.
The adjacent ZIP `.sha256` file records the complete archive hash.

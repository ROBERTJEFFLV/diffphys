# Structured L2F pipeline

This is a Torch-only pipeline for migrating the Q2 controller.  It does not
modify or retrain the historical Q2 baseline.  Every stage uses seed `7`; the
formal calibration tool declares its independent bank seeds from that seed.
The runner stops on a command failure, an upstream stage/gate mismatch,
calibration promotion failure, or Q2 paired migration failure.  There is no
failed-gate override in these configs.

### Cadence semantics (current)

Observation and policy call index `t` means that exactly `t` physics
transitions have completed before the call.  Call0 has no physical response;
the first response is consumed by call1.  Call25 updates the persistent
identifier state but its six capability outputs are explicitly unavailable
and cannot affect control.  The first capability/trim publication is call50,
after responses from all probe commands `u0` through `u49` have been consumed;
call75 is the next registered publication.  Before call50 the allocator keeps
the conservative maximum-effectiveness prior and zero contextual gain.

All structured checkpoints, reports, calibration metadata, and smoke outputs
created before cadence version `call50_first_publication_v3` are stale for
current cadence-sensitive evidence and must be regenerated.  The Q2 baseline
and its artifacts are unaffected.

## Dry run

```bash
python3 tools/run_structured_pipeline.py --stage all --device auto --dry-run
```

Use `--work-dir /path/to/artifacts` to relocate all generated checkpoints and
reports.  The dry run only expands configs and prints commands.

The first stage is the read-only causal identifier oracle. It is shown in dry
runs but deliberately blocked for execution until its formal schema and blind
gate are accepted. This prevents a K35 banked Phase-A run from starting on a
screening artifact. A banked A1/A2 report is expected at
`{work}/causal_identifier_oracle.json` and is checked for the source-checkpoint
hash, cadence semantics, v2/K35 representation, and the registered `.005`
probe.

Probe v4 promotion is an explicit, reproducible operation rather than an
edited JSON flag:

```bash
python3 tools/diagnose_probe_v4.py \
  --checkpoint reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt \
  --output reports/probe_v4_formal.json \
  --scenarios 16 --horizon 125 --n-jobs 4 --formal-freeze
```

`--formal-freeze` can promote only for the fixed H125×16 protocol, when every
registered training seed and the independent validation seed pass and the
selected waveform hash exactly equals `probe_contract.py`; it never consumes
blind seeds.  The record binds the Q2 checkpoint SHA, producer-code SHA,
complete candidate-family evidence, fixed seed split, and empty blind set.
Any formal failure writes an audit report but exits nonzero. The currently
checked `probe_v4_q2_debug.json` has no passing candidate, so this command is
documented for pipeline closure but the formal chain remains blocked today.

The `identifier_pretrain` stage is the production artifact producer. It
requires the v4 frozen/eligible probe and the collector's independent
`requested_formal_shape`/`pretraining_gate_passed` contract (including K35,
physics, finite, and zero-parity checks); it does not require the collector's
`formal_eligible` sequence-artifact field. The stage trains only the
deployable identifier, optional bank adapter, and capability mean head using
teacher-executed trajectories, then writes `identifier_init.pt` through
`structured_checkpoint.write_identifier_init_artifact`. A structurally valid
tensor file is not enough: independent validation and final banks must both
pass the registered call50/call75 capability-mean/equilibrium gate before the
artifact is written. Phase A1 requires the hash-bound producer report as well
as the artifact, and revalidates the current formal-freeze file immediately
before any model or optimizer initialization. Training uses a fresh
deterministic 4x4 bank per update;
validation/final banks are disjoint and blind seeds remain unused.

The `identifier_revalidation` stage is a read-only gate between A1 and A2. It
binds the exact A1 checkpoint, identifier artifact, Q2 checkpoint, probe
waveform, cadence contract, and feature schema, then evaluates two fresh 4x4
authority-stratified banks: teacher-forced (`beta=1`) and fully on-policy
(`beta=0`). Both must pass the Phase-A physical/equilibrium mean gate at calls
50 and 75, including capability/effectiveness metrics, finite rollout, and no
latched identifier failure. Their scenario seeds use separate +30k/+40k
ranges, distinct from pretraining and A1. No blind bank is used;
a failed gate cannot authorize A2.

### Production identifier-init binding

Phase A1 additionally requires `{work}/identifier_init.pt`. This is a
production `StructuredRecurrentPolicy` artifact, not the oracle's diagnostic
GRU. Its `weights` mapping must contain exactly `identifier.cell.*`, the
optional K15/K35 `bank_adapter.weight`, and `capability_head.{weight,bias}`
for the requested architecture. The loader binds those tensors to the exact
Q2 checkpoint hash, complete policy config, current cadence contract, v4
probe hash, and shared feature-schema hash. Missing, extra, stale, non-finite,
or label-bearing payloads fail closed. The artifact contains no capability
labels and no runtime oracle model.

An A1 run updates the production identifier, so its report always sets
`causal_gate_status=stale_revalidation_required` and
`causal_revalidation_required=true`. A2 is blocked until an independently
produced `causal_identifier_revalidation.json` binds the A1 checkpoint and
post-update identifier hash. This is intentionally a reversible interface:
the artifact can be regenerated from a production policy with
`structured_checkpoint.build_identifier_init_artifact`, while no formal or
blind experiment is started automatically.

## Stages

| Stage | Config | Artifact | Contract |
|---|---|---|---|
| Identifier oracle | `structured_identifier_oracle.args` | `causal_identifier_oracle.json` | read-only K35 causal gate; currently blocked |
| Production identifier pretraining | `structured_identifier_pretrain.args` | `identifier_init.pt`, `identifier_pretrain_report.json` | deployable identifier artifact producer; v4/collector gates required |
| Phase A1 DAgger | `structured_phase_a_dagger.args` | `phase_a1.pt` | causal capability mean and physical equilibrium fit; uncertainty frozen |
| Post-A1 causal revalidation | `structured_identifier_revalidation.args` | `causal_identifier_revalidation.pt`, `causal_identifier_revalidation.json` | independent teacher-forced/on-policy Phase-A gate |
| Phase A2 uncertainty | `structured_phase_a2_uncertainty.args` | `phase_a2.pt` | scale-only fit and smoke conformal calibration; mean/identifier frozen |
| Phase B local JVP | `structured_phase_b_local_jvp.args` | `phase_b.pt` | contextual-gain local derivative fit |
| Residual oracle | `structured_residual_oracle.args` | `residual_oracle.pt` | independent annulus action-RMS threshold; screening only |
| Phase C residual DAgger | `structured_phase_c_residual_dagger.args` | `phase_c.pt` | residual fit with runner-injected, hash-checked oracle |
| Formal q calibration | `structured_formal_q_calibration.args` | `calibrated.pt` | held-out self-consistent promotion |
| Frozen postcheck | `structured_postcalibration_evidence.args` | `postcalibration_evidence.pt` | same-hash equilibrium/JVP/annulus/allocator/H250 evidence |
| Paired migration | `structured_paired_migration.args` | `migrated.pt` | natural+balanced 256, H250+H500 |
| Full-space 2xH250 | `structured_fullspace_2xH250.args` | `fullspace_2xH250.pt` | B=64, alpha=.8, stride=1 |
| Full-space 4xH250 | `structured_fullspace_4xH250.args` | `fullspace_4xH250.pt` | independent horizon-scaling test from original migrated |
| Post-MS calibration | `structured_postms_calibration.args` | `postms_calibrated.pt` | fresh q calibration after accepted MS |
| Post-MS postcheck | `structured_postms_postcheck.args` | `postms_postcalibration_evidence.pt` | frozen fresh evidence |
| Post-MS migration | `structured_postms_migration.args` | `postms_migrated.pt` | complete paired release gate |

The formal A1/A2 configurations select `allocator_solver=box_qp`.  This solves
the bounded four-motor allocation problem on an active face and applies the
registered post-burn-in command slew limit of 50 normalized action units/s.
The older smooth DLS allocator remains available only for backward-compatible
diagnostics and old experimental checkpoints.

Run a single stage with the same runner, for example:

```bash
python3 tools/run_structured_pipeline.py --stage phase_a1 --device auto
```

The residual oracle uses normalized feedback-feature radius `[0.25, 3.0]` by
default.  Its `e<r_min` Q2 bias is recorded separately and is not a parity
target; the registered Phase-C threshold is
`max(2*heldout_oracle_rms, measurement_floor)`.

The formal full-space configs use only `residual_head.` as the trainable safe
prefix, `--pre-rollout-steps 50`, and probe every action (`stride=1`).  Their
risk update is fixed at `alpha=.8` over a deterministic 4x4 authority-stratified
physical-fit batch of 64 scenarios (13 effective 20% tail samples); the
migration validator must promote the checkpoint before either full-space stage
can start.

The 4xH250 stage is an independent horizon-scaling test from the original
`migrated.pt`; it does not consume the 2xH250 candidate weights.  If the 4x
solver accepts an MS update, all pre-MS calibration/postcheck/migration
evidence is stale.  The post-MS stages therefore recalibrate, rerun frozen
evidence, and rerun the paired migration gate.  The immutable residual-oracle
definition and its Phase-B hash are carried forward only for same-hash
revalidation; they do not silently promote the changed controller.

The checked-in 64-scenario DAgger banks and one-full-space-step settings are
screening configurations.  Phase A is deliberately split: A1 trains only the
identifier/capability mean and does not install conformal calibration; A2 can
start only after the A1 physical-mean gate passes, freezes both identifier and
mean, and trains only the uncertainty scale before installing smoke calibration.
Both use stratified recurrent replay across all five DAgger rounds, so a large
sigma cannot substitute for a scenario-dependent mean.  A screening run may
fail its formal gates and is stopped rather than being presented as a formal
promotion.  For Phase C, the runner first requires Phase B's active phase,
local gate, and held-out H250 report before running the independent residual
oracle.  It then injects `--external-residual-oracle-report`, and the DAgger
tool verifies the oracle's Phase-B deployment hash, annulus, threshold, and
gate before enabling residual training.

The formal A1 config targets the registered v2 35-candidate fixed multi-tau motor
observer bank and the `.005` zero-sum identification probe.  The legacy
single-observer mode remains the config default for strict old-checkpoint
loading.  K15/v1 is retained only as a failed diagnostic control: its mean
tracking improved, but its strict p95 coverage and capability oracle did not
pass.  `tools/upgrade_structured_multi_tau.py` is the only supported way to
add the bank to an existing structured checkpoint; it verifies zero-adapter
action/identifier parity, invalidates old capability calibration, and never
marks the upgraded artifact as promoted.  The read-only K35 coverage,
physics-ceiling, and causal-sequence oracle gates must all pass before spending
a full A1 training budget.  At present this is a fail-closed experimental
configuration, not a promoted trainer path.

To migrate an old K0 structured checkpoint, choose the target grid explicitly:

```bash
python3 tools/upgrade_structured_multi_tau.py \
  --input old_structured.pt \
  --output structured_k35_v2.pt \
  --report structured_k35_v2_upgrade.json \
  --target-version 2
```

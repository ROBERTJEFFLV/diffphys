# Historical Phase 1 diagnostic: nominal airframe, candidate search

The current Phase 1 baseline is [task-value Actor–Critic with Adam](response_phase1_task_value.md).
This document describes the preserved candidate-search experiment only.

Run the current ResponseMotorPolicy and mean-risk Critic from model seed 7.
Every proposal samples two fresh independent 32-state TRAIN banks, pooled before
all objectives, CVaR weights and candidate selection. Only position, velocity,
attitude and omega are randomized. The simulator uses its existing nominal
parameters with sample_dynamics=False and zero external force.

The complete chain remains H500 no-grad collection -> Critic fit -> 10 H50
windows with numerical continuity and detached boundaries -> five gradient rows
-> at most five orthonormal basis vectors -> +/- each at rho and rho/4 -> real
H500 candidate evaluations on identical pooled initial states.

With `--phase1-train-only`, `CriticConfig.acceptance_mode` is `train-objective`:
choose the finite candidate with lowest true task_objective, and retain it only
if its objective is strictly lower than the current Actor. Hard-risk values,
flight-bound diagnostics and success do not veto finite proposals. No DEV bank
is constructed, evaluated or consulted. There is no periodic DEV evaluation,
DEV acceptance or DEV rollback. Completed finite Critic fits survive rejection.

The ordinary response chain defaults to `train-and-dev`; its existing acceptance
rules are unchanged. The Phase 1 configuration does not use MS, PETSc, FD,
common-descent gates, broad dynamics, capability supervision or hidden-reset
experiments. Training-only capability truth remains a Critic input, not an
Actor input or supervised capability objective.

```
python3 tools/check_response_training_contract.py --device cuda \
  --scenario-mode fixed-airframe \
  --output reports/phase1_single_airframe/contract.json
python3 tools/train_response_control.py $(cat configs/response_phase1_search.args)
```

The independent config requests CUDA float32, 50 proposals, H500/H50, 64 TRAIN
states per proposal and model seed 7. Its 51-rejection/minimum-update settings
keep the default plateau from truncating this bounded experiment. The wall
budget is 1800 seconds. Use a new work directory for a fresh run. Exact resume
binds the acceptance mode; do not mix TRAIN/DEV and TRAIN-only experiments.

## Probes, not optimization objectives

- Compare every physical and policy dataclass field at each H50 boundary to
  a full no-grad H500 reference call. This includes GRU memory, integral,
  motor state, all action/response history and calls. Float32 tolerances are
  atol=1e-6 and rtol=1e-5; float64 uses 1e-12 and 1e-10. Record exact equality
  and each field's absolute error. Reference snapshots never replace state.
- Decompose both task_objective and minimal training objective into position,
  velocity, omega and regularization. Use one CVaR tail selected from total
  scene cost for additive attribution. Preserve the existing distinction
  between gradient performance loss and true evaluation/task_objective.
- Record each window's raw performance parameter-gradient norm and four risk
  gradient rows before the 1/10 window average. Risk norm is Frobenius across
  the four rows. Terminal state-gradient probes use independent detached leaves
  and the frozen target: d(mean-scene cumulative risk_j)/d(dynamic closed state),
  including the H-t factor. Their last-window value is zero. They do not change
  Actor gradients and are not a certificate of accurate future-risk derivatives.
- Record baseline/best/retained H500 objectives, candidate count, basis rank,
  relative improvement, retained update count, and search trial details.
- Record Critic loss, four component MAEs, mean/cumulative true risk ranges and
  prediction ranges. Range summaries exclude the conventionally zero Z_H row.
- Record retained Actor position/velocity/omega RMS, success, soft risk and hard
  warning exposure on that proposal's same 64 TRAIN states. Every next proposal
  uses new initial states; distinguish paired improvement within a proposal
  from variation between different batches when interpreting learning curves.
- Record synchronized proposal wall time and peak CUDA allocated/reserved memory.

A boundary inconsistency or nonfinite state/gradient/candidate stops and saves
failure evidence. Other probes add no gate. A missing improving candidate is
one proposal rejection, not a numerical failure. TRAIN-objective-only acceptance
does not establish safety; objective reductions can coexist with worse risk.

The final report is `reports/phase1_single_airframe/SUMMARY.md`. An optional
start/end replay on the first TRAIN pool is a descriptive comparison only:
it is not DEV, independent generalization evidence, or an update selection rule.
The archived prior DEV-gated run is under `phase1_single_airframe_dev_gated`.

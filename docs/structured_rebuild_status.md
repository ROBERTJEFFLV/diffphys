# Structured controller rebuild status (2026-09-03)

The implementation path is complete through the fail-closed staged runner, but
no structured policy has been promoted.  The registered v4 identification
probe failed its Q2 safety gate, so the pipeline correctly stops before K35
collection, identifier pretraining, A1, or any blind evaluation.

## Implemented path

- deployable fast motor observer plus a slow recurrent dynamics identifier;
- disturbance-aware equilibrium/trim estimation;
- equilibrium-centered fast wrench feedback and a second-order residual;
- differentiable bounded/rate-constrained four-motor box-QP allocation;
- recurrent Q2 behavior distillation with separate A1/A2/B/C release gates;
- candidate phase-space metric and sampled augmented fixed-point/JVP evidence;
- independent SO(3) shooting nodes and a matrix-free joint GN/LM/KKT step;
- action/parameter trust regions, nonlinear rollout restoration, and a disjoint
  held-out acceptance bank;
- smooth large-batch CVaR with at least eight effective tail samples in formal
  configurations;
- post-update recalibration, local evidence, and paired migration stages.

`strict_multiple_shooting.py` remains only as a compatibility implementation
behind `checkpointed_exact_bptt.py`.  It is not imported by the formal
full-space trainer.

## Formal v4 stop

Command:

```bash
python3 tools/diagnose_probe_v4.py \
  --checkpoint reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt \
  --output reports/probe_v4_formal.json \
  --scenarios 16 --horizon 125 --n-jobs 4 --formal-freeze
```

The command exited with status 1 and wrote an audit record.  No blind seed was
consumed.

| Candidate | Failing train seed(s) | Binding angular-rate failure |
|---:|---|---|
| 0, registered | 3707 | H125 mean: 1.235575 vs allowed 1.227841 (1.05661× Q2) |
| 1, sign inverse | 3707, 4707, 6707 | up to H125 p99: 9.063390 vs allowed 7.865975 |
| 2, time reverse | 6707 | H125 p99: 9.391561 vs allowed 7.865975 |

The freeze record binds the Q2 SHA, producer-code SHA, exact H125×16 protocol,
candidate family, train/validation split, numeric gate evidence, and an empty
blind set.  Failed formal invocations retain their report but return nonzero.

## Solver smoke

The CPU 2-segment smoke exercised independent nodes and a nonzero joint policy
step.  The free-node trial reduced merit by 28.2617.  Exact continuous-rollout
restoration had zero continuity defect and reduced merit by 0.10034, with an
action RMS change of 8.56e-5.  The update was correctly rejected because the
intentionally uncalibrated smoke policy latched identifier failures and the
held-out risk ratio was 1.000010 > 1.0.  Therefore `accepted_steps=0`; this is a
solver-path test, not evidence of controller improvement.

The same structured path also executed on CUDA.  Its exact restored trajectory
again had zero continuity defect and a positive merit reduction (2.53125), but
the uncalibrated smoke identifier latched failure and held-out risk was
1.0000044× baseline, so the update was likewise rejected.

## Verification

- full Python test suite: 337 passed;
- CUDA baseline smoke: two finite updates, no skips;
- structured full-space CPU and CUDA smokes: complete paths executed and were
  rejected by the declared held-out gate;
- 16-stage pipeline dry-run: all paths and artifacts resolve;
- real `identifier_oracle` stage: stops before collection with
  `formal causal oracle is not ready`.

The next scientific action is not to relax the 5% threshold after observing
this result.  A different probe or identification protocol must be designed
and pre-registered as a new contract before collecting new train/validation
evidence.  Until then, the correct project state is implemented but not
promoted.

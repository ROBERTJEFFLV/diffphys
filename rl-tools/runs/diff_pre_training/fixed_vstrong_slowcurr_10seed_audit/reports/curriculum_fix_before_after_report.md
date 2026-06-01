# Fixed SO(3) Curriculum Stability Fix Report

This report compares the original 500-step horizon/state curriculum against the slower 1000-step curriculum. It only evaluates fixed-dynamics SO(3) differentiable pretraining stability. It does not prove teacher-cost reduction, sampled-dynamics adaptation, Sim2Real transfer, or full L2F dynamics equivalence.

## Diagnosis

The instability is most consistent with curriculum/update-scale stress after horizon increases, especially at H=128. The failed baseline seeds had no invalid rollouts, no NaN/Inf flags, and zero skipped actor steps, so this is not a simulator/VJP validity failure. Action saturation was also near zero in the failed seeds, making saturation an unlikely primary cause.

Evidence from the original 10-seed audit:

| Seed | Symptom |
| ---: | --- |
| 1 | H=128 step 1500 loss 3094.7, terminal velocity loss 1926.1, actor grad 117913 -> 50; final loss 6296.5 with Euler/L2F success 0/0. |
| 3 | H=128 transition raised loss above 2200, but the run recovered partially; final eval stayed below both baselines. |
| 6 | H=128 transition raised loss to 1932.2 then 2670.4; late angular velocity remained high and L2F success stayed 0.0167. |
| 9 | H=128 step 1500 loss 3445.7, terminal velocity loss 2029.2, actor grad 327652 -> 50; final L2F success stayed 0.0167. |

Actor gradients were clipped on essentially every step, but clipping alone did not prevent destabilizing updates after the curriculum reached long horizons. The smallest supported intervention was to slow the existing curriculum without changing physics, loss terms, actor observation, or legacy RL paths.

## Implemented Fix

Changed the default curriculum stage lengths:

| Setting | Before | After |
| --- | ---: | ---: |
| `HORIZON_STAGE_STEPS` | 500 | 1000 |
| `STATE_CURRICULUM_STAGE_STEPS` | 500 | 1000 |

The audit script now defaults to the same 1000-step stage lengths while still exposing `HORIZON_STAGE_STEPS` and `STATE_CURRICULUM_STAGE_STEPS` as environment overrides.

## Focused Failed-Seed Validation

Run root: `runs/diff_pre_training/fixed_vstrong_slowcurr_failedseed_validation`

| Seed | Before Euler/L2F success | After Euler/L2F success | Result |
| ---: | --- | --- | --- |
| 1 | 0 / 0 | 0.523333 / 0.35 | PASS_STRONG |
| 3 | 0.05 / 0.04 | 0.52 / 0.363333 | PASS_STRONG |
| 6 | 0.036667 / 0.016667 | 0.216667 / 0.096667 | PASS_WEAK |
| 9 | 0.023333 / 0.016667 | 0.156667 / 0.10 | PASS_WEAK |

Focused validation result: all four previously failed seeds beat both old fixed first-order success baselines with the slower curriculum.

## Full 10-Seed Before/After

| Metric | 500-step curriculum | 1000-step curriculum |
| --- | ---: | ---: |
| Euler fixed mean/std success | 0.148 / 0.124474 | 0.233667 / 0.163105 |
| L2F fixed mean/std success | 0.127667 / 0.117266 | 0.150333 / 0.117042 |
| Seeds beating Euler baseline | 6 / 10 | 7 / 10 |
| Seeds beating L2F baseline | 6 / 10 | 7 / 10 |
| Seeds beating both baselines | 6 / 10 | 7 / 10 |
| Catastrophic zero-success seeds | 1 | 1 |
| Skipped actor steps | 0 | 0 |
| Any NaN/Inf | false | false |
| Mean invalid_or_nan_rate | 0 | 0 |
| Audit decision | UNSTABLE_NEEDS_FIX | ACCEPTABLE_BUT_UNSTABLE |

## After-Fix Seed Table

| Seed | Euler success | L2F success | Failure class |
| ---: | ---: | ---: | --- |
| 0 | 0.283333 | 0.233333 | PASS_WEAK |
| 1 | 0.523333 | 0.35 | PASS_STRONG |
| 2 | 0 | 0 | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 3 | 0.52 | 0.363333 | PASS_STRONG |
| 4 | 0.09 | 0.056667 | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 5 | 0.233333 | 0.12 | PASS_WEAK |
| 6 | 0.216667 | 0.096667 | PASS_WEAK |
| 7 | 0.11 | 0.08 | TYPE_B_CURRICULUM_TRIGGERED_COLLAPSE |
| 8 | 0.203333 | 0.103333 | PASS_WEAK |
| 9 | 0.156667 | 0.10 | PASS_WEAK |

## Decision

PARTIAL_FIX

The slower curriculum is supported by the failed-seed validation and improves the full 10-seed audit from 6/10 to 7/10 seeds beating both baselines. It does not fully fix stability: one catastrophic zero-success seed remains, and three seeds still fail both-baseline repeatability. The next targeted experiment should continue along the same axis, such as freezing H=64 longer, extending the state curriculum further, or adding a stability-gated curriculum advance.

# Fixed-Dynamics H32-H64-H128 Conservative Diagnostic

## Run Directory

Remote:

```text
reports/fixed_h32_h64_h128_conservative_b8192
```

## Purpose

This run tests the curriculum mechanics and H128 stability before re-enabling major dynamics randomization.

The previous broad-sampled diagnostic was stopped because the active methodology requires major dynamics randomization to remain disabled until H128 is stable.

## Fixed-Dynamics Semantics

CUDA `--fixed-dynamics` was corrected before this run.

Before:

- Fixed mode reused one broad-randomized batch across optimizer steps.

After:

- Fixed mode uses a nominal dynamics batch.
- Initial states remain randomized across the batch.
- Dynamics bins/group metadata are nominal and not broad-balanced.
- `--sample-dynamics` remains the broad/randomized path.

## Command

```bash
./build_stage96_cuda/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --diff-model euler \
  --gpu-rollout \
  --steps 9000 \
  --horizon-curriculum 32,64,128 \
  --curriculum-success-threshold 0.90 \
  --curriculum-min-steps 2000,2000,2000 \
  --curriculum-gate-check-interval 1000 \
  --curriculum-learning-rate-scale 1,0.1,0.05 \
  --curriculum-diff-loss-scale 1,0.25,0.125 \
  --curriculum-terminal-loss-scale 1,0.25,0.125 \
  --curriculum-actor-grad-clip-norm 100,0.5,0.5 \
  --reset-optimizer-on-curriculum-transition \
  --batch-size 8192 \
  --fixed-dynamics \
  --w-v 3.0 \
  --w-terminal-v 20 \
  --w-u 0.1 \
  --w-sat 1.0 \
  --action-saturation-start 0.85 \
  --success-velocity-threshold 0.5 \
  --actor-grad-clip \
  --actor-grad-clip-norm 100 \
  --save-optimizer \
  --log-path reports/fixed_h32_h64_h128_conservative_b8192/log.csv \
  --save-path reports/fixed_h32_h64_h128_conservative_b8192/checkpoint.ckpt
```

After training, the wrapper runs:

```bash
--eval-only --fixed-dynamics --eval-horizon 128 --eval-episodes 1000
```

## Horizon Gates

- H32 requires at least 2 chunks and chunk mean success > 0.90.
- H64 requires at least 2 chunks and chunk mean success > 0.90.
- H128 receives the remaining chunks.

## Transition Audit

| Transition | Optimizer | LR Scale | Physics Scale | Terminal Scale | Actor Grad Clip | Smoothness | Saturation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H32 -> H64 | reset | 0.1 | 0.25 | 0.25 | 0.5 | 0.1 | 1.0 at safe 0.85 |
| H64 -> H128 | reset | 0.05 | 0.125 | 0.125 | 0.5 | 0.1 | 1.0 at safe 0.85 |

## Pass Criteria

H128 is accepted only if:

- H128 eval success >= 0.90.
- H128 final training success >= 0.90.
- action saturation < 0.05.
- no NaN/Inf.
- no gradient explosion.
- no delayed collapse across H128 chunks.

## Stop Criteria

Stop and diagnose if:

- action saturation > 0.20.
- recent H64/H128 saturation > 0.05 after warmup.
- success collapses after a passing chunk.
- grad norm clips continuously.
- NaN/Inf appears.

## Status

Started on remote at:

```text
2026-06-04T23:06:25+08:00
```

Completed on remote at:

```text
2026-06-05T00:22:16+08:00
```

## Results

| Chunk | Horizon | Global Step | Mean Success | Final Success | Saturation | Grad Norm | Final p | Final v | Final w | Finite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 32 | 1000 | 1.000000 | 1.000000 | 0 | 0.0165833 | 0.194486 | 0.0646919 | 0.0304088 | true |
| 1 | 32 | 2000 | 1.000000 | 1.000000 | 0 | 0.00677542 | 0.190879 | 0.0629430 | 0.0247720 | true |
| 2 | 64 | 3000 | 0.859147 | 0.871338 | 0 | 0.5 | 0.194561 | 0.133768 | 0.0322073 | true |
| 3 | 64 | 4000 | 0.977267 | 0.993530 | 0 | 0.5 | 0.184751 | 0.0754798 | 0.0216656 | true |
| 4 | 128 | 5000 | 0.934258 | 1.000000 | 0 | 0.5 | 0.161914 | 0.113981 | 0.0369318 | true |
| 5 | 128 | 6000 | 1.000000 | 1.000000 | 0 | 0.5 | 0.143929 | 0.0925269 | 0.0234092 | true |
| 6 | 128 | 7000 | 1.000000 | 1.000000 | 0 | 0.5 | 0.145720 | 0.0880664 | 0.0200283 | true |
| 7 | 128 | 8000 | 1.000000 | 1.000000 | 0 | 0.5 | 0.142128 | 0.0813268 | 0.0244407 | true |
| 8 | 128 | 9000 | 1.000000 | 1.000000 | 0 | 0.5 | 0.119744 | 0.0830654 | 0.0443460 | true |

## H128 Evaluation

1000-episode fixed-dynamics H128 evaluation from the final checkpoint:

| Metric | Value |
| --- | ---: |
| success rate | 1.000000 |
| near success p | 1.000000 |
| near success p+v | 1.000000 |
| mean final p | 0.120634 |
| mean final v | 0.083895 |
| mean final w | 0.045008 |
| p90 final p | 0.187166 |
| p90 final v | 0.132099 |
| p90 final w | 0.0809048 |
| action saturation | 0 |
| invalid/NaN rate | 0 |

## Pass/Fail

| Gate | Result | Evidence |
| --- | --- | --- |
| H32 -> H64 transition | PASS | H32 chunk 1 mean success 1.0; transitioned to H64. |
| H64 -> H128 transition | PASS | H64 chunk 3 mean success 0.977267; transitioned to H128. |
| H128 training success >= 0.90 | PASS | H128 chunks 4-8 mean success 0.934258, then 1.0 for all remaining chunks. |
| H128 eval success >= 0.90 | PASS | Eval success 1.0 over 1000 episodes. |
| action saturation < 0.05 | PASS | Training and eval saturation both 0. |
| no NaN/Inf | PASS | Training finite=true for all chunks; eval invalid/NaN rate 0. |
| no delayed collapse | PASS | H128 chunks improved after chunk 4 and stayed at success 1.0. |

## Interpretation

This fixed-dynamics diagnostic supports the curriculum-scale diagnosis:

- The earlier broad-sampled H64 collapse was not because H64/H128 evaluation is intrinsically unflyable.
- With optimizer reset on horizon transition, lower actor LR, lower physics/terminal loss scale, stronger action smoothness/saturation penalties, and per-chunk checkpoint preservation, H32 -> H64 -> H128 reaches stable fixed-dynamics H128 control.
- The H64 and H128 actor gradient norms still hit the configured 0.5 clip frequently, so the clip is actively shaping updates. This is acceptable for this safety diagnostic, but should be tracked in later sampled-dynamics runs.
- This result does not prove broad sampled dynamics stability. Major dynamics randomization should be reintroduced gradually from the stable H128 checkpoint or from the stable H64 checkpoint.

## Checkpoints

Stable checkpoints preserved in:

```text
reports/fixed_h32_h64_h128_conservative_b8192/checkpoint.ckpt.chunk_3_h64.ckpt
reports/fixed_h32_h64_h128_conservative_b8192/checkpoint.ckpt.chunk_8_h128.ckpt
reports/fixed_h32_h64_h128_conservative_b8192/checkpoint.ckpt
```

## Next Step

Use this fixed H128 result as the stable curriculum baseline. The next sampled-dynamics run should not jump directly back to the previous broad setup; it should re-enable dynamics variation in controlled stages and keep the same transition-reset, low H64/H128 LR, low long-horizon loss scale, saturation guard, and per-chunk checkpoints.

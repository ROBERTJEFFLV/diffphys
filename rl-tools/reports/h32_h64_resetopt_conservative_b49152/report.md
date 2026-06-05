# H32-H64 Conservative Transition Report

## Run Directory

Remote:

```text
reports/h32_h64_resetopt_conservative_b49152
```

## Motivation

The previous `gpu_curriculum_h32_h64_h128_b49152` run was stopped after H64 degradation:

| Chunk | Horizon | Mean Success | Final Success | Action Saturation | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 32 | 0.533218 | 0.535563 | 0 | stable but below gate |
| 1 | 32 | 0.601767 | 0.586975 | 0 | stable but below gate |
| 2 | 32 | 0.626763 | 0.716288 | 0.002393 | stable but below gate |
| 3 | 32 | 0.927402 | 1.000000 | 0 | promoted to H64 |
| 4 | 64 | 0.744608 | 0.238220 | 0.722989 | collapsed late in chunk |
| 5 | 64 | partial 0.258706 | 0.167725 | 0.701396 | run stopped |

The failure mode was not NaN/Inf or gradient explosion. The visible failure was delayed action saturation after the H32 to H64 transition.

## Loss Audit

CUDA rollout loss audit:

- Per-step position, velocity, attitude, angular-velocity, action magnitude, action smoothness, and saturation losses already use `1 / horizon` normalization.
- Terminal loss was not horizon-scaled. A `terminal_loss_scale` control was added.
- Saturation penalty was weak by default: weight `0.05`, start `0.95`. Runtime controls were added for saturation weight and start threshold.
- The previous curriculum overwrote the latest checkpoint each chunk, so the stable H32 chunk checkpoint was lost after H64 training. Per-chunk checkpoint preservation was added.
- The previous curriculum carried Adam optimizer state across horizon transitions. An option was added to reset optimizer state on horizon transition.

## Implemented Controls

Added runtime controls:

```text
--curriculum-learning-rate-scale
--curriculum-diff-loss-scale
--curriculum-terminal-loss-scale
--curriculum-actor-grad-clip-norm
--reset-optimizer-on-curriculum-transition
--w-u
--w-sat
--action-saturation-start
--terminal-loss-scale
--no-load-optimizer
```

Checkpoint preservation:

```text
checkpoint.ckpt
checkpoint.ckpt.chunk_<n>_h<H>.ckpt
```

## Supporting Ablations

Base checkpoint:

```text
/tmp/raptor_stage11_20260604_161141/stage11_final.ckpt_h32.ckpt
```

H64 eval with no training:

| Eval | Success | Saturation | Mean Final p/v/w |
| --- | ---: | ---: | ---: |
| H64 no train | 0.781 | 0 | 0.193749 / 0.358543 / 0.132136 |

One-update H64 ablations:

| Case | LR | Physics Scale | Grad Clip | Eval Success | Saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| B | 1e-4 | 1.0 | 100 | 0.953 | 0 |
| C | 3e-5 | 1.0 | 100 | 0.851 | 0 |
| D | 1e-5 | 0.5 | 100 | 0.803 | 0 |
| E | 1e-5 | 0.25 | 0.5 | 0.794 | 0 |

Two-hundred-update H64 ablations:

| Case | LR | Physics Scale | Grad Clip | Train Mean Success | Eval Success | Saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B200 | 1e-4 | 1.0 | 100 | 0.992455 | 1.000 | 0 |
| E200 | 1e-5 | 0.25 | 0.5 | 0.989960 | 1.000 | 0 |

Conclusion from these short ablations:

- H64 rollout itself is viable.
- H64 does not collapse after one update.
- H64 does not collapse after 200 updates from the old H32 checkpoint.
- The previous collapse is most likely tied to curriculum transition state, checkpoint contamination, or delayed dynamics during the long chunk, not an immediate H64 incompatibility.

## Current Diagnostic Run

Command:

```bash
./build_stage96_cuda/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --diff-model euler \
  --gpu-rollout \
  --steps 8000 \
  --horizon-curriculum 32,64 \
  --curriculum-success-threshold 0.80 \
  --curriculum-gate-check-interval 1000 \
  --curriculum-learning-rate-scale 1,0.1 \
  --curriculum-diff-loss-scale 1,0.25 \
  --curriculum-terminal-loss-scale 1,0.25 \
  --curriculum-actor-grad-clip-norm 100,0.5 \
  --reset-optimizer-on-curriculum-transition \
  --batch-size 49152 \
  --sample-dynamics \
  --sampled-dynamics-level broad \
  --balanced-dynamics-sampling \
  --w-v 3.0 \
  --w-terminal-v 20 \
  --w-u 0.1 \
  --w-sat 1.0 \
  --action-saturation-start 0.85 \
  --success-velocity-threshold 0.5 \
  --actor-grad-clip \
  --actor-grad-clip-norm 100 \
  --save-optimizer \
  --log-path reports/h32_h64_resetopt_conservative_b49152/log.csv \
  --save-path reports/h32_h64_resetopt_conservative_b49152/checkpoint.ckpt
```

## Decision Gates

Stop and diagnose if:

- NaN/Inf appears.
- Any H64 chunk has action saturation > 0.20.
- H64 recent-100 action saturation > 0.05 after the first 100 H64 steps.
- H64 mean success drops below 0.50 after a previously stable H64 chunk.
- actor gradient norm clips continuously at the configured limit.

Accept H64 as documented-stable if:

- At least two completed H64 chunks.
- Both H64 chunks have mean success >= 0.90.
- Both H64 chunks have final success >= 0.90.
- Both H64 chunks have action saturation < 0.05.
- No NaN/Inf.
- No delayed collapse in the later H64 chunk.

Only after H64 passes this gate should H128 be entered.

## Next Ablation

If H64 passes:

- Run controlled H64 -> H128 transition.
- Reset optimizer on transition.
- Use LR scale <= 0.1 for H128.
- Use physics scale <= 0.25 for H128.
- Use terminal scale <= 0.25 for H128.
- Keep actor grad clip <= 0.5.
- Keep `w_u=0.1`, `w_sat=1.0`, `action_saturation_start=0.85`.

If H64 fails:

- Resume from the last stable H32/H64 chunk checkpoint.
- Compare optimizer carryover on/off.
- Compare saturation penalty start 0.85 vs 0.90.
- Compare terminal loss scale 0.25 vs 0.1.

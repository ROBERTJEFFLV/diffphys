# H64 vs H128 Long Continuous Control Compare

## Purpose

Compare the stable H64 checkpoint and final H128 checkpoint under the same fixed-dynamics evaluation protocol at horizons 64, 128, and 256.

## Checkpoints

```text
H64 policy:  reports/fixed_h32_h64_h128_conservative_b8192/checkpoint.ckpt.chunk_3_h64.ckpt
H128 policy: reports/fixed_h32_h64_h128_conservative_b8192/checkpoint.ckpt.chunk_8_h128.ckpt
```

## Evaluation Command Pattern

```bash
./build_stage96_cuda/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --eval-only \
  --gpu-rollout \
  --fixed-dynamics \
  --load-path <checkpoint> \
  --eval-horizon <64|128|256> \
  --eval-episodes 1000 \
  --success-velocity-threshold 0.5 \
  --log-path reports/h64_h128_long_control_compare/eval_<policy>_h<horizon>.csv
```

## Results

| Policy | Eval horizon | Success | Mean loss | Mean final p | Mean final v | Mean final w | P90 p | P90 v | P90 w | Saturation | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H64 | 64 | 0.989 | 2.39565 | 0.186161 | 0.0770698 | 0.021911 | 0.262969 | 0.118654 | 0.0330678 | 0 | 0 |
| H64 | 128 | 0.939 | 3.37768 | 0.175257 | 0.175514 | 0.0636208 | 0.227232 | 0.275583 | 0.0864631 | 0 | 0 |
| H64 | 256 | 0.939 | 11.2021 | 0.336054 | 0.154444 | 0.105225 | 0.363933 | 0.221934 | 0.135299 | 0 | 0 |
| H128 | 64 | 1.000 | 2.22508 | 0.176758 | 0.100236 | 0.0343773 | 0.243575 | 0.148171 | 0.0568562 | 0 | 0 |
| H128 | 128 | 1.000 | 1.44681 | 0.120634 | 0.083895 | 0.045008 | 0.187166 | 0.132099 | 0.0809048 | 0 | 0 |
| H128 | 256 | 1.000 | 3.07302 | 0.0932814 | 0.103313 | 0.246404 | 0.155472 | 0.173037 | 0.483671 | 0 | 0 |

## Interpretation

- H64 is mostly stable at its own horizon, but its success drops from 0.989 at H64 to 0.939 at H128/H256.
- H64 shows long-horizon drift: at H256 the mean final position norm rises to 0.336054 and mean loss rises to 11.2021.
- H128 keeps success at 1.0 for all tested horizons and has much lower long-horizon position error. At H256, mean final position norm is 0.0932814 versus H64's 0.336054.
- H128 also improves H128/H256 velocity control relative to H64, but it carries more residual angular velocity at H256: mean final w 0.246404 versus H64's 0.105225. This is still far below the angular velocity success threshold and occurs without action saturation.
- Neither checkpoint uses saturated actions in these fixed-dynamics evaluations.

## Conclusion

H64 learns a controller that is viable beyond 64 steps, but it is not as robust under longer continuous closed-loop rollout. H128 training mainly improves long-horizon position/velocity regulation and prevents the H64 policy's longer-horizon drift. The tradeoff observed in this fixed test is a larger residual angular-rate norm at H256 for the H128 policy, but it does not cause failure or saturation.

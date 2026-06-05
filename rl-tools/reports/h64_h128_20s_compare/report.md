# H64 vs H128 20s+ Continuous Evaluation

## Purpose

Check what happens when the fixed-dynamics controller is evaluated far beyond the H128 training horizon.

At 100 Hz:

| Horizon | Duration |
| ---: | ---: |
| 256 | 2.56 s |
| 512 | 5.12 s |
| 1024 | 10.24 s |
| 2000 | 20.00 s |
| 3000 | 30.00 s |

The CUDA eval-only horizon limit was raised from the CPU validation limit to `GPU_EVAL_MAX_HORIZON = 4096`. Training and CPU parity/validation limits remain unchanged.

## Checkpoints

```text
H64 policy:  reports/fixed_h32_h64_h128_conservative_b8192/checkpoint.ckpt.chunk_3_h64.ckpt
H128 policy: reports/fixed_h32_h64_h128_conservative_b8192/checkpoint.ckpt.chunk_8_h128.ckpt
```

## Important Limitation

The current eval success metric is a final-time classification. It does not prove that the trajectory stayed inside bounds for the full 20 seconds. Mean trajectory loss and action saturation help, but a true continuous-flight gate should add throughout-window metrics such as max position/velocity/angular-rate over the full rollout.

## H128 Failure Onset

H128 is stable at 2.56 s, starts degrading around 3.84 s, and fails hard after 5.12 s.

| Horizon | Duration | Episodes | Success | Mean final p | Mean final v | Mean final w | P90 p | P90 v | P90 w | Saturation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 2.56 s | 256 | 1.000000 | 0.096984 | 0.100770 | 0.244199 | 0.159852 | 0.167827 | 0.461653 | 0 |
| 320 | 3.20 s | 256 | 0.996094 | 0.122722 | 0.156748 | 0.475960 | 0.180356 | 0.277570 | 0.847334 | 0 |
| 384 | 3.84 s | 256 | 0.792969 | 0.153803 | 0.327528 | 0.851753 | 0.246914 | 0.620510 | 1.52055 | 0 |
| 448 | 4.48 s | 256 | 0.437500 | 0.216129 | 0.594438 | 1.43741 | 0.383753 | 1.08264 | 2.56827 | 0 |
| 512 | 5.12 s | 256 | 0.277344 | 0.372522 | 1.06301 | 2.29010 | 0.756676 | 2.04159 | 4.15159 | 0 |
| 1024 | 10.24 s | 256 | 0 | 86.1104 | 51.7774 | 6.66885 | 158.177 | 92.9321 | 10.3039 | 0.211876 |
| 1500 | 15.00 s | 256 | 0 | 541.094 | 141.198 | 7.21551 | 914.151 | 225.456 | 10.5009 | 0.458132 |
| 2000 | 20.00 s | 256 | 0 | 1474.7 | 231.944 | 8.03324 | 2369.62 | 355.265 | 11.5527 | 0.593599 |
| 3000 | 30.00 s | 64 | 0 | 4100.52 | 351.63 | 10.1224 | 6628.26 | 546.175 | 15.8888 | 0.730448 |

## H64 vs H128 At Long Horizons

| Policy | Horizon | Duration | Success | Mean final p | Mean final v | Mean final w | P90 p | P90 v | P90 w | Saturation | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H64 | 512 | 5.12 s | 0.984375 | 0.326417 | 0.147230 | 0.0969328 | 0.402720 | 0.211395 | 0.148260 | 0 | 0 |
| H64 | 1024 | 10.24 s | 0.980469 | 0.350823 | 0.0677511 | 0.0597027 | 0.392366 | 0.0962094 | 0.0732544 | 0 | 0 |
| H64 | 2000 | 20.00 s | 0.984375 | 0.454070 | 0.0156206 | 0.0268415 | 0.473680 | 0.0225822 | 0.0358478 | 0 | 0 |
| H128 | 512 | 5.12 s | 0.277344 | 0.372522 | 1.06301 | 2.29010 | 0.756676 | 2.04159 | 4.15159 | 0 | 0 |
| H128 | 1024 | 10.24 s | 0 | 86.1104 | 51.7774 | 6.66885 | 158.177 | 92.9321 | 10.3039 | 0.211876 | 0 |
| H128 | 2000 | 20.00 s | 0 | 1474.7 | 231.944 | 8.03324 | 2369.62 | 355.265 | 11.5527 | 0.593599 | 0 |

## Interpretation

- H128 is better at the trained regime and near-term extension: H128/H256 were strong in earlier reports.
- H128 is not a 20-second controller. It begins losing velocity/angular-rate control between 3.2 s and 3.84 s. By 10 s it has diverged and begins using saturated actions.
- H64 is less accurate at short H128/H256 endpoints, but it behaves like a more conservative hover/settling controller under long fixed-dynamics rollout. Its final-state success stays near 0.98 at 20 s with no saturation.
- H64's good final-time H2000 result is not yet a full continuous-flight guarantee because the eval does not report max-over-time violations.

## Conclusion

For 20 seconds or longer, the current final H128 checkpoint is not acceptable. It is stable for roughly 2.5-3.2 seconds, degrades around 3.8-5.1 seconds, and diverges by 10-20 seconds.

The older H64 checkpoint is currently the better long-duration fixed-dynamics candidate, but it still needs a stricter throughout-trajectory safety evaluation before claiming continuous 20-second flight.

## Next Action

Add long-horizon throughout-window eval metrics:

- max position norm over all timesteps
- max velocity norm over all timesteps
- max angular velocity norm over all timesteps
- fraction of timesteps inside success bounds
- first failure time
- action saturation over time

Then train or fine-tune with a long-horizon stability objective, using the H64 checkpoint as a possible baseline and the H128 checkpoint as a short-horizon tracking baseline.

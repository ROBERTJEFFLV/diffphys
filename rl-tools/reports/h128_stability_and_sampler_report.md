# H128 Stability And Size-Mass Sampler Report

## Stable H128 Configuration

Run directory:

```text
reports/fixed_h32_h64_h128_conservative_b8192
```

Stable training command characteristics:

- CUDA GPU rollout.
- GRU actor with exactly 4 motor outputs.
- Fixed nominal dynamics during the stability proof.
- H32 -> H64 -> H128 curriculum.
- Horizon transitions require multiple stable chunks.
- H32 -> H64 transition: optimizer reset, LR x0.1, physics loss x0.25, terminal loss x0.25, actor grad clip 0.5.
- H64 -> H128 transition: optimizer reset, LR x0.05, physics loss x0.125, terminal loss x0.125, actor grad clip 0.5.
- Smoothness penalty `--w-u 0.1`.
- Saturation penalty `--w-sat 1.0 --action-saturation-start 0.85`.
- Velocity objective `--w-v 3.0 --w-terminal-v 20`.
- Success velocity threshold `0.5`.

## H128 Result

Final fixed-dynamics training and evaluation passed the active stability gates.

| Gate | Result | Evidence |
| --- | --- | --- |
| H32 -> H64 transition | PASS | H32 chunk 1 mean success 1.0; transitioned after the minimum two chunks. |
| H64 -> H128 transition | PASS | H64 chunk 3 mean success 0.977267; transitioned after the minimum two chunks. |
| H128 final training success >= 0.90 | PASS | H128 chunks 4-8 mean success: 0.934258, 1.0, 1.0, 1.0, 1.0. |
| H128 eval success >= 0.90 | PASS | 1000-episode H128 eval success 1.0. |
| action saturation < 0.05 | PASS | Training and eval saturation both 0. |
| no delayed collapse | PASS | H128 success stayed at 1.0 after the first H128 chunk. |
| no NaN/Inf | PASS | Training finite=true for all chunks; eval invalid/NaN rate 0. |
| no gradient explosion | PASS | Actor grad was clipped at 0.5 in H64/H128; no skip/NaN/explosion was observed. |

H128 eval metrics:

| Metric | Value |
| --- | ---: |
| success rate | 1.000000 |
| mean final position norm | 0.120634 |
| mean final velocity norm | 0.083895 |
| mean final angular velocity norm | 0.045008 |
| p90 final position norm | 0.187166 |
| p90 final velocity norm | 0.132099 |
| p90 final angular velocity norm | 0.0809048 |
| action saturation rate | 0 |
| invalid/NaN rate | 0 |

Preserved checkpoints:

```text
reports/fixed_h32_h64_h128_conservative_b8192/checkpoint.ckpt
reports/fixed_h32_h64_h128_conservative_b8192/checkpoint.ckpt.chunk_3_h64.ckpt
reports/fixed_h32_h64_h128_conservative_b8192/checkpoint.ckpt.chunk_8_h128.ckpt
```

## Observed Failure Mode

The earlier broad-sampled run collapsed immediately after entering H64:

- H32 chunk 3: mean success 0.927, final success 1.0, saturation 0.
- H64 chunk 4: mean success 0.745, final success 0.238, saturation 0.723.
- H64 chunk 5: mean success 0.259, final success 0.168, saturation 0.701.

The fixed-dynamics conservative run did not reproduce this collapse. That supports the diagnosis that the failure was caused by horizon-transition update scale and broad sampled-domain pressure, not by H64/H128 rollout being intrinsically unflyable.

## Supporting Ablations

- H64 eval-only from an H32 checkpoint stayed flyable.
- Short H64 update-scale tests did not show immediate saturation when the update was controlled.
- The fixed H32 -> H64 -> H128 run with transition optimizer reset and reduced H64/H128 update scale reached H128 success 1.0 with saturation 0.
- H64/H128 actor gradients frequently hit the configured 0.5 clip, so this clip is an active stabilizer and should remain logged in later sampled-dynamics runs.

## Size-Mass Correlated Sampler

Implemented as an optional CUDA sampler path. It is disabled by default.

Relevant files:

```text
src/foundation_policy/diff_pre_training/gpu_rollout.h
src/foundation_policy/diff_pre_training/gpu_rollout.cu
src/foundation_policy/diff_pre_training/cuda_main.cu
```

Runtime flag:

```text
--correlated-size-mass-sampling
```

Default behavior:

- `FullGpuTrainingOptions::correlated_size_mass_sampling = false`.
- `GpuPolicyEvalOptions::correlated_size_mass_sampling = false`.
- Calling `generate_validation_batch(..., correlated_size_mass_sampling=false)` is unchanged.
- Fixed nominal dynamics remain nominal even if the correlated flag is passed internally.

Implemented relationships:

```text
mass_new = clamp(size_new^3, mass_min, mass_max)
scale_relative = cbrt(mass_new / nominal_mass)
rotor_distance_factor = scale_relative * size_factor
factor_thrust_coefficients = factor_thrust_to_weight * factor_mass
J /= inertia_factor
J_inv *= inertia_factor
```

## Sampler Validation

Build:

```bash
cmake --build /tmp/raptor_stage96_cuda_build --target foundation_policy_diff_pre_training_cuda -j2
```

Validation:

```bash
./build_stage96_cuda/src/foundation_policy/foundation_policy_diff_pre_training_cuda \
  --gpu-rollout \
  --gpu-validate-correlated-size-mass-sampler \
  --batch-size 1024 \
  --horizon 16 \
  --seed 123
```

Result:

| Check | Result |
| --- | --- |
| validation passed | true |
| default disabled | true |
| fixed nominal unchanged | true |
| correlated formula close | true |
| finite | true |
| formula mismatch count | 0 |
| NaN/Inf count | 0 |
| sampled mass range | 0.0200512 to 0.249179 |
| sampled size-factor range | 0.833433 to 1.19963 |
| max default batch abs error | 0 |
| max nominal abs error | 0 |
| max thrust factor abs error | 1.43051e-06 |
| max J inverse abs error | 5.96046e-08 |

Smoke tests:

| Run | Result | Final Success | Saturation | NaN/Inf |
| --- | --- | ---: | ---: | ---: |
| default fixed 2-step smoke | PASS | 1.0 | 0 | 0 |
| correlated sampled 2-step smoke | PASS | 1.0 | 0 | 0 |

Smoke logs:

```text
reports/size_mass_correlated_sampler_validation/fixed_smoke.csv
reports/size_mass_correlated_sampler_validation/correlated_smoke.csv
```

## Conclusion

The stable H128 fixed-dynamics curriculum now meets the requested stability criteria. The size-mass correlated dynamics sampler is implemented, unit-tested through a dedicated validation gate, verified not to alter default/fixed behavior, and remains disabled by default.

Major dynamics randomization should be reintroduced only through small staged robustness ablations from the stable H128 checkpoint, not by returning directly to the previous broad sampled setup.

# CUDA efficiency audit, 2026-10-05

The Actor remains native B: GRUCell(16,64), hidden-only Linear(64,4), tanh.
These changes accelerate the existing H500 computation; they do not change
the controller, loss, physical sampler, group clipping or Adam.

## Applied changes

- Cache the read-only gravity and rotor-spin vectors by device/dtype. Values
  and RK4 arithmetic order are unchanged.
- Omit unused observation storage during training. The noisy Actor observation
  is still calculated at its original Time Decay location. EVAL and replay
  record complete observations by default.
- Convert local probe contributions directly into the existing FP64 accumulators
  with `add_`, avoiding separate FP64 temporaries. Local float32 matmuls/sums
  remain float32; this does not cure overflow.
- Compile only the original quaternion-to-rotation expression inside physics
  RK4. Actor/GRU, sensing, other physics, compaction and probe remain eager.

The main config explicitly uses `--rotation-backend compile`. Bare CLI and
simulator defaults remain `eager`. There is no automatic compiler fallback.
The backend/version/options are checkpoint-bound and reproduced in audit replay.
Precision-cast and division-rounding emulation are enabled, CUDA Graphs disabled.
This was verified with PyTorch 2.10.0+cu126, not arbitrary compiler releases.

## Measured on RTX 4060 Ti, 8 GiB

Float32, mature B weights, 2048 TRAIN scenes, 128 groups, H500, Time Decay=1.
The frozen batch executes 936,700 valid transitions. First samples are excluded
from warm medians; these are not cheap random-Actor early-failure timings.

| Comparison | Before | After | Scope |
| --- | ---: | ---: | --- |
| Actual update | 5.35 s | 4.29 s | Sampling through Adam/audit, 3 paired updates |
| Eager cleanup | 5.13 s | 4.75 s | Frozen forward/backward, 5 paired samples |
| Rotation compile vs optimized eager | 4.64 s | 4.03 s | Frozen forward/backward, 5 paired samples |
| NVML busy-time mean | 33.9% | 28.2% | Rotation comparison; not SM occupancy |

Actual update latency fell about 20%, approximately 25% more updates per unit
time. NVML busy-time did **not** rise. It measures sampled time with at least
one executing kernel, not the fraction of compute capacity used. Fusion can
complete work faster with less GPU active time. See [NVIDIA's definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
and [PyTorch's profiling guidance](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html).

The 3 paired updates produced bitwise-identical Actor, Adam and RNG states.
A separate 256-scene H500 EVAL had identical complete trajectories and costs.
Full and Adam-only update replay passed. The full regression run passed 414
tests, including actual CPU/CUDA checks. This is bounded numerical/performance
evidence, not proof of better control or prevention of gradient explosions.

Whole-RK4 compilation failed H500 cost/gradient tolerances and is not integrated.
Probe-gate compilation gave no additional benefit over rotation fusion and is
not integrated. A 3072-scene eager batch improved valid physical transitions/s
by about 44%, with 5.97 GiB allocation, but changes batch statistics; the default
remains 2048. 4096 scenes were not tested.

## Reproduce without updating weights

```bash
python tools/profile_response_training.py --device cuda --scenes 2048 \
  --horizon 500 --rotation-backend eager --warmup 1 --repeats 5 \
  --checkpoint /path/to/native_B.pt \
  --epsilon-p .01 --epsilon-a .01 --lambda-R .2 --report /tmp/eager.json

python tools/profile_response_training.py --device cuda --scenes 2048 \
  --horizon 500 --rotation-backend compile --warmup 1 --repeats 5 \
  --checkpoint /path/to/native_B.pt \
  --epsilon-p .01 --epsilon-a .01 --lambda-R .2 --report /tmp/compile.json

python -m pytest -q tests
```

Profiler `--scenes` means total scenes, while training `--scenarios` means per
bank. The profiler never calls Adam or writes a model; sampling and warm-up
are separate from compute timings. Cold compiler caches increase initial cost.
CPU compilation additionally needs a C++ toolchain and `setuptools`.

Exact resume still requires identical source/config. Do not rewrite old run
hashes: preserve original source/checkpoints and use a new directory with
explicit weights-only `--init-checkpoint` when initializing from old weights.
That creates fresh Adam/sampling, not exact continuation. Eager and compile
have different backend bindings. Local raw results and the performance-only
patch are under `reports/gpu_efficiency_20261005/`, ignored by Git.

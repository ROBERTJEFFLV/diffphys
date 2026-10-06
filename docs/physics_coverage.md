# Fixed physical coverage for TRAIN

Adapted-loss branch: this sampler/group layout is unchanged. The task now uses
equal scene weights; historical CVaR statements below describe the prior objective.

Scope: change which valid initial scenes are sampled, not the controller, task
loss, physics, disturbances, Time Decay, CVaR, or optimizer.
The group-gradient probe reuses these cells; see [group_gradient_probe.md](group_gradient_probe.md).
The existing random sampler remains available; fixed EVAL is always unchanged.

## Cells and pool size

The production argument file selects `--train-sampling coverage128` and
`--scenarios 512`. The latter retains its original four-bank size convention:
4 * 512 = 2048 selected scenes in ONE pool, not four separate quota pools.

- TTI: [40,330), [330,620), [620,910), [910,1200].
- Motor rise time (s): [.03,.0475), [.0475,.065), [.065,.0825), [.0825,.10].
- Motor fall time (s): [.03,.0975), [.0975,.165), [.165,.2325), [.2325,.30].
- Yaw: below / at-or-above a fixed conditional median for that TTI interval.

Thus 4 * 4 * 4 * 2 = 128 cells, with 16 scenes per cell at 2048 TRAIN scenes.
Cell ID is `(((tti_bin*4 + rise_bin)*4 + fall_bin)*2 + yaw_bin)`.
Interior equality belongs to the higher bin. FP32 edges use FP32 representation.
Any positive multiple of 128 is supported as the total TRAIN size; incomplete
quotas are rejected, not silently rounded. Bare CLI calls still default to the
original random sampler, including small CPU tests and existing custom scripts.

Originally the 128 cells only controlled sampling. The later clipping change now
also uses these SAME fixed IDs for 128 gradient groups. At N=2048 each contains
16 initial scenes. The frozen yaw medians and original sampler are unchanged;
`response_groups.py` no longer re-splits coverage pools into 16 adaptive groups.
Legacy random-pool API calls can still request the adaptive layout, with the new
shrink-only fixed cap. No batch median normalization remains.

## What yaw means

We use the single-rotor physical authority scale

`K_yaw = km * T_max_single_rotor / Jz`

where maximum thrust comes from the original thrust polynomial at motor_max.
All four rotors share these parameters in this reference family. This is not
a measured closed-loop gain or stability certificate. Under the current sampler,
an equivalent expression is `TTI * (km/arm_length) * (Jx/Jz)`.
**Do not divide the first expression by arm length again.** The earlier verbal
proposal `km/(arm_length*Jz)` omitted thrust and is not the implemented measure.

A single global yaw threshold would strongly change the low/high proportions
inside each TTI interval. Instead, `configs/physics_coverage.json` freezes four
conditional medians, calibrated once from 262144 independent original clean
reset draws (no Actor, training trajectories, fixed EVAL, or failure labels).
The medians are approximately 21.4040, 60.3551, 97.8052 and 135.9149 rad/s^2.
These bins describe relatively low/high yaw authority WITHIN each TTI interval;
they are not identical absolute yaw ranges across all TTI intervals.

The calibration is empirical, not an exact analytical quantile. Equal quotas
therefore approximate the original joint distribution rather than preserving
it mathematically exactly. On a separate 131072-scene prior check, low-yaw
fractions were 0.5040, 0.4979, 0.5033 and 0.4971 in the four TTI intervals.
Mass, size, thrust-to-weight and other physical quantities follow the original
conditional distribution in each cell. No extra independent parameter draw,
clipping, or recoupling is introduced. Rare finer corners are not guaranteed.

Reproduce the definition (prints JSON; never overwrites the live file):

```bash
python tools/calibrate_physics_coverage.py > /tmp/physics_coverage_reproduced.json
```

The released JSON, sampler implementation, candidate seed scheme, and physical
generator fingerprint enter the new checkpoint source/configuration binding.
Changing the generator requires an explicit review and recalibration; thresholds
are never silently refitted on a training batch or an evaluation set.

## Selection, randomness, and memory

`response_sampling.sample_coverage` draws candidates with the unchanged
`RaptorSimulator.reset(..., disturbances=clean(), horizon=1)` on CPU. It retains
the first required rows in each cell. Whole rows, including random initial
kinematics and hidden motors, are selected WITHOUT modifying their values or
filtering by flight outcome. A separate seeded permutation shuffles the pool.
There are no duplicate-row fills, jittered copies, or parameter averages.

Candidate batches contain `max(1024, 2*N)` scenes. At most 16 batches are tried;
if a cell cannot be filled, training fails explicitly before an update instead
of falling back to biased or duplicate samples. The per-update report records
cell counts, quota, candidate count, rounds, seeds, and definition digest.

Candidate and shuffle seeds are derived deterministically from the existing
TRAIN update seeds, in a separate high-valued namespace; no global RNG is used.
After selection, `noise_row` is reset to 0..N-1, and the original noise/pulse
sampler attaches ONE full-horizon tape. Extra candidates never carry H500 tapes
or GPU rollout graphs. The same source/count/update/dtype is repeatable; changing
noise, pulse switches or horizon does not change the selected physical rows.

## Compatibility and verification

`sample_training_scenarios` retains its original two-value return, with optional
sampling mode/report arguments. `sample_pool`, fixed EVAL seeds/distribution,
Actor observations, noise implementation, and physical generator are unchanged.
The original `--train-sampling random` path reproduces old pool values exactly.
Old compatible checkpoints can still be evaluated using stored loss settings.

Exact resume remains strict. Switching source or sampling is an explicit new
experiment, not an exact continuation. Do not rewrite checkpoint bindings. Keep
the original worktree/run and use an independent checkout and empty work directory
with the existing `--init-checkpoint PATH` option. This imports Actor weights but
starts fresh Adam, update numbering, and TRAIN sampling, as before. No old Adam
state is silently converted, and no existing local service is restarted.

```bash
python tools/train_response_control.py @configs/response_raptor_multi_airframe.args \
    --init-checkpoint PATH_TO_CHECKPOINT --work-dir runs/coverage128/seed7
python tools/verify_physics_coverage.py --batches 32
python -m pytest -q tests/test_physics_coverage.py tests/test_core_contract.py
```

The independent 32-batch N=2048 CPU audit found 5..32 rows per cell for random
sampling and exactly 16 for coverage sampling. Neither mode had an empty coarse
cell in that audit. This demonstrates quota balancing, NOT unseen-airframe
learning or reduced oscillation. The complete regression tests additionally
check original-row equality, causal tape indexing, fixed EVAL, invalid sizes,
exhaustion, source binding, and same-source Actor/Adam resume. CPU checks do not
establish CUDA throughput or the user's 11507/171-family control performance.

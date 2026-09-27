# DiffPhys: multi-airframe recurrent motor control

One trainable Actor, one RAPTOR-style multi-airframe simulator, one train/evaluate
entry. Training uses full-horizon BPTT, backward-only Time Decay, physical-group
gradient normalization and persistent Adam. There is no critic, teacher,
auxiliary network or single-airframe mode.

## Run

Python 3.11+, PyTorch 2.10, NumPy and pytest. Install the PyTorch build matching
your CUDA runtime; no native extension build is required.

```bash
python tools/train_response_control.py @configs/response_raptor_multi_airframe.args
python tools/train_response_control.py --mode evaluate \
    --checkpoint runs/raptor_gaussian/seed7/best.pt \
    --work-dir runs/raptor_gaussian/evaluation
python -m pytest -q tests
```

The checked-in config uses one GPU, 4 x 128 = **512 TRAIN** scenes per update,
2 x 128 = **256 fixed EVAL** scenes, H500 at 100 Hz, memory dimension 64,
Time Decay 1 s^-1, Adam 3e-4, gradient scale 0.1 and global clip 10.
It stops at 50 updates or 1800 seconds. `--scenarios` and `--eval-scenarios` are
**per bank**, not totals. No training runs on import.

The 2048-scene configuration uses the same source and protocol:

```bash
python tools/train_response_control.py @configs/response_raptor_multi_airframe.args \
    --scenarios 512 --updates 1000000000 --max-seconds 1000000000 \
    --work-dir runs/raptor_gaussian_b2048/seed7
```

The large limits describe a manually stopped run, not a recommendation to run
without monitoring. Numerical failures still use transactional rollback.
This command starts fresh with seed 7 and fresh Adam in an empty directory;
checkpoints, training logs and local visualization exports are not bundled here.

## One disturbance protocol for TRAIN and EVAL

| Component | Distribution | Lifetime |
|---|---|---|
| External world force | Pinned RAPTOR Gaussian law below | One vector per episode |
| Position measurement | Independent Gaussian, sigma = 0.001 m per axis | Each acquired sample |
| World-velocity measurement | Independent Gaussian, sigma = 0.002 m/s per axis | Each acquired sample |
| Attitude measurement | Gaussian rotation vector, sigma = 0.001 rad per axis | Each acquired sample |
| Body angular velocity | Independent Gaussian, sigma = 0.002 rad/s per axis | Each acquired sample |
| Velocity measurement latency | Uniform [0.010, 0.030] seconds | Fixed per episode |

For sampled mass m and thrust-to-weight ratio TWR:

```
r ~ Uniform(0, 0.3 * max(TWR - 1, 0))
sigma_F = r * TWR * m / 3
F_x, F_y, F_z ~ Normal(0, sigma_F^2)
```

This reproduces the numerical force law in RAPTOR's pinned source. Do not add an
extra g factor or interpret division by 3 as clipping. Gaussian samples are
unbounded. There is **no shared 10% budget**, no added external-torque noise and
no additive motor-command error. Existing randomized motor-response dynamics
remain. The retained external-torque state field is zero in this protocol.

TRAIN resamples airframes, initial states, force, latency and measurement tapes
for each update. The **single** EVAL uses the same distribution, with its own
fixed seeds, airframes and tapes. EVAL reads saved checkpoint settings, not new
CLI noise flags. `--disable-disturbances` is a zero-noise regression fixture;
it does not create another evaluator or turn off multi-airframe randomization.

The four measurement scales use the L2F low-level code as a reference. The SO(3)
attitude model is deliberately different from L2F's matrix-element corruption;
the same numeric standard deviation is not a claim of equivalent covariance.
Uniform 10-30 ms latency is our design choice motivated by RAPTOR's delayed-velocity
diagnosis, not an identified hardware latency distribution or a stability proof.
See [protocol and source audit](docs/disturbance_budget.md).

## Actor visibility and causal history

The Actor still receives 22 values, in this order:

```
measured position (3), delayed measured world velocity (3),
current measured rotation matrix (9), current measured body angular velocity (3),
last known motor command (4)
```

The 16 control features feed `GRUCell(16,64)` and the affine readout
`[features,memory] -> 4 -> tanh`: **16,068 trainable parameters**.
"GRU16" is the input feature count, not the hidden-state dimension.
Commands are absolute normalized values in [-1,1], FR/BR/BL/FL, FLU.

The random initial motor state is **not** encoded in `previous_action` anymore.
The initial history command is a fixed zero placeholder, independent of motor
truth; following steps store the Actor's issued command. Actual motor state,
airframe parameters, force, noise samples/scales, latency, group IDs and boundary
metadata are not Actor inputs. The initial placeholder is not a hover estimate.

Latency uses current plus three past world-velocity samples and their **original
acquisition noise**. Pre-reset history is held at the first noisy measurement.
Interpolation handles both 0 and 30 ms without a future or fifth-frame read.
The current measured attitude is used to express this world vector in body axes.
Only velocity is delayed; position, attitude, gyro and known command stay current.

Noise tapes and SO(3) matrices are precomputed on CPU with independent random
streams and shared by stable row IDs during live-scene compaction. Forward steps
never draw randomness. Delayed velocity history is differentiable and remains
connected across the 50-step metrics chunks. No physics or GRU memory is detached.
Losses, boundaries and metrics use physical truth, not noisy observations.

## Checkpoints and unchanged training rules

This is a **new environment/protocol**, not an exact continuation of bounded-noise
training. Use a new work directory. Old bounded checkpoints are rejected for
`--resume` and direct evaluation under this protocol; use their original source
for their original evaluation. Do not rewrite hashes to bypass this restriction.

`--init-checkpoint PATH` can explicitly import interface-compatible Actor weights
only, with fresh Adam, fresh sampling and the new protocol. Such a run is
fine-tuning from old weights, not a from-scratch comparison. The architecture,
action convention, memory size, timestep and model digest must pass checks.

For a new-protocol run, `--resume PATH` requires identical source, environment,
noise settings and optimizer configuration. Update/time budgets may be extended.

Joint RK4, the size-scaled initial kinematics, position-only first-failure rule,
Huber/task/CVaR costs, physical-group gradient aggregation, Time Decay and Adam
are retained. Changing initial command history also intentionally changes the
first action-change cost's reference; there is no change to the cost formula.
Time Decay > 0 remains a surrogate backward gradient; 0 is exact BPTT.

## Transient disturbances and scope

The official RAPTOR submodule pins `rl-tools/rl-tools` at
`e43ae4bcda4556321a63f4eb5dcc826cd637aa39`. Its audited training path samples force
at reset and copies it unchanged after integration. The Langevin process changes
the **reference trajectory**, not external force. Real-world poking and fan tests
are not evidence of a training pulse schedule. No transient-force mechanism was
copied or added in this change. See the source-path audit in the protocol document.

Motor-command noise and external torque were omitted to keep this experiment
small, **not** because parameter randomization mathematically subsumes them.
Bias/drift, dropout, time-varying wind and complete firmware latency are not modeled.
No setting here certifies learned recovery, successful Sim2Real or deployment.
Checkpoints retain `deployment_authorized: false`.

## Source and verification

`env_raptor.py` owns physics; `response_noise.py` owns sampling and sensing;
`response_policy.py` is the deployable Actor; `response_task.py` owns causal
rollouts and true-state costs; `response_adjoints.py` and `response_groups.py`
own the retained grouped full-BPTT update; `response_training.py` owns banks,
Adam and checkpoint transactions; `tools/train_response_control.py` is the only CLI.

Tests cover source-kernel contracts, independent NumPy RK4, action and delayed
history finite differences, exact noisy resume, first-failure compaction,
Actor input noninterference, original noise timestamps and H500 graph continuity.
CUDA tests skip explicitly when unavailable. CPU tests and constructed hover
fixtures do not establish GPU throughput or learned flight performance.

`tests/core_contract.json` remains unchanged. [Physics provenance](docs/raptor_reference.md),
[third-party notices](THIRD_PARTY_NOTICES.md), `reference/`, `物理配置/` and historical
images are retained; they are not additional runtime entry points.

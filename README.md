# DiffPhys: multi-airframe recurrent motor control

One trainable Actor, one RAPTOR-style multi-airframe simulator, one train/evaluate
entry. Training uses full-horizon BPTT, backward-only Time Decay, physical-group
shrink-only gradient clipping and persistent Adam. There is no critic, teacher,
auxiliary network or single-airframe mode.

## Run

Python 3.11+, PyTorch 2.10, NumPy and pytest. Install the PyTorch build matching
your CUDA runtime; no native extension build is required.

```bash
python tools/train_response_control.py @configs/response_raptor_multi_airframe.args
python tools/train_response_control.py --mode evaluate \
    --checkpoint runs/pulsed_recovery/seed7/best.pt \
    --work-dir runs/pulsed_recovery/evaluation
python -m pytest -q tests
```

The checked-in config uses one GPU, 4 x 512 = **2048 TRAIN** scenes per update,
2 x 128 = **256 fixed EVAL** scenes, H500 at 100 Hz, memory dimension 64,
Time Decay 1 s^-1, Adam 3e-4, gradient scale 0.1 and global clip 10.
It stops at 50 updates or 1800 seconds. `--scenarios` and `--eval-scenarios` are
**per bank**, not totals. No training runs on import.

TRAIN now uses [fixed physical coverage](docs/physics_coverage.md):
TTI 4 x rise 4 x fall 4 x conditional yaw 2 = 128 cells, 16 scenes each.
Candidate aircraft retain the original physical coupling and initial conditions;
noise is attached after selection. Fixed EVAL does not change. The 128 fixed
sampling cells are now also the 128 gradient groups: 16 initial scenes each,
including terminated scenes. Each group is capped at `--group-clip-norm 1.0`
after CVaR/gradient-scale, never amplified, then averaged. Groups are processed
in chunks of 16 (8 VJP calls), not 128 adjoints at once. This increases backward
work relative to the former 16 groups; CUDA throughput is not yet measured.
See [clipping and update evidence](docs/cell_clipping_audit.md).

Bare CLI calls retain the original random sampler. To explicitly use that sampler
with the checked-in config and the former smaller pool, use a NEW run directory:

```bash
python tools/train_response_control.py @configs/response_raptor_multi_airframe.args \
    --train-sampling random --scenarios 128 --group-max-groups 16 \
    --group-min-scenarios 32 --work-dir runs/random_sampling/seed7
```

Use an empty directory for fresh Actor/Adam initialization. Extend `--updates`
and `--max-seconds` explicitly for a longer run. Numerical failures still use
transactional rollback. Generated checkpoints/logs/visualizations are not bundled.

The independent [60-second surface-force EVAL and switchable replay UI](docs/long_hover_eval.md)
are also retained. This stress test uses a 4 m arena, a concentric 3 m target cube,
target changes every 10 s, and 0.1 s surface-force pulses every second at 20% of
vehicle weight. It is separate from the trainer's fixed EVAL and never selects
training checkpoints.

```bash
python tools/evaluate_response_long.py @configs/response_long_eval.args \
    --checkpoint runs/pulsed_recovery/seed7/best.pt \
    --work-dir runs/long_hover_eval/seed20260927
python tools/play_response_long.py --run-dir runs/long_hover_eval/seed20260927
```

New evaluations require a checkpoint compatible with the current protocol.
Existing exported replays remain viewable; reproduce older trajectories with
their archived evaluator sources rather than rewriting checkpoint bindings.

## Uniform per-step training loss

New training uses `--steady-weight 0`: every valid physical step has the same
coefficient `1/H`. The former H500 final-100-step amplification (0.022 versus
0.002) is disabled in both the dataclass default and the checked-in config.
Passing a nonzero steady weight to new training fails before creating run files.
The legacy fields and scoring kernel remain solely to reproduce stored evaluation
settings; archived scores are not silently recalculated with a new objective.

The uniform-time change itself did **not** alter sampling or physical groups,
CVaR scenario weighting, Time Decay, Actor, simulator, failure costs or GUI.
Uniform time weighting is not uniform scenario weighting or an exact, undecayed
BPTT gradient. Compare physical metrics, not raw old/new objective values.

An old-source run cannot be exactly resumed with a different objective/source.
Keep its checkpoint and checkout intact; use `--init-checkpoint PATH` with an
empty work directory for an explicit weights-only fork (fresh Adam and sampling).
For 2048 scenes, keep `--scenarios 512` (four-bank size convention). Exact resume
inside a new run remains supported with identical source and sampler settings. See the
[verification and training-design audit](docs/uniform_time_loss_audit.md).

## Online metrics without live training rendering

The optional [read-only Pygame dashboard](docs/training_gui.md) follows existing
TRAIN/EVAL JSONL logs and manually opens the saved-flight player. It never loads
checkpoints, runs extra EVAL, or changes the training process/source binding.
Already-running training does not need to restart for this GUI-only addition.

```bash
python -m pip install -r requirements-gui.txt
python tools/monitor_response_training.py \
    --run-dir runs/pulsed_recovery_b2048/seed7 \
    --replay-root runs/long_hover_eval
```

Only exported `playback/playlist.json` + `.npz` pairs are opened. To convert an
existing finished long EVAL, run `tools/play_response_long.py --run-dir PATH
--export-only` separately. The dashboard never launches export or simulation.
Polling is bounded and read-only; shared CPU/disk/display resources still have
some overhead. Run it on another machine with synced files for best isolation.

## One disturbance protocol for TRAIN and EVAL

| Component | Distribution | Lifetime |
|---|---|---|
| Constant external world force | Pinned RAPTOR Gaussian law below | One vector per episode |
| Force-at-point pulse | Independent Gaussian with the same airframe sigma_F | Constant world vector for 0.1 s |
| Pulse application point | Random arm, uniform fraction from COM to rotor | New body-fixed point per pulse |
| Position measurement | Gaussian, sigma = 0.001 m per axis | Each acquired sample |
| World-velocity measurement | Gaussian, sigma = 0.002 m/s per axis | Each acquired sample |
| Attitude measurement | Gaussian rotation vector, sigma = 0.001 rad per axis | Each acquired sample |
| Body angular velocity | Gaussian, sigma = 0.002 rad/s per axis | Each acquired sample |
| Velocity measurement latency | Uniform [0.010, 0.030] seconds | Fixed per episode |

For sampled mass m and thrust-to-weight ratio TWR:

```
r ~ Uniform(0, 0.3 * max(TWR - 1, 0))
sigma_F = r * TWR * m / 3
F_const ~ Normal(0, sigma_F^2 I_3)
F_pulse[k] ~ Normal(0, sigma_F^2 I_3)
```

The constant force numerical law is RAPTOR's; there is no extra g or clipping.
Each pulse gets a new force vector, NOT a copy of F_const. Gaussian force
components have no hard bound. During a pulse, F_total = F_const + F_pulse;
otherwise F_total = F_const. There is no shared 10% budget, independently sampled
external torque, or additive motor-command noise. Motor-response dynamics remain.

The first pulse onset is uniformly sampled at steps 0..99 (0.00-0.99 s).
Subsequent onset intervals are uniform integer steps 80..120 (0.8-1.2 s).
Each pulse lasts 10 transitions (0.1 s), unless episode end cuts it short.
Each scene has an independent phase/schedule. The minimal body geometry uses
existing COM-to-rotor arms, not a fictitious fuselage mesh. At each RK4 stage,
`tau_body = r_body x (R_true.T @ F_pulse_world)` is added to rotor torque. Thus a
pulse can push AND rotate the aircraft; its torque is not frozen as attitude changes.

TRAIN resamples airframes, initial states, constant force, pulses, latency and
measurement tapes each update. The **single** fixed EVAL retains original random
airframe sampling and fixed seeds/tapes; the conditional disturbance laws above
are shared. EVAL reads saved checkpoint settings, not new CLI noise flags. `--disable-pulses` is a regression switch; `--disable-disturbances`
turns off all disturbances for regression. Neither undoes the harder reset below
or creates another evaluator.

Measurement scales are L2F-inspired. SO(3) rotation-vector noise is not identical
to L2F's matrix-element corruption. Uniform 10-30 ms latency is a modeling choice
motivated by RAPTOR's diagnosis, not an identified hardware law or stability proof.
See [protocol, equations and source audit](docs/disturbance_budget.md).

## Recovery initialization

For non-guidance scenes, random-axis orientation angle is uniform in **0-120 degrees**, each world-velocity component in **[-2.5,2.5] m/s**, and each body
angular-velocity component in **[-2.2,2.2] rad/s**. These velocity limits are per
axis, not vector magnitudes. Position remains uniform within +/-10*arm_length per
axis; motor coordinates remain independently uniform in [0,0.5].

The 10% guidance draw still sets position, velocities and attitude to the target,
but does not reset motors to hover. Position-only first failure is strict
per-axis exceedance of 30*arm_length. No filtering or automatic curriculum hides
difficult initial states. These settings may include unrecoverable
samples and require actual training evidence before making performance claims.

## Actor visibility and causal history

The Actor still receives 22 values in this order:

```
measured position (3), delayed measured world velocity (3),
current measured rotation matrix (9), current measured body angular velocity (3),
last known motor command (4)
```

The 16 features feed `GRUCell(16,64)` and `[features,memory] -> Linear(80,4) -> tanh`:
**16,068 trainable parameters**. GRU16 denotes input size, not hidden dimension.
Commands remain absolute [-1,1], FR/BR/BL/FL, FLU.

Initial previous_action is a fixed zero placeholder, independent of random motor
truth; following steps store the Actor's issued command. The Actor never receives
pulse force, lever, schedule/mask, countdown, sigma, actual motor state, physical
parameters, force truth, latent delay, group IDs or boundary metadata. Motion
observed after a pulse is legitimate feedback, not a hidden answer.

Latency uses current plus three past world velocities with ORIGINAL acquisition
noise. Pre-reset history holds the first noisy measurement. Only velocity is
delayed; other measurements remain current. Current measured attitude converts
the delayed world vector to body features. Noise/pulse tapes are precomputed on
CPU and shared by stable original-row IDs during compaction; step/observe draw
no RNG. No physics, GRU memory or velocity history detaches at 50-step metric
boundaries. Posture-dependent pulse torque remains differentiable through RK4.
Loss, termination and flight metrics use true physical state.

## Checkpoints and retained training rules

Use a **new work directory**. Environment/noise versions and source bindings
changed; older Gaussian-without-pulse or bounded-noise checkpoints cannot be
silently resumed/evaluated under this environment. Their original source is
required for exact original evaluation. Never rewrite checkpoint hashes.

`--init-checkpoint PATH` explicitly imports only interface-compatible Actor
weights, with fresh Adam/new sampling. This is fine-tuning, not from-scratch
training. The checkpoint serialization schema remains unchanged; the new
environment and disturbance settings are stored in the existing strict binding.
For new-protocol runs, exact resume still requires matching source/configuration;
update/time budgets can be extended.

Joint RK4 and motor laws are retained with the requested additional force/torque
terms. Apart from the uniform time-weight default documented above, Huber/CVaR costs,
position-only first-failure semantics, Adam and Time Decay are unchanged.
Physical-group gradients now use fixed-cell membership and shrink-only caps. Time Decay > 0 is a
surrogate backward gradient; 0 is exact BPTT. No auxiliary controller is added.

## Provenance and verification

The official RAPTOR submodule pins rl-tools at
`e43ae4bcda4556321a63f4eb5dcc826cd637aa39`. Its audited force is constant within
episodes; its Langevin process changes reference trajectories. The pulses and
expanded reset here are **requested DiffPhys extensions**, not claimed features
of RAPTOR's original training or an exact reproduction of its benchmark.

Tests cover pulse timing/geometry/Gaussian scale, analytic COM impulse, independent
NumPy force-at-point RK4, attitude/action finite differences, privileged-input
noninterference, pre-pulse causality, original-row compaction, terminal freezing,
grouped full BPTT across metric boundaries and exact noisy checkpoint resume.
Intentional dynamics RHS, uniform-time default, and fixed-cell/clipping changes
are recorded with old/new provenance in the core contract; other protected hashes are retained. CPU tests are
not CUDA throughput, convergence, trained recovery or real-flight safety results.
Checkpoints retain `deployment_authorized: false`.

`env_raptor.py` owns physics; `response_noise.py` sampling/sensing/pulse tapes;
`response_policy.py` the deployable Actor; `response_task.py` rollout/true losses;
`response_adjoints.py` and `response_groups.py` grouped full BPTT;
`response_training.py` banks/Adam/checkpoints; `response_sampling.py` TRAIN coverage;
`response_audit.py` bounded Actor/Adam/RNG update evidence;
`tools/train_response_control.py`
is the only train/evaluate CLI. [Physics provenance](docs/raptor_reference.md),
[third-party notices](THIRD_PARTY_NOTICES.md), `reference/`, `物理配置/` and historical
images remain; they are not additional runtime entry points.

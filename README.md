# DiffPhys: multi-airframe recurrent motor control

One trainable Actor, one RAPTOR-style multi-airframe simulator, one train/evaluate
entry. Training uses full-horizon BPTT, optional backward-only Time Decay,
physical-group shrink-only gradient clipping and persistent Adam. There is no
critic, teacher, auxiliary network or single-airframe mode. The hover Actor is
GRUCell(16,64) + hidden-only Linear(64,4); the agile task adds 20 goal features
to the same recurrent/readout structure. Wc is removed, with no mask or
alternative direct-feature branch. No geometric feedback controller is added.

## Agile flight: three-point continuous-reference training

The `Agile-Flight-training` branch adds `--task agile`: start at point 1 with
initial linear velocity, follow a smooth three-point
position/attitude/velocity reference, then brake at the final point.
The v4 defaults request a **105-degree midpoint tilt**, half-span `0.70 * position_limit`
and bend magnitude `0.38 * position_limit`, inside the original boundary.
Every update uses the same target-generation rules; there is no
difficulty schedule or success-gated promotion. A fixed clock moves the reference
from point 1 at reset, through point 2 at half the motion duration, to point 3.
After motion ends, every goal slot describes the final point with zero velocity
and level attitude; no fixed hold duration is reserved. The reference keeps advancing if the aircraft misses a point;
continuous tracking losses penalize the deviation. The original world-fixed
`30 * arm_length` boundary and sampled airframes remain intact. Agile disables
constant external force and force-at-point pulses by default, while retaining
measurement noise and velocity latency. Hover keeps its existing force default.

```bash
python tools/train_response_control.py @configs/agile_flight.args \
    --init-checkpoint /absolute/path/hover.pt \
    --work-dir runs/agile/from_hover_seed7
```

Replace the checkpoint placeholder before running. A compatible structured hover
checkpoint copies the existing 16 input columns and zero-initializes the 20 added
columns, with fresh Adam and sampling progress. All Actor weights remain trainable.
Each output checkpoint contains the complete updated Actor and can be loaded
without the original hover checkpoint. Inference still receives measured state
and the time-varying goal packet. Exact `--resume` remains separate and strict.
The agile argument file uses **H500 at 100 Hz (5 seconds)** and Time Decay 0.
All 500 steps are available to the motion reference. CUDA still compiles each original RK4 physics
step's forward and backward through strict Inductor/AOTAutograd; the full rollout
remains connected by autograd, with native GRU, loss, group probe and Adam.
CPU physics remains the native reference. The original hover argument file is
unchanged.

The projected-heading attitude frame is continuous at and beyond 90 degrees;
`tilt_degrees` accepts `[0, 180)`. The midpoint target uses the actual acceleration
after timing and actuator checks, so an aircraft may receive a smaller tilt than
the requested 105 degrees. World-vertical thrust may be negative in an inverted
segment while collective body-axis rotor thrust remains positive.

For each original airframe, the planner first keeps the complete path and searches
longer motion durations on a fixed grid until individual rotor thrust and motor
response pass the nominal audit. It uses geometric shrink only if no full path
duration passes within the horizon. Reports expose actual midpoint tilt/speed,
path length, initial speed, duration stretch and any geometric fallback. This is
a finite candidate search, not a global or continuous-time optimality claim, and
does not use training progress or flight success. The default task mix requests
three-point motion for every aircraft, with no separate hover replay samples.

The nominal S9 reference still starts at rest. Actual initial velocity is an
initial tracking error handled by the continuous loss, with matching delayed
velocity history; it is not an extra waypoint or recovery gate. Existing nonzero
velocities are retained. Originally stationary guidance rows receive an
independent seeded per-axis `Uniform[-2.5, 2.5] m/s` draw, matching the original
non-guidance range, without replacing the aircraft or its noise tapes. Reports
record this zero-velocity replacement count. An initial velocity can point
outward, so reference containment alone does not establish actual recovery.

See the [Chinese mathematical and usage guide](docs/agile_flight.md) for the S9
construction, continuous loss, CPU example, initialization, checkpoint rules and
limitations. Unless explicitly marked agile, the loss/reset descriptions and
commands below document the existing hover task.

## Repository layout

```text
diffphys/    Core Python package: simulator, sensing, Actor, loss and training
tools/       Command-line tools, replay players and training monitor
tests/       Deterministic regressions and protected kernel contract
configs/     Training/evaluation arguments and frozen physical coverage
docs/        Protocol, numerical methods and usage documentation
```

Import library code through the package, for example
`from diffphys.env_raptor import RaptorSimulator`. Run the commands below from
the repository root; no package installation is needed.

The package move updates imports, source paths and archive manifests. Numerical
kernels and protected contract hashes are retained. Source bindings change:
exact resume/evaluation of pre-move checkpoints requires their original checkout.
Existing checkpoint hashes must not be rewritten to accept the new layout.

Production CUDA training uses strict full-step Compiled physics, original
survivor compaction, GC deferral over forward/backward, and one persistent
`spawn` CPU sampler with one next-batch cache. CPU uses native physics as a
numerical reference. All execution runs through `tools/train_response_control.py`;
there is no separate experiment trainer or A/C/P selector. See
[current execution and measured qualifications](docs/gpu_efficiency.md).

The retired experimental implementations and detailed reports are preserved
in the [source archive](docs/archive/README.md). Existing generated runs,
checkpoints, licenses, source audit evidence and local agent configuration
are retained. Old exact resume uses its archived source; no checkpoint hash
is rewritten to accept the new production source.

## Run

Python 3.11+, PyTorch 2.10, NumPy and pytest. Install the PyTorch build matching
your CUDA runtime; no native extension build is required.

```bash
python tools/train_response_control.py @configs/response_raptor_multi_airframe.args
python tools/train_response_control.py --mode evaluate \
    --checkpoint runs/compiled_prefetch_3072/fresh_seed7/best.pt \
    --work-dir runs/compiled_prefetch_3072/evaluation
python -m pytest -q tests
```

The checked-in config uses one GPU, 4 x 768 = **3072 TRAIN** scenes per update,
2 x 128 = **256 fixed EVAL** scenes, H500 at 100 Hz, memory dimension 64,
Time Decay 1 s^-1, Adam 3e-4, gradient scale 0.1 and global clip 10.
It stops at 50 updates or 1800 seconds. `--scenarios` and `--eval-scenarios` are
**per bank**, not totals. No training runs on import.

TRAIN now uses [fixed physical coverage](docs/physics_coverage.md):
TTI 4 x rise 4 x fall 4 x conditional yaw 2 = 128 cells, 24 scenes each.
Candidate aircraft retain the original physical coupling and initial conditions;
noise is attached after selection. Fixed EVAL does not change. The 128 fixed
sampling cells are now also the 128 gradient groups: 24 initial scenes each,
including terminated scenes. Each group is capped at `--group-clip-norm 1.0`
after equal-scene weighting/gradient-scale, never amplified, then averaged. Groups are processed
using a training-only native GRU/Linear gradient probe in one backward graph
traversal. The former eight-chunk VJP path is an explicit reference backend only.
This preserves per-group clipping, not an approximation of the group norms.
See [one-pass probes and profiling](docs/group_gradient_probe.md). Update evidence
is retained locally by `diffphys/response_audit.py`. CUDA throughput must
be measured on the target GPU; fewer traversals do not promise a specific latency.

The checked-in config starts new runs under `runs/compiled_prefetch_3072/fresh_seed7`.
CUDA uses the strict compiler directly; the former `--rotation-backend` option
is removed. Sampling, EVAL, audit, checkpoints, worker startup/shutdown and
final cleanup remain within the run time budget.

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
    --checkpoint runs/compiled_prefetch_3072/fresh_seed7/best.pt \
    --work-dir runs/long_hover_eval/seed20260927
python tools/play_response_long.py --run-dir runs/long_hover_eval/seed20260927
```

New evaluations require a checkpoint compatible with the current protocol.
Existing exported replays remain viewable; reproduce older trajectories with
their archived evaluator sources rather than rewriting checkpoint bindings.

## Adapted task loss, equal scene weights

Every valid control step contributes true pre-action position smooth L2,
`25.0 * (1 - cos(relative rotation from R_t to R_(t+1)))`, and smooth L2 of
the final four-command difference. A constant necessary tilt is free; actual
roll, pitch and yaw changes all count. This is a soft motion cost, not a test
for instability, and is not divided by dt.
The fixed coefficient 25.0 is the current experimental setting (previously 0.2),
not a validated stability bound. Changing it requires a new run and best-score
record; physical flight metrics remain comparable across coefficients.
The fixed smoothing scales are `epsilon_p=0.01 m` and `epsilon_a=0.01` in raw
[-1,1] command units. Time normalization is always `1/H`; all initial scenes
have weight `1/N`. Velocity/omega penalties, Huber, CVaR and tail weighting
are removed from the task objective. Read-only velocity/omega/risk metrics remain.

A failed scene still pays `[3*(H-T)+200]/H` once, including failure on the last
step. This is detached bookkeeping and supplies no direct avoidance gradient.
Position, attitude_delta, action_delta, dead and terminal logs use the same cost as TRAIN
and EVAL. See [the exact loss and checkpoint rules](docs/adapted_loss_no_cvar.md).

Removing Wc changes only the Actor's readout and architecture binding. The adapted
loss, physics, sensor/pulse tapes, physical coverage, termination, Time Decay,
group clipping and Adam retain their definitions. Equal scene weights do not make the
Time-Decay/group-clipped update an exact unprocessed gradient. New and old task
scores cannot be compared directly; use the retained physical flight metrics.

An old-source/objective run cannot be exactly resumed with this source. Keep its
checkpoint and checkout intact; `--init-checkpoint PATH` explicitly imports only
same-architecture weights into a fresh Adam/sampling/best-score run. For 3072
scenes, use `--scenarios 768` and `--group-min-scenarios 24` (four-bank size convention). Exact resume inside
this new run requires matching source, objective and sampler settings.

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

The dashboard opens saved short-EVAL or long-EVAL playlists and their matching
`.npz` files. For the ordinary H500 fixed EVAL, enable the separate exporter:

```bash
python tools/short_eval_replay.py \
    --run-dir runs/compiled_prefetch_3072/fresh_seed7 --device cuda
python tools/monitor_response_training.py \
    --run-dir runs/compiled_prefetch_3072/fresh_seed7
```

The exporter retains the latest complete fixed-EVAL trace, including the exact
constant force and force-at-point pulses. It does one additional EVAL forward
pass per new EVAL/checkpoint pair and does not update Actor or Adam. Its source
must match the checkpoint. The monitor automatically discovers new short
replays; opening one uses saved arrays without policy inference.

To convert an existing finished long EVAL, run
`tools/play_response_long.py --run-dir PATH --export-only` separately.
The dashboard itself never launches export or simulation.
Polling is bounded and read-only; shared CPU/disk/display resources still have
some overhead. Run it on another machine with synced files for best isolation.

## Disturbance settings shared by TRAIN and EVAL

The force-enabled protocol below remains the default for **hover** and is
available to agile with `--enable-external-forces`. Agile defaults to force off:
constant force, pulse force and its induced torque are zero; sensor noise and
latency remain as listed. `--disable-external-forces` selects force off explicitly
for either task. The selected settings are saved with the checkpoint and reused
for fixed EVAL and exact replay.

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
are shared when external forces are enabled. EVAL reads saved checkpoint settings,
not new CLI noise flags. `--disable-pulses` disables pulses while retaining any
enabled constant force and sensors; `--disable-disturbances` turns off all
disturbances, including sensors and latency, for regression. Neither undoes the
harder hover reset below or creates another evaluator.

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

The 16 features feed `GRUCell(16,64)` and `memory -> Linear(64,4) -> tanh`:
**16,004 trainable parameters**. GRU16 denotes input size, not hidden dimension.
Current state feedback and action history reach the motors through the GRU; there
is no Wc shortcut. The first observation updates memory before the first command.
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
training. A historical hidden-only B checkpoint can be imported explicitly this
way; an A checkpoint with an 80-column readout is rejected. Neither historical
source is treated as an exact resume of the new mainline. Fresh initialization
uses the native 64-column layer's RNG sequence; use a B weight import to preserve
an existing B Actor exactly. The checkpoint serialization schema remains unchanged; the new
environment and disturbance settings are stored in the existing strict binding.
For new-protocol runs, exact resume still requires matching source/configuration;
update/time budgets can be extended.

Joint RK4 and motor laws are retained with the requested additional force/torque
terms. The task objective is the adapted loss documented above;
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

`diffphys/env_raptor.py` owns physics; `diffphys/response_noise.py` sampling/sensing/pulse tapes;
`diffphys/response_policy.py` the deployable Actor; `diffphys/response_task.py` rollout/true losses;
`diffphys/response_adjoints.py` and `diffphys/response_groups.py` grouped full BPTT;
`diffphys/response_training.py` banks/Adam/checkpoints; `diffphys/response_sampling.py` TRAIN coverage;
`diffphys/response_grad_probe.py` one-traversal layer-local group derivatives;
`diffphys/response_audit.py` bounded Actor/Adam/RNG update evidence;
`tools/train_response_control.py`
is the only train/evaluate CLI. [Physics provenance](docs/raptor_reference.md),
[third-party notices](THIRD_PARTY_NOTICES.md) are retained. Retired source and
alternative implementations are preserved in Git history and the documented
source archive. CUDA production uses strict whole-step compilation and bounded CPU prefetch;
CPU keeps the native numerical reference.

# DiffPhys: multi-airframe recurrent motor control

One trainable Actor, one RAPTOR-style multi-airframe simulator, one training entry.
The Actor is trained through differentiable physics with full-horizon BPTT,
backward-only Time Decay and physical-group gradient normalization. There is no
critic, teacher, auxiliary network, single-airframe mode or alternative optimizer path.

## Run

Python 3.11+ with PyTorch 2.10, NumPy and pytest. Install the PyTorch build matching
the machine's CUDA runtime; the repository does not compile native extensions.

```bash
python tools/train_response_control.py @configs/response_raptor_multi_airframe.args
python tools/train_response_control.py --mode evaluate \
    --checkpoint runs/raptor_multi_airframe/seed7/best.pt \
    --work-dir runs/raptor_multi_airframe/evaluation
python -m pytest -q tests
```

The checked-in config uses **one GPU**, 4 x 128 = **512 TRAIN** scenes per update,
2 x 128 = **256 fixed EVAL** scenes, H500 at 100 Hz, memory dimension 64,
Time Decay 1 s^-1, Adam 3e-4, gradient scale 0.1 and global clip 10.
It limits a run to 50 updates / 1800 seconds, whichever occurs first.
`--scenarios` and `--eval-scenarios` are **per bank**, not totals.
No training is launched by importing modules.

Resume with the identical config plus `--resume PATH`. The update limit and
wall-clock budget can be extended. Exact resume requires matching source,
objective, environment, disturbances and optimizer configuration. Exact resume and evaluation of checkpoints
from the former L2F/RAPTOR split are **rejected**, not silently reinterpreted.
The saved model shape alone is not an environment compatibility certificate.
`--init-checkpoint` is an explicit **weights-only** import: compatible direct-readout
Actor weights (including this branch's parent checkpoints) can initialize a new
run, after checking architecture, memory size, timestep, command convention,
finite tensors and model digest. It always uses the new simulator/noise protocol,
fresh Adam and fresh sampling. It does not resume an old training run.
EVAL uses the checkpoint's saved disturbance distribution, not new CLI noise flags.

## Deployed policy

Observation (22): position 3, world velocity 3, measured rotation matrix 9,
body angular velocity 3, previous **known motor command** 4.

Control features (16): body-frame position 3, body-frame velocity / 3,
world-up expressed in body frame 3, body angular velocity / 10, previous command 4.
The existing GRUCell(16, 64) and affine readout [features, memory] -> 4 -> tanh
are unchanged: 16,068 trainable parameters at the default memory size.
"GRU16" denotes the input feature count here, not a 16-dimensional hidden state.
Actions are absolute normalized motor commands in [-1,1], FR/BR/BL/FL, FLU.
No true motor RPM, airframe parameters, noise samples or hidden executed commands
are provided to the Actor. All state/cost/termination measurements use physical
truth; only the Actor observation is corrupted.

## Bounded randomization

The existing multi-airframe dynamics distribution is retained. Every episode
also samples a disturbance severity and randomly allocates a **single total
budget <= 10%** across force, torque, command error, position, velocity, attitude,
angular velocity and velocity delay. These are not eight separate 10% budgets.
The existing unbounded Gaussian external force is replaced too, so it cannot
invalidate the joint bound. All random disturbance draws have bounded support.

```bash
# Default: equally likely total budgets 0%, 2.5%, 5%, 7.5%, 10%.
--disturbance-budget 0.1 --disturbance-pool 1 1 1 1 1
# Change pool percentages, without introducing another trainer or simulator.
--disturbance-budget 0.1 --disturbance-pool 10 20 30 25 15
# Disable all disturbances for regression checks; dynamics remain multi-airframe.
--disturbance-budget 0
```

Physics-dependent bounds use each vehicle's two-sided hover thrust reserve and
full thrust-to-wrench allocation matrix, including yaw. Sensor and delay bounds
are common to all vehicles in the pool and use its most restrictive reference
error-model envelope; their sampled severity is independent of vehicle identity.
A normalized motor command is dimensionless, but its physical thrust sensitivity
is not: action-error limits therefore belong to the physics-dependent class.

The budget certifies **static hover allocation and a stated reference error
model**. It does **not** prove the learned Actor can recover from arbitrary
initial states or tolerate a 10% disturbance everywhere. Logs and checkpoints
keep `deployment_authorized: false`. A random/untrained Actor can fail with zero
noise. See [the complete derivation](docs/disturbance_budget.md) before interpreting
this number as a safety claim.

## Maintained source

| File | Responsibility |
|---|---|
| `env_raptor.py` | Multi-airframe sampling, motor dynamics, joint RK4, physical state |
| `response_noise.py` | Shared percentage sampler, mathematical bounds, immutable tapes, sensing/execution |
| `response_policy.py` | Deployable GRU + direct readout |
| `response_task.py` | First-failure rollout, unchanged task/Huber/CVaR loss, metrics |
| `response_adjoints.py` | One full retained graph and group backward update |
| `response_groups.py` | Physical partition and median-norm group gradient aggregation |
| `response_training.py` | Adam, deterministic TRAIN/EVAL banks, checkpoint/resume/rollback |
| `response_execution.py` | Exit status classification |
| `tools/train_response_control.py` | The sole train/evaluate CLI |

Metrics are streamed in 50-step chunks, but **no physics, memory or velocity
history is detached**. H500 still backpropagates through all 500 steps. No reverse
window recomputation, AGC, action slew projection or private GRU dispatcher patch
remains. Time Decay > 0 deliberately uses a surrogate backward gradient while
leaving the forward flight and objective unchanged; 0 is the exact BPTT check.

Immutable noise and precomputed SO(3) tapes are shared across compacted live
states using stable row IDs: termination never copies an entire [batch,horizon]
tape at each step. Randomization and its double-precision bounds are computed once
on CPU after pooling, followed by one bank transfer to the selected device. The
rollout never samples RNG and never evaluates trigonometric sensor-noise kernels.

## Verification and provenance

Tests cover allocation corners and joint bounds across 2048 sampled airframes,
both dtypes, an independent NumPy RK4 reference, action finite differences,
measurement replay, acquisition-time velocity delay and its gradients,
50-step boundary continuity, first-failure costs, grouped VJPs, transactional Adam,
and exact noisy save/resume. CUDA-specific tests are explicitly skipped when CUDA
is unavailable. CPU CI is not a GPU throughput or flight-performance result.

`tests/core_contract.json` freezes mathematical kernels from source commit
`c15ca41824ecca400069636bdd9546d137b8f30d`, ignoring only source positions,
docstrings and the L2F -> Raptor type rename. It is not regenerated to make a
changed algorithm pass. [Physics provenance](docs/raptor_reference.md) and
[third-party notices](THIRD_PARTY_NOTICES.md) are retained. `reference/`,
`物理配置/` and historical images are provenance/assets, not additional runtime
entry points. Removed code remains in Git history; other Git branches are untouched.

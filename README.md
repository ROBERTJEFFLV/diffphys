# DiffPhys response motor control

A response encoder, GRU memory and motor controller learn from differentiable
quadrotor flight. The Actor consumes 25 deployable observation entries and outputs
four **absolute normalized motor commands** in [-1,1]. No mass, inertia, motor
state or external-force truth is given to the Actor.

## Reference environment v3

The production path now uses two explicit reference protocols:

| `--scenario-mode` | Environment and initialization |
| --- | --- |
| `l2f` | 2024 L2F Crazyflie, original quadratic RPM thrust curve, 150 ms motor response, source observation noise/force/torque, position +/-0.2 m inside +/-0.6 m boundaries |
| `raptor` (default) | RAPTOR source joint dynamics distribution, **paper-first** 90-degree initialization, per-axis position +/-10 rotor radii and termination +/-20 radii, speed/omega initialization +/-1, source force/noise settings |

Both use FLU X geometry, motor order **front-right, back-right, back-left,
front-left**, 100 Hz and joint RK4 dynamics. Motor zero is no longer a universal
hover point. A command maps by `u = min + (action+1)/2*(max-min)`; motor lag acts
on `u` and the original thrust polynomial acts on that delayed state. The
RAPTOR profile samples rising/falling delays independently. The default Actor
has no extra slew-rate projection (`--action-rate 0`).

**Breaking change:** v2 and older hover-centered / plus-layout checkpoints cannot
be resumed, used for weight initialization or evaluated under v3. Retrain in a
new work directory. Editing checkpoint hashes is not a migration. Removed
`fixed-airframe` / `physical-fit` CLI names fail explicitly instead of silently
changing the meaning of old commands.

The definitive numerical settings, source pins, initialization discrepancies and
limits of this comparison are in [the reference protocol](docs/raptor_reference.md).
`docs/response_control_v1.md` describes the **pre-v3** environment and its migration
path; it is historical, not the current motor/scene contract.

## Production path

The seven production files remain `env_l2f.py`, `response_policy.py`,
`response_task.py`, `response_adjoints.py`, `response_training.py`,
`response_execution.py` and `tools/train_response_control.py`.

Each aircraft stops at its first boundary violation or the horizon cap (default
500). The crossing transition is retained; other aircraft continue independently.
Stopped rows are not passed to Actor or RK4 again and are not reset/replaced.
`full` BPTT retains every executed transition. H50 windows do not detach the graph.
`windowed` recomputes windows in reverse with exact boundary covectors. Each
scene's observation noise is pre-sampled once, held fixed for that rollout and
reused during recomputation. Immutable noise tapes are shared across snapshots.
Both modes keep finite checks, optional AGC, global clipping, atomic checkpoints,
RNG restoration and persistent Adam. The old implicit angular solver is replaced
by reference RK4; there are no implicit-solve status checks to defer.

`--scenarios` is per TRAIN bank (four banks); `--eval-scenarios` is per fixed EVAL
bank (two banks) and is independent of TRAIN batch size. Defaults are 512 TRAIN
and 256 EVAL trajectories, seed 7, H500, lr=3e-4, gradient scale 0.1 and clip 10.
Periodic evaluation/checkpoint cadence remains 50 updates. Checkpoints bind the
protocol, environment source hash, action convention and sampling configuration.
Only compatible v3 checkpoints can be rescored with newer metric code.

## Run

Python, PyTorch and NumPy are required; no native extension is needed. Launch
training only with an explicit time/update budget:

```bash
# Multi-airframe, paper-first initialization
python3 tools/train_response_control.py $(cat configs/response_raptor_multi_airframe.args)

# Single-airframe L2F baseline
python3 tools/train_response_control.py $(cat configs/response_phase1_single_airframe.args)
```

`--mode profile` executes at most one update. Use `--mode evaluate --checkpoint
PATH --work-dir NEW_DIR` to evaluate a v3 checkpoint; it uses the checkpoint's
protocol, horizon and fixed EVAL count, not unrelated CLI defaults.

Bounded verification (no long training or external downloads):

```bash
OMP_NUM_THREADS=1 python3 -m pytest tests -q
```

## Scope and provenance

This changes the requested scene, sensor/disturbance and motor semantics, not the
learning algorithm. The Actor-only response/GRU architecture and existing
Huber/CVaR terms and weights remain. Only post-termination padding is excluded
from costs and statistics; horizon normalization and the original steady window
are unchanged. **Failure penalties and loss redesign are deferred:** the present
positive-cost objective can favor short failures and is not a finished episodic
training objective. Reference metrics retain the **first** failure and publish
only the matching profile. We do not reproduce RAPTOR's teachers, distillation, reward,
Langevin trajectory curriculum or seven-airframe published evaluation set. Fresh
TRAIN episodes sample the same source distribution, not a byte-identical copy of
the original 1000-airframe archive. Therefore these are reference-environment
changes, not a claim of reproducing the full RAPTOR experiment or its performance.

Historical source/provenance and `物理配置/` remain available. Earlier baselines
and diagnostic code remain at `76b3a02857e122fbfdef5ece5d0ae7dbf98a870b`.
See [third-party notices](THIRD_PARTY_NOTICES.md). No new license is asserted for
the whole repository. Exact gradients can still explode over long horizons.
Passing numerical tests does not prove hover/adaptation or authorize real flight.

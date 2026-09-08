# Minimal response-conditioned development training

## Current primary trainer (2026-09-07)

The default is now a training-only Monte Carlo Critic with continuous H500
collection, ten H50 physics-gradient windows, two independent 64-scene TRAIN
banks, Huber physical costs, and per-proposal continuous TRAIN/both-DEV acceptance.
Actor architecture and deployment inputs remain unchanged. See
[the current algorithm, configuration and checkpoint contract](response_critic_training.md).
Use `configs/response_control_critic_seed7.args`; MS/PETSc is a debug path.

The sections below document the earlier full-length Adam/MS experiments. Select
`--optimizer adam` explicitly for their commands; they are not the current
short-window training procedure. Historical quadratic comparisons also require
`--huber-delta 0` and matching newly generated contract evidence.

## Scope

This is teacher-free physical-task learning, not Q2 behavior migration. The
deployable response memory and motor controller jointly receive task gradients.
No teacher action, teacher Jacobian, privileged dynamics input, calibration
authorization multiplier, or detached startup is required. Auxiliary response
prediction uses detached self-collected observations/actions.

The objective covers full-trajectory position, velocity, all angular-velocity
axes, motor effort/smoothness, sustained performance, and difficult scenarios.
There is no direct horizontal-attitude or absolute-yaw target. Network weights
are fixed in deployment; recurrent memory adapts to executed action/response
history. Initial bad control performance does not prohibit simulator learning.

Historical V4/V5 failures, validation claims and Q2 migration artifacts remain
historical. They do not certify this architecture. The older migration method
requires `--historical-q2-distillation`; the primary entry is
`tools/train_response_control.py`.

## Only the minimal sequence

1. Run the one-time task-gradient/information contract.
2. Run H16-H32, batch16, a few updates, save/load, and one more update.
3. Train initialization seed7 for 150 updates, then check the preflight DEV gate.
4. If it passes, continue the same checkpoint to the 2000-update budget.
5. After training, run one normal-memory versus one-time hidden-reset experiment.
6. Use MS only if ordinary training reaches a long-horizon bottleneck.
7. Freeze the design before any explicit FINAL evaluation.

No multi-seed training, teacher parity, parameter-identification exams, history
curves, wrong-platform memory swaps, linear probes, temporal gradient curves, or
per-layer daily gradient monitoring are part of this run. No FINAL data is read
during the sequence above. Simulator/VJP evidence must cover the actual invoked
path and boundary cases, not just an unchanged source file. The SO(3) zero-angle
backward repair changes simulator code; local zero/small-angle derivative and
relevant short-rollout parity checks are required before new performance training.
No new test pass or training result is claimed by this code revision.

## Commands

```bash
python tools/check_response_training_contract.py --device cuda \
  --output runs/response_pooled_v1/contract.json

python tools/train_response_control.py --device cuda \
  --work-dir runs/response_pooled_v1/smoke --horizon 32 --scenarios 16 \
  --updates 3 --minimum-updates 1 --development-every 100 --checkpoint-every 1
python tools/train_response_control.py --device cuda \
  --work-dir runs/response_pooled_v1/smoke --horizon 32 --scenarios 16 \
  --updates 4 --minimum-updates 1 --development-every 100 --checkpoint-every 1

python tools/train_response_control.py $(cat configs/response_preflight_seed7.args)
python tools/check_response_preflight.py

# Only after the preflight gate passes:
python tools/train_response_control.py $(cat configs/response_control_seed7.args)

# Only after ordinary training:
python tools/train_response_control.py --mode hidden-reset --device cuda \
  --work-dir runs/response_pooled_v1/seed7 --reset-step 100 --reset-horizon 250
```

Budgets are explicit caps, not claims that 2000 updates guarantee success.
Preflight is part of the same training run, not discarded warmup. Latest/best
checkpoints include model, optimizer, all random state, stage progress, and
source/configuration bindings. Exact resume allows changed stopping budgets,
but rejects changed model/objective/runtime. More than five training updates
requires matching successful contract evidence.

## Pooled TRAIN proposal (Adam only)

The default `--adam-train-batches 2 --scenarios 64` draws two independently
seeded TRAIN banks with the existing 4x4 strata, concatenates their physical
initial states, and runs both under one unchanged set of policy parameters.
Each episode initializes its own complete recurrent state. One loss/backward,
one gradient clip, and one AdamW proposal follow; there is no update between
banks. The catastrophe guard replays the same complete pooled batch.

The existing objective is evaluated on all128 scenario costs at once:
mean cost plus the unchanged CVaR weight times the pooled worst fraction.
With tail_fraction0.2 this selects26 of128 scenes globally, with no per-bank
tail quota. It is NOT the average of two independently selected bank CVaRs.
Auxiliary prediction retains detached observations/actions and its existing
weight and normalization, now over the pooled records. Task weights and
success criteria are unchanged.

This minimal implementation uses one128-scene graph, not memory-saving
microbatch accumulation. Memory use can increase; `--mode profile` now measures
the same pooled proposal and acceptance replay. Profile output distinguishes
total scenarios from scenarios per bank. No new profile or performance result
is claimed by this revision.

For zero-based attempt i and B TRAIN banks, seeds are
TRAIN_SEED_BASE + B*i + j, with j in0..B-1. Both seeds, the pooled count and
the sampling rule are recorded in logs/checkpoint bindings. Rejected attempts
still consume their recorded seed pair. Oversized budgets are rejected before
the schedule can enter the reserved DEV/FINAL seed range. Only initialization
seed7 is trained; sampling seeds do not mean separate model-training runs.

DEV remains64 scenarios per registered bank in the checked-in configs and
never enters this gradient objective. FINAL stays unopened. Existing aggregate
and tail loss definitions and non-monotone catastrophe guards are preserved;
there is no requirement that each DEV bank improve after every update.
A harmful direction on different sampled banks demonstrates a sample-level
tradeoff, not conflict between all possible directions. Initial conditions
and disturbances also differ, so the effect cannot be attributed solely to
airframe dynamics or used to declare a unified controller impossible.

The new `runs/response_pooled_v1/` output namespace preserves old runs. Old
source-bound contracts and exact-resume checkpoints are not silently reused.
An explicitly requested `--initialize-from` can load the old best825 weights
into a new experiment; it does not restore or relabel the old optimizer history.
Changing the TRAIN bank count also invalidates exact resume. No training,
evaluation or solver experiment is automatically launched by this change.

## Minimal daily metrics and preflight decision

Record task loss, position RMS, velocity RMS, full omega RMS, steady-success
rate, and motor saturation fraction. Also record raw gradient norm and finite
loss/gradient/physical/recurrent-state checks. The logged gradient norm is the
combined task plus weighted auxiliary gradient BEFORE clipping, not an isolated
task gradient or parameter-step norm. Raw auxiliary loss and combined loss are
recorded separately. The task objective retains CVaR; there is no separate
tail/frequency/JVP monitoring gate.

Steady success means position <0.05, velocity <0.10, and full omega <0.50 for
the last 100 steps (or the available shorter smoke window). Constant arbitrary
yaw is allowed; continued spinning is not.

The preflight decision is preregistered in `tools/check_response_preflight.py`:
at least100 updates, finite numerics, no tenfold loss explosion, improved fixed-
DEV position/omega RMS relative to the untrained baseline, and recent average
motor saturation below0.95. It uses the latest three DEV checks and last25
training updates. These are simulation-continuation criteria, not deployment
thresholds or changes to historical Q2+5% safety gates. Success rate is monitored
without demanding a random network already pass a deployment exam.

Per-scenario failures and strata are retained in DEV reports. Large trajectory
artifacts are saved for explicit evaluations, not every training checkpoint.

## AdamW catastrophic-update rollback

A finite gradient or parameter tensor is not a closed-loop stability guarantee.
The differentiated TRAIN rollout still starts at call0 and uses grad_decay=1.
No loss scaling, CVaR weight, recurrent gradient path or control architecture is
changed by the acceptance guard.

After each clipped AdamW proposal, start again from the SAME physical initial
state and freshly initialized recurrent state and run a no-grad continuous
acceptance rollout. This is not a detached startup in the training graph.
Reject a non-finite model, optimizer state or trajectory, or either of:

- task objective >2 times the pre-update objective, using a1e-6 reference floor;
- omega RMS >2 times the pre-update omega RMS, using a0.5 reference floor.

A rejected proposal restores model, AdamW moments/step counters and RNG. It does
not count as an accepted update. Its before/candidate states, optimizer, scenario,
source binding and candidate trajectory are kept under `rejected_adam/`.
Stop on non-finite proposals or three consecutive finite rejections. No automatic
learning-rate change or guard bypass is performed.

At the existing DEV cadence, compare task objective and omega RMS to the
best-loss checkpoint with the same2x tolerances and floors. Catastrophic DEV
regression restores that complete best checkpoint and stops, instead of merely
incrementing patience. The rejected candidate is saved under
`rejected_development/`. Attempts and sampled seeds are retained; intervening
provisionally accepted history rows are marked rolled back and the retained
update count is restored. The rejected DEV report is not relabeled as a score
for the restored model. Resume retains both this provenance and optimizer state.

These are non-monotone catastrophe guards, not requirements that every DEV
metric improve each update, stability certificates, or replacements for the
historical Q2+5% gate. Tolerances are explicit configuration/source bindings.
There is one extra no-grad TRAIN replay per proposal; `--mode profile` includes
that cost, but not periodic DEV cost. The `--updates` cap bounds proposal
attempts; accepted/retained updates are reported separately.

The earlier guard revision used `runs/response_guarded_v1/`; those artifacts
remain untouched. Current configs use `runs/response_pooled_v1/` as described
above. Do not edit old checkpoint hashes or relabel interrupted or rejected
runs as successful. The observed failures and weak hidden-reset effect remain
unresolved; the SO(3) zero-angle bug is not established as their cause.

## One hidden-reset counterfactual

The evaluator uses the same checkpoint and initial conditions on DEV parameter
draws not used for gradient updates. It first runs a shared self-generated
prefix. At t=100 it branches:

- normal: retain response memory;
- reset: zero only response memory once, then resume normal updates.

Physical state, integral, measured-response alignment, action history, and clock
are unchanged. Compare subsequent H250/H500 position RMS, omega RMS and steady
success. The result is a paired mechanism diagnostic, not independent FINAL
evidence or an optimality certificate. If the two results are nearly identical,
do not claim online adaptation has been demonstrated.

## MS: one continuous-performance acceptance gate

All policy parameters remain trainable. Startup is part of the differentiated
problem. Independent shooting nodes contain full dynamic physical/recurrent
state; clocks are fixed, policy-generated action history is not.

MS requires an existing learned task checkpoint. After a proposal, discard free
nodes and rerun the real candidate policy from the original physical initial
state and freshly initialized recurrent state, for both TRAIN and acceptance
DEV. Accept only finite continuous task improvement without DEV deterioration.

Daily MS evidence includes continuous loss before/after, held-out loss
before/after, continuity defect after restoration, and accepted/rejected.
The PETSc follow-up also always records the backend, fixed PC, iterations,
reason and independently recomputed KKT residuals. Additional LM, trust and
predicted-reduction diagnostics are available via `--ms-debug`.
Linear certification is enforced before a direction enters the trust step;
continuous TRAIN improvement and DEV non-deterioration remain the performance
acceptance criteria.
MS remains a research MVP; no throughput or convergence claim is made.

### Historical solver deferral and PETSc follow-up

The deferral below describes the earlier sampling revision. The subsequent
PETSc revision implements the reduced, symmetrically scaled full KKT with
fixed-SPD-preconditioned MINRES as the response MS default. See
[PETSc backend and staged validation](petsc_kkt_backend.md) for dependencies,
residual certification, legacy selection and validation commands.

The earlier sampling revision kept the single TRAIN bank and nested CG
solver. `--adam-train-batches` still does not alter MS. The PETSc follow-up
implements the previously deferred fixed KKT operator, symmetric scaling,
clock removal and independent relative-residual criteria. CPU/CUDA dense
oracles pass, but its bounded 825-checkpoint H250 run still fails the linear
certificate on all three proposals at the 200-iteration limit. This does not
establish that a certified SQP direction lacks continuous/generalization
benefit. See the [completed experiment report](../reports/petsc_kkt_implementation/SUMMARY_ZH.md).

## Data separation and research boundaries

Only model initialization seed7 is trained. Batches still cover a4x4 physical-
fit capability stratification, varied mass/inertia, thrust/torque authority,
rise/fall lag and persistent external disturbance. TRAIN draws start at31000007.
DEV uses32000007/32010007; MS acceptance uses33000007 and is also DEV.
FINAL seeds34000007/34010007 remain unopened until an explicit frozen-candidate
evaluation. Evaluation reuse cannot be made independent by deleting claims or
copying a workspace. No tool here grants flight deployment authorization.

The simulator actuator coordinate is airframe-normalized around hover. A real
motor adapter may require airframe-specific hover/thrust calibration; the method
does not establish calibration-free arbitrary-airframe deployment. Its parameter
family, symmetry and fixed-per-episode assumptions remain limits.

Per-platform best-known reference controllers are deferred. No claim of being
near each airframe's optimum follows from this first mechanism experiment.

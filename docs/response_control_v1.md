# Minimal response-conditioned development training

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
during the sequence above. Do not repeat simulator/VJP parity unless that lower
layer changes; its prior numerical checks are separate evidence.

## Commands

```bash
python tools/check_response_training_contract.py --device cuda \
  --output runs/response_minimal_v1/contract.json

python tools/train_response_control.py --device cuda \
  --work-dir runs/response_minimal_v1/smoke --horizon 32 --scenarios 16 \
  --updates 3 --minimum-updates 1 --development-every 100 --checkpoint-every 1
python tools/train_response_control.py --device cuda \
  --work-dir runs/response_minimal_v1/smoke --horizon 32 --scenarios 16 \
  --updates 4 --minimum-updates 1 --development-every 100 --checkpoint-every 1

python tools/train_response_control.py $(cat configs/response_preflight_seed7.args)
python tools/check_response_preflight.py

# Only after the preflight gate passes:
python tools/train_response_control.py $(cat configs/response_control_seed7.args)

# Only after ordinary training:
python tools/train_response_control.py --mode hidden-reset --device cuda \
  --work-dir runs/response_minimal_v1/seed7 --reset-step 100 --reset-horizon 250
```

Budgets are explicit caps, not claims that 2000 updates guarantee success.
Preflight is part of the same training run, not discarded warmup. Latest/best
checkpoints include model, optimizer, all random state, stage progress, and
source/configuration bindings. Exact resume allows changed stopping budgets,
but rejects changed model/objective/runtime. More than five training updates
requires matching successful contract evidence.

## Minimal daily metrics and preflight decision

Record task loss, position RMS, velocity RMS, full omega RMS, steady-success
rate, and motor saturation fraction. Also record raw gradient norm and finite
loss/gradient/physical/recurrent-state checks. The task objective may contain a
CVaR term; there is no separate tail/frequency/JVP monitoring gate.

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

Daily MS evidence is limited to continuous loss before/after, held-out loss
before/after, continuity defect after restoration, and accepted/rejected.
KKT/CG/LM/predicted-reduction telemetry is available via `--ms-debug` only.
The solver still uses its numerical algorithm/trust safeguards, but those
debugging statistics are not additional method-level performance gates.
MS remains a research MVP; no throughput or convergence claim is made.

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

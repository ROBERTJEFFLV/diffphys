# Response Actor-only training

The production path is `tools/train_response_control.py → response_training.py`.
It trains the existing response encoder, GRU and motor controller with the exact
full-flight task gradient. `response_predictor` and its unused auxiliary output
are removed. The observation, recurrent state, integral, action histories, motor
lag, action mapping and physics are preserved.

## One forward collection and one reverse traversal

1. Sample two independent TRAIN banks and pool their initial states.
2. Hold Actor parameters fixed, fly H500 continuously and store complete states
   at 0,50,…,500. Store only per-scene total costs and fixed scene weights.
3. Recompute windows in reverse order. One `autograd.grad` per window returns
   both Actor parameter contributions and the boundary covector for the preceding
   window. The covector already contains scene weights; never multiply them again.
4. Accumulate parameter contributions using the explicit `--gradient-scale 0.1`.
   This multiplier does **not** change with `--window-steps`.
5. Check finite gradients, optionally apply AGC, globally clip using a float64
   norm, then take one persistent Adam update. Parameters with no gradient retain
   `grad=None`. No optimizer update or gradient clipping occurs inside a window.

Window boundaries limit graph storage, **not credit horizon**. The propagated
covectors give full H500 BPTT, so true long-range gradient explosion remains
possible. Boundary mismatch/nonfinite stops execution. A finite cost increase
does not trigger a replay, search, rejection or rollback.

There is no Critic/target, derivative dataset, readiness check, candidate search,
finite difference or MS/PETSc import in this path. The simulator's own implicit
angular-velocity integration is unchanged.

## Task and observations

`response_task.step_costs(..., start=absolute_step, horizon=500)` is the sole
training cost. It preserves position/velocity/omega tracking, motor effort,
first action difference, omega difference, existing Huber transformation and
the absolute final steady interval. No new shaping or weight changes are applied.
`risk_weights()` still selects the full-flight pooled worst 20% once, giving
mean cost plus the existing CVaR coefficient. Banks and windows never select
their own independent tail.

Logs attribute the objective to position, velocity, omega and regularization.
They also report full-flight RMS, steady success, saturation and warning-zone
omega/saturation risk. These are observations, not update vetoes. Success retains
the last-100-step p<0.05 m, v<0.10 m/s and omega<0.50 rad/s rule.

The Phase 1 config uses seed7, nominal L2F dynamics, no external force, new
random position/velocity/attitude/omega for each episode, 2×32 pooled TRAIN states,
H500/W50, dt=.01, Adam lr=.0003, no weight decay, clip10 and AGC disabled.
TRAIN seeds are `31000007 + 2*zero_based_update + bank`; fixed development seeds
are 32000007/32010007, also pooled 2×32. Development evaluation is periodic and
only observes/saves models. FINAL is not part of this entry point.

## Commands

Run jobs only with an explicit experiment budget:

```bash
python tools/check_response_training_contract.py --device cuda
python tools/check_response_boundary_adjoints.py --device cuda \
  --windows 25 50 100 --scenarios 32 \
  --output reports/actor_only_boundary.json

python tools/train_response_control.py $(cat configs/response_phase1_single_airframe.args) \
  --mode profile --work-dir runs/response_actor_only/profile

python tools/train_response_control.py $(cat configs/response_phase1_single_airframe.args) \
  --updates 10 --max-seconds 600 --work-dir runs/response_actor_only/smoke

python tools/train_response_control.py $(cat configs/response_phase1_single_airframe.args) \
  --resume runs/response_actor_only/smoke/latest.pt --updates 20 \
  --work-dir runs/response_actor_only/smoke

python tools/train_response_control.py --mode evaluate --device cuda \
  --checkpoint runs/response_actor_only/smoke/best.pt \
  --work-dir reports/response_actor_only_eval
```

`profile` executes at most one update, including cold setup/initial EVAL; update
timings separately report collection and forward+backward+Adam, with CUDA
synchronization. `max-seconds` is per invocation and checked between updates.
SIGINT/SIGTERM likewise finish the current transaction and save. The update
budget is the total target index, including resumed updates.

## Checkpoint contract

`latest.pt`, `best.pt`, and `best_success.pt` contain actual Actor weights,
Adam, explicit parameter names, primitive/tensor RNG metadata, next update index,
source/config bindings and a small progress record. History resides in JSONL,
not in every checkpoint. Every strictly lower evaluated objective refreshes best;
best-success is independent, with cost breaking success ties. No plateau gate.

`--resume` requires the exact current algorithm/source/config and preserves Adam,
sampling index and RNG. On recovery, log records beyond the saved update and a
partial trailing record are discarded before appending. Budgets and logging
cadence may change; dynamics/loss/Actor/optimizer semantics may not.

`--init-checkpoint` explicitly imports effective Actor weights into a new run,
drops only the four known prediction-head tensors and creates fresh Adam. Unknown
or missing control tensors are errors. It does not claim resume equivalence.

`--migrate-checkpoint` is for an audited **old Actor-only exact-BPTT** checkpoint.
`--migration-metadata` must supply its SHA256, `algorithm=actor-only-exact-bptt`,
the intended `binding` (excluding source hash), nested
`optimizer_parameter_names` per group and `next_update`. Adam states map by names,
never inferred IDs. A Critic-era checkpoint can initialize weights; it cannot be
silently relabeled as an Actor-only optimizer continuation. Legacy import reads
local trusted pickle data; new checkpoints load with `weights_only=True`.

On numerical failure, the failed update's Actor/Adam/RNG are restored and
`failure.pt` plus an error summary are saved. No further update is attempted.
Source binding covers only production dependencies, including physics. Editing
an archived Critic or solver does not invalidate an Actor-only checkpoint.

## Historical scope

The old response Critic/value/proposal/MS trainers, their dedicated configs,
tests, derivative diagnostics and readiness rules are available at Git commit
`e6559debd1cf7c821a1af408e45734f2c18d79cf`. They are not parallel production paths.
The local user audit ZIP/plan and pre-edit archive stay outside published code.
Generic full-space/PETSc kernels, other baselines, MATLAB and deployment tools
are retained. `tools/validate_petsc_kkt.py` retains its generic artificial oracle;
its former response-specific B case belongs to that historical revision.

Correct gradient equivalence and reproducible optimizer state do not establish
reliable hover, generalization across airframes or deployment safety. Numerical
checks, learned performance and real-flight authorization remain separate.

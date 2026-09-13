# Response Actor-only training

The production path is `tools/train_response_control.py → response_training.py`.
It trains the existing response encoder, GRU and motor controller with the exact
full-flight task gradient. `response_predictor` and its unused auxiliary output
are removed. The observation, recurrent state, integral, action histories, motor
lag, action mapping and physics are preserved.

## One full graph by default; optional exact window recomputation

1. Sample four independent TRAIN banks and pool their initial states (default 4×128=512).
2. Hold Actor parameters fixed and fly H500 continuously with gradients enabled.
   H50 chunks retain the complete computation graph across physical, memory,
   integral and history states. Compute per-scene full-flight costs and choose
   the pooled CVaR tail once.
3. With `--backprop-mode full` (default), differentiate that graph once without
   repeating its forward flight. With `--backprop-mode windowed`, collect without
   a graph, then recompute windows in reverse order. Each window returns Actor
   contributions and the preceding boundary covector. That covector already
   contains scene weights; never multiply them again.
4. Apply the explicit `--gradient-scale 0.1` to the total Actor gradient.
   This multiplier does **not** change with `--window-steps`.
5. Check finite gradients, optionally apply AGC, globally clip using a float64
   norm, then take one persistent Adam update. Parameters with no gradient retain
   `grad=None`. No optimizer update or gradient clipping occurs inside a window.

Both modes provide full H500 BPTT, so true long-range gradient explosion remains
possible. Windowed mode limits graph storage and retains runtime recomputation
checks. Full mode has no recomputation to compare: logs explicitly record
`boundary_checks=0`, `boundary_exact=null`, `boundary_max_error=null`, rather
than claiming a boundary check passed. Continuous-state finite checks still run
at every H50 chunk. The reference path and offline full/windowed gradient
comparison remain available. A finite cost increase does not trigger rejection.

The implicit midpoint method still uses four Newton solves forward and one
transpose solve backward, with unchanged stabilization and derivative formulas.
`L2FSimulator.defer_solve_errors()` scopes the whole forward/backward transaction.
`solve_ex(check_errors=False)` status tensors remain on-device and are checked
together before clipping/Adam. Evaluation checks its forward statuses too.
Calls outside this scope check immediately, including a backward performed after
its forward scope has closed. Any reported solve error aborts the transaction
even when its returned solution is finite. Tensor finite checks aggregate per
device before host reads; CPU Adam step counters are included.

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
random position/velocity/attitude/omega for each episode, 4×128=512 pooled TRAIN states,
H500/W50, dt=.01, Adam lr=.0003, no weight decay, clip10 and AGC disabled.
TRAIN seeds are `31000007 + 4*zero_based_update + bank`; fixed development seeds
are 32000007/32010007, pooled 2×128=256. Development evaluation and periodic
checkpoint saving both occur every 50 updates by default. Development evaluation
only observes/saves models. Initial and final/interrupted saves remain in place.
FINAL is not part of this entry point.

## Commands

Run jobs only with an explicit experiment budget:

```bash
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
The explicit `backprop_mode` is bound in checkpoints too. Switching full/windowed
changes floating-point accumulation order, so it requires a new audited execution
configuration even though the mathematical full-horizon objective is identical.
No automatic OOM fallback silently changes this setting.

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

Source cleanup changes this binding even when numerical behavior is equivalent.
An earlier checkpoint therefore cannot use `--resume` or standalone evaluation
under the new source. Preserve its source and checkpoint together. After an
explicit equivalence audit, `--migrate-checkpoint` can carry named Adam moments,
RNG and the update index into a new run with matching semantics. Do not rewrite
the old checkpoint's hash or disable source checks. Weights-only initialization
does not provide exact optimizer continuation.

Legacy pickle data can reference modules removed from the production tree.
Such artifacts require conversion to dictionaries, tensors and primitive
metadata in their original environment before import here. The legacy loader
does not promise to recreate arbitrary historical Python objects.

## Production interfaces

`sample_scenarios()` returns only `L2FState`. Fixed-airframe uses nominal dynamics
and zero disturbance; physical-fit retains the same random draws, candidate pool
and outer 4x4 thrust-to-weight / roll-authority selection. Internal alternative
samplers and capability-label outputs are absent.

`rollout()` calls the Actor once per physical step and executes its returned
action. It has no parameter injection, memory ablation or boundary callback.
`ResponsePolicyOutput` contains `action` and `next_state`; recurrent memory stays
in `next_state.memory`. Complete state schemas and control parameter names are
unchanged. `compare_boundary()` and all finite-gradient checks remain runtime
requirements of the reverse traversal.

## Historical scope

The seven production Python files and one training config are the only active
code. Baselines, optional native CUDA backends, reference implementations,
external tests and diagnostics are archived at Git commit
`76b3a02857e122fbfdef5ece5d0ae7dbf98a870b`; restore them with their matching source
when needed. The older learned-Critic path is available at `e6559deb`.
Local snapshots and cleanup acceptance scripts stay outside the production tree.
Existing models, logs, audit evidence, physical-fit provenance and applicable
license/source notices must be preserved separately from source pruning.

Correct gradient equivalence and reproducible optimizer state do not establish
reliable hover, generalization across airframes or deployment safety. Numerical
checks, learned performance and real-flight authorization remain separate.

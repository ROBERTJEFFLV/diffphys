# Response Actor-only training

The production entry is `tools/train_response_control.py → response_training.py`.
It trains the existing response encoder, GRU and four-motor controller. There
is no stability Metric MLP, learned Critic, auxiliary prediction loss, candidate
search, finite-difference solver or MS/PETSc dependency in the production path.

## Rollout and gradient

1. Sample four independent TRAIN banks and pool them (default 4×128=512).
2. Keep Actor parameters fixed while each aircraft flies until its first
   boundary violation or the H500 cap. Include the crossing transition. Ended
   aircraft are not passed to the Actor or physics again and are not replaced.
3. Compute each aircraft's full-flight task cost and select the pooled CVaR tail
   once. All windows share those weights.
4. Backpropagate with the configured Time Decay. Default `--time-decay 1`
   multiplies incoming physical and recurrent-state derivatives by
   `exp(-alpha * dt)` at each control step. It changes gradients only: forward
   states, actions, history, noise, first failure and cost are unchanged.
5. Apply `--gradient-scale 0.1` once, check finite gradients, optionally apply
   AGC, globally clip using a float64 norm, and perform one persistent Adam
   update. Parameters without a gradient retain `grad=None`.

Default `--backprop-mode full` retains all executed transition graphs across
H50 chunks. `windowed` collects without gradients, then recomputes windows in
reverse and propagates complete boundary covectors. Covectors already include
scene weights; do not multiply the weights twice. Window boundaries introduce
no additional decay, reset, clipping or optimizer update. Each scene's sensor
noise tape is sampled once and reused during recomputation.

Time Decay is a surrogate backward rule, not physical damping or exact H500
BPTT. `--time-decay 0` retains the exact derivative for diagnostics. Both modes
can still produce extreme gradients. Reverse-window recomputation keeps its
runtime boundary checks; full mode reports zero boundary comparisons rather
than claiming a check was performed.

## Actor and physics

The Actor still consumes deployable observations and action-response history,
updates its GRU memory and integral, and produces four motor commands. Dynamics
truth does not enter the Actor. Control-network parameters, dimensions, memory,
integral, action/response histories and action limits are unchanged.

Physics uses the selected reference protocol, joint RK4, its native motor curve,
X-frame convention and protocol-specific motor dynamics. The production path
does not use implicit midpoint integration. Protocol definitions and
initialization are in `env_l2f.py`; their source hashes are checkpoint-bound.

The L2F configuration selects the single nominal airframe. RAPTOR selects
variable airframes, paper initialization, sensor noise and external-force
sampling. No auxiliary network changes either sampler.

## Task and evaluation

`response_task.step_costs(..., start=absolute_step, horizon=500)` supplies the
task objective, retaining position, velocity, angular velocity, motor effort,
first action difference, omega difference, Huber weights and the absolute final
steady interval. Costs after a scene's first failure are excluded. First-failure
accounting adds `d * (dead_cost * (H-X) + terminal_cost) / H` once, where X
includes the crossing transition and d is false for a clean horizon timeout.
Defaults are dead_cost=3 and terminal_cost=200.

`risk_weights()` selects the full pooled worst 20% and combines the mean cost
with the configured CVaR weight. Banks and H50 windows do not select separate
tails. Loss weights, failure accounting and CVaR are unchanged by removal of
the stability teacher.

Fixed EVAL uses two banks (default 2×128=256), every 50 updates. It reports task
objective and components, valid-transition position/velocity/omega RMS, motor
saturation, physical risk and the selected protocol's episode-length and
termination metrics. Survival to H500 is not a hover-success certificate.
Evaluation observes performance and refreshes `best.pt` on any strictly lower
task objective; it does not approve or reject finite Actor updates.

TRAIN seeds are `31000007 + 4*zero_based_update + bank`. Fixed EVAL seeds are
32000007 and 32010007. The seed-overlap guard prevents TRAIN sampling from entering
the reserved EVAL range. FINAL is not consumed by this entry.

## Execution

The two maintained configs are:

- `configs/response_phase1_single_airframe.args`: nominal L2F.
- `configs/response_raptor_multi_airframe.args`: multi-airframe RAPTOR.

Both use seed7, CUDA float32, H500/W50, dt=.01, lr=.0003, gradient scale=.1,
clip10, no AGC, and Time Decay=1. `--scenarios` and `--eval-scenarios` count
scenes per bank. Default EVAL and checkpoint cadence is 50 updates.

Run only with an explicitly authorized budget:

```bash
python3 tools/train_response_control.py $(cat configs/response_raptor_multi_airframe.args)
python3 tools/train_response_control.py $(cat configs/response_phase1_single_airframe.args)
```

`--mode profile` limits the same trainer to one update. `--max-seconds` is a
per-invocation budget checked between updates. `--updates` is the total update
index including resumed progress. SIGINT/SIGTERM requests completion of the
current transaction and an atomic save.

## Checkpoints and recovery

New checkpoints contain only Actor weights, named Actor Adam state, RNG,
sampling/update index, source/config bindings and progress. Full history lives
in JSONL. `latest.pt`, `best.pt`, failure recovery and fixed EVAL remain.
There is no second network, optimizer or auxiliary payload in new checkpoints.

`--resume` requires exact source/config bindings and preserves Adam, RNG and
sampling index. Budgets and logging cadence can change; physical, task, Actor,
gradient-decay and optimizer semantics cannot. Changing full/windowed mode also
requires a new execution binding because floating-point accumulation order can
differ. There is no automatic OOM fallback or source-hash bypass.

`--init-checkpoint` explicitly loads compatible Actor weights into a new run
with fresh Adam. `--mode evaluate --checkpoint PATH --work-dir NEW_DIR` uses
the stored Actor, environment, task loss, horizon and fixed EVAL bank. Compatible
v3 archives containing a removed auxiliary network can still be scored: that
payload is ignored, while Actor/environment contract checks remain in force.
Evaluation reports checkpoint_source_sha256, evaluator_source_sha256 and
source_match. This does not permit resuming an old experiment under new source.

On numerical failure, the current update's Actor, Adam and RNG are restored,
and failure.pt plus an error summary are saved. No further update is attempted.
All previously generated checkpoints, logs and frozen experiment source remain
historical evidence. Reproduce those runs using their matching source snapshot.

## Verification

Existing tests cover first failure and frozen rows, reference physics, noise
replay, failure cost/CVaR, Time Decay's independent backward rule, full/windowed
agreement, CUDA gradients, checkpoint recovery and numeric rollback. Small
integration tests verify that training and evaluation work without the removed
auxiliary module. Production source binding includes only the seven active
Python files.

Passing these checks establishes implementation behavior. It does not establish
reliable hovering, broad airframe adaptation or authorization for deployment.

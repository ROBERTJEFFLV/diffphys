# Phase 1: unified task-value Actor–Critic

The default Phase 1 configuration is `configs/response_phase1_single_airframe.args`.
It uses the current ResponseMotorPolicy, fixed nominal L2F dynamics, zero external
force, and fresh random position, velocity, attitude and omega on each episode.
Two banks of 32 initial states are pooled before computing any task/CVaR weights.

## One objective

`response_task.step_costs()` supplies both supervision and Actor window costs.
It retains all existing evaluation weights, Huber residuals, the full-flight mean
and the final steady interval. A window calls it with `start=b, horizon=H`; it
never restarts the final interval at the local window boundary. Risk metrics are
observations only. `training_step_costs()` and risk scalarization are not used.

The Critic has the existing dimensionless closed-loop/capability inputs, two
256-wide SiLU hidden layers, and one signed linear output. It predicts cumulative
remaining **task** cost directly, without Softplus, asinh, output calibration, or
an extra remaining-time multiplier. The deployable Actor receives no new inputs.

For a fixed Actor, one no-gradient H500 rollout produces costs c[t] and labels
G[t] = sum(c[t:]); G[H] = 0. Huber regression updates only the Critic. No TD,
ranking directions, replay, optimal-policy labels, or risk supervision are used.
After fitting, target = (1-tau)*target + tau*critic. Target parameters are frozen;
its input derivatives remain enabled during the Actor pass.

For each H50 window, the objective is the same task cost over that absolute time
slice plus V_target(Z_end, end), with zero terminal value at H. Full-flight
mean+CVaR scenario weights are selected once from sum(c) and reused for all ten
windows. The Critic predicts per-scenario value, never batch CVaR. The ten window
gradients are averaged; physical state, memory, integral and history remain
numerically continuous and only the computation graph is detached at boundaries.
One gradient clip and one persistent Actor Adam step follow the complete episode.

## Execution and evaluation

`response_value.py` owns collection, scalar value learning and window gradients.
`response_value_training.py` owns persistent optimizers, checkpoints and periodic
EVAL. The existing entry point dispatches `--optimizer task-adam` to this path.
No candidate search, H500 backtracking, risk veto or EVAL rollback runs here.
Ordinary finite cost increases do not stop training or restore Adam moments.

EVAL uses a fixed pooled 2x32 nominal-state bank at updates 0,5,...,50. It reports
the same objective, CVaR, loss decomposition and original success criteria. It
only records trends and saves `best.training.pt` / `best_success.training.pt`.
`latest.training.pt` always follows the ongoing Adam trajectory. EVAL preserves
RNG, so reporting does not alter training sampling or Critic minibatches.

Checkpoints bind the task-value schema, loss, source, hyperparameters and dtype.
They preserve Actor Adam, Critic Adam, target, RNG and sampling position. Old Risk
Critic or optimizer states cannot be resumed into this objective. An explicitly
requested weights-only Actor initialization starts both optimizers and Critic anew.
Nonfinite gradients do not reach the Actor step; numerical failure saves its
stage and inputs in `failure.pt`, saves training state, and stops. Boundary
continuity failure also stops. No performance plateau gate is active.

## Reproduce the trial

First generate a fresh nominal training contract using the existing
`tools/check_response_training_contract.py` CLI. Then:

```bash
python3 tools/train_response_control.py $(cat configs/response_phase1_single_airframe.args) --mode profile
python3 tools/train_response_control.py $(cat configs/response_phase1_single_airframe.args)
```

Profile is a fresh isolated update; it does not initialize the subsequent run.
The seed-7 trial uses 50 updates, CUDA float32, Actor Adam lr=3e-4, Critic Adam
lr=1e-3, clips=10, one full-data Critic epoch, minibatch=1024, target tau=0.6,
and an 1800-second limit. The configuration contains no weight decay objective.
Diagnostics include complete boundary states, window local/total parameter
gradients, terminal dV/dZ, Critic target/error ranges and synchronized update time.

The prior TRAIN-only search is still available through
`configs/response_phase1_search.args`; it is a separate diagnostic experiment.
Neither trial is a deployment or generalization certificate.

## Reference and deliberate differences

[Official NVIDIA SHAC implementation](https://github.com/NVlabs/DiffRL/blob/main/algorithms/shac.py)
uses Actor/Critic Adam, clipping and a smoothed target. This adaptation uses exact
Monte Carlo labels from a full episode, no discount, the repository's task/CVaR,
and one recurrent Actor update after all ten windows. It does not reproduce the
original TD-based SHAC algorithm or claim its published results.

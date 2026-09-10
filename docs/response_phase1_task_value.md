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
G[t] = sum(c[t:]); G[H] = 0. Huber value regression and the continuation derivative
supervision below update only the Critic. No TD, replay buffer, optimal-policy
labels, or risk supervision are used.
After fitting, target = (1-tau)*target + tau*critic. Target parameters are frozen;
its input derivatives remain enabled during the Actor pass.

For each H50 window, the objective is the same task cost over that absolute time
slice plus V_target(Z_end, end), with zero terminal value at H. Full-flight
mean+CVaR scenario weights are selected once from sum(c) and reused for all ten
windows. The Critic predicts per-scenario value, never batch CVaR. The ten window
gradients are averaged; physical state, memory, integral and history remain
numerically continuous and only the computation graph is detached at boundaries.
One gradient clip and one persistent Actor Adam step follow the complete episode.

## Continuation derivative supervision

The task-value objective is now `task-value-v2-memory-cosine-lognorm`. The MLP,
value labels, Actor objective, and optimizer hyperparameters are unchanged.
Default supervision selects **only `policy.memory`**. At t=50,250,450 of the
current H500 trajectory, a small set of states continues to H500 with Actor
parameters frozen but the selected state field differentiable. The suffix cost
uses `step_costs(..., start=t, horizon=H)`. Labels are detached per-scene gradients
of that remaining task cost; they exclude batch CVaR, like the scalar value.

Both true and predicted raw-state gradients are multiplied by the same fixed
state scale: for x_hat=x/s, dV/dx_hat=s*dV/dx. Memory uses s=1. The selectable
single-field interface also supports the explicit fixed-scale fields listed in
`DERIVATIVE_STATE_SCALES`; it never enables all-state supervision implicitly.
The Critic prediction uses the canonical `critic_features` mapping and absolute
t/H. No running normalization or learned state units are added.

Each value minibatch (1024 by default) draws 32 derivative samples with
replacement from a fresh 32-sample pool. It minimizes:

```
L = L_value + 0.5 * wd * (1 - cosine(g_pred, g_true))
            + wm * SmoothL1(log(norm(g_pred)+eps) - log(norm(g_true)+eps))
```

Cosine and log norms are computed per sample in float64 for stable reductions.
Zero true gradients have no direction; they contribute only the magnitude loss.
`create_graph=True` lets these losses update Critic parameters through its state
gradient. On the first derivative minibatch, BEFORE clipping, fix:
`wd=norm(grad L_value)/norm(grad L_dir)` and
`wm=norm(grad L_value)/norm(grad L_mag)`. The 0.5 direction factor is applied after
this measurement. A denominator floor of eps=1e-8 handles a zero gradient norm;
the original measured norms and floor are logged. This is a one-time gradient
balance, not the adaptive GradNorm algorithm. No weights are recalibrated later.

Critic clipping remains 10 and target EMA remains 0.6. The checkpoint stores
fixed weights, initial gradient norms, selected state field, boundary steps,
sample counts/scene IDs and sampler configuration. Failed Critic transactions
restore the weights and sampling metadata with the Critic/target/optimizer.
Completed fits still survive a subsequent Actor failure.

An additional 16 derivative samples use disjoint TRAIN scene IDs, across every
selected boundary. They are excluded from derivative optimization and weight
calibration. `response_phase1.py` records target cosine, norm ratio, and absolute
log-norm error before/after fitting, plus post-fit online-Critic metrics. These
are **held-out derivative labels within TRAIN**: ordinary value regression uses
all trajectory scenes. They are not independent EVAL/generalization evidence.
None of these metrics creates a new Actor gate.

For small test horizons the default boundaries are the first/middle/last distinct
nonterminal window boundaries. A single terminal window has no bootstrap
derivative pool. Explicit boundary indices must be nonterminal window boundaries.

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
They preserve Actor Adam, Critic Adam, target, RNG and sampling position. Old v1
task-value and Risk Critic states cannot be resumed into this objective. An explicitly
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
The derivative-supervised configuration writes to a new experiment directory,
`runs/phase1_task_value_derivatives/seed7`, and requires a newly generated contract
at `reports/phase1_task_value_derivatives/contract.json`; previous runs are kept.
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

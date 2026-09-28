# Uniform time loss: change, verification and training-design audit

Base inspected: `bb9c140f4a181a229fce9e0270bd002c2671729d` (2026-09-28).
Scope: remove the new-training final-window loss amplification. Do not implement
32-cell sampling, change physics, add a controller, or retune other losses.

## Applied change

`TaskLossConfig.steady_weight` defaults to 0 instead of 2, and the production
argument file explicitly sets `--steady-weight 0`. New `train()` calls reject
nonzero steady weights before allocating a device or creating a run directory.
The scoring kernel and serialized loss fields are retained for historical EVAL.

For H500, the continuous state/action terms previously had coefficients
`1/500 = 0.002` for steps 1-400 and `1/500 + 2/100 = 0.022` for steps 401-500.
They now have coefficient 0.002 at every valid step. Frozen post-failure padding
still carries no physical loss. Existing dead and one-off terminal penalties
are unchanged. These event penalties are not ordinary per-step physical errors.

This intentionally changes the objective, not just its display. For identical
errors throughout a successful rollout, the summed continuous-loss weight falls
from 3 to 1; unchanged failure penalties consequently have a larger relative
weight. A smaller new objective is not evidence of a better flying policy.

## Verification

The exact base source was obtained from the repository's successful Actions run
36386321647, source artifact 10954801707. The archive SHA256 was verified as
`4faad7a84621f50c5e8d1be6d635834868ff4d9ebfd0e56373f55798f0dca4a2`, and its
git-archive comment matches the base commit. No credentials were used by tests.

Local environment: Python 3.13.5, PyTorch 2.10.0+cpu. CUDA was unavailable.

- Before editing: 221 tests passed, 6 skipped.
- New regressions before the fix: 20 failed, 1 passed.
- After the fix: 242 tests passed, 6 skipped.
- Independent H500 constant-error probe: late/early loss ratio changed from 11
  to 1; direct gradient with respect to the same error changed from 11 to 1.
  This is not a claim that full Actor gradients at different times must match.
- Tests cover float32/float64 and H=1,8,100,101,500,751, time permutation, irregular
  metric slices, legacy scoring, rejected nonzero training weights and exact
  same-source resume. Existing physics, sensing, group-VJP, failure, rollback,
  long-evaluation and dashboard regressions remain in the suite.
- A real base-source CPU fixture (random Actor, 32 TRAIN / 8 EVAL, H120, one
  update) was saved before editing. After editing, its stored EVAL objective
  reproduced exactly: 42.73256290312841. Its bytes were unchanged.
- Importing that Actor with zero new updates kept position/velocity/omega metrics
  and termination fraction identical. Its newly defined objective was
  15.917540231652584: a scoring change, not learned improvement.
- Two bounded uniform-time CPU updates were finite. Split resume matched
  continuous training in both model and Adam state. Weights-only import kept
  the model intact and started a fresh optimizer as documented.

Run `python -m pytest -q tests` to repeat the checked-in regressions. These checks
do not establish CUDA throughput, reduced drift, high-TTI convergence, stability
of a trained 11150 Actor, or deployment safety. The user's local trained weights
and running training service were not available to these tests.

## Compatibility and operational boundary

All Actor interfaces, environment/noise semantics, checkpoint fields, CLI modes,
per-bank scene counts, group sampling/normalization, replay UI and log formats
are retained. The scoring kernel has not been redefined for old stored configs.
Only the intentional `TaskLossConfig` default AST hash is updated; other protected
kernel hashes are unchanged by this patch.

Exact `--resume` continues to require identical source/configuration. Changing
the objective is not an exact continuation; do not rewrite checkpoint hashes to
bypass that guard. Preserve the old worktree/checkpoint and use a separate
worktree and a new work directory with `--init-checkpoint PATH` to start an
explicit uniform-time fine-tuning run. That existing option deliberately resets
Adam, update numbering and TRAIN sampling. It is not evidence that resetting
Adam is scientifically preferable. Preserve the old run for a fair comparison.

An older checkpoint with different environment/noise contracts still needs its
original evaluator; this patch does not relax that pre-existing constraint.

## Unchanged designs worth examining next

These are observed design choices and testable risks, not diagnosed causes of
the user's oscillation. None was changed in this patch.

### 1. Failure penalties do not directly teach recovery

In `response_task.weighted_task_features`, the termination mask and both failure
costs are constructed under `torch.no_grad()`. A constant death/terminal score
adds no direct Actor gradient. It can change the selected CVaR scenarios; their
remaining physical-loss gradients still train the Actor. Existing
`test_constants_do_not_add_state_gradient_or_post_terminal_calls` verifies this.
No Actor/physics steps occur after the first crossing.

Also, a fixed replacement cost is not a no-suicide guarantee. As an objective-
only counterexample at H500, a sustained per-step physical cost of 4 totals 4;
a first-step failure at that same cost totals
`4/500 + 499*3/500 + 200/500 = 3.402`. This is NOT a dynamically feasible flight
experiment and does not prove that the trained Actor seeks termination. It shows
why a large-looking raw `terminal_cost=200` is not sufficient evidence of the
intended incentive. Its normalized H500 contribution is only 0.4.

### 2. CVaR changes weights across aircraft, not across time

`risk_weights` forms mean scenario cost plus 0.5 times the mean cost of the worst
20 percent. With 2048 scenes it selects ceil(409.6)=410, giving selected scenes
about 3.49756 times the base coefficient before physical-group normalization.
This remains active. It is not the removed final-100-step weight, and equal
numbers of scenes per physical group would not remove it.

### 3. Physical-group normalization can amplify as well as suppress gradients

`response_groups.normalize_group_rows` rescales each nonzero whole-Actor group
gradient toward the median norm before averaging. For raw norms 1, 100, 0.01,
the multipliers are 1, 0.01, 100. This is not ordinary average-loss BPTT, and it
does not resolve directional conflicts. Its intended purpose is to prevent
sensitive groups from dominating; whether it improves control must be measured.
In history logs, `raw_gradient_norm` is AFTER this normalization and BEFORE the
global clip. Inspect `group_gradient.raw_gradient_norm` for original group norms.

### 4. Uniform time loss still uses a deliberately modified backward gradient

`response_task._GradientDecay` has identity forward and scales backward state
edges by exp(-alpha*dt), including recurrent memory and delayed velocity history.
At alpha=1, the decay gates alone contribute exp(-1)=0.3679 over one second and
exp(-5)=0.006738 over five seconds, before the dynamics Jacobians. It does not
truncate H500 or erase forward GRU memory. It is a training-stability device,
not physical damping or a contraction guarantee. It is retained unchanged.

### 5. Saving frequently does not preserve every evaluated Actor

`response_training.train` overwrites `latest.pt` at checkpoints and saves
`best.pt` only for a better pooled task objective. It does not save an immutable
checkpoint for every evaluation. Thus an old JSONL row may not be replayable,
as happened with update 11100. In addition, 'best' means best configured scalar
score, not necessarily lowest early drift or best performance for every group.
A future checkpoint-retention change should be explicit and separately scoped.

### 6. Smoothness penalties use step differences, not time derivatives

`omega_deltas = omega[t+1]-omega[t]` and `action_deltas` are not divided by dt;
the loss coefficients therefore act on per-step changes, not rad/s^2 or action/s.
At the fixed 100 Hz rate this is a valid convention, but it should not be
mistaken for a calibrated angular-acceleration penalty or guaranteed suppression
of low-frequency oscillation. It must be retuned if the control period changes.

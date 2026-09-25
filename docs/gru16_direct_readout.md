# GRU16 direct-readout Actor

Base: `codex/time-decay-bounded-influence-20260925` at
`ea5a8f2eb1dbe8edba85cfa3ae26aa0e5834f76d`.
Branch: `codex/gru16-direct-readout-20260926`.

## Architecture

The only learned modules are a native PyTorch `GRUCell(16,64)` and
`Linear(80,4)`. The persistent policy state contains only `memory`.

```text
22D observation: p, v, measured R, omega, previous executed action
                           |
       c = [R.T p, R.T v/3, R.T e_z, omega/10, previous_action] (16D)
                           |
                 +---------+----------------------+
                 |                                |
                 v                                |
h_(t-1) --> native GRU(16,64)                      |
                 |                                |
                 h_t (64D) ----> next timestep     |
                 |                                |
                 +----------- concat(c_t,h_t) <----+
                                   |
                             Linear(80,4)
                                   |
                                  tanh
                                   |
                   4 absolute normalized motor commands
```

`h_t = GRU(c_t,h_(t-1))`

`a_t = tanh(W_c c_t + W_h h_t + b)`

`W_c` and `W_h` are column blocks of ONE readout matrix, not two extra
networks. `W_c` is initialized to zero and stays trainable. `W_h` uses a
small nonzero Xavier initialization (gain 0.1); output bias starts at zero.
This supplies a direct gradient path, not a hand-tuned flight controller or
a stability certificate. Both blocks and the GRU enter the same group
parameter-gradient vector and the same Adam optimizer.

The first observation updates hidden state before producing the first action.
There is no `calls > 0` gate. Reset hidden only when starting a new episode or
explicitly reactivating the policy. A zero initial hidden state does not erase
trained weights, nor provide knowledge of previously unseen dynamics.

Removed: the 26D response-difference features, external encoder/SiLU, separate
19D control MLP, explicit integral, previous-velocity/omega/rotation caches,
older-action cache and call counter. The actual previous executed action is
still supplied by the environment/execution interface. Physical motor state
and action history inside the simulator are not deleted or exposed as truth.

Default parameter count: **16,068**. This is 64 parameters more than the
GRU16 hidden-only readout and fewer than the parent's 36,484. Parameter/MAC
counts are not end-to-end training-time measurements.

## Preserved training and environment contract

No changes to `env_l2f.py`, `response_groups.py`, training CLI, exit semantics
or the three checked-in argument files. All task loss expressions, Huber/CVaR,
steady-window weights, failure accounting, sampler/noise and metric functions
are inherited. Full and reverse-window BPTT retain their original algorithms.
Only the list of recurrent state leaves now contains `memory` instead of the
removed architecture caches. Per-step Time Decay still covers every existing
physical/recurrent state edge, including the physical previous action which
enters both GRU and direct readout. No detach or extra boundary decay is added.

With `--group-balance`, groups still use the four physical parameters, whole-
Actor group norms still use the lower positive median, normalized gradients
are still averaged before ONE Adam update, and all 64 entries of W_c are
included. Default native/sparse VJP switches and constraints stay unchanged:
group balancing requires `full` BPTT and `agc=0`; there is no silent fallback.
Without group balancing, the original reverse-window mode remains available.

**Termination is position-only in this parent branch**, strict per-axis
exceedance. Velocity/omega still contribute to loss but do not end an episode.
The crossing transition is retained; subsequent Actor/RK4/hidden/noise updates
are frozen for that aircraft. There is no reset or replacement mid-rollout.

The optional existing `action_rate` projection is preserved (default zero).
Actions remain absolute [-1,1], motor order FR/BR/BL/FL, FLU X geometry,
100 Hz and joint RK4 including motor delay. No hover normalization is added.

## Checkpoints and launch

The new architecture tag is `gru16-direct-readout-absolute-motor-policy-v4`.
The environment/protocol schema is not changed. Old response-encoder/MLP
checkpoints are rejected for resume, weight initialization and evaluation.
They are not equivalent tensors; train this architecture in a NEW work
directory. New-architecture checkpoints retain strict source/config guards
and persistent-Adam/RNG exact resume. Do not edit hashes to bypass checks.

Removed architecture flags (`--hidden-dim`, `--integral-limit`,
`--integral-leak`) fail explicitly. `--memory-dim` controls the GRU width.
All non-architecture defaults and checked-in configs are unchanged, including
their output paths. Override the work directory to avoid existing runs:

```bash
# Bounded profile: same H500 and group settings, one update at most.
python3 tools/train_response_control.py \
  $(cat configs/response_raptor_group_gradient_smoke.args) \
  --work-dir runs/gru16_direct_readout_smoke/seed7

# Training command; execute only with an intentionally chosen budget.
python3 tools/train_response_control.py \
  $(cat configs/response_raptor_multi_airframe.args) \
  --work-dir runs/gru16_direct_readout/seed7
```

## Verification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q tests
python3 tools/verify_vjp_acceleration.py --device cpu \
  --horizon 8 --scenarios 64 --output gru16-vjp-audit.json
```

`tests/gru16_parent_contract.json` records the unchanged parent file blobs and
50 unchanged source definitions, including task losses, decay gates, BPTT and
the training loop. Tests cover first-call gradients, W_c/W_h algebra,
nonzero-feedback H500 full/windowed gradients, original termination masks,
group VJPs/normalization, one Adam step, strict old-checkpoint rejection,
checkpoint resume and rollback. A constructed near-hover RK4 H500 fixture is
an implementation check, not learned control performance.

The existing group/VJP diagnostic scripts work with the new readout. Their
optional `--baseline-root` equivalence checks require the SAME architecture;
comparing new GRU16 weights bitwise to the old MLP is deliberately rejected.
The historical `probe_group_capture.py` stays research-only and is not enabled
in production. Historical architecture images show the parent, not this Actor.

CPU results do not exercise the CUDA fused GRU kernels. No GPU speedup,
long-training convergence improvement, sim-to-real validation or deployment
safety is asserted by these changes. Keep the old branch/checkpoints for the
controlled performance comparison; never overwrite a running experiment.

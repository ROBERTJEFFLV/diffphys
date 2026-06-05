# Stage 9.6 Objective Audit

Scope: CPU production path in `src/foundation_policy/diff_pre_training/main.cpp`, `rdac_operations.h`, `config.h`, `checkpoint_io.h`, and `eval_utils.h`.

## CPU RDAC Critic/Value Path

- Critic input: `rdac_operations.h` assembles `critic_input[t,b] = observation[t,b] || hidden[t,b]` in `assemble_head_input_sequence`. The observation part has dimension `ENVIRONMENT::Observation::DIM = 22`; the recurrent hidden part has `HIDDEN_DIM = 16`, so `CRITIC_HEAD_INPUT_SHAPE` is 38.
- Critic output dimension: `config.h` defines `CRITIC_DIM = 1`, and `CRITIC_OUTPUT_SHAPE = [DIFF_TRAINING_SEQUENCE_LENGTH, DIFF_TRAINING_BATCH_SIZE, 1]`.
- Critic forward: `rdac_forward` runs the shared `model.trunk`, copies trunk output into `buffer.hidden`, assembles both actor and critic head inputs, then evaluates `model.actor_head` into `buffer.action` and `model.critic_head` into `buffer.q`.
- Critic target definition: `main.cpp` computes per-step cost from the next rollout state (`horizon_i + 1`) plus action magnitude, then discounted return with `AC_GAMMA = 0.99`:
  `running_return = -step_cost + 0.99 * running_return`.
  Targets are normalized per dynamics group:
  `q_target = (q_return - group_mean_return) / group_std_return`.
- Critic target weight: `main.cpp` sets `q_weight[t,b,0] = sample_group_weights[b] / current_horizon`. `sample_group_weights[b] = 1 / (num_dynamics_groups * group_count)` when group counts are valid, otherwise `1 / batch_size`.
- Critic loss formula: `rdac_backward` computes
  `0.5 * LOSS_AC_CRITIC_WEIGHT * q_weight * (q - q_target)^2`
  for every `(t,b)`.
- Critic loss weight: `config.h` defines `LOSS_AC_CRITIC_WEIGHT = 0.10`.
- Where critic loss is added to total loss: `main.cpp` adds `rdac_ac_terms.actor_critic_actor + rdac_ac_terms.actor_critic_critic` to `mean_terms.total` after `rdac_backward`.
- Critic backward: `rdac_backward` sets `buffer.d_q_critic = LOSS_AC_CRITIC_WEIGHT * q_weight * (q - q_target)`, then calls `rlt::backward_full` on `model.critic_head`.
- How critic gradient reaches shared GRU/trunk: after critic-head backward, `rdac_backward` adds the hidden slice of `buffer.d_critic_input` into `buffer.d_hidden`. It then backprops the actor head, adds the actor hidden slice into the same `buffer.d_hidden`, and finally calls `rlt::backward(device, model.trunk, input, buffer.d_hidden, ...)`.
- Whether critic affects actor head directly: no. Critic loss only backprops through `critic_head` and the shared trunk hidden slice. It does not add gradient to `actor_head`.
- Whether critic affects actor shared trunk: yes. The hidden component of `critic_input` is injected into `buffer.d_hidden`, which is the shared trunk/GRU BPTT input.

## Logging And Checkpoint

- CPU logs `critic_loss`, `actor_critic_loss`, `value_loss`, `actor_critic_actor_loss`, `actor_critic_critic_loss`, and RDAC gradient diagnostics including `rdac_critic_head_grad_norm_before_clip` and `rdac_critic_head_grad_norm_after_clip`.
- CPU checkpoint I/O in `checkpoint_io.h` saves and loads `critic_head_weights` and `critic_head_biases` in the text checkpoint magic `foundation_policy_diff_pre_training_rdac_hidden_actor_v3`.
- Current CPU checkpoint I/O does not save optimizer state.
- Current CUDA checkpoint I/O was a separate binary actor checkpoint before this Stage 9.6 patch series; it was extended to include critic head in its private format, but it is still not a unified CPU/CUDA production checkpoint.

## Bin Logging Check

The requested duplicated CPU bin count bug was audited. Current `main.cpp` contains one active `dynamics_bin_sample_count[group_key]++` in the per-sample bin logging block; a duplicate increment was not present in the inspected source.

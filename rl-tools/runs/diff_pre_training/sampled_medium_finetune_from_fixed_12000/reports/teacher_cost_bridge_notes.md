# Teacher-Cost Bridge Notes

Run root: `runs/diff_pre_training/sampled_medium_finetune_from_fixed_12000`

These notes only describe the bridge required to test whether differentiable pretraining reduces later teacher cost. They do not claim teacher-cost reduction, RAPTOR replacement, sampled-dynamics success in general, Sim2Real transfer, or full L2F dynamics equivalence.

## Current Compatibility

The differentiable pretraining actor and the post-training student use the same RAPTOR-style dense-GRU-dense actor shape:

- input dense ReLU layer
- GRU layer
- output dense identity layer
- hidden dimension 16
- action dimension from the L2F environment
- observation remains the implicit-identification observation; no explicit physical parameters are added

Relevant definitions:

- `src/foundation_policy/diff_pre_training/config.h`
- `src/foundation_policy/post_training/config.h`

## Current Gap

The gap is checkpoint format and training-entry wiring, not actor architecture.

Differentiable pretraining currently writes an actor-only text checkpoint with magic:

```text
foundation_policy_diff_pre_training_actor_v1
```

It stores:

- input dense weights and biases
- GRU input weights and biases
- GRU hidden weights and biases
- GRU initial hidden state
- output dense weights and biases

Post-training currently loads teacher actors from RLtools HDF5 checkpoint groups named `actor`, and exports/checks code checkpoints through RLtools persistence. Existing post-training save calls use:

```text
rlt::rl::loop::steps::checkpoint::save(...)
```

## Bridge Options

Preferred bridge:

1. Add a small converter executable that compiles the post-training actor type, loads `foundation_policy_diff_pre_training_actor_v1` with the existing diff-pretraining checkpoint reader, and writes an RLtools checkpoint containing the actor group expected by post-training.
2. Add a verification command that loads the converted checkpoint and checks last-step actor outputs against the original diff-pretraining text checkpoint on a fixed batch of observations.

Lower-friction bridge:

1. Add an opt-in post-training `--init-actor-path` argument that loads the diff-pretraining text checkpoint directly into the post-training student actor before dataset/few-teacher updates.
2. Initialize the post-training optimizer from scratch. The diff checkpoint is actor-only and has no optimizer state.
3. Keep existing post-training checkpoint save/export paths unchanged after initialization.

## Few-Teacher Test Design

A fair teacher-cost test should compare fixed-initialized and scratch-initialized post-training runs under identical teacher IDs, seeds, environment settings, and update budgets.

Suggested first sweep:

| condition | teacher counts | initialization |
| --- | --- | --- |
| scratch | 0, 8, 16, 32, 64 | random actor |
| diff-init | 0, 8, 16, 32, 64 | selected fixed/sampled diff checkpoint |

Recommended metrics:

- original L2F fixed-dynamics return/success
- sampled narrow and medium evaluation success
- teacher count needed to hit a predeclared target
- area under success-vs-teacher-count curve
- invalid/NaN rate and skipped update count

Use the same teacher subset order for scratch and diff-init. Do not select teacher IDs separately per condition.

## Current Best Candidate

For a sampled-dynamics bridge starting point, the strongest current candidate is:

```text
runs/diff_pre_training/sampled_medium_finetune_from_fixed_12000/checkpoints/
```

This run reached nonzero sampled success on both Euler and original L2F evaluation for all five tested seeds, with zero invalid rollouts and zero dynamics rejection at the medium level. It is not proof of teacher-cost reduction; it is only a reasonable initialization candidate for the next controlled post-training experiment.


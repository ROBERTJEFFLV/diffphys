# Stage 9.6 Full Production Parity Report

## Implemented In This Pass

- Extended CUDA replay to v2 production-parity schema with:
  initial state, dynamics parameters, dynamics bin ids, group key, group weight, rejection count, references, reset mask, hidden reset mask, sampler seed, and replay metadata.
- Implemented formal CUDA Stage 9.6 gates:
  `--stage9-6-checkpoint-parity`, `--stage9-6-objective-parity`, `--stage9-6-sampler-parity`, and `--stage9-6-eval-parity`.
- Implemented CUDA v4 checkpoint parity for actor, critic, GRU, Adam first/second moments, optimizer age, and metadata.
- Updated objective parity to use replay v2 and balanced group weighting:
  `q_weight = group_weight / horizon`.
- Updated CPU reference objective path inside CUDA parity to use the same group weighting, so diff loss, critic loss, gradients, and Adam update are compared under the same replay weights.
- Added sampler replay validation for per-sample bins, group key reconstruction, group-weight sums, rejection counts, reset masks, and replay metadata.
- Added eval replay validation for same replay initial states/dynamics/actions/references, final state statistics, success/failure classification, action saturation, and loss parity.
- Kept CUDA critic forward/backward, actor backward, closed-loop, action-gradient injection, and Adam parity validations available.

## Verification Run

- CUDA build passed:
  `cmake --build /tmp/raptor_stage96_cuda_build --target foundation_policy_diff_pre_training_cuda -j2`
- Checkpoint parity passed:
  `foundation_policy_diff_pre_training_cuda --stage9-6-checkpoint-parity --seed 68 --save-path /tmp/stage9_6_checkpoint_final_v4.ckpt`
  - saved/loaded: true
  - metadata ok: true
  - weight/moment max abs errors: `0`
- Objective 1000-step replay parity passed:
  `foundation_policy_diff_pre_training_cuda --stage9-6-objective-parity --stage9-6-steps 1000 --batch-size 16 --horizon 8 --seed 65 --stage9-6-replay-path /tmp/stage9_6_objective_1000.bin --log-path /tmp/stage9_6_objective_1000.csv`
  - final CPU/GPU loss: `0.030236/0.030236`
  - max loss abs error: `7.45058e-09`
  - max weight L2 relative error: `2.43338e-06`
  - NaN/Inf count: `0`
- Sampler replay parity passed:
  `foundation_policy_diff_pre_training_cuda --stage9-6-sampler-parity --stage9-6-steps 64 --batch-size 64 --horizon 8 --seed 66 --stage9-6-replay-path /tmp/stage9_6_sampler_4096.bin`
  - samples: `4096`
  - bins balanced: true
  - group weights close: true
  - reset masks replayed: true
  - rejected total: `0`
  - metadata mismatch count: `0`
- Eval replay parity passed:
  `foundation_policy_diff_pre_training_cuda --stage9-6-eval-parity --stage9-6-steps 1000 --batch-size 16 --horizon 8 --seed 67 --stage9-6-replay-path /tmp/stage9_6_eval_1000.bin`
  - final state close: true
  - loss close: true
  - success close: true
  - action saturation close: true
  - max final state abs error: `8.9407e-06`
  - max loss abs error: `4.50611e-05`
  - success mismatch count: `0`
  - NaN/Inf count: `0`
- Final non-replay smoke passed:
  CPU fixed, CPU sampled, CUDA fixed, and CUDA sampled all ran finite 2-step smoke tests and saved checkpoints.

## Remaining Gaps

- No Stage 9.6 parity blockers remain for the CUDA formal gates listed above.
- The repository does not contain a Stage 11w specification or command definition, so Stage 11w execution still needs an explicit task definition.

## Stage 11 Gate

Stage 9.6 PASSED for the formal CUDA CPU/GPU checkpoint, objective, sampler, and eval parity gates.

Stage 11 is allowed, pending a concrete Stage 11w objective/command definition.

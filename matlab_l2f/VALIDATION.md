# Validation report

Validation target: the modified `matlab_l2f` folder in this archive.

## Checks completed in the build container

1. **MATLAB syntax parse**
   - Parsed every `.m` file with the tree-sitter MATLAB grammar.
   - Files parsed: 72
   - Syntax error files: 0

2. **Physics-preservation check**
   - SHA-256 compared the following files against the uploaded archive:
     - `l2f_step.m`
     - `l2f_implicit_midpoint_omega.m`
     - `l2f_so3_exp.m`
     - `l2f_motor_wrench.m`
     - `l2f_sample_dynamics.m`
   - All five are byte-for-byte unchanged.

3. **Numerical equivalence checks**
   - Vectorized observation construction equals the former per-vehicle loop exactly.
   - Pre-transposed MotorGRU weights and broadcast biases equal the former transpose/repmat implementation exactly.
   - O(TN) settle-time calculation equals the former O(T^2N) definition, including NaN handling.
   - Direct non-finite counting equals the former large-array concatenation for every tested prefix.
   - Streaming H500 persistence, RMS/max, action-delta and hidden-state accumulators equal full-array formulas on randomized trajectories.

4. **Memory estimate for N=1024, H=10000**
   - Former full logger: approximately 3.90 GiB before temporary post-processing copies.
   - Streaming evaluator: designed to remain below 100 MiB working storage, excluding MATLAB/runtime overhead and policy weights.

The machine-readable results are in `validation_results.json`.

## MATLAB runtime regression test included

The archive includes:

```matlab
addpath('matlab_l2f/tests');
results = run_fast_eval_tests;
```

This creates deterministic synthetic MotorGRU weights, runs both the full-log and streaming evaluators from the exact same initial state, and compares final physical state, return, success masks, control energy, invalid fraction and all long-horizon summary fields.

The build container does not contain a licensed MATLAB or GNU Octave runtime, so this MATLAB-native regression test could not be executed here. It is included to make the final platform-side verification reproducible before production H10000 evaluation.

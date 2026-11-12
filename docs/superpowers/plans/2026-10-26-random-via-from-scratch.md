# Random START state and bounded PASS targets

Use the current single-agent thread. The user authorizes implementation and fresh training, then explicitly requests continuous local training until they say stop. Keep the original pool/horizon scale, use a new run directory, and do not initialize from a checkpoint. Reuse the existing continuous wrapper, which only removes operational update/time budgets from the unchanged trainer. Do not import the previous adaptive certificate implementation.

## Contract

- Keep START/STOP positions at their original fractions of the original fence.
- Preserve the original sampled velocity, orientation, angular velocity, motors and delayed-velocity history when random initialization is enabled, including the source guidance mixture. Retain zero command placeholder and the task's existing force/pulse disabling.
- Sample PASS inside the cube with per-axis half-width B−7L. Reject overlapping START/PASS/STOP geometry without discarding aircraft or changing the fence.
- Sample complete target poses with tilt inside 60–120 degrees. Reserve the retained 1-degree pose tolerance by sampling 61–119 degrees; randomize azimuth and twist. Use a 1.5 m/s speed floor with no penalty above it.
- Retain Actor, BPTT, physical sampling/groups, original loss formulas and STOP requirements. Read each row's target quaternion in the existing pose loss.
- Reproduce targets from local RNG and stable scene IDs; use discarded original spawn positions as reset entropy. Never seed from retained hidden motors, velocities, attitudes, sensor noise or noise settings.

## Work

1. Add deterministic failing tests for random state preservation, target geometry/tilt, speed threshold, row/seed/noise invariance and per-row loss gradients.
2. Extend task configuration/preparation and per-row target lookup; add an explicit fresh-run preset. Preserve the fixed-task defaults and reject unsupported random specifications.
3. Check replay and native group backward, run CPU regressions and a bounded CUDA smoke check. Update documentation and source-fork checkpoint expectations without editing frozen hashes.
4. Start fresh CUDA training with the checked configuration and a new directory. Observe a completed update and saved checkpoint; report actual progress and budget without claiming learned success.

## Review focus

Target range must describe true thrust-axis tilt, not Euler magnitude. Pose tolerance must not admit tilt outside the requested band. Subminimum speed must remain penalized. Random initial attitude/motor truth must not become public task features. Stable IDs must survive row compaction and replay. Static geometry checks do not prove every random task dynamically feasible.

## Execution evidence

Implementation and review completed. CPU suite: 607 passed, 25 CUDA-only skips. A separate 64-scene H8 CUDA probe confirmed compiled/eager costs and gradients within 3e-4 tolerance. All 3072 TRAIN and 256 EVAL members passed static task preparation. Actor source and 23 protected source/config/contract files remain unchanged; the original loss body differs only in reading per-row target quaternions.

Continuous CUDA training is running in `runs/pass_stop/random_via_from_scratch_seed7/`, PID 2758097. The update-zero checkpoint matches the independently reproduced seed-7 random Actor digest and has empty Adam state. At 2026-10-26 15:15:01 UTC, update 35 was saved with finite gradients and all 128 physical groups. The user explicitly requested continued training until stop; do not stop it at the preset's original 50-update/1800-second limits. Exact launch and live evidence are in `launch.json` and `docs/random_via_validation.json`.

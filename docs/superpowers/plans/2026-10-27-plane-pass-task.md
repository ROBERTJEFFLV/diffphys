# Plane PASS task implementation plan

**Goal:** Replace the complete-pose PASS target with an unoriented body-plane target and terminal-directed normal velocity, and change the task geometry to 50L/40L/30L cubes.

**Execution:** Native single-agent thread. The user confirmed that this plane task replaces the old 60–120 degree tilt requirement and authorized continued local training until stop. Preserve the previous run and its exact source snapshot; initialize the new task explicitly from its Actor weights with fresh Adam.

**Architecture:** Keep the original physical sampler, RK4, noise, GRU69/64/4, BPTT, grouping and optimizer implementation. Task preparation sets a 50L position limit and samples START/STOP on the 40L cube surface and PASS inside the 30L cube. A task helper supplies canonical public goal frames and geometry-only plane/direction ratios. Extend the existing loss for this explicit task mode and keep legacy pose mode behavior.

**Contract:** The plane normal is parallel to STOP−PASS. Body normals ±n are equivalent in the task loss, and rotation about the normal is free. The physical thrust vector remains directed along actual body +z. PASS speed is at least 1.5 m/s, directed towards STOP, and perpendicular to the actual body plane, using the retained angular tolerance. START keeps the original random kinematics/motors; STOP retains its existing thresholds. First measured exit still switches to STOP without a success gate. A new objective/source binding records the fork.

**Review focus:** Plane geometry is not a symmetry of positive-thrust dynamics. Reject backwards, lateral and subminimum crossings. Avoid zero-speed division and singular 180-degree quaternion construction. Keep all actual coverage128 members and scene/noise identities. Source family fence changes are explicit task preparation, never a hidden aircraft replacement. Verify replay labels and actual targets agree with training.

- [x] Save and stop the previous trainer/exporter; confirm the final checkpoint and archived source.
- [x] Add failing tests for plane sign/spin invariance, velocity direction/orthogonality, finite gradients, cube geometry and actual members.
- [x] Implement helpers, task configuration/preparation, loss and checkpoint source binding; add the new preset.
- [x] Run task/replay/group tests, CPU regressions and bounded CUDA gradient comparison; prepare full production pools and review the diff.
- [x] Launch continuous training with explicit weights-only initialization and switch GUI/replay to the new run; verify actual updates and checkpoints.

## Final crossing semantics and evidence

The production preset enables `plane_crossing_only=true`: score the first true plane intersection within the first measured 2L traversal, including forward crossing, and permit pose/velocity changes outside that intersection. The original whole-ball plane mode remains an explicit reference. This choice follows the stated interpretation of passing through a plane after optional clarification; the source and checkpoints of the whole-ball trial are retained. The horizontal-crossing gravity example is recorded in `reports/plane_task_20261027/whole_ball_direction_coupling.json` and is not a general impossibility proof.

Final verification: 634 CPU tests passed with 25 CUDA-only skips, separate 64-scene H8 CUDA cost/gradient equivalence passed, and an actual-hit CUDA plane-intersection gradient test passed. Actor/training/backward AST and 22 protected files were reviewed unchanged. Geometry preparation preserves all 3072/256 source members.

Continuous training runs at `runs/pass_stop/plane_crossing_to_stop_seed7/`, PID 2955684. At 2026-10-27 05:16:13 UTC, update 72 was saved with finite native one-pass gradients across 128 groups. Initial weights match the previous trial's Actor exactly and initial Adam is empty. GUI and replay point to this run; CUDA replay update 60 matches its logged evaluation. Evidence is in `docs/plane_task_validation.json`. Learned task success is not claimed; the process continues until the user requests stop.

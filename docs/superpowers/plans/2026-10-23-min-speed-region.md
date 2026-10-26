# Region minimum-speed and geometry fixes

**Goal:** Replace constant-vector PASS speed supervision with a strict scalar speed floor, fix FP32 crossing fractions/gradients, and bring the saved-flight display up to date.

**Scope:** Native RK4, noise, Actor69 layout, group clipping, Time Decay implementation, and Adam remain unchanged. No formal training or GPU load test. The current three-point entry/traverse/exit state machine remains; initial and terminal target velocities remain zero. The PASS vector norm defines the minimum speed (default 2 m/s); its direction is not a region constraint.

- [x] Tests first: millimetre crossings over FP32/FP64 and scales; analytic position gradients; speeds 1.9/2/3.5 m/s and alternate directions; first-visit retention; replay stage/radius/floor telemetry.
- [x] Repair only degenerate intersection branches, preserving all legitimate positive discriminants.
- [x] Version the new objective; replace vector errors with normalized speed shortfall everywhere (loss, recovery, temporal proxy, success, diagnostics). Remove the mean-velocity target and its obsolete configuration.
- [x] Export actual Actor phase, scalar minimum speed and per-aircraft region radius. Validate new replay fields; retain historical replay support. Display region/center radii, phase and lower-bound speed separately.
- [x] Update affected tests/docs, verify numerical/gradient and replay behavior, run all CPU regressions, and inspect a Pygame screenshot. Record protected-core hashes and limitations.

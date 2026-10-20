# Fixed-point task update

For the event-waypoint task, the latest user request supersedes the root document's older S9/continuous-time-target instructions. `waypoint_*.py` implements fixed commands, 0.1 m measured-position switches, H500, force-free initial velocity, native physical BPTT, event constraints, per-bank augmented-Lagrangian duals, and Gaussian hard-program control-variate gradients. See `docs/waypoint_training.md`.

Do not reintroduce moving references, performance curricula, attitude-gated switches, silent tilt downgrades, detached physical/GRU state at switches, or stage-dependent exemptions from final stopping. The new uniform scalar objective intentionally does not use physical-group gradient reweighting; original hover/group modules remain unchanged.

Keep `env_raptor.py`, `compiled_physics.py`, `strict_inductor.py`, original sensors and physical sampler numerically intact. Fractional RK4 is audit-only. Source-bound old runs require their original source. Changes in task/schema cannot be disguised as exact resume.

Do not delegate, spawn agents, or start long training without explicit user authorization. Use targeted deterministic tests; distinguish code correctness, derivative correctness, learning outcomes and deployment safety. No finite loss or limited test suite certifies all sampled aircraft or safe deployment.

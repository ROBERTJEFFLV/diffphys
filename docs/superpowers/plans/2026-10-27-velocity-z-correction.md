# Correct velocity orthogonal to body z

User clarification: velocity must be perpendicular to body z, directed toward STOP, with a 1.5 m/s floor. Keep the current Actor, training algorithm, geometry bounds, random initial states, timing weight 10 and STOP conditions.

1. Preserve the old source and checkpoint, then stop the mistaken training objective cooperatively.
2. Regress target z dot heading = 0 across both dtypes and all physical groups; regress acceptance at 90 degrees and rejection at parallel/antiparallel. Choose the upward projection perpendicular to travel, with world-x fallback for exactly vertical travel.
3. Correct target generation and the actual-velocity/body-z score. Keep the navigation crossing plane normal equal to the STOP heading; this plane is distinct from the aircraft plane.
4. Update the viewer, regress its target/current angle distinction, and keep old replay semantics readable.
5. Run deterministic CPU regressions, a small CUDA forward/backward check and the original-size pool geometry check. Review the actual diff.
6. Start a new random Actor and Adam in a new run directory at the original 3072/128/24, EVAL256, H500 scale; verify checkpoint provenance and live updates. Continue until the user says stop.

Review focus: exactly vertical headings; flipped normals; wrong parallel acceptance; actual orthogonality despite separate pose/direction tolerances; missing/backward/late crossings. No delegation or deployment.

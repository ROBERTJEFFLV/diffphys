# Apply pose/velocity region patch

**Goal:** Apply the supplied `aft_pose_velocity_patch_42e5f17.zip` to the current AFT checkout, retaining local checkpoint/replay fixes and all existing runs.

**Architecture:** Verify the original 14-file payload and nine protected core files on its exact baseline in a temporary local checkout. Three-way merge the verified payload with the current working files. The new protocol uses GRU69 and first-region-entry/exit semantics; old runs remain bound to their archived source.

**Spec:** Supplied archive README_CN.md, manifest.json, and changed_files/docs/pass_stop.md. The latest explicit patch request supersedes earlier proposed removal of all waypoint switching.

**Constraints:** Single agent; no new training/GPU jobs, no checkpoint hash rewrites, no commit/push. Keep existing runtime source/checkpoint provenance. Preserve the patch's zero spawn velocity, region radius 2L, midpoint velocity (2,0,0), and time-decay default 0.

## Tasks

- [x] Back up the working source and local diff; verify protected hashes and apply the guarded package to its pristine baseline.
- [x] Merge response_training.py while retaining local diagnostics fixes; accept the explicitly changed region configuration and update related documentation.
- [x] Run supplied region/state/feature tests and bridge numerical tests. Check the existing typed replay against the new public goal protocol; add the smallest failing regression for required adapters before editing them.
- [x] Run the repository CPU tests, investigate failures by tracing the new schema/fields, and repair integration errors without changing protected kernels or the requested task.
- [x] Verify final payload/protected hashes, record deliberate local integration differences, and report checks and unmeasured GPU/training limits.

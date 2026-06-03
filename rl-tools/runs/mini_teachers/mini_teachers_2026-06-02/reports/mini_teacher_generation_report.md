# Mini Teacher Generation Report

Experiment: `mini_teachers_2026-06-02`
Teacher training budget: `10000` SAC environment steps per teacher; final checkpoint saved manually at step 10000.
Teacher dynamics domain: `medium` nominal-centered sampled dynamics.
Artifact layout matches post-training `find_latest_run` loader.

Dynamics JSON count: `32`
Trained teacher checkpoint count at step 10000: `32`
Teacher index: `/home/lvmingyang/raptor_diff_20260531_215150/diffphys/rl-tools/runs/mini_teachers/mini_teachers_2026-06-02/checkpoints_mini_teachers_2026-06-02.txt`
Dynamics manifest: `/home/lvmingyang/raptor_diff_20260531_215150/diffphys/rl-tools/runs/mini_teachers/mini_teachers_2026-06-02/reports/mini_teacher_dynamics_manifest.csv`
Training manifest: `/home/lvmingyang/raptor_diff_20260531_215150/diffphys/rl-tools/runs/mini_teachers/mini_teachers_2026-06-02/reports/mini_teacher_training_manifest.csv`
Logged pre-training eval step: `0`
Logged return mean over teachers: `-69.3314`
Logged return min/max: `-71.8421` / `-64.761`

Teacher index order: fixed dynamics IDs `0..31`; no final-return sorting is claimed because the shortened 10000-step run only logs the initial pre-training evaluation before the manual final checkpoint.

Compatibility smoke: 8 teachers loaded successfully; post-training produced 32 minibatches in 1 epoch with finite loss 0.0296194 after short-episode batching compatibility was added.

Scope: local mini-teacher artifacts only, not official RAPTOR teacher artifacts.

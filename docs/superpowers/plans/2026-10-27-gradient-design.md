# Repair gradient incentives with controlled local experiments

Keep GRU69→64→4, native physics/noise, all 128 physical groups, task geometry,
speed≥1.5, velocity perpendicular body z, event/STOP hard criteria, H500,
Time Decay=0, native group probe and shrink-only clipping. User authorizes loss
gradient design experiments and repair; no delegation or deployment.

1. Archive source/checkpoint and stop the old producer before edits.
2. Regress radial monotonicity of angular scoring, finite derivatives at zero
   and the floor, and unchanged valid-speed angular acceptance in both dtypes.
3. Extend low-speed direction scoring to its worst angular error; extend
   orthogonality scoring likewise. At valid speeds use the original angular
   metric. Lower speed may increase error, never reduce it at fixed pose/heading.
4. Compare baseline, angular-only repair, and ordered robust penalties. Let
   P be weighted log1p squared PASS violations and S the corresponding STOP
   aggregate. Use P+log(1+S/(1+P)); its partial derivatives are nonnegative,
   so degrading PASS cannot disable an expensive STOP penalty.
5. Probe total/time/PASS/STOP parameter derivatives on all 128 physical cells,
   including contributions after the same total-gradient cell clipping.
6. Run paired 50-update CUDA pilots: same parent Actor, new Adam, seed/tapes,
   512 TRAIN scenes, 256 fixed EVAL, H500, lr0.001, time weight3. Compare hard
   metrics and skipping/slowdown behavior; do not compare unlike scalar losses.
7. Review the diff, run focused/full CPU and bounded CUDA checks, restore the
   original 3072/128×24 continuous training and GUI with explicit provenance.

Review focus: zero speed, floor boundary, reverse/sideways velocity, sign-flipped
body normal, valid set/tolerances, first failed crossing, bypassing PASS,
per-cell clipping, source-bound checkpoints, and numerical failures.

# Seed 7 training execution

This run starts from `codex/safe-probe-v5` commit
`859dda49ffc9e8dd1f5cade3704c5eca01e62296`. Historical v4 failures remain
failures. Only the final post-MS paired migration gate can authorize the
updated controller. A training checkpoint, development gate, or privileged
physics ceiling is not a deployment artifact.

## Measurements and budgets

`tools/profile_structured_training.py` measures the production observation,
executed-action replay, motor observer and A1 backward pass. On the RTX 4060
Ti with PyTorch 2.2.2+cu121, B64/H126 collection took about 3.9 seconds and
one backward/update about 1.1 seconds; peak allocated memory was 195 MiB.
CPU collection/update was about 2.4/0.8 seconds. CUDA is retained for this run.
These measurements do not predict the cost of JVP/MS solves: their actual
timings and solver iterations must be reported when their prerequisites pass.

| Stage | Maximum updates | Walltime limit |
|---|---:|---:|
| identifier pretrain | 3000 | 10800 s |
| A1 | 5 x 120 | 3600 s |
| A2 | 5 x 60 | 1800 s |
| B | 300 | 1800 s |
| residual oracle | 400 | 1800 s |
| C | 5 x 120 | 3600 s |
| 2xH250 MS | 12 outer attempts | 1800 s |
| 4xH250 MS | 12 outer attempts | 1800 s |

The total pipeline limit is 28800 seconds. Nonfinite updates stop immediately.
Identifier pretraining checks its separate development bank every 50 updates,
requires at least 300 updates, and stops after 12 checks without 0.2% relative
improvement in the worst normalized physical acceptance metric. Passing that
development gate also allows stopping. The numerical acceptance thresholds
are unchanged. DAgger preserves all five beta rounds and its two final beta=0
rounds. MS retains its original rejection, KKT, trust-region, restoration and
nonlinear acceptance rules. Upper budgets are limits, not promised completed
updates. Every report must use actual update counts.

## Data separation

The model seed is always 7. Scenario seeds are independent dataset identifiers;
the registered 4x4 authority cells and scenario counts are retained. Scenario
generation no longer reseeds the model RNG.

The probe registration retains training seeds 3707/4707/5707/6707 and its
single validation seed 7707. K35 collection now consumes only its registered
training seeds 13707/14707/15707/16707. Its validation/final/blind seeds remain
reserved; no production mean-accuracy claim comes from this collection.

| Stage | Development scenario seeds | Final/calibration scenario seeds |
|---|---|---|
| identifier | 2000007 | validation 10007, final 20007 |
| A1 | 2200007 | 100008/100009/100010 |
| A2 | 2210007/2210008 | 110008/110009/110010 |
| causal revalidation | none | 30008/40009 |
| B | 2200007 | 120008 |
| residual oracle | 2300007 | 140009 |
| C | 2230007 | 130008/130009/130010 |
| calibration | calibration 500007..500010 | terminal validation 500110 |
| postcheck | none | 510007 |
| paired migration | none | 520007/520008 |
| MS acceptance | 1000010 | not final release data |
| post-MS calibration | calibration 700007..700010 | terminal validation 700110 |
| post-MS postcheck | none | 710007 |
| post-MS migration | none | 720007/720008 |

Existing A1/A2/C/B/calibration configurations reused small +1/+2/+3 offsets
across stages and overlapped the pretraining training range. Those overlapping
banks cannot be called independent final validation. The table preregisters
separate final namespaces while preserving the bank layout and stage purposes.
Development banks may be reused for diagnosis and are never called final.

## Resumption and final evaluation

Each trainer writes `*.training.pt` containing the complete model, optimizer
(or MS solver state), Python/NumPy/Torch/CUDA random states, completed update
count, elapsed time and phase progress. DAgger includes its replay buffer and
round position; MS includes theta, boundaries and trust-region state. Saves
use a temporary file, fsync and atomic replacement. SIGINT/SIGTERM request a
stop at a complete update boundary. Repeating the same command resumes matching
progress. Runtime/source/config bindings reject stale checkpoints.

Development reports have a `_development.json` suffix and cannot authorize
downstream stages. Only a passing development candidate is frozen to a hashed
`*.candidate.pt` record. The runner then invokes the same tool with
`--final-evaluation`, which performs no optimizer updates and exclusively
claims its final dataset before collection. Claims are under
`reports/structured_seed7_final_claims/`. A failed final bank is not reused
for tuning or presented as independent on another attempt. This remains a
workspace audit, not a cross-machine anti-reuse service.

The runner records exact commands, per-stage logs, walltime, output digests,
gates and reuse decisions in `execution_manifest.json`. It only reuses a
passed, matching result, and stops on a failed current-stage gate.

## Training prerequisites repaired before protocol freezing

The allocator still solves all 81 box-QP faces, now using one batched linear
solve. The strict allocator continues to reject incompatible supplied bounds.
The production policy explicitly handles an inherited command outside its
nominal startup trim box: its temporary trust radius contains the command
already executed, cannot increase that deviation, and retains the hard motor
box and slew bound. `inherited_command_recovery` reports this case. Once the
nominal box is reached the original radius applies. Passive Q2 commands,
simulator equations and the +5% paired probe safety gate remain unchanged.

The first valid disturbance window initializes the force estimate without
blending with an unmeasured zero prior. Subsequent windows retain the original
low-pass cadence. A1 now explicitly supervises the differentiable production
motor observer, using detached simulator motor targets only in the loss.
Privileged motor/parameter/force targets never enter deployment observations.
The A1 identifier-loader keyword is passed to the provenance validator that
actually owns it.

## Commands

```bash
python tools/profile_structured_training.py --output runs/structured_seed7_v5/profile.json
python tools/diagnose_probe_v5.py --formal-freeze --output reports/probe_v5_formal.json --n-jobs 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u tools/run_structured_pipeline.py \
  --stage all --device cuda --work-dir runs/structured_seed7_v5 --max-total-seconds 28800
```

Run formal freezing only after the source has been fixed and tests pass. A
previous claim must be inspected and preserved, not removed to get another
validation attempt. Actual run outcomes belong in the generated execution
report; the budget table is not an assertion of successful training.

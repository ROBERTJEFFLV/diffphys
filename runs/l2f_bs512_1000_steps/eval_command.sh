#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/L2F-in-seconds"
export CUDA_HOME=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:$PATH
PY=/home/lvmingyang/miniconda3/envs/barn-env/bin/python
RUN_ROOT="$HOME/L2F-in-seconds/runs/l2f_bs512_1000_steps"
$PY train.py \
  --eval-only \
  --device cuda \
  --sim-backend cuda \
  --seed 10000 \
  --eval-seeds "10000,10001,10002,10003,10004,10005,10006,10007,10008,10009,10010,10011,10012,10013,10014,10015,10016,10017,10018,10019" \
  --horizon 500 \
  --batch-size 512 \
  --tail-start-step 450 \
  --trajectory-count 5 \
  --log-path "$RUN_ROOT/eval.csv" \
  --trajectory-path "$RUN_ROOT/trajectories.csv" \
  --checkpoint-path "$RUN_ROOT/step1000_checkpoint.pt"

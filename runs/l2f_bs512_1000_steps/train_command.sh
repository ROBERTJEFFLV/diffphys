#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/L2F-in-seconds"
export CUDA_HOME=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:$PATH
PY=/home/lvmingyang/miniconda3/envs/barn-env/bin/python
RUN_ROOT="$HOME/L2F-in-seconds/runs/l2f_bs512_1000_steps"
$PY train.py \
  --device cuda \
  --sim-backend cuda-full \
  --seed 7 \
  --horizon 500 \
  --batch-size 512 \
  --steps 1000 \
  --state-grad-decay 0.5 \
  --hidden-grad-decay 0.7 \
  --tail-steps 50 \
  --tail-start-step 450 \
  --log-every 10 \
  --save-every 0 \
  --log-path "$RUN_ROOT/train.csv" \
  --checkpoint-path "$RUN_ROOT/step1000_checkpoint.pt"

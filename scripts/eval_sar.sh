#!/bin/bash
# scripts/eval_sar.sh

WEIGHT_PATH="./checkpoints/best_model_weight.pth"

echo "Starting KIDA Evaluation using weights: ${WEIGHT_PATH}"

python test.py \
    -wp "./checkpoints/best_model_weight.pth" \
    -w 4 \
    -gpu_id "cuda:0"
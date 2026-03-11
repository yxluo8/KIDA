#!/bin/bash
# scripts/train_sar.sh

echo "Starting KIDA Training on SAR-RARP50..."

python train.py \
    -exp "KIDA_SAR" \
    -lb 0.1 \
    -bs 1 \
    -e 120 \
    -l 2e-5 \
    -w 4 \
    -gpu_id "cuda:0"
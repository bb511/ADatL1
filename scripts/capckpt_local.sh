#!/bin/bash
python3 src/capckpt.py \
    experiment=axov4 \
    ckpt_path=/data/deodagiu/adl1t/logs/train/multiruns/2025-07-09_23-13-30/0/checkpoints/ \
    model.loss.alpha=0.6 \
    data.batch_size=1024 \
    logger=none \
    +gpu_nb=3 \
    +save_results=true

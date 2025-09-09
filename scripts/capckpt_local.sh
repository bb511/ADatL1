#!/bin/bash
python3 src/capckpt.py \
    experiment=axov4 \
    ckpt_path=/data/deodagiu/adl1t/logs/train/multiruns/2025-07-09_23-14-05/0/checkpoints/ \
    model.loss.alpha=0.8 \
    data.batch_size=1000 \
    logger=none \
    +gpu_nb=1 \
    +save_results=true
    # logger=none \

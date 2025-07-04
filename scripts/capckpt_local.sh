#!/bin/bash
python3 src/capckpt.py \
    experiment=axov4 \
    ckpt_path=/data/deodagiu/adl1t/logs/train/multiruns/2025-07-04_16-38-21/0/checkpoints/ \
    model.loss.alpha=0.6 \
    data.batch_size=1000 \
    # logger=none \

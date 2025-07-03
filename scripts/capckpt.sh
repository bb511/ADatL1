#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=4:00:00

# srun python3 src/train.py \
python3 src/capckpt.py \
    experiment=axov4 \
    ckpt_path=2025-07-03_01-15-11 \
    model.loss.alpha=0.5 \
    logger=none \
    data.batch_size=100 \

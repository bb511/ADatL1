#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=4:00:00

# python3 src/capckpt.py \
srun python3 src/capckpt.py \
    experiment=axov4 \
    ckpt_path=/cluster/home/vjimenez/ADatL1/logs/train/multiruns/2025-07-03_01-15-11/0/checkpoints \
    model.loss.alpha=0.5 \
    data.batch_size=100 \
    # logger=none \

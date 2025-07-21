#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=4:00:00

# srun python3 src/capckpt.py \
python3 src/capckpt.py \
    experiment=axov4 \
    checkpoint_filter.model.loss.alpha=0.0 \
    data.batch_size=100 \
    logger=none \
    trainer=cpu

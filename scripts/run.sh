#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=24:00:00

srun python3 src/train.py \
    --multirun \
    experiment=vicreg \
    trainer=gpu \
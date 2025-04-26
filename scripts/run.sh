#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=24:00:00

# srun python3 src/train.py \
python3 src/train.py \
    --multirun \
    experiment=vicreg \
    trainer=cpu \
    trainer.limit_train_batches=0.0000001 \
    trainer.limit_val_batches=0.00000001 \
    trainer.limit_test_batches=0.0000001 \
    trainer.max_epochs=1 \
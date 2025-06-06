#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=30G
#SBATCH --time=24:00:00

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    experiment=qvae \
    trainer=gpu \
    model.loss.alpha=0.1,0.2,0.3 \
    model.loss.beta=0.1,0.2,0.3 \ 
    trainer.limit_train_batches=0.0000001 \
    trainer.limit_val_batches=0.00000001 \
    trainer.limit_test_batches=0.0000001 \
    trainer.max_epochs=1 \
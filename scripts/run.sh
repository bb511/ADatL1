#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=24:00:00

# python3 src/train.py \
srun python3 src/train.py \
    --cfg=job \
    experiment=qvae \
    trainer=gpu \
    model.loss.alpha=0.5 \
    trainer.limit_train_batches=1.23e-4 \
    trainer.limit_val_batches=9.8261e-4 \
    trainer.limit_test_batches=1.46e-3 \
    trainer.max_epochs=5 \
    trainer.num_sanity_val_steps=0 \
    data.num_workers=0
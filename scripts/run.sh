#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=4:00:00

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    experiment=axov4 \
    trainer=gpu \
    model.loss.alpha=0.5736625791372814 \
    trainer.max_epochs=10 \
    +trainer.num_sanity_val_steps=0 \
    +experiment_id=null \
    # data.batch_size=100 \
    # logger=none \
    # data.loader.num_workers=1 \
    # trainer.limit_train_batches=1.23e-4 \
    # trainer.limit_val_batches=9.8261e-4 \
    # trainer.limit_test_batches=1.46e-3 \
    # +callbacks@callbacks.approximation_capacity=cap \
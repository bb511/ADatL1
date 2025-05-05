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
    experiment=vicreg \
    trainer=ddp \
    trainer.devices=4 \
    trainer.max_epochs=100 \
    data.batch_size=16384 \
    data.num_workers=50 \
    # trainer.limit_train_batches=1.23e-5 \
    # trainer.limit_val_batches=9.8261e-5 \
    # trainer.limit_test_batches=1.46e-4 \

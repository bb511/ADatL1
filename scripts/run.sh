#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    experiment=axov4 \
    callbacks=basic \
    paths.raw_data_dir=data/adl1t_data/parquet_files \
    algorithm=svdd \
    trainer=gpu \
    data.batch_size=16276 \
    trainer.max_epochs=1 \
    +trainer.limit_train_batches=2 \
    +trainer.limit_val_batches=2 \
    +trainer.limit_test_batches=2 \
    # algorithm.loss.kl.scale=0.4 \

    # data.batch_size=100 \
    # logger=none \
    # data.loader.num_workers=1 \
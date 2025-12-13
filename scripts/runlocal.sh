#!/bin/bash
#SBATCH --mem-per-cpu=30G

srun python3 src/train.py \
    --multirun \
    experiment=axov4 \
    callbacks=basic \
    paths.raw_data_dir=data/adl1t_data/parquet_files \
    algorithm=svdd \
    trainer=cpu \
    data.batch_size=16 \
    trainer.max_epochs=2 \
    +trainer.limit_train_batches=1 \
    +trainer.limit_val_batches=1 \
    +trainer.limit_test_batches=1 \
    logger=none \
    # algorithm.loss.kl.scale=0.4 \

    # data.batch_size=100 \
    # logger=none \
    # data.loader.num_workers=1 \
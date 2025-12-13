#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00

srun python3 src/train.py \
    --multirun \
    experiment=vae \
    paths.raw_data_dir=data/adl1t_data/parquet_files \
    trainer=gpu \
    trainer.max_epochs=150 \
    data.batch_size=4096,16276,65536 \
    algorithm.optimizer._target_=torch.optim.SGD,torch.optim.AdamW \
    algorithm.optimizer.lr=1e-5,1e-4,1e-3 \
    algorithm.optimizer.weight_decay=1e-5,1e-4,1e-3,1e-2 \
    algorithm.loss.scale=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    # trainer.max_epochs=1 \
    # +trainer.limit_train_batches=2 \
    # +trainer.limit_val_batches=2 \
    # +trainer.limit_test_batches=2 \
    # logger=none
    # algorithm.loss.kl.scale=0.4 \

    # data.batch_size=100 \
    # logger=none \
    # data.loader.num_workers=1 \
# Short debug script.
taskset -c 0-2 \
python3 src/train.py \
    -m \
    experiment=debug \
    experiment_name=debug \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    trainer=gpu \
    trainer.max_epochs=1 \
    hparams_search=vicreg_optuna \
    hydra.sweeper.study_name=test \
    +trainer.limit_train_batches=10 \
    trainer.devices=[0]

    # logger=none
    # hydra/launcher=submitit_local \
    # hydra.launcher.cpus_per_task=1 \
    # hydra.launcher.gpus_per_node=4 \

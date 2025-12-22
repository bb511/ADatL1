# taskset -c 0-1 \
# python3 src/train.py \
#     -m \
#     experiment=debug \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     data.val_batches=4
    # hparams_search=vicreg_optuna \

taskset -c 5-6 \
python3 src/train.py \
    -m \
    hydra/launcher=submitit_local \
    hydra.launcher.cpus_per_task=1 \
    hydra.launcher.gpus_per_node=4 \
    hparams_search=vicreg_optuna \
    experiment=vicreg \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    data.val_batches=4 \
    trainer.devices=[3]
    # trainer=cpu

# taskset -c 0-1 \
# python3 src/train.py \
#     -m \
#     hparams_search=vicreg_optuna \
#     experiment=debug \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     data.val_batches=4

taskset -c 37-39 \
python3 src/train.py \
    -m \
    hydra/launcher=submitit_local \
    hydra.launcher.cpus_per_task=1 \
    hydra.launcher.gpus_per_node=4 \
    hparams_search=vicreg_optuna \
    experiment=vicreg \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    data.val_batches=4 \
    trainer.devices=[2]
    # trainer=cpu

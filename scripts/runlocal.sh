# Short debug script.
# taskset -c 0-9 \
python3 src/train.py \
    -m \
    experiment=debug \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    trainer=gpu \
    trainer.devices=[0]
    # hparams_search=vae_optuna \
    # hydra/launcher=submitit_local \
    # hydra.launcher.cpus_per_task=1 \

# vicreg hyperparameter search on gpu.
# taskset -c 37-39 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     hparams_search=vicreg_optuna \
#     experiment=vicreg \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     data.val_batches=4 \
#     trainer.devices=[2]

# vae hyperparameter search on cpu.
# taskset -c 60-69 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hparams_search=vae_optuna \
#     experiment=vae \
#     experiment_name=vae_search_broad_bs16k \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files

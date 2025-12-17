# taskset -c 0-9 \
python3 src/train.py \
    -m \
    hparams_search=vae_optuna \
    experiment=debug \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files

# taskset -c 84-95 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hparams_search=vae_optuna \
#     experiment=vae \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     trainer=cpu

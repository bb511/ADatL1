# Short debug script.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     experiment=debug \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     trainer=gpu \
#     trainer.devices=[0] \
    # hparams_search=vicreg_optuna \
    # hydra/launcher=submitit_local \
    # hydra.launcher.cpus_per_task=1 \
    # hydra.launcher.gpus_per_node=4 \

# vicreg hyperparameter search on gpu.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     hparams_search=vicreg_optuna \
#     experiment=vicreg \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     trainer=gpu \
#     trainer.devices=[0]

# vae hyperparameter search on gpu.
# taskset -c 15-17 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     hparams_search=vae_optuna \
#     experiment=vae \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     trainer=gpu \
#     trainer.devices=[1]

# vae gpu bs study run.
# 4096 8192
taskset -c 24-31 \
python3 src/train.py \
    -m \
    experiment=vae \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment_name=vae_batch_size_study \
    trainer.gradient_clip_val=1.0 \
    algorithm.optimizer.betas='[0.95, 0.999]' \
    algorithm.optimizer.eps=1e-08 \
    +lr=0.0009388 \
    algorithm.loss.kl_scale=0.0020871 \
    algorithm.kl_warmup_frac=0.05 \
    data.batch_size=49152

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

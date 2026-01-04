# Short debug script.
taskset -c 0-2 \
python3 src/train.py \
    -m \
    experiment=debug \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    trainer=gpu \
    trainer.devices=[0] \
    # hparams_search=vicreg_optuna \
    # hydra/launcher=submitit_local \
    # hydra.launcher.cpus_per_task=1 \
    # hydra.launcher.gpus_per_node=4 \

# VICreg runs.
# ========================================================================
# vicreg hyperparameter search.
# taskset -c 21-23 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     hparams_search=vicreg_optuna \
#     experiment=vicreg \
#     experiment_name=vicreg_search \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     trainer=gpu \
#     trainer.devices=[3]

# vicreg bs study.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     experiment=vicreg \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment_name=vicreg_feats_ckpt \
#     trainer.gradient_clip_val=2.0 \
#     +lr=0.006496 \
#     algorithm.optimizer.weight_decay=3.5e-06 \
#     algorithm.optimizer.betas='[0.9, 0.999]' \
#     algorithm.scheduler.scheduler.warmup_ratio=0.05 \
#     algorithm.scheduler.scheduler.min_lr_ratio=0.01 \
#     algorithm.feature_blur.prob=0.101196 \
#     algorithm.feature_blur.magnitude=0.112294 \
#     algorithm.feature_blur.strength=0.27725 \
#     algorithm.object_mask.prob=0.0479 \
#     algorithm.lorentz_rotation.prob=0.0 \
#     algorithm.loss.var_coef=75 \
#     algorithm.loss.cov_coef=17.5 \
#     algorithm.model.out_dim=16 \
#     trainer=gpu \
#     trainer.devices=[0]
    # data.batch_size=4096,8192,16384,32768,49152


# VAE runs.
# ========================================================================
# vae hyperparameter search.
# taskset -c 15-17 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     hparams_search=vae_optuna \
#     experiment=vae \
#     experiment_name=vae_search \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     trainer=gpu \
#     trainer.devices=[1]

# vae bs study.
# taskset -c 10-19 \
# python3 src/train.py \
#     -m \
#     experiment=vae \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment_name=vae_batch_size_study \
#     trainer.gradient_clip_val=1.0 \
#     algorithm.optimizer.betas='[0.9, 0.999]' \
#     algorithm.optimizer.eps=1e-08 \
#     +lr=0.000307997 \
#     algorithm.loss.kl_scale=0.0001222 \
#     algorithm.kl_warmup_frac=0.3 \
#     data.batch_size=16384,32768,49152

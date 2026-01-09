# Short debug script.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     experiment=debug \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     trainer=gpu \
#     trainer.devices=[0] \
    # hparams_search=vae_optuna \
    # hydra/launcher=submitit_local \
    # hydra.launcher.cpus_per_task=1 \
    # hydra.launcher.gpus_per_node=4 \

# VICreg runs.
# ========================================================================
# vicreg hyperparameter search.
taskset -c 21-23 \
python3 src/train.py \
    -m \
    hydra/launcher=submitit_local \
    hydra.launcher.cpus_per_task=1 \
    hydra.launcher.gpus_per_node=4 \
    hparams_search=vicreg_optuna \
    experiment=vicreg \
    experiment_name=vicreg_b16k_archold_inputsv4_hpsearch \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    trainer=gpu \
    trainer.devices=[3]

# vicreg training.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     experiment=vicreg \
#     experiment_name=vicreg_best_models \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.optimizer.lr=0.0003814576334244987 \
#     algorithm.optimizer.weight_decay=0.0002733098092067108 \
#     algorithm.optimizer.betas=[0.95,0.99] \
#     algorithm.scheduler.scheduler.warmup_ratio=0.03 \
#     algorithm.scheduler.scheduler.min_lr_ratio=0.1 \
#     algorithm.feature_blur.prob=0.11487916123917022 \
#     algorithm.feature_blur.magnitude=0.2853105068240043 \
#     algorithm.feature_blur.strength=0.5719591690886635 \
#     algorithm.object_mask.prob=0.0016834348224807894 \
#     algorithm.lorentz_rotation.prob=0.6 \
#     algorithm.loss.inv_coef=7.663054422626061 \
#     algorithm.loss.var_coef=75.46034266179656 \
#     algorithm.loss.cov_coef=13.160806547367477 \
#     algorithm.model.out_dim=16 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     algorithm.diagnosis_metrics=true \
#     trainer=gpu \
#     trainer.devices=[0]

# VAE runs.
# ========================================================================
# vae hyperparameter search.
# taskset -c 3-5 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     hparams_search=vae_optuna \
#     experiment=vae \
#     experiment_name=vae_embvicreg_b16k_search \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     trainer=gpu \
#     trainer.devices=[0]

# vae training.
# taskset -c 3-5 \
# python3 src/train.py \
#     -m \
#     experiment=vae \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment_name=vae_raw_best \
#     trainer.gradient_clip_val=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.eps=1e-08 \
#     algorithm.optimizer.lr=0.0013038046803867584 \
#     algorithm.loss.kl_scale=0.0003813132147973469 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.encoder.out_dim=16 \
#     trainer=gpu \
#     trainer.devices=[1]
    # algorithm.optimizer.weight_decay=0.00019832498920758287 \

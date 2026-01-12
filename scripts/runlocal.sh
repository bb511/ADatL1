# Short debug script.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     experiment=debug \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     trainer=gpu \
#     trainer.devices=[2] \
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
#     experiment_name=vicreg_b16k_archold_inputsv4_hpsearch \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     trainer=gpu \
#     trainer.devices=[3]


# vicreg training.
# taskset -c 3-5 \
# python3 src/train.py \
#     -m \
#     experiment=vicreg \
#     experiment_name=qvicreg_best_models \
#     trainer.gradient_clip_val=5.0 \
#     algorithm.optimizer.lr=0.00016579676425253136 \
#     algorithm.optimizer.weight_decay=2.8140536953097513e-05 \
#     algorithm.optimizer.betas='[0.95, 0.99]' \
#     algorithm.scheduler.scheduler.warmup_ratio=0.0 \
#     algorithm.scheduler.scheduler.min_lr_ratio=0.01 \
#     algorithm.feature_blur.prob=0.03131763472052267 \
#     algorithm.feature_blur.magnitude=0.32874110770580367 \
#     algorithm.feature_blur.strength=0.1074715159617419 \
#     algorithm.object_mask.prob=0.015748705771184444 \
#     algorithm.lorentz_rotation.prob=0.3 \
#     algorithm.loss.inv_coef=33.106544984974654 \
#     algorithm.loss.rvar_coef=1.27363319503812489 \
#     algorithm.loss.rcov_coef=0.8804065020151917 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     algorithm.diagnosis_metrics=true \
#     trainer=gpu \
#     trainer.devices=[0]

# VAE runs.
# ========================================================================
# vae hyperparameter search.
taskset -c 21-23 \
python3 src/train.py \
    -m \
    hydra/launcher=submitit_local \
    hydra.launcher.cpus_per_task=1 \
    hydra.launcher.gpus_per_node=4 \
    hparams_search=vae_optuna \
    experiment=vae \
    experiment_name=vae_b16k_archold_inputsv4_hpsearch \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    trainer=gpu \
    trainer.devices=[3]

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

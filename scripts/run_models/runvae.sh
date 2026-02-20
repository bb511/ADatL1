# VAE running commands.
# ========================================================================

# TRAINING.
# =======================

# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     experiment=vicreg_qvae_mse \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment_name=vicreg_qvae_best_pure \
#     run_name=mse_best_model_1 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas='[0.95,0.99]' \
#     algorithm.optimizer.weight_decay=0.00010499023396929063 \
#     algorithm.optimizer.eps=1e-07 \
#     algorithm.optimizer.lr=0.0027189473918282627 \
#     algorithm.loss.kl_scale=7.517098739653426e-05 \
#     algorithm.kl_warmup_frac=0.1 \
#     "algorithm.ckpt='/data/deodagiu/adl1t/checkpoints/vicreg_vae_best/mse_best_model_1/single/loss_reco_full_rate0.25kHz/max/ds=GluGluHto2G_Par-MH-125__metric=loss_reco_full_rate0.25kHz__value=766.731384__epoch=89.ckpt'" \
#     trainer=gpu \
#     trainer.devices=[0]


# HYPERPARAMETER SEARCHES.
# =======================
# Normal VAE hyperparam search.
taskset -c 0-2 \
python3 src/train.py \
    -m \
    hydra/launcher=submitit_local \
    hydra.launcher.cpus_per_task=1 \
    hydra.launcher.gpus_per_node=4 \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment=vae \
    experiment_name=vae_search \
    callbacks.max_rate_mse_ckpt=null \
    logger=none \
    hparams_search=vae_optuna \
    hydra.sweeper.study_name=efficiency_vs_rmse95_b16k \
    trainer=gpu \
    trainer.max_epochs=50 \
    trainer.devices=[0]

# VICreg VAE hyperparameter search.
# taskset -c 3-5 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     hparams_search=vae_optuna \
#     experiment=vicreg_vae_kl \
#     hydra.sweeper.study_name=vicreg_vae_b16k_archold_kl \
#     experiment_name=vicreg_vae_b16k_archold_inputsv4_hpsearch_kl \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     trainer=gpu \
#     trainer.devices=[0]

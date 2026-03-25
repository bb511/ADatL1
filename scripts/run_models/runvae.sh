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


# AXOv4 training...
# taskset -c 18-20 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=axov4 \
#     run_name=axov4_fp \
#     trainer=gpu \
#     trainer.devices=[0]


# HYPERPARAMETER SEARCHES.
# =======================


# Semi-Supervised Searches
# ------------------------

# AE hyperparameter search semi-supervised.
taskset -c 0-2 \
python3 src/train.py \
    -m \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment=vae \
    experiment_name=vae_cvar_vs_kl_search \
    callbacks.max_rate_kl_ckpt=null \
    callbacks.stable_kl_ckpt=null \
    evaluator.ckpts.last=false \
    evaluator.ckpts.single=null \
    evaluator_callbacks.reco=null \
    logger=none \
    hparams_search=vae_optuna \
    hydra.sweeper.study_name=cvar25eff_vs_kl_b16k \
    hydra.sweeper.n_trials=100 \
    hydra.sweeper.sampler.n_startup_trials=150 \
    trainer=gpu \
    trainer.max_epochs=1 \
    trainer.devices=[0]
    # hydra/launcher=submitit_local \
    # hydra.launcher.cpus_per_task=1 \
    # hydra.launcher.gpus_per_node=4 \


# AE hyperparameter search semi-supervised mse q99.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae \
#     experiment_name=ae_cvar_vs_mseq99_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.stable_mse_ckpt=null \
#     evaluator.ckpts.last=false \
#     evaluator.ckpts.single=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_optuna \
#     optimized_metric_config.sec_metric.callback.name=mse_q99 \
#     hydra.sweeper.study_name=cvar25eff_vs_mseq99_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# AE hyperparameter search semi-supervised cvar 10%.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae \
#     experiment_name=ae_cvar10_vs_mse_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.stable_mse_ckpt=null \
#     evaluator.ckpts.last=false \
#     evaluator.ckpts.single=null \
#     evaluator_callbacks.reco=null \
#     evaluator_callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     logger=none \
#     hparams_search=ae_optuna \
#     hydra.sweeper.study_name=cvar10eff_vs_mse_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=1 \
#     trainer.devices=[0]


# CAP Agnostic Searches
# ------------------------

# AE agnostic hyperparameter search - CAP vs MSE.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_cap_vs_mse_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_agnostic_optuna \
#     hydra.sweeper.study_name=cap_vs_mse_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# AE agnostic hyperparameter search - CAP vs MSE q99.
# taskset -c 64-66 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_cap_vs_mseq99_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_agnostic_optuna \
#     optimized_metric_config.sec_metric.callback.name=mse_q99 \
#     hydra.sweeper.study_name=cap_vs_mseq99_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# Stability Agnostic Searches
# ---------------------------

# AE agnostic MSE and threshold stability.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_mse_vs_thres_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.last=true \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_agnostic_optuna \
#     optimized_metric_config.main_metric.callback.name=mse_mean_top_vals \
#     +optimized_metric_config.main_metric.callback.params.ckpt_name=last \
#     +optimized_metric_config.main_metric.callback.params.test_ds=zerobias \
#     optimized_metric_config.main_metric.direction=minimize \
#     optimized_metric_config.sec_metric.callback.name=thres_transfer \
#     optimized_metric_config.sec_metric.callback.params.ckpt_name=null \
#     +optimized_metric_config.sec_metric.callback.params.target_rate=0.25 \
#     ~optimized_metric_config.sec_metric.callback.params.test_ds \
#     optimized_metric_config.sec_metric.direction=minimize \
#     hydra.sweeper.study_name=mse_vs_drift_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# AE agnostic MSE and threshold stability.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_mseq99_vs_thres_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.last=true \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_agnostic_optuna \
#     optimized_metric_config.main_metric.callback.name=mse_q99 \
#     +optimized_metric_config.main_metric.callback.params.ckpt_name=last \
#     +optimized_metric_config.main_metric.callback.params.test_ds=zerobias \
#     optimized_metric_config.main_metric.direction=minimize \
#     optimized_metric_config.sec_metric.callback.name=thres_transfer \
#     optimized_metric_config.sec_metric.callback.params.ckpt_name=null \
#     +optimized_metric_config.sec_metric.callback.params.target_rate=0.25 \
#     ~optimized_metric_config.sec_metric.callback.params.test_ds \
#     optimized_metric_config.sec_metric.direction=minimize \
#     hydra.sweeper.study_name=mseq99_vs_drift_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# Wasserstein Agnostic Searches
# -----------------------------

# AE agnostic MSE and wasserstein distance search.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_mse_vs_thres_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.last=true \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_agnostic_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_mse_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# AE agnostic MSE q99 and wasserstein distance search.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_mse_vs_thres_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.last=true \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_agnostic_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     optimized_metric_config.sec_metric.callback.name=mse_q99 \
#     hydra.sweeper.study_name=wasserstein_vs_mseq99_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

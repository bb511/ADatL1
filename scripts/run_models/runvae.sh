# VAE running commands.
# ========================================================================

# TRAINING.
# =======================

# Semi-supervised training.
# taskset -c 3-5 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae \
#     experiment_name=vae_semisup_models \
#     run_name=trial_560 \
#     algorithm.optimizer.lr=0.001135281648361112 \
#     algorithm.loss.delta=3.0 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.encoder.nodes='[64,32,32]' \
#     algorithm.input_noise_std=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]


# Agnostic training.
# taskset -c 6-8 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae_agnostic \
#     experiment_name=vae_cap \
#     run_name=trial_452 \
#     algorithm.optimizer.lr=0.0017073088946215253 \
#     algorithm.loss.delta=3.0 \
#     trainer.gradient_clip_val=5.0 \
#     algorithm.optimizer.betas='[0.9, 0.99]' \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.encoder.nodes='[64, 32, 16]' \
#     algorithm.input_noise_std=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]


# Agnostic MSE-threshold training.
# taskset -c 9-11 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae_agnostic \
#     experiment_name=vae_stability \
#     run_name=trial_339 \
#     evaluator_callbacks.cap_sn_zb=null \
#     algorithm.optimizer.lr=0.0018127512953324415 \
#     algorithm.loss.delta=3.0 \
#     trainer.gradient_clip_val=5.0 \
#     algorithm.optimizer.betas='[0.9, 0.999]' \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.encoder.nodes='[64,32,16]' \
#     algorithm.input_noise_std=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]


# Agnostic MSE-wasserstein training.
# taskset -c 6-8 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae_agnostic \
#     experiment_name=vae_wasserstein \
#     run_name=trial_105 \
#     algorithm.optimizer.lr=0.00023624225721440126 \
#     algorithm.loss.delta=3.0 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas='[0.9, 0.99]' \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.input_noise_std=0.0001 \
#     trainer=gpu \
#     trainer.devices=[2]


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

# VAE hyperparameter search semi-supervised.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae \
#     experiment_name=vae_cvar_vs_kl_search \
#     callbacks.max_rate_kl_ckpt=null \
#     callbacks.stable_kl_ckpt=null \
#     evaluator.ckpts.last=false \
#     evaluator.ckpts.single=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=vae_optuna \
#     hydra.sweeper.study_name=cvar25eff_vs_kl_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# VAE hyperparameter search semi-supervised kl q99.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae \
#     experiment_name=vae_cvar_vs_klq99_search \
#     callbacks.max_rate_kl_ckpt=null \
#     callbacks.stable_kl_ckpt=null \
#     evaluator.ckpts.last=false \
#     evaluator.ckpts.single=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=vae_optuna \
#     optimized_metric_config.sec_metric.callback.name=kl_raw_q99 \
#     hydra.sweeper.study_name=cvar25eff_vs_klq99_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# VAE hyperparameter search semi-supervised cvar 10%.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae \
#     experiment_name=vae_cvar10_vs_kl_search \
#     callbacks.max_rate_kl_ckpt=null \
#     callbacks.stable_kl_ckpt=null \
#     evaluator.ckpts.last=false \
#     evaluator.ckpts.single=null \
#     evaluator_callbacks.reco=null \
#     evaluator_callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     logger=none \
#     hparams_search=vae_optuna \
#     hydra.sweeper.study_name=cvar10eff_vs_kl_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# CAP Agnostic Searches
# ------------------------

# VAE agnostic hyperparameter search - CAP vs kl.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae_agnostic \
#     experiment_name=vae_agnostic_cap_vs_kl_search \
#     callbacks.max_rate_kl_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=vae_agnostic_optuna \
#     hydra.sweeper.study_name=cap_vs_kl_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# VAE agnostic hyperparameter search - CAP vs kl q99.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae_agnostic \
#     experiment_name=vae_agnostic_cap_vs_klq99_search \
#     callbacks.max_rate_kl_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=vae_agnostic_optuna \
#     optimized_metric_config.sec_metric.callback.name=kl_raw_q99 \
#     hydra.sweeper.study_name=cap_vs_klq99_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# Stability Agnostic Searches
# ---------------------------

# VAE agnostic kl and threshold stability.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae_agnostic \
#     experiment_name=vae_agnostic_kl_vs_thres_search \
#     callbacks.max_rate_kl_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.last=true \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=vae_agnostic_optuna \
#     optimized_metric_config.main_metric.callback.name=kl_raw_mean_top_vals \
#     +optimized_metric_config.main_metric.callback.params.ckpt_name=last \
#     +optimized_metric_config.main_metric.callback.params.test_ds=zerobias \
#     optimized_metric_config.main_metric.direction=minimize \
#     optimized_metric_config.sec_metric.callback.name=thres_transfer \
#     optimized_metric_config.sec_metric.callback.params.ckpt_name=null \
#     +optimized_metric_config.sec_metric.callback.params.target_rate=0.25 \
#     ~optimized_metric_config.sec_metric.callback.params.test_ds \
#     optimized_metric_config.sec_metric.direction=minimize \
#     hydra.sweeper.study_name=kl_vs_drift_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# AE agnostic kl and threshold stability.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae_agnostic \
#     experiment_name=vae_agnostic_klq99_vs_thres_search \
#     callbacks.max_rate_kl_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.last=true \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=vae_agnostic_optuna \
#     optimized_metric_config.main_metric.callback.name=kl_raw_q99 \
#     +optimized_metric_config.main_metric.callback.params.ckpt_name=last \
#     +optimized_metric_config.main_metric.callback.params.test_ds=zerobias \
#     optimized_metric_config.main_metric.direction=minimize \
#     optimized_metric_config.sec_metric.callback.name=thres_transfer \
#     optimized_metric_config.sec_metric.callback.params.ckpt_name=null \
#     +optimized_metric_config.sec_metric.callback.params.target_rate=0.25 \
#     ~optimized_metric_config.sec_metric.callback.params.test_ds \
#     optimized_metric_config.sec_metric.direction=minimize \
#     hydra.sweeper.study_name=klq99_vs_drift_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# Wasserstein Agnostic Searches
# -----------------------------

# AE agnostic kl and wasserstein distance search.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae_agnostic \
#     experiment_name=vae_agnostic_kl_vs_thres_search \
#     callbacks.max_rate_kl_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.last=true \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=vae_agnostic_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_kl_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# AE agnostic kl q99 and wasserstein distance search.
taskset -c 0-2 \
python3 src/train.py \
    -m \
    hydra/launcher=submitit_local \
    hydra.launcher.cpus_per_task=1 \
    hydra.launcher.gpus_per_node=4 \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment=vae_agnostic \
    experiment_name=vae_agnostic_kl_vs_thres_search \
    callbacks.max_rate_kl_ckpt=null \
    callbacks.cvar25_ema_ckpt=null \
    evaluator.ckpts.last=true \
    evaluator.ckpts.summary=null \
    evaluator_callbacks.cap_sn_zb=null \
    evaluator_callbacks.reco=null \
    logger=none \
    hparams_search=vae_agnostic_optuna \
    optimized_metric_config.main_metric.callback.name=wasserstein \
    optimized_metric_config.main_metric.direction=minimize \
    optimized_metric_config.sec_metric.callback.name=kl_raw_q99 \
    hydra.sweeper.study_name=wasserstein_vs_klq99_b16k \
    hydra.sweeper.direction='[minimize, minimize]' \
    hydra.sweeper.n_trials=100 \
    hydra.sweeper.sampler.n_startup_trials=150 \
    trainer=gpu \
    trainer.max_epochs=50 \
    trainer.devices=[0]

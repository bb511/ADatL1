# VAE running commands.
# ========================================================================

# TRAINING.
# =======================

# Semi-supervised cvar25 training.
# taskset -c 12-14 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae \
#     run_name=cvar25_q99_trial_339 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.optimizer.lr=0.0002713322501269118 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.loss.kl_scale=0.002 \
#     algorithm.kl_warmup_frac=0.0 \
#     algorithm.encoder.nodes='[64,32,24]' \
#     algorithm.encoder.activation=silu \
#     trainer=gpu \
#     trainer.devices=[1]


# Semi-supervised cvar10 training.
# taskset -c 3-5 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae \
#     run_name=cvar10_trial_339 \
#     evaluator_callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     trainer.gradient_clip_val=0.0 \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.optimizer.lr=0.0007725899485830742 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.loss.kl_scale=0.002 \
#     algorithm.kl_warmup_frac=0.0 \
#     algorithm.encoder.nodes='[48,24,24]' \
#     algorithm.encoder.activation=relu \
#     trainer=gpu \
#     trainer.devices=[0]


# Agnostic training.
# taskset -c 9-11 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae_agnostic \
#     run_name=cap_trial_180 \
#     trainer.gradient_clip_val=0.0 \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.optimizer.lr=0.0003403153803133397 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.loss.kl_scale=0.002 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.encoder.nodes='[32,16,4]' \
#     algorithm.encoder.activation=gelu \
#     trainer=gpu \
#     trainer.devices=[1]


# Agnostic KL-threshold training.
# taskset -c 9-11 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae_agnostic \
#     run_name=stability_trial_491 \
#     trainer.gradient_clip_val=1.0 \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.optimizer.lr=0.00018077873064930474 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.loss.kl_scale=0.002 \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.encoder.nodes='[48,24,8]' \
#     algorithm.encoder.activation=gelu \
#     trainer=gpu \
#     trainer.devices=[3]


# Agnostic KL-wasserstein training.
# taskset -c 15-17 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae_agnostic \
#     run_name=wasserstein_q99_trial_572 \
#     trainer.gradient_clip_val=1.0 \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.optimizer.lr=0.000657267315408614 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.loss.kl_scale=0.002 \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.encoder.nodes='[16,4]' \
#     algorithm.encoder.activation=gelu \
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

# VAE hyperparameter search semi-supervised.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae \
#     experiment_name=vae_cvar25_vs_kl_search \
#     callbacks.max_rate_kl_ckpt=null \
#     callbacks.stable_kl_q99_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluator.ckpts.summary.cvar10_ema \
#     ~evaluator.ckpts.single.loss_kl_raw_q99 \
#     ~evaluator.ckpts.single.eff__ascore_loss_kl_raw_full__brate_0_25kHz \
#     evaluator_callbacks.kl_loss_q99=null \
#     evaluator_callbacks.thres_transfer=null \
#     evaluator_callbacks.wasserstein=null \
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
#     experiment_name=vae_cvar25_vs_klq99_search \
#     callbacks.max_rate_kl_ckpt=null \
#     callbacks.stable_kl_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluator.ckpts.summary.cvar10_ema \
#     ~evaluator.ckpts.single.loss_kl_raw_mean_top_vals \
#     ~evaluator.ckpts.single.eff__ascore_loss_kl_raw_full__brate_0_25kHz \
#     evaluator_callbacks.kl_operational_mean=null \
#     evaluator_callbacks.thres_transfer=null \
#     evaluator_callbacks.wasserstein=null \
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
#     callbacks.stable_kl_q99_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     ~evaluator.ckpts.summary.cvar25_ema \
#     ~evaluator.ckpts.single.loss_kl_raw_q99 \
#     ~evaluator.ckpts.single.eff__ascore_loss_kl_raw_full__brate_0_25kHz \
#     evaluator_callbacks.kl_loss_q99=null \
#     evaluator_callbacks.thres_transfer=null \
#     evaluator_callbacks.wasserstein=null \
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
#     hparams_search=vae_optuna \
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
#     hparams_search=vae_optuna \
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
#     experiment_name=vae_agnostic_drift_vs_kl_search \
#     callbacks.anomaly_eff=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_kl_q99_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluator.ckpts.single.loss_kl_raw_q99 \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.kl_loss_q99=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.wasserstein=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=vae_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     +optimized_metric_config.main_metric.callback.params.target_rate=0.25 \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=drift_vs_kl_b16k \
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
#     experiment_name=vae_agnostic_drift_vs_klq99_search \
#     callbacks.anomaly_eff=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_kl_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluator.ckpts.single.loss_kl_raw_mean_top_vals \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.kl_operational_mean=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.wasserstein=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=vae_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     +optimized_metric_config.main_metric.callback.params.target_rate=0.25 \
#     optimized_metric_config.main_metric.direction=minimize \
#     optimized_metric_config.sec_metric.callback.name=kl_raw_q99 \
#     hydra.sweeper.study_name=drift_vs_klq99_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# Wasserstein Agnostic Searches
# -----------------------------

# AE agnostic kl and wasserstein distance search.
taskset -c 0-2 \
python3 src/train.py \
    -m \
    hydra/launcher=submitit_local \
    hydra.launcher.cpus_per_task=1 \
    hydra.launcher.gpus_per_node=4 \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment=vae_agnostic \
    experiment_name=vae_agnostic_wasserstein_vs_kl_search \
    callbacks.anomaly_eff=null \
    callbacks.thres_drift=null \
    callbacks.cap_sn_zb=null \
    callbacks.stable_kl_q99_ckpt=null \
    callbacks.thres_drift_ema_ckpt=null \
    callbacks.cap_sn_zb_ema_ckpt=null \
    ~evaluator.ckpts.single.loss_kl_raw_q99 \
    ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
    ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
    evaluator_callbacks.anomaly_efficiency=null \
    evaluator_callbacks.kl_loss_q99=null \
    evaluator_callbacks.cap_sn_zb=null \
    evaluator_callbacks.thres_drift=null \
    evaluator_callbacks.reco=null \
    logger=none \
    hparams_search=vae_optuna \
    optimized_metric_config.main_metric.callback.name=wasserstein \
    optimized_metric_config.main_metric.direction=minimize \
    hydra.sweeper.study_name=drift_vs_kl_b16k \
    hydra.sweeper.direction='[minimize, minimize]' \
    hydra.sweeper.n_trials=100 \
    hydra.sweeper.sampler.n_startup_trials=150 \
    trainer=gpu \
    trainer.max_epochs=50 \
    trainer.devices=[0]

# AE agnostic kl q99 and wasserstein distance search.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vae_agnostic \
#     experiment_name=vae_agnostic_drift_vs_klq99_search \
#     callbacks.anomaly_eff=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_kl_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluator.ckpts.single.loss_kl_raw_mean_top_vals \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.kl_operational_mean=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=vae_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     optimized_metric_config.sec_metric.callback.name=kl_raw_q99 \
#     hydra.sweeper.study_name=drift_vs_klq99_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

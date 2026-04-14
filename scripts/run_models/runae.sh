# Run the AE models.


# MODEL TRAINING
# =======================

# Semi-supervised cvar25 training.
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/ae \
#     experiment_name=debug \
#     run_name=test \
#     algorithm.optimizer.lr=0.001457369500608365 \
#     algorithm.loss.delta=7.0 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.encoder.nodes='[64,32,32]' \
#     algorithm.input_noise_std=0.001 \
#     trainer=gpu \
#     trainer.devices=[0]


# Semi-supervised cvar10 training.
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/ae \
#     run_name=cvar10_trial_339 \
#     evaluation/callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.optimizer.lr=0.0019789545082545034 \
#     algorithm.loss.delta=10.0 \
#     trainer.gradient_clip_val=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.encoder.nodes='[64,32,32]' \
#     algorithm.input_noise_std=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]


# CAP training.
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/ae_agnostic \
#     experiment_name=ae_models_q99 \
#     run_name=cap_t507_low \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.ckpts.single.ascore_operational \
#     ~evaluation.ckpts.summary.operational_drift_ema \
#     ~evaluation.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.0027927024120831816 \
#     algorithm.loss.delta=10.0 \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.encoder.nodes='[64,32,16]' \
#     algorithm.input_noise_std=0.01 \
#     trainer=gpu \
#     trainer.devices=[1]


# Agnostic stability training.
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/ae_agnostic \
#     experiment_name=ae_models_q99 \
#     run_name=stability_t560_high \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.ckpts.single.ascore_operational \
#     ~evaluation.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.000814981343573229 \
#     algorithm.loss.delta=10.0 \
#     trainer.gradient_clip_val=1.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.encoder.nodes='[64,32,32]' \
#     algorithm.input_noise_std=0.0003 \
#     trainer=gpu \
#     trainer.devices=[2]


# Agnostic KL-wasserstein training.
# taskset -c 12-14 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/ae_agnostic \
#     experiment_name=ae_models_q99 \
#     run_name=wasserstein_t585_medium \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.ckpts.single.ascore_operational \
#     ~evaluation.ckpts.summary.operational_drift_ema \
#     ~evaluation.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.00047124714609726086 \
#     algorithm.loss.delta=5.0 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.encoder.nodes='[64,32,32]' \
#     algorithm.input_noise_std=0.001 \
#     trainer=gpu \
#     trainer.devices=[3]


# HYPERPARAMETER SEARCHES.
# =====================================================================================


# Semi-Supervised Searches
# ------------------------
# AE hyperparameter search semi-supervised.
taskset -c 0-2 \
python3 src/train.py \
    -m \
    hydra/launcher=submitit_local \
    hydra.launcher.cpus_per_task=1 \
    hydra.launcher.gpus_per_node=4 \
    paths.raw_data_dir=/iopsstor/scratch/cscs/podagiu/adl1t_data/parquet_files \
    experiment=physics/ae \
    experiment_name=ae_cvar25_vs_mse_search \
    callbacks.max_rate_ckpt=null \
    callbacks.cvar10_ema_ckpt=null \
    ~evaluation.ckpts.single.eff__ascore_full__brate_0_25kHz \
    ~evaluation.ckpts.summary.cvar10_ema \
    evaluation/callbacks.thres_drift=null \
    evaluation/callbacks.wasserstein=null \
    evaluation/callbacks.reco=null \
    logger=none \
    hparams_search=ae_optuna \
    hydra.sweeper.study_name=cvar25eff_vs_mse \
    hydra.sweeper.n_trials=100 \
    hydra.sweeper.sampler.n_startup_trials=150 \
    trainer=gpu \
    trainer.max_epochs=50 \
    trainer.devices=[0]


# AE hyperparameter search semi-supervised cvar 10%.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/ae \
#     experiment_name=ae_cvar10_vs_mse_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     ~evaluation.ckpts.single.eff__ascore_full__brate_0_25kHz \
#     ~evaluation.ckpts.summary.cvar25_ema \
#     evaluation/callbacks.thres_drift=null \
#     evaluation/callbacks.wasserstein=null \
#     evaluation/callbacks.reco=null \
#     evaluation/callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     logger=none \
#     hparams_search=ae_optuna \
#     hydra.sweeper.study_name=cvar10eff_vs_mse \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# CAP Agnostic Searches
# ------------------------

# AE agnostic hyperparameter search - CAP vs MSE.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.timeout_min=100 \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/ae_agnostic \
#     experiment_name=ae_agnostic_cap_vs_mse_search \
#     callbacks.anomaly_eff=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     ~evaluation.ckpts.summary.operational_drift_ema \
#     ~evaluation.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     evaluation/callbacks.anomaly_efficiency=null \
#     evaluation/callbacks.thres_drift=null \
#     evaluation/callbacks.wasserstein=null \
#     evaluation/callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_optuna \
#     hydra.sweeper.study_name=cap_vs_mse \
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
#     experiment=physics/ae_agnostic \
#     experiment_name=ae_agnostic_drift_vs_mse_search \
#     callbacks.anomaly_eff=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     evaluation/callbacks.anomaly_efficiency=null \
#     evaluation/callbacks.cap_sn_zb=null \
#     evaluation/callbacks.wasserstein=null \
#     evaluation/callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     +optimized_metric_config.main_metric.callback.params.target_rate=0.25 \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=drift_vs_mse \
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
#     experiment=physics/ae_agnostic \
#     experiment_name=ae_agnostic_wasserstein_vs_mse_search \
#     callbacks.anomaly_eff=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.ckpts.summary.operational_drift_ema \
#     ~evaluation.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     evaluation/callbacks.anomaly_efficiency=null \
#     evaluation/callbacks.cap_sn_zb=null \
#     evaluation/callbacks.thres_drift=null \
#     evaluation/callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_mse \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

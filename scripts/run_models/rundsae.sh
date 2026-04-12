# Run the AE models.


# MODEL TRAINING
# =======================

# Semi-supervised cvar25 training.
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=dsae \
#     run_name=cvar25_t599_low \
#     algorithm.optimizer.lr=0.0021320360922839053 \
#     algorithm.loss.delta=10.0 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.input_noise_std=0.001 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.add_counts=false \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     trainer=gpu \
#     trainer.devices=[2]


# Semi-supervised cvar10 training.
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=dsae \
#     run_name=cvar10_trial_339 \
#     evaluator_callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.optimizer.lr=0.002081733354631208 \
#     algorithm.loss.delta=10.0 \
#     trainer.gradient_clip_val=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.input_noise_std=0.003 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.add_counts=true \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,8],jets:[16,8],muons:[8],taus:[16,8]}' \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     trainer=gpu \
#     trainer.devices=[0]



# CAP training.
# taskset -c 12-14 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=dsae_agnostic \
#     run_name=cap_t570_high \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.stable_mse_q99_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_mse_q99 \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.0028585937890006803 \
#     algorithm.loss.delta=7.0 \
#     trainer.gradient_clip_val=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.input_noise_std=0.01 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.add_counts=false \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[32,16],jets:[32,16],muons:[8,8],taus:[32,16]}' \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     trainer=gpu \
#     trainer.devices=[0]


# Agnostic stability training.
# taskset -c 12-14 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=dsae_agnostic \
#     run_name=stability_t565_high \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_mse_q99_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_mse_q99 \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.002156643622954745 \
#     algorithm.loss.delta=10.0 \
#     trainer.gradient_clip_val=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.input_noise_std=0.003 \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.add_counts=true \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     trainer=gpu \
#     trainer.devices=[2]



# Agnostic KL-wasserstein training.
# taskset -c 12-14 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=dsae_agnostic \
#     run_name=wasserstein_t383_high \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_mse_q99_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_mse_q99 \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.0012119364214017788 \
#     algorithm.loss.delta=4.0 \
#     trainer.gradient_clip_val=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.input_noise_std=0.01 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.add_counts=false \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     trainer=gpu \
#     trainer.devices=[3]



# HYPERPARAMETER SEARCHES.
# =====================================================================================


# Semi-Supervised Searches
# ------------------------

# AE hyperparameter search semi-supervised.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=dsae \
#     experiment_name=dsae_cvar_vs_mse_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.stable_mse_q99_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluator.ckpts.single.loss_mse_q99 \
#     ~evaluator.ckpts.single.eff__ascore_loss_mse_full__brate_0_25kHz \
#     ~evaluator.ckpts.summary.cvar10_ema \
#     evaluator_callbacks.mse_loss_q99=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.wasserstein=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=dsae_optuna \
#     hydra.sweeper.study_name=cvar25eff_vs_mse_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# AE hyperparameter search semi-supervised mse q99.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=dsae \
#     experiment_name=dsae_cvar_vs_mseq99_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.stable_mse_mean_top_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluator.ckpts.single.loss_mse_mean_top_vals \
#     ~evaluator.ckpts.single.eff__ascore_loss_mse_full__brate_0_25kHz \
#     ~evaluator.ckpts.summary.cvar10_ema \
#     evaluator_callbacks.mse_operational_mean=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.wasserstein=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=dsae_optuna \
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
#     experiment=dsae \
#     experiment_name=dsae_cvar10_vs_mse_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.stable_mse_q99_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     ~evaluator.ckpts.single.loss_mse_q99 \
#     ~evaluator.ckpts.single.eff__ascore_loss_mse_full__brate_0_25kHz \
#     ~evaluator.ckpts.summary.cvar25_ema \
#     evaluator_callbacks.mse_loss_q99=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.wasserstein=null \
#     evaluator_callbacks.reco=null \
#     evaluator_callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     logger=none \
#     hparams_search=dsae_optuna \
#     hydra.sweeper.study_name=cvar10eff_vs_mse_b16k \
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
#     hydra.launcher.timeout_min=200 \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=dsae_agnostic \
#     experiment_name=dsae_agnostic_cap_vs_mse_search \
#     callbacks.anomaly_eff=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.stable_mse_q99_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_mse_q99 \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.mse_loss_q99=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.wasserstein=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=dsae_optuna \
#     hydra.sweeper.study_name=cap_vs_mse_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# dsae agnostic hyperparameter search - CAP vs MSE q99.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.timeout_min=200 \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=dsae_agnostic \
#     experiment_name=dsae_agnostic_cap_vs_mseq99_search \
#     callbacks.anomaly_eff=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.stable_mse_mean_top_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     ~evaluator.ckpts.single.loss_mse_mean_top_vals \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.mse_operational_mean=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.wasserstein=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=dsae_optuna \
#     optimized_metric_config.sec_metric.callback.name=mse_q99 \
#     hydra.sweeper.study_name=cap_vs_mseq99_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# Stability Agnostic Searches
# ---------------------------

# dsae agnostic MSE and threshold stability.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=dsae_agnostic \
#     experiment_name=dsae_agnostic_drift_vs_mse_search \
#     callbacks.anomaly_eff=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.stable_mse_q99_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluator.ckpts.single.loss_mse_q99 \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.mse_loss_q99=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.wasserstein=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=dsae_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     +optimized_metric_config.main_metric.callback.params.target_rate=0.25 \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=drift_vs_mse_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# dsae agnostic MSE and threshold stability q99.
taskset -c 0-2 \
python3 src/train.py \
    -m \
    hydra/launcher=submitit_local \
    hydra.launcher.cpus_per_task=1 \
    hydra.launcher.gpus_per_node=4 \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment=dsae_agnostic \
    experiment_name=dsae_agnostic_drift_vs_mseq99_search \
    callbacks.anomaly_eff=null \
    callbacks.cap_sn_zb=null \
    callbacks.wasserstein_dist=null \
    callbacks.stable_mse_mean_top_ckpt=null \
    callbacks.wasserstein_dist_ema_ckpt=null \
    callbacks.thres_drift_ema_ckpt=null \
    callbacks.cap_sn_zb_ema_ckpt=null \
    ~evaluator.ckpts.single.loss_mse_mean_top_vals \
    ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
    ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
    ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
    evaluator_callbacks.anomaly_efficiency=null \
    evaluator_callbacks.mse_operational_mean=null \
    evaluator_callbacks.cap_sn_zb=null \
    evaluator_callbacks.wasserstein=null \
    evaluator_callbacks.reco=null \
    logger=none \
    hparams_search=dsae_optuna \
    optimized_metric_config.main_metric.callback.name=thres_drift \
    +optimized_metric_config.main_metric.callback.params.target_rate=286.0 \
    optimized_metric_config.main_metric.direction=minimize \
    optimized_metric_config.sec_metric.callback.name=mse_q99 \
    hydra.sweeper.study_name=drift_vs_mseq99_b16k \
    hydra.sweeper.direction='[minimize, minimize]' \
    hydra.sweeper.n_trials=100 \
    hydra.sweeper.sampler.n_startup_trials=150 \
    trainer=gpu \
    trainer.max_epochs=50 \
    trainer.devices=[0]


# Wasserstein Agnostic Searches
# -----------------------------

# dsae agnostic MSE and wasserstein distance search.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=dsae_agnostic \
#     experiment_name=dsae_agnostic_wasserstein_vs_mse_search \
#     callbacks.anomaly_eff=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift=null \
#     callbacks.stable_mse_q99_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_mse_q99 \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.mse_loss_q99=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=dsae_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_mse_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# dsae agnostic MSE q99 and wasserstein distance search.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=dsae_agnostic \
#     experiment_name=dsae_agnostic_wasserstein_vs_mseq99_search \
#     callbacks.anomaly_eff=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift=null \
#     callbacks.stable_mse_mean_top_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_mse_mean_top_vals \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.mse_operational_mean=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=dsae_optuna \
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

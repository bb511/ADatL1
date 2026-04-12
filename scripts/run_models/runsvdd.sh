# svdd running commands.
# ========================================================================

# TRAINING.
# =======================

# Semi-supervised cvar25 training.
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=svdd \
#     run_name=cvar25_t739_mid \
#     algorithm.optimizer.lr=0.0025808010156689754 \
#     trainer.gradient_clip_val=0.0 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.loss.weight_decay=1e-7 \
#     algorithm.loss.soft_boundary=False \
#     algorithm.loss.nu=0.01 \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.nodes=[48,24,8] \
#     algorithm.encoder.activation=gelu \
#     trainer=gpu \
#     trainer.devices=[0]


# Semi-supervised cvar10 training.
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=svdd \
#     run_name=cvar10_trial_339 \
#     evaluator_callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     trainer.gradient_clip_val=0.0 \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.optimizer.lr=0.0007725899485830742 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.loss.distance_scale=0.002 \
#     algorithm.distance_warmup_frac=0.0 \
#     algorithm.encoder.nodes='[48,24,24]' \
#     algorithm.encoder.activation=relu \
#     trainer=gpu \
#     trainer.devices=[0]


# CAP training.
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=svdd_agnostic \
#     run_name=cap_t176_high \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.stable_distance_q99_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_distance_q99 \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.001359990012421445 \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.optimizer.betas=[0.9,0.999] \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.loss.weight_decay=1e-07 \
#     algorithm.loss.soft_boundary=False \
#     algorithm.loss.nu=0.05 \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.nodes=[48,16] \
#     algorithm.encoder.activation=gelu \
#     trainer=gpu \
#     trainer.devices=[1]


# Agnostic stability training.
# taskset -c 12-14 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=svdd_agnostic \
#     run_name=stability_t419_high \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_distance_q99_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_distance_q99 \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.0009833375412439862 \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.optimizer.betas=[0.9,0.999] \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.loss.weight_decay=1e-05 \
#     algorithm.loss.soft_boundary=True \
#     algorithm.loss.nu=0.1 \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.nodes=[32,16,8] \
#     algorithm.encoder.activation=relu \
#     trainer=gpu \
#     trainer.devices=[1]


# Agnostic distance-wasserstein training.
# taskset -c 15-17 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=svdd_agnostic \
#     run_name=wasserstein_t539_high \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_distance_q99_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_distance_q99 \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.001983105654764409 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.loss.weight_decay=1e-06 \
#     algorithm.loss.soft_boundary=True \
#     algorithm.loss.nu=0.1 \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.nodes=[16,8] \
#     algorithm.encoder.activation=gelu \
#     trainer=gpu \
#     trainer.devices=[1]


# HYPERPARAMETER SEARCHES.
# =====================================================================================

# Semi-Supervised Searches
# ------------------------

# svdd hyperparameter search semi-supervised.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=svdd \
#     experiment_name=svdd_cvar25_vs_dist_search \
#     callbacks.max_rate_distance_ckpt=null \
#     callbacks.stable_distance_q99_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluator.ckpts.summary.cvar10_ema \
#     ~evaluator.ckpts.single.loss_distance_q99 \
#     ~evaluator.ckpts.single.eff__ascore_loss_distance_full__brate_0_25kHz \
#     evaluator_callbacks.distance_loss_q99=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=svdd_optuna \
#     hydra.sweeper.study_name=cvar25eff_vs_dist_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[1]


# svdd hyperparameter search semi-supervised distance q99.
# taskset -c 18-20 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=svdd \
#     experiment_name=svdd_cvar25_vs_distq99_search \
#     callbacks.max_rate_distance_ckpt=null \
#     callbacks.stable_distance_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluator.ckpts.summary.cvar10_ema \
#     ~evaluator.ckpts.single.loss_distance_mean_top_vals \
#     ~evaluator.ckpts.single.eff__ascore_loss_distance_full__brate_0_25kHz \
#     evaluator_callbacks.distance_operational_mean=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=svdd_optuna \
#     optimized_metric_config.sec_metric.callback.name=distance_q99 \
#     hydra.sweeper.study_name=cvar25eff_vs_distq99_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# svdd hyperparameter search semi-supervised cvar 10%.
# taskset -c 18-20 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=svdd \
#     experiment_name=svdd_cvar10_vs_dist_search \
#     callbacks.max_rate_distance_ckpt=null \
#     callbacks.stable_distance_q99_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     ~evaluator.ckpts.summary.cvar25_ema \
#     ~evaluator.ckpts.single.loss_distance_q99 \
#     ~evaluator.ckpts.single.eff__ascore_loss_distance_full__brate_0_25kHz \
#     evaluator_callbacks.distance_loss_q99=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.wasserstein=null \
#     evaluator_callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     logger=none \
#     hparams_search=svdd_optuna \
#     hydra.sweeper.study_name=cvar10eff_vs_dist_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# CAP Agnostic Searches
# ------------------------

# svdd agnostic hyperparameter search - CAP vs distance.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=svdd_agnostic \
#     experiment_name=svdd_agnostic_cap_vs_dist_search \
#     callbacks.anomaly_eff=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.stable_distance_q99_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_distance_q99 \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.distance_loss_q99=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=svdd_optuna \
#     hydra.sweeper.study_name=cap_vs_dist_b16k \
#     hydra.sweeper.n_trials=150 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# svdd agnostic hyperparameter search - CAP vs distance q99.
# taskset -c 18-20 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=svdd_agnostic \
#     experiment_name=svdd_agnostic_cap_vs_distq99_search \
#     callbacks.anomaly_eff=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.stable_distance_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_distance_mean_top_vals \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.distance_operational_mean=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=svdd_optuna \
#     optimized_metric_config.sec_metric.callback.name=distance_q99 \
#     hydra.sweeper.study_name=cap_vs_distq99_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# Stability Agnostic Searches
# ---------------------------

# svdd agnostic distance and threshold stability.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=svdd_agnostic \
#     experiment_name=svdd_agnostic_drift_vs_dist_search \
#     callbacks.anomaly_eff=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_distance_q99_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_distance_q99 \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.distance_loss_q99=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=svdd_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     +optimized_metric_config.main_metric.callback.params.target_rate=0.25 \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=drift_vs_dist_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# svdd agnostic distance and threshold stability.
# taskset -c 18-20 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=svdd_agnostic \
#     experiment_name=svdd_agnostic_drift_vs_distq99_search \
#     callbacks.anomaly_eff=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_distance_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_distance_mean_top_vals \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.distance_operational_mean=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=svdd_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     +optimized_metric_config.main_metric.callback.params.target_rate=286.0 \
#     optimized_metric_config.main_metric.direction=minimize \
#     optimized_metric_config.sec_metric.callback.name=distance_q99 \
#     hydra.sweeper.study_name=drift_vs_distq99_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# Wasserstein Agnostic Searches
# -----------------------------

# svdd agnostic distance and wasserstein distance search.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=svdd_agnostic \
#     experiment_name=svdd_agnostic_wasserstein_vs_dist_search \
#     callbacks.anomaly_eff=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_distance_q99_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_distance_q99 \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.distance_loss_q99=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.thres_drift=null \
#     logger=none \
#     hparams_search=svdd_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_dist_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# svdd agnostic distance q99 and wasserstein distance search.
# taskset -c 18-20 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=svdd_agnostic \
#     experiment_name=svdd_agnostic_drift_vs_distq99_search \
#     callbacks.anomaly_eff=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_distance_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluator.ckpts.single.loss_distance_mean_top_vals \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.distance_operational_mean=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.thres_drift=null \
#     logger=none \
#     hparams_search=svdd_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     optimized_metric_config.sec_metric.callback.name=distance_q99 \
#     hydra.sweeper.study_name=wasserstein_vs_distq99_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

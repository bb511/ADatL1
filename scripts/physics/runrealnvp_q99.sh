# ========================================================================
# RealNVP RUNNING COMMANDS
# ========================================================================
# These are the running commands for the q99 background rate study.


# ========================================================================
# TRAINING
# ========================================================================
# ------------------------------------------------------------------------
# Semi-supervised cvar25 training
# ------------------------------------------------------------------------
# taskset -c 48-50 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/realnvp \
#     experiment_name=physics_realnvp_models_q99 \
#     run_name=cvar25_t466 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.optimizer.lr=3.7803380868254345e-05 \
#     trainer.gradient_clip_val=0.0 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=5.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# AGNOSTIC TRAINING
# ========================================================================
# ------------------------------------------------------------------------
# CAP training
# ------------------------------------------------------------------------
# taskset -c 51-53 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_models_q99 \
#     run_name=cap_t200 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.0008856132495375707 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas=[0.9,0.999] \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.activation=relu \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=5.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# Stability training
# ------------------------------------------------------------------------
# taskset -c 54-56 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_models_q99 \
#     run_name=stability_t595 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.0008426900182503831 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.activation=relu \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=5.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# Wasserstein training
# ------------------------------------------------------------------------
# taskset -c 57-59 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_models_q99 \
#     run_name=wasserstein_t580 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.0007229552409929307 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.activation=relu \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=5.0 \
#     trainer=gpu \
#     trainer.devices=[3]



# ========================================================================
# HYPERPARAMETER SEARCHES
# =========================================================================
# ------------------------------------------------------------------------
# Semi-supervised cvar25 training
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/realnvp \
#     experiment_name=realnvp_cvar25_vs_logpq99_search \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational \
#     ~evaluation.evaluator.ckpts.summary.cvar10_ema \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=realnvp_optuna \
#     hydra.sweeper.study_name=cvar25eff_vs_logpq99 \
#     hydra.sweeper.n_trials=150 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# CAP training
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.timeout_min=200 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=realnvp_agnostic_cap_vs_logpq99_search \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.anomaly_eff=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=realnvp_optuna \
#     hydra.sweeper.study_name=cap_vs_logpq99 \
#     hydra.sweeper.n_trials=150 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Stability training
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=realnvp_agnostic_drift_vs_logpq99_search \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.anomaly_eff=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.cap_sn_zb=null \
#     evaluation.callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=realnvp_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=drift_vs_logpq99 \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=150 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Wasserstein training
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=realnvp_agnostic_wasserstein_vs_logpq99_search \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.anomaly_eff=null \
#     callbacks.anomaly_eff=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.cap_sn_zb=null \
#     evaluation.callbacks.thres_drift=null \
#     logger=none \
#     hparams_search=realnvp_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_logpq99 \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=150 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ========================================================================
# VAE RUNNING COMMANDS
# ========================================================================

# These are the running commands for the 250 Hz background rate study.
# To switch to q99 studies, add the following lines to each of the commands below:
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \

# ========================================================================
# TRAINING
# ========================================================================
# ------------------------------------------------------------------------
# Semi-supervised cvar25 training
# ------------------------------------------------------------------------
# taskset -c 12-14 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/vae \
#     run_name=cvar25_t345 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.optimizer.lr=5.236233832409967e-05 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.kl_scale=0.002 \
#     algorithm.kl_warmup_frac=0.0 \
#     algorithm.encoder.nodes='[32,16,16]' \
#     algorithm.encoder.activation=gelu \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Semi-supervised cvar10 training
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/vae \
#     run_name=cvar10_t100 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     trainer.gradient_clip_val=0.0 \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.optimizer.lr=0.0007725899485830742 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.kl_scale=0.002 \
#     algorithm.kl_warmup_frac=0.0 \
#     algorithm.encoder.nodes='[48,24,24]' \
#     algorithm.encoder.activation=relu \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# CAP training
# ------------------------------------------------------------------------
# taskset -c 15-17 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     run_name=cap_t179 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.optimizer.lr=6.358827890772183e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.encoder.nodes='[32,16,16]' \
#     algorithm.encoder.activation=silu \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# Stability training
# ------------------------------------------------------------------------
# taskset -c 18-20 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     run_name=stability_t529 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.optimizer.lr=0.0008437643709861688 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.kl_scale=0.002 \
#     algorithm.kl_warmup_frac=0.0 \
#     algorithm.encoder.nodes='[48,16,8]' \
#     algorithm.encoder.activation=gelu \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# Wasserstein training
# ------------------------------------------------------------------------
# taskset -c 21-23 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     run_name=wasserstein_t539 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.optimizer.lr=6.005882209363415e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.kl_scale=0.002 \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.encoder.activation=relu \
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
#     experiment=physics/vae \
#     experiment_name=vae_cvar25_vs_kl_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational \
#     ~evaluation.evaluator.ckpts.summary.cvar10_ema \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=vae_optuna \
#     hydra.sweeper.study_name=cvar25eff_vs_kl \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Semi-supervised cvar10 training
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/vae \
#     experiment_name=vae_cvar10_vs_kl_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational \
#     ~evaluation.evaluator.ckpts.summary.cvar25_ema \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.reco=null \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     logger=none \
#     hparams_search=vae_optuna \
#     hydra.sweeper.study_name=cvar10eff_vs_kl \
#     hydra.sweeper.n_trials=100 \
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
#     experiment=physics/vae_agnostic \
#     experiment_name=vae_agnostic_cap_vs_kl_search \
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
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=vae_optuna \
#     hydra.sweeper.study_name=cap_vs_kl \
#     hydra.sweeper.n_trials=100 \
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
#     experiment=physics/vae_agnostic \
#     experiment_name=vae_agnostic_drift_vs_kl_search \
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
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=vae_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=drift_vs_kl \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
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
#     experiment=physics/vae_agnostic \
#     experiment_name=vae_agnostic_wasserstein_vs_kl_search \
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
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=vae_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_kl \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

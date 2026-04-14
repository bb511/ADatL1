# ========================================================================
# DeepSets VAE RUNNING COMMANDS
# ========================================================================


# ========================================================================
# TRAINING
# ========================================================================
# ------------------------------------------------------------------------
# Semi-supervised cvar25 training
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=debug \
#     run_name=test \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.encoder.clamp_zlogvar_range=[-6,4] \
#     algorithm.optimizer.lr=0.00011979051144559894 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,16],jets:[16,16],muons:[8,8],taus:[16,16]}' \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Semi-supervised cvar10 training
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=debug \
#     run_name=test \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.encoder.clamp_zlogvar_range=[-6,4] \
#     algorithm.optimizer.lr=0.00011979051144559894 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,16],jets:[16,16],muons:[8,8],taus:[16,16]}' \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# CAP training
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=debug \
#     run_name=test \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.encoder.clamp_zlogvar_range=[-6,4] \
#     algorithm.optimizer.lr=0.00011979051144559894 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,16],jets:[16,16],muons:[8,8],taus:[16,16]}' \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# Stability training
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=debug \
#     run_name=test \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.encoder.clamp_zlogvar_range=[-6,4] \
#     algorithm.optimizer.lr=0.00011979051144559894 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,16],jets:[16,16],muons:[8,8],taus:[16,16]}' \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# Wasserstein training
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=debug \
#     run_name=test \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.encoder.clamp_zlogvar_range=[-6,4] \
#     algorithm.optimizer.lr=0.00011979051144559894 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,16],jets:[16,16],muons:[8,8],taus:[16,16]}' \
#     algorithm.encoder.rho_nodes='[16,8]' \
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
#     experiment=physics/dsvae \
#     experiment_name=vae_cvar25_vs_kl_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_0_25kHz \
#     ~evaluation.evaluator.ckpts.summary.cvar10_ema \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=dsvae_optuna \
#     hydra.sweeper.study_name=cvar25eff_vs_kl \
#     hydra.sweeper.n_trials=150 \
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
#     experiment=physics/dsvae \
#     experiment_name=vae_cvar10_vs_kl_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_0_25kHz \
#     ~evaluation.evaluator.ckpts.summary.cvar25_ema \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.reco=null \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     logger=none \
#     hparams_search=dsvae_optuna \
#     hydra.sweeper.study_name=cvar10eff_vs_kl \
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
#     experiment=physics/dsvae_agnostic \
#     experiment_name=dsvae_agnostic_cap_vs_kl_search \
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
#     hparams_search=dsvae_optuna \
#     hydra.sweeper.study_name=cap_vs_kl \
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
#     experiment=physics/dsvae_agnostic \
#     experiment_name=dsvae_agnostic_drift_vs_klq99_search \
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
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=dsvae_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=drift_vs_klq99 \
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
#     experiment=physics/dsvae_agnostic \
#     experiment_name=dsvae_agnostic_wasserstein_vs_klq99_search \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
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
#     hparams_search=dsvae_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_klq99 \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=150 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

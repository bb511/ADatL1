# ========================================================================
# DeepSets VAE RUNNING COMMANDS
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
# taskset -c 36-38 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     run_name=cvar25_t372 \
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
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     run_name=cvar10_t100 \
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


# ========================================================================
# AGNOSTIC TRAINING
# ========================================================================
# ------------------------------------------------------------------------
# CAP training
# ------------------------------------------------------------------------
# taskset -c 39-41 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     run_name=cap_t324 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.encoder.clamp_zlogvar_range=[-20,10] \
#     algorithm.optimizer.lr=7.02318412358252e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.kl_scale=0.002 \
#     algorithm.kl_warmup_frac=0.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.add_counts=true \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.rho_nodes='[48,24,8]' \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# Stability training
# ------------------------------------------------------------------------
# taskset -c 42-44 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     run_name=stability_t445 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.encoder.clamp_zlogvar_range=[-20,10] \
#     algorithm.optimizer.lr=5.00019324835396e-05 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# Wasserstein training
# ------------------------------------------------------------------------
# taskset -c 45-47 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     run_name=wasserstein_t503 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     trainer.gradient_clip_val=0.0 \
#     algorithm.encoder.clamp_zlogvar_range=[-10,6] \
#     algorithm.optimizer.lr=5.3007600884915246e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.kl_scale=3e-05 \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     trainer=gpu \
#     trainer.devices=[3]



# =========================================================================
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
#     experiment_name=dsvae_cvar25_vs_kl_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational \
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
#     experiment_name=dsvae_cvar10_vs_kl_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational \
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

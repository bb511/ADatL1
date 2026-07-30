# ========================================================================
# DTE HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# These are the running commands for the 250 Hz background rate study.
#
# One driver per strategy: hydra.sweeper.n_jobs=6 runs six trials at a time as
# slurm jobs, so the whole 600-trial study completes in a single invocation.
#
# BEFORE THE FIRST LAUNCH the processed data must already exist on clariden.
# configs/experiment/physics/dte*.yaml override /data/data_normalizer to
# `standard`, which has its own cache tree that nothing else in the repo builds.
# Build it on olqti with scripts/physics/rundte_pareto.sh and copy the trees
# listed there; six drivers starting against a missing cache race on the same
# directory and leave partial parquet behind. Then run one trial
# (hydra.sweeper.n_trials=1) to confirm 720 minutes is enough.


# ========================================================================
# HYPERPARAMETER SEARCHES
# =========================================================================
# ------------------------------------------------------------------------
# Semi-supervised cvar25 training
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=720 \
#     hydra.sweeper.n_jobs=6 \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=dte_cvar25_vs_ascore_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational \
#     ~evaluation.evaluator.ckpts.summary.cvar10_ema \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.study_name=cvar25eff_vs_ascore \
#     hydra.sweeper.n_trials=600 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Semi-supervised cvar10 training
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=720 \
#     hydra.sweeper.n_jobs=6 \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=dte_cvar10_vs_ascore_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational \
#     ~evaluation.evaluator.ckpts.summary.cvar25_ema \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     logger=none \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.study_name=cvar10eff_vs_ascore \
#     hydra.sweeper.n_trials=600 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# CAP training
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=720 \
#     hydra.sweeper.n_jobs=6 \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=dte_agnostic_cap_vs_ascore_search \
#     callbacks.anomaly_eff=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.consistency_sn_zb=null \
#     logger=none \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.study_name=cap_vs_ascore \
#     hydra.sweeper.n_trials=600 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Stability training
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=720 \
#     hydra.sweeper.n_jobs=6 \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=dte_agnostic_drift_vs_ascore_search \
#     callbacks.anomaly_eff=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.cap_sn_zb=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.consistency_sn_zb=null \
#     logger=none \
#     hparams_search=physics/dte_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=drift_vs_ascore \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=600 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Wasserstein training
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=720 \
#     hydra.sweeper.n_jobs=6 \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=dte_agnostic_wasserstein_vs_ascore_search \
#     callbacks.anomaly_eff=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.cap_sn_zb=null \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.consistency_sn_zb=null \
#     logger=none \
#     hparams_search=physics/dte_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_ascore \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=600 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Consistency search
# ------------------------------------------------------------------------
# Consistency-only CAP: the beta-free limit of CAP's posterior log-cosine,
# i.e. CAP with the informative (posterior-norm) component removed.
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=720 \
#     hydra.sweeper.n_jobs=6 \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=dte_agnostic_consistency_vs_ascore_search \
#     callbacks.anomaly_eff=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.cap_sn_zb=null \
#     logger=none \
#     hparams_search=physics/dte_optuna \
#     optimized_metric_config.main_metric.callback.name=consistency \
#     hydra.sweeper.study_name=consistency_vs_ascore \
#     hydra.sweeper.n_trials=600 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

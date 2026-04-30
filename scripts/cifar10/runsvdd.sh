# ========================================================================
# SVDD RUNNING COMMANDS
# ========================================================================


# ========================================================================
# TRAINING
# ========================================================================
# ------------------------------------------------------------------------
# Semi-supervised cvar25 training
# ------------------------------------------------------------------------
# taskset -c 36-38 \
# python3 src/train.py \
#     experiment=cifar10/svdd \
#     run_name=cvar25_t324_high \
#     algorithm.optimizer.lr=0.002487596640834986 \
#     trainer.gradient_clip_val=1.0 \
#     algorithm.optimizer.betas=[0.9,0.999] \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.weight_decay=1e-07 \
#     algorithm.soft_boundary=True \
#     algorithm.nu=0.01 \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.nodes=[64,128,256] \
#     algorithm.encoder.strides=[2,2] \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# Semi-supervised cvar10 training
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd \
#     run_name=cvar10_t339_high \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.optimizer.lr=0.0019789545082545034 \
#     algorithm.delta=10.0 \
#     trainer.gradient_clip_val=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.encoder.nodes='[64,32,32]' \
#     algorithm.input_noise_std=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]


# ========================================================================
# AGNOSTIC TRAINING
# ========================================================================
# ------------------------------------------------------------------------
# CAP training
# ------------------------------------------------------------------------
# taskset -c 40-42 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     run_name=cap_t240_high \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.optimizer.lr=0.001938086961586347 \
#     trainer.gradient_clip_val=1.0 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.weight_decay=1e-06 \
#     algorithm.soft_boundary=True \
#     algorithm.nu=0.2 \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.nodes=[16,32,64] \
#     algorithm.encoder.strides=[2,1] \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# Stability training
# ------------------------------------------------------------------------
# taskset -c 43-45 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     run_name=stability_t390_high \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.optimizer.lr=0.0029638593347981854 \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.optimizer.betas=[0.9,0.999] \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.weight_decay=1e-08 \
#     algorithm.soft_boundary=True \
#     algorithm.nu=0.01 \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.nodes=[16,64,128] \
#     algorithm.encoder.strides=[2,1] \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# Wasserstein training
# ------------------------------------------------------------------------
# taskset -c 46-48 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     run_name=wasserstein_t390_high \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.optimizer.lr=0.002998149016072265 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.nu=0.05 \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.nodes=[32,64,128] \
#     algorithm.encoder.strides=[2,1] \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     trainer=gpu \
#     trainer.devices=[2]


# ========================================================================
# HYPERPARAMETER SEARCH
# ========================================================================
# ------------------------------------------------------------------------
# Semi-supervised search (cvar25)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.sweeper.n_jobs=6 \
#     experiment=cifar10/svdd \
#     experiment_name=cifar10_svdd_cvar25_vs_dist_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational \
#     ~evaluation.evaluator.ckpts.summary.cvar10_ema \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=imagesvdd_optuna \
#     hydra.sweeper.study_name=cvar25eff_vs_dist \
#     hydra.sweeper.n_trials=600 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Semi-supervised search (cvar10)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.sweeper.n_jobs=6 \
#     experiment=cifar10/svdd \
#     experiment_name=cifar10_svdd_cvar10_vs_dist_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational \
#     ~evaluation.evaluator.ckpts.summary.cvar25_ema \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     logger=none \
#     hparams_search=imagesvdd_optuna \
#     hydra.sweeper.study_name=cvar10eff_vs_dist \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# CAP search
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.sweeper.n_jobs=6 \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_agnostic_cap_vs_dist_search \
#     callbacks.anomaly_eff=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=imagesvdd_optuna \
#     hydra.sweeper.study_name=cap_vs_dist \
#     hydra.sweeper.n_trials=600 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Stability search
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.sweeper.n_jobs=6 \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_agnostic_drift_vs_dist_search \
#     callbacks.anomaly_eff=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.cap_sn_zb=null \
#     evaluation.callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=imagesvdd_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=drift_vs_dist \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=600 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Wasserstein search
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.sweeper.n_jobs=6 \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_agnostic_wasserstein_vs_dist_search \
#     callbacks.anomaly_eff=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.cap_sn_zb=null \
#     evaluation.callbacks.thres_drift=null \
#     logger=none \
#     hparams_search=imagesvdd_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_dist \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=600 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ========================================================================
# AE RUNNING COMMANDS
# ========================================================================


# ========================================================================
# TRAINING
# ========================================================================
# ------------------------------------------------------------------------
# Semi-supervised cvar25 training
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/ae \
#     run_name=cvar25_t592 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.delta=5.0 \
#     algorithm.input_noise_std=0.05 \
#     algorithm.encoder.nodes=[64,128,256] \
#     algorithm.encoder.strides=[1,2] \
#     algorithm.encoder.activation=silu \
#     algorithm.optimizer.lr=0.0017368497078662237 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=0.0005 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# AGNOSTIC TRAINING
# ========================================================================
# ------------------------------------------------------------------------
# CAP training
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/ae_agnostic \
#     run_name=cap_t211 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     trainer.gradient_clip_val=1.0 \
#     algorithm.delta=3.0 \
#     algorithm.input_noise_std=0.0 \
#     algorithm.encoder.nodes=[32,64,128] \
#     algorithm.encoder.strides=[2,1] \
#     algorithm.encoder.activation=gelu \
#     algorithm.optimizer.lr=0.0028555698070213114 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# Stability training
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/ae_agnostic \
#     run_name=stability_t241 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     trainer.gradient_clip_val=1.0 \
#     algorithm.delta=1.0 \
#     algorithm.input_noise_std=0.05 \
#     algorithm.encoder.nodes=[64,128,256] \
#     algorithm.encoder.strides=[2,2] \
#     algorithm.encoder.activation=silu \
#     algorithm.optimizer.lr=0.0008065877157877547 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=0.0005 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# Wasserstein training
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/ae_agnostic \
#     run_name=wasserstein_t279 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.delta=5.0 \
#     algorithm.input_noise_std=0.03 \
#     algorithm.encoder.nodes=[64,128,256] \
#     algorithm.encoder.strides=[2,2] \
#     algorithm.encoder.activation=silu \
#     algorithm.optimizer.lr=0.002250553692978696 \
#     algorithm.optimizer.betas=[0.9,0.999] \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer=gpu \
#     trainer.devices=[3]


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
#     experiment=cifar10/ae \
#     experiment_name=cifar10_ae_cvar25_vs_mse_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational \
#     ~evaluation.evaluator.ckpts.summary.cvar10_ema \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=imageae_optuna \
#     hydra.sweeper.study_name=cvar25eff_vs_mse \
#     hydra.sweeper.n_trials=600 \
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
#     experiment=cifar10/ae_agnostic \
#     experiment_name=cifar10_ae_agnostic_cap_vs_mse_search \
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
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=imageae_optuna \
#     hydra.sweeper.study_name=cap_vs_mse \
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
#     experiment=cifar10/ae_agnostic \
#     experiment_name=cifar10_ae_agnostic_drift_vs_mse_search \
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
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=imageae_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=drift_vs_mse \
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
#     experiment=cifar10/ae_agnostic \
#     experiment_name=cifar10_ae_agnostic_wasserstein_vs_mse_search \
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
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=imageae_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_mse \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=600 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

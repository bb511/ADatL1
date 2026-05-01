# ========================================================================
# RealNVP RUNNING COMMANDS
# ========================================================================


# ========================================================================
# TRAINING
# ========================================================================
# ------------------------------------------------------------------------
# Semi-supervised cvar25 training
# ------------------------------------------------------------------------
# taskset -c 24-26 \
# python3 src/train.py \
#     experiment=cifar10/realnvp \
#     run_name=cvar25_t339 \
#     algorithm.optimizer.lr=0.0010115191329924005 \
#     trainer.gradient_clip_val=1.0 \
#     algorithm.optimizer.betas=[0.9,0.999] \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.activation=silu \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ========================================================================
# AGNOSTIC TRAINING
# ========================================================================
# ------------------------------------------------------------------------
# CAP training
# ------------------------------------------------------------------------
# taskset -c 27-29 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     run_name=cap_t211 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.optimizer.lr=0.00014437689046359488 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas=[0.9,0.999] \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=3.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# Stability training
# ------------------------------------------------------------------------
# taskset -c 30-32 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     run_name=stability_t104 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.optimizer.lr=0.000359807400273336 \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.activation=silu \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=3.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# Wasserstein training
# ------------------------------------------------------------------------
# taskset -c 33-35 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     run_name=wasserstein_t475 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.optimizer.lr=0.0008371472615602258 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas=[0.9,0.999] \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.activation=silu \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=3.0 \
#     trainer=gpu \
#     trainer.devices=[1]


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
#     experiment=cifar10/realnvp \
#     experiment_name=cifar10_realnvp_cvar25_vs_logp_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational \
#     ~evaluation.evaluator.ckpts.summary.cvar10_ema \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=imagerealnvp_optuna \
#     hydra.sweeper.study_name=cvar25eff_vs_logp \
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
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_agnostic_cap_vs_logp_search \
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
#     hparams_search=imagerealnvp_optuna \
#     hydra.sweeper.study_name=cap_vs_logp \
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
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_agnostic_drift_vs_logp_search \
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
#     hparams_search=imagerealnvp_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=drift_vs_logp \
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
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_agnostic_wasserstein_vs_logp_search \
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
#     hparams_search=imagerealnvp_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_logp \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=600 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

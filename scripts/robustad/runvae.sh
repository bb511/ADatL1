# ========================================================================
# VAE RUNNING COMMANDS
# ========================================================================


# ========================================================================
# TRAINING
# ========================================================================
# ------------------------------------------------------------------------
# Semi-supervised cvar25 training
# ------------------------------------------------------------------------
# taskset -c 12-14 \
# python3 src/train.py \
#     experiment=robustad/vae \
#     run_name=cvar25_t62 \
#     trainer.gradient_clip_val=1.0 \
#     algorithm.encoder.clamp_zlogvar_range=[-8,6] \
#     algorithm.optimizer.lr=3.99357685959721e-05 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.kl_scale=0.0003 \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.encoder.nodes=[8,16,32] \
#     algorithm.encoder.strides=[2] \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# AGNOSTIC TRAINING
# ========================================================================
# ------------------------------------------------------------------------
# CAP training
# ------------------------------------------------------------------------
# taskset -c 15-17 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     run_name=cap_t402 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.encoder.clamp_zlogvar_range=[-6,4] \
#     algorithm.optimizer.lr=3.0060074034422283e-05 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.kl_scale=1e-05 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.encoder.nodes=[8,16,16] \
#     algorithm.encoder.strides=[2] \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     trainer.max_epochs=1 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# Stability training
# ------------------------------------------------------------------------
# taskset -c 18-20 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     run_name=stability_t389 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.encoder.clamp_zlogvar_range=[-8,6] \
#     algorithm.optimizer.lr=4.048861860453195e-05 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.kl_scale=0.0003 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.encoder.nodes=[8,16,16] \
#     algorithm.encoder.strides=[2] \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# Wasserstein training
# ------------------------------------------------------------------------
# taskset -c 21-23 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     run_name=wasserstein_t587 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     trainer.gradient_clip_val=0.0 \
#     algorithm.encoder.clamp_zlogvar_range=[-8,6] \
#     algorithm.optimizer.lr=4.2327787012183054e-05 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.encoder.nodes=[8,16,16] \
#     algorithm.encoder.strides=[2] \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     trainer=gpu \
#     trainer.devices=[3]


# ========================================================================
# HYPERPARAMETER SEARCH
# ========================================================================
# ------------------------------------------------------------------------
# Semi-supervised search (cvar25)
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.timeout_min=200 \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     hydra.sweeper.storage='sqlite:///logs/optuna/robustad/vae.db' \
#     experiment=robustad/vae \
#     experiment_name=robustad_vae_cvar25_vs_kl_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational \
#     ~evaluation.evaluator.ckpts.summary.cvar10_ema \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=imagevae_optuna \
#     hydra.sweeper.study_name=cvar25eff_vs_kl \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# CAP search
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.timeout_min=200 \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     hydra.sweeper.storage='sqlite:///logs/optuna/robustad/vae.db' \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_agnostic_cap_vs_kl_search \
#     callbacks.anomaly_eff=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=imagevae_optuna \
#     hydra.sweeper.study_name=cap_vs_kl \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Stability search
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.timeout_min=200 \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     hydra.sweeper.storage='sqlite:///logs/optuna/robustad/vae.db' \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_agnostic_drift_vs_kl_search \
#     callbacks.anomaly_eff=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.cap_sn_zb=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=imagevae_optuna \
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
# Wasserstein search
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.timeout_min=200 \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     hydra.sweeper.storage='sqlite:///logs/optuna/robustad/vae.db' \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_agnostic_wasserstein_vs_kl_search \
#     callbacks.anomaly_eff=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.cap_sn_zb=null \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=imagevae_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_kl \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

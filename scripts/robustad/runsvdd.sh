# ========================================================================
# svdd RUNNING COMMANDS
# ========================================================================


# ========================================================================
# TRAINING
# ========================================================================
# ------------------------------------------------------------------------
# Semi-supervised cvar25 training
# ------------------------------------------------------------------------
# taskset -c 48-50 \
# python3 src/train.py \
#     experiment=robustad/svdd \
#     run_name=cvar25_t339 \
#     algorithm.optimizer.lr=0.0008931169751749622 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas=[0.9,0.999] \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.nu=0.2 \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.nodes=[8,16,32] \
#     algorithm.encoder.strides=[2,2] \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
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
#     experiment=robustad/svdd_agnostic \
#     run_name=cap_t240 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.optimizer.lr=0.0009585626245761539 \
#     trainer.gradient_clip_val=1.0 \
#     algorithm.optimizer.betas=[0.9,0.99] \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.nu=0.05 \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.nodes=[32,64,128] \
#     algorithm.encoder.strides=[2,2] \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# Stability training
# ------------------------------------------------------------------------
# taskset -c 42-44 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     run_name=stability_t525 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.optimizer.lr=0.0006495587687856368 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas=[0.9,0.999] \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.nu=0.1 \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.nodes=[8,16,32] \
#     algorithm.encoder.strides=[2,2] \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# Wasserstein training
# ------------------------------------------------------------------------
# taskset -c 45-47 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     run_name=wasserstein_t581 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.optimizer.lr=0.0006079271679609457 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas=[0.9,0.999] \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.weight_decay=1e-07 \
#     algorithm.soft_boundary=True \
#     algorithm.nu=0.1 \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.nodes=[8,16,32] \
#     algorithm.encoder.strides=[2,2] \
#     algorithm.encoder.activation=silu \
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
#     hydra.sweeper.storage='sqlite:///logs/optuna/robustad/svdd.db' \
#     experiment=robustad/svdd \
#     experiment_name=robustad_svdd_cvar25_vs_dist_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational \
#     ~evaluation.evaluator.ckpts.summary.cvar10_ema \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=imagesvdd_optuna \
#     hydra.sweeper.study_name=cvar25eff_vs_dist \
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
#     hydra.sweeper.storage='sqlite:///logs/optuna/robustad/svdd.db' \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_agnostic_cap_vs_dist_search \
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
#     logger=none \
#     hparams_search=imagesvdd_optuna \
#     hydra.sweeper.study_name=cap_vs_dist \
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
#     hydra.sweeper.storage='sqlite:///logs/optuna/robustad/svdd.db' \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_agnostic_drift_vs_dist_search \
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
#     logger=none \
#     hparams_search=imagesvdd_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=drift_vs_dist \
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
#     hydra.sweeper.storage='sqlite:///logs/optuna/robustad/svdd.db' \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_agnostic_wasserstein_vs_dist_search \
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
#     logger=none \
#     hparams_search=imagesvdd_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_dist \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

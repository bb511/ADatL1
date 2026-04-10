# RealNVP running commands.
# ========================================================================

# TRAINING.
# =======================

# Semi-supervised cvar25 training.
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=realnvp \
#     run_name=cvar25_t513_high \
#     trainer.gradient_clip_val=0.0 \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.optimizer.lr=0.0009593961598492453 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.loss.nll_scale=0.002 \
#     algorithm.nll_warmup_frac=0.0 \
#     algorithm.encoder.nodes='[48,24,8]' \
#     algorithm.encoder.activation=gelu \
#     trainer=gpu \
#     trainer.devices=[0]


# Semi-supervised cvar10 training.
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=realnvp \
#     run_name=cvar10_trial_339 \
#     evaluator_callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     trainer.gradient_clip_val=0.0 \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.optimizer.lr=0.0007725899485830742 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.loss.nll_scale=0.002 \
#     algorithm.nll_warmup_frac=0.0 \
#     algorithm.encoder.nodes='[48,24,24]' \
#     algorithm.encoder.activation=relu \
#     trainer=gpu \
#     trainer.devices=[0]


# CAP training.
# taskset -c 12-14 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=realnvp_agnostic \
#     run_name=cap_t156_low \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.stable_nll_q99_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_nll_q99 \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.optimizer.lr=7.822713922010138e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.loss.nll_scale=0.002 \
#     algorithm.nll_warmup_frac=0.05 \
#     algorithm.encoder.nodes='[16,4]' \
#     algorithm.encoder.activation=relu \
#     trainer=gpu \
#     trainer.devices=[2]


# Agnostic stability training.
# taskset -c 18-20 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=realnvp_agnostic \
#     run_name=stability_t491_high \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_nll_q99_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_nll_q99 \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.optimizer.lr=0.0008423594125278336 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.loss.nll_scale=0.002 \
#     algorithm.nll_warmup_frac=0.0 \
#     algorithm.encoder.nodes='[48,16,8]' \
#     algorithm.encoder.activation=gelu \
#     trainer=gpu \
#     trainer.devices=[1]


# Agnostic nll-wasserstein training.
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=realnvp_agnostic \
#     run_name=wasserstein_t539_high \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_nll_q99_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_nll_q99 \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.optimizer.lr=6.005882209363415e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.loss.nll_scale=0.002 \
#     algorithm.nll_warmup_frac=0.2 \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.encoder.activation=relu \
#     trainer=gpu \
#     trainer.devices=[1]


# HYPERPARAMETER SEARCHES.
# =====================================================================================

# Semi-Supervised Searches
# ------------------------

# RealNVP hyperparameter search semi-supervised.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=realnvp \
#     experiment_name=realnvp_cvar25_vs_logp_search \
#     callbacks.max_rate_nll_ckpt=null \
#     callbacks.stable_nll_q99_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluator.ckpts.summary.cvar10_ema \
#     ~evaluator.ckpts.single.loss_nll_q99 \
#     ~evaluator.ckpts.single.eff__ascore_loss_nll_full__brate_0_25kHz \
#     evaluator_callbacks.nll_loss_q99=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=realnvp_optuna \
#     hydra.sweeper.study_name=cvar25eff_vs_logp_b16k \
#     hydra.sweeper.n_trials=150 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# RealNVP hyperparameter search semi-supervised nll q99.
# taskset -c 18-20 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=realnvp \
#     experiment_name=realnvp_cvar25_vs_logpq99_search \
#     callbacks.max_rate_nll_ckpt=null \
#     callbacks.stable_nll_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluator.ckpts.summary.cvar10_ema \
#     ~evaluator.ckpts.single.loss_nll_mean_top_vals \
#     ~evaluator.ckpts.single.eff__ascore_loss_nll_full__brate_0_25kHz \
#     evaluator_callbacks.nll_operational_mean=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=realnvp_optuna \
#     optimized_metric_config.sec_metric.callback.name=nll_q99 \
#     hydra.sweeper.study_name=cvar25eff_vs_logpq99_b16k \
#     hydra.sweeper.n_trials=150 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# RealNVP hyperparameter search semi-supervised cvar 10%.
# taskset -c 18-20 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=realnvp \
#     experiment_name=realnvp_cvar10_vs_logp_search \
#     callbacks.max_rate_nll_ckpt=null \
#     callbacks.stable_nll_q99_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     ~evaluator.ckpts.summary.cvar25_ema \
#     ~evaluator.ckpts.single.loss_nll_q99 \
#     ~evaluator.ckpts.single.eff__ascore_loss_nll_full__brate_0_25kHz \
#     evaluator_callbacks.nll_loss_q99=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.wasserstein=null \
#     evaluator_callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     logger=none \
#     hparams_search=realnvp_optuna \
#     hydra.sweeper.study_name=cvar10eff_vs_logp_b16k \
#     hydra.sweeper.n_trials=150 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# CAP Agnostic Searches
# ------------------------

# RealNVP agnostic hyperparameter search - CAP vs nll.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=realnvp_agnostic \
#     experiment_name=realnvp_agnostic_cap_vs_logp_search \
#     callbacks.anomaly_eff=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.stable_nll_q99_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_nll_q99 \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.nll_loss_q99=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=realnvp_optuna \
#     hydra.sweeper.study_name=cap_vs_logp_b16k \
#     hydra.sweeper.n_trials=150 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# RealNVP agnostic hyperparameter search - CAP vs nll q99.
# taskset -c 18-20 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=realnvp_agnostic \
#     experiment_name=realnvp_agnostic_cap_vs_logpq99_search \
#     callbacks.anomaly_eff=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.stable_nll_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_nll_mean_top_vals \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.nll_operational_mean=null \
#     evaluator_callbacks.thres_drift=null \
#     evaluator_callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=realnvp_optuna \
#     optimized_metric_config.sec_metric.callback.name=nll_q99 \
#     hydra.sweeper.study_name=cap_vs_logpq99_b16k \
#     hydra.sweeper.n_trials=150 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# Stability Agnostic Searches
# ---------------------------

# RealNVP agnostic nll and threshold stability.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=realnvp_agnostic \
#     experiment_name=realnvp_agnostic_drift_vs_logp_search \
#     callbacks.anomaly_eff=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_nll_q99_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_nll_q99 \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.nll_loss_q99=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=realnvp_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     +optimized_metric_config.main_metric.callback.params.target_rate=0.25 \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=drift_vs_logp_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=150 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# RealNVP agnostic nll and threshold stability.
# taskset -c 18-20 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=realnvp_agnostic \
#     experiment_name=realnvp_agnostic_drift_vs_logpq99_search \
#     callbacks.anomaly_eff=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_nll_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.single.loss_nll_mean_top_vals \
#     ~evaluator.ckpts.summary.w1dist_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.nll_operational_mean=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.wasserstein=null \
#     logger=none \
#     hparams_search=realnvp_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     +optimized_metric_config.main_metric.callback.params.target_rate=286.0 \
#     optimized_metric_config.main_metric.direction=minimize \
#     optimized_metric_config.sec_metric.callback.name=nll_q99 \
#     hydra.sweeper.study_name=drift_vs_logpq99_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=150 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# Wasserstein Agnostic Searches
# -----------------------------

# RealNVP agnostic nll and wasserstein distance search.
taskset -c 0-2 \
python3 src/train.py \
    -m \
    hydra/launcher=submitit_local \
    hydra.launcher.cpus_per_task=1 \
    hydra.launcher.gpus_per_node=4 \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment=realnvp_agnostic \
    experiment_name=realnvp_agnostic_wasserstein_vs_logp_search \
    callbacks.anomaly_eff=null \
    callbacks.thres_drift=null \
    callbacks.cap_sn_zb=null \
    callbacks.stable_nll_q99_ckpt=null \
    callbacks.thres_drift_ema_ckpt=null \
    callbacks.cap_sn_zb_ema_ckpt=null \
    callbacks.thres_drift_q99_ema_ckpt=null \
    ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
    ~evaluator.ckpts.single.loss_nll_q99 \
    ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
    ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
    evaluator_callbacks.anomaly_efficiency=null \
    evaluator_callbacks.nll_loss_q99=null \
    evaluator_callbacks.cap_sn_zb=null \
    evaluator_callbacks.thres_drift=null \
    logger=none \
    hparams_search=realnvp_optuna \
    optimized_metric_config.main_metric.callback.name=wasserstein \
    optimized_metric_config.main_metric.direction=minimize \
    hydra.sweeper.study_name=wasserstein_vs_logp_b16k \
    hydra.sweeper.direction='[minimize, minimize]' \
    hydra.sweeper.n_trials=150 \
    hydra.sweeper.sampler.n_startup_trials=150 \
    trainer=gpu \
    trainer.max_epochs=1 \
    trainer.devices=[0]


# RealNVP agnostic nll q99 and wasserstein distance search.
# taskset -c 18-20 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=realnvp_agnostic \
#     experiment_name=realnvp_agnostic_drift_vs_logpq99_search \
#     callbacks.anomaly_eff=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.stable_nll_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluator.ckpts.single.loss_nll_mean_top_vals \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.summary.cap_ema_zerobias_vs_SingleNeutrino_E-10-gun \
#     evaluator_callbacks.anomaly_efficiency=null \
#     evaluator_callbacks.nll_operational_mean=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.thres_drift=null \
#     logger=none \
#     hparams_search=realnvp_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     optimized_metric_config.sec_metric.callback.name=nll_q99 \
#     hydra.sweeper.study_name=wasserstein_vs_logpq99_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=150 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

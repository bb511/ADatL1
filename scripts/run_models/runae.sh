# Run the AE models.


# MODEL TRAINING
# =======================

# Semi-supervised cvar25 training.
# taskset -c 15-17 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae \
#     run_name=cvar25_q99_trial_532 \
#     algorithm.optimizer.lr=0.001713976189208681 \
#     algorithm.loss.delta=5.0 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.encoder.nodes='[64,32,8]' \
#     algorithm.input_noise_std=0.001 \
#     trainer=gpu \
#     trainer.devices=[0]


# Semi-supervised cvar10 training.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae \
#     run_name=cvar10_trial_561 \
#     evaluator_callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.optimizer.lr=0.0017124104726253574 \
#     algorithm.loss.delta=10.0 \
#     trainer.gradient_clip_val=1.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.encoder.nodes='[64,32,8]' \
#     algorithm.input_noise_std=0.0003 \
#     trainer=gpu \
#     trainer.devices=[0]


# Agnostic CAP training.
# taskset -c 6-8 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     run_name=cap_q99_trial_430 \
#     algorithm.optimizer.lr=0.0005490035000565241 \
#     algorithm.loss.delta=4.0 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas='[0.9, 0.999]' \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.encoder.nodes='[48,24]' \
#     algorithm.input_noise_std=0.003 \
#     trainer=gpu \
#     trainer.devices=[0]


# Agnostic stability training.
# taskset -c 3-5 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     run_name=stability_q99_trial_535 \
#     algorithm.optimizer.lr=0.0027011957629331706 \
#     algorithm.loss.delta=5.0 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas='[0.9, 0.99]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.encoder.nodes='[64,32,32]' \
#     algorithm.input_noise_std=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]


# Agnostic MSE-wasserstein training.
# taskset -c 9-11 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     run_name=wasserstein_q99_trial_487 \
#     algorithm.optimizer.lr=0.000679024998102717 \
#     algorithm.loss.delta=10.0 \
#     trainer.gradient_clip_val=0.0 \
#     algorithm.optimizer.betas='[0.9, 0.999]' \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.encoder.nodes='[64,32,32]' \
#     algorithm.input_noise_std=0.0003 \
#     trainer=gpu \
#     trainer.devices=[2]


# HYPERPARAMETER SEARCHES.
# =======================


# Semi-Supervised Searches
# ------------------------

# AE hyperparameter search semi-supervised.
taskset -c 0-2 \
python3 src/train.py \
    -m \
    hydra/launcher=submitit_local \
    hydra.launcher.cpus_per_task=1 \
    hydra.launcher.gpus_per_node=4 \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment=ae \
    experiment_name=ae_cvar_vs_mse_search \
    callbacks.max_rate_mse_ckpt=null \
    callbacks.stable_mse_q99_ckpt=null \
    evaluator_callbacks.mse_loss_q99=null \
    evaluator_callbacks.thres_transfer=null \
    evaluator_callbacks.reco=null \
    logger=none \
    hparams_search=ae_optuna \
    hydra.sweeper.study_name=cvar25eff_vs_mse_b16k \
    hydra.sweeper.n_trials=100 \
    hydra.sweeper.sampler.n_startup_trials=150 \
    trainer=gpu \
    trainer.max_epochs=50 \
    trainer.devices=[0]


# AE hyperparameter search semi-supervised mse q99.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae \
#     experiment_name=ae_cvar_vs_mseq99_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.stable_mse_ckpt=null \
#     evaluator.ckpts.last=false \
#     evaluator.ckpts.single=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_optuna \
#     optimized_metric_config.sec_metric.callback.name=mse_q99 \
#     hydra.sweeper.study_name=cvar25eff_vs_mseq99_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# AE hyperparameter search semi-supervised cvar 10%.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae \
#     experiment_name=ae_cvar10_vs_mse_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.stable_mse_ckpt=null \
#     evaluator.ckpts.last=false \
#     evaluator.ckpts.single=null \
#     evaluator_callbacks.reco=null \
#     evaluator_callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     logger=none \
#     hparams_search=ae_optuna \
#     hydra.sweeper.study_name=cvar10eff_vs_mse_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# CAP Agnostic Searches
# ------------------------

# AE agnostic hyperparameter search - CAP vs MSE.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_cap_vs_mse_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_agnostic_optuna \
#     hydra.sweeper.study_name=cap_vs_mse_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# AE agnostic hyperparameter search - CAP vs MSE q99.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_cap_vs_mseq99_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_agnostic_optuna \
#     optimized_metric_config.sec_metric.callback.name=mse_q99 \
#     hydra.sweeper.study_name=cap_vs_mseq99_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# Stability Agnostic Searches
# ---------------------------

# AE agnostic MSE and threshold stability.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_mse_vs_thres_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.last=true \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_agnostic_optuna \
#     optimized_metric_config.main_metric.callback.name=mse_mean_top_vals \
#     +optimized_metric_config.main_metric.callback.params.ckpt_name=last \
#     +optimized_metric_config.main_metric.callback.params.test_ds=zerobias \
#     optimized_metric_config.main_metric.direction=minimize \
#     optimized_metric_config.sec_metric.callback.name=thres_transfer \
#     optimized_metric_config.sec_metric.callback.params.ckpt_name=null \
#     +optimized_metric_config.sec_metric.callback.params.target_rate=0.25 \
#     ~optimized_metric_config.sec_metric.callback.params.test_ds \
#     optimized_metric_config.sec_metric.direction=minimize \
#     hydra.sweeper.study_name=mse_vs_drift_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# AE agnostic MSE and threshold stability q99.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_mseq99_vs_thres_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.last=true \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_agnostic_optuna \
#     optimized_metric_config.main_metric.callback.name=mse_q99 \
#     +optimized_metric_config.main_metric.callback.params.ckpt_name=last \
#     +optimized_metric_config.main_metric.callback.params.test_ds=zerobias \
#     optimized_metric_config.main_metric.direction=minimize \
#     optimized_metric_config.sec_metric.callback.name=thres_transfer \
#     optimized_metric_config.sec_metric.callback.params.ckpt_name=null \
#     +optimized_metric_config.sec_metric.callback.params.target_rate=0.25 \
#     ~optimized_metric_config.sec_metric.callback.params.test_ds \
#     optimized_metric_config.sec_metric.direction=minimize \
#     hydra.sweeper.study_name=mseq99_vs_drift_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# Wasserstein Agnostic Searches
# -----------------------------

# AE agnostic MSE and wasserstein distance search.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_mse_vs_thres_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.last=true \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_agnostic_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_mse_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# AE agnostic MSE q99 and wasserstein distance search.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_mse_vs_thres_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.last=true \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_agnostic_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     optimized_metric_config.sec_metric.callback.name=mse_q99 \
#     hydra.sweeper.study_name=wasserstein_vs_mseq99_b16k \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

#!/usr/bin/env bash

# ========================================================================
# AE RUNNING COMMANDS
# ========================================================================
# These are the running commands for the 250 Hz background rate study.

# ========================================================================
# Training on the NGT cluster
# ========================================================================
# set -euo pipefail

# : "${PROJECT_ROOT:=/shared/adatl1}"
# : "${CODE_DIR:=/tmp/ADatL1}"
# : "${RAW_DATA_DIR:=${PROJECT_ROOT}/raw/parquet_files}"
# : "${RUN_NAME:?Set RUN_NAME, for example: Bernoulli_MI_No_FET_Run_01}"
# : "${MAX_EPOCHS:=100}"
# : "${MI_GAMMA:=0.1}"
# : "${MI_NUM_BINS:=50}"
# : "${DATA_WORKERS:=3}"
# : "${CKPT_PATH:=}"
# : "${MPLCONFIGDIR:=/scratch/adatl1/matplotlib}"

# [[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
#   echo "Invalid RUN_NAME."
#   exit 2
# }
# [[ "$MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
#   echo "MAX_EPOCHS must be a positive integer."
#   exit 2
# }
# [[ "$MI_NUM_BINS" =~ ^[1-9][0-9]*$ ]] && (( MI_NUM_BINS >= 2 )) || {
#   echo "MI_NUM_BINS must be an integer of at least 2."
#   exit 2
# }
# [[ "$DATA_WORKERS" =~ ^[1-9][0-9]*$ ]] || {
#   echo "DATA_WORKERS must be a positive integer."
#   exit 2
# }
# if [[ -n "$CKPT_PATH" && ! -f "$CKPT_PATH" ]]; then
#   echo "Checkpoint not found: $CKPT_PATH"
#   exit 2
# fi

# for dir in extracted processed mlready; do
#   test -d "${PROJECT_ROOT}/data/data_2025E+G/${dir}" || {
#     echo "Missing staged data directory: ${PROJECT_ROOT}/data/data_2025E+G/${dir}"
#     exit 1
#   }
# done

# export PROJECT_ROOT MPLCONFIGDIR
# export NUMEXPR_MAX_THREADS="$DATA_WORKERS"
# export NUMEXPR_NUM_THREADS="$DATA_WORKERS"
# export OMP_NUM_THREADS="$DATA_WORKERS"
# export MKL_NUM_THREADS="$DATA_WORKERS"
# export OPENBLAS_NUM_THREADS="$DATA_WORKERS"
# mkdir -p "$MPLCONFIGDIR"

# test -d "${CODE_DIR}/src" || {
#   echo "Missing source directory: ${CODE_DIR}/src"
#   exit 1
# }
# cd "$CODE_DIR"

# resume_args=()
# if [[ -n "$CKPT_PATH" ]]; then
#   echo "Resuming training from: $CKPT_PATH"
#   # Keep prior checkpoints and plots when resuming the same run.
#   resume_args=(
#     "ckpt_path=$CKPT_PATH"
#     "callbacks.clear_ckpts=null"
#   )
# fi

# exec python3 src/train.py \
#   paths.root_dir="$PROJECT_ROOT" \
#   paths.raw_data_dir="$RAW_DATA_DIR" \
#   experiment=physics/ae \
#   run_name="$RUN_NAME" \
#   logger=mlflow \
#   algorithm.optimizer.lr=0.0019859329798336714 \
#   algorithm.delta=1.0 \
#   algorithm.mi_gamma="$MI_GAMMA" \
#   algorithm.mi_temperature=6.0 \
#   algorithm.mi_sensitive_num_bins="$MI_NUM_BINS" \
#   trainer.gradient_clip_val=5.0 \
#   algorithm.optimizer.betas='[0.9,0.999]' \
#   algorithm.optimizer.weight_decay=1e-06 \
#   algorithm.encoder.nodes='[64,32,8]' \
#   algorithm.input_noise_std=0.0 \
#   data.data_awkward2torch.workers="$DATA_WORKERS" \
#   trainer.max_epochs="$MAX_EPOCHS" \
#   trainer=gpu \
#   trainer.devices='[0]' \
#   "${resume_args[@]}"



# ========================================================================
# TRAINING
# ========================================================================
# ------------------------------------------------------------------------
# Semi-supervised cvar25 training
# ------------------------------------------------------------------------
# taskset -c 0-2 \

python3 src/train.py \
    paths.raw_data_dir=../../03_Data/adl1t_data/parquet_files \
    experiment=physics/ae \
    run_name="Test_Min-Mean-Efficiency_Run_1" \
    logger=mlflow \
    algorithm.optimizer.lr=0.0019859329798336714 \
    algorithm.delta=1.0 \
    algorithm.mi_gamma=0.1 \
    algorithm.mi_temperature=6.0 \
    trainer.gradient_clip_val=5.0 \
    algorithm.optimizer.betas='[0.9,0.999]' \
    algorithm.optimizer.weight_decay=1e-06 \
    algorithm.encoder.nodes='[64,32,8]' \
    algorithm.input_noise_std=0.0 \
    trainer.max_epochs=2 \
    trainer=gpu \
    trainer.devices='[0]'

# ------------------------------------------------------------------------
# Semi-supervised cvar10 training
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/ae \
#     run_name=cvar10_t339 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.optimizer.lr=0.0019789545082545034 \
#     algorithm.delta=10.0 \
#     trainer.gradient_clip_val=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.encoder.nodes='[64,32,32]' \
#     algorithm.input_noise_std=0.0 \
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
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/ae_agnostic \
#     run_name=cap_t175 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.0028379676477468516 \
#     algorithm.delta=10.0 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.encoder.nodes='[64,32,16]' \
#     algorithm.input_noise_std=0.0001 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# Stability training
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/ae_agnostic \
#     run_name=stability_t564 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.002583753082224847 \
#     algorithm.delta=7.0 \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.encoder.nodes='[64,32,24]' \
#     algorithm.input_noise_std=0.0003 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# Wasserstein training
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/ae_agnostic \
#     run_name=wasserstein_t584 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.00047124714609726086 \
#     algorithm.delta=5.0 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.encoder.nodes='[64,32,32]' \
#     algorithm.input_noise_std=0.001 \
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
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/ae \
#     experiment_name=ae_cvar25_vs_mse_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar10_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational \
#     ~evaluation.evaluator.ckpts.summary.cvar10_ema \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_optuna \
#     hydra.sweeper.study_name=cvar25eff_vs_mse \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Semi-supervised search (cvar10)
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/ae \
#     experiment_name=ae_cvar10_vs_mse_search \
#     callbacks.max_rate_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.single.eff__ascore_full__brate_operational \
#     ~evaluation.evaluator.ckpts.summary.cvar25_ema \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.reco=null \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     logger=none \
#     hparams_search=ae_optuna \
#     hydra.sweeper.study_name=cvar10eff_vs_mse \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# CAP search
# ------------------------------------------------------------------------
# AE agnostic hyperparameter search - CAP vs MSE.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.timeout_min=200 \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/ae_agnostic \
#     experiment_name=ae_agnostic_cap_vs_mse_search \
#     callbacks.anomaly_eff=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_optuna \
#     hydra.sweeper.study_name=cap_vs_mse \
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
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/ae_agnostic \
#     experiment_name=ae_agnostic_drift_vs_mse_search \
#     callbacks.anomaly_eff=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.cap_sn_zb=null \
#     evaluation.callbacks.wasserstein=null \
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_optuna \
#     optimized_metric_config.main_metric.callback.name=thres_drift \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=drift_vs_mse \
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
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/ae_agnostic \
#     experiment_name=ae_agnostic_wasserstein_vs_mse_search \
#     callbacks.anomaly_eff=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     evaluation.callbacks.anomaly_efficiency=null \
#     evaluation.callbacks.cap_sn_zb=null \
#     evaluation.callbacks.thres_drift=null \
#     evaluation.callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_optuna \
#     optimized_metric_config.main_metric.callback.name=wasserstein \
#     optimized_metric_config.main_metric.direction=minimize \
#     hydra.sweeper.study_name=wasserstein_vs_mse \
#     hydra.sweeper.direction='[minimize, minimize]' \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=150 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]

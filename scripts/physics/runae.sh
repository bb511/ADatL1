#!/usr/bin/env bash
# ===========================================================================
# STAGE 1 -- train the autoencoder and evaluate it on the validation split
# ===========================================================================
# Fits the model, then (run_validation: true) replays the validation split
# through the best checkpoint and runs every evaluation callback: efficiencies,
# correlation matrices (objective E), latent collapse, AUROC, reconstruction
# plots, anomaly score, Wasserstein and threshold drift, all under
# .../<RUN_NAME>/plots/val/loss_total/. The checkpoint itself is
#
#   checkpoints/<experiment_name>/<RUN_NAME>/loss_total.ckpt
#
# What remains for the Pareto study:
#
#   RUN_NAME=$RUN_NAME bash scripts/physics/runae_pareto_runprobes.sh   # stage 2, leakage L
#   bash scripts/physics/runae_pareto_runcollect.sh                     # stage 3, Pareto front
#
# Usage:
#   bash scripts/physics/runae.sh                      # cvar25_t169 on GPU 0
#   RUN_NAME=AE_30ep MAX_EPOCHS=30 bash scripts/physics/runae.sh
#
# Defaults reproduce the cvar25_t169 reference training:
#
#   python3 src/train.py \
#       paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#       experiment=physics/ae \
#       experiment_name=Pareto-Front-261002 \
#       run_name=cvar25_t169 \
#       algorithm.encoder.nodes='[64,32,8]' \
#       algorithm.input_noise_std=0.0 \
#       algorithm.delta=10.0 \
#       algorithm.optimizer.betas='[0.9,0.999]' \
#       algorithm.optimizer.lr=0.0019859329798336714 \
#       algorithm.optimizer.weight_decay=1e-06 \
#       trainer.gradient_clip_val=5.0 \
#       trainer=gpu \
#       trainer.devices=[0]
#
# The hyperparameters live in configs/experiment/physics/ae.yaml (and
# experiment_name there), not here, so stage 2 and the test evaluation see the same model.
# Only the run name and the trainer are defaulted in this script. NOTE: a new
# training clears checkpoints/<experiment_name>/<RUN_NAME>, so set RUN_NAME for
# anything you want to keep next to an existing cvar25_t169.
#
# Every knob is an environment variable; see scripts/physics/_stage_common.sh
# for the shared ones (EXPERIMENT, TRAINER, CPU_THREADS, RAW_DATA_DIR and the
# model hyperparameters, which MUST match in stage 2 and runae_test.sh).

: "${RUN_NAME:=cvar25_t169}"
# The batch wrappers export their own TRAINER (cpu on the cluster), which wins.
: "${TRAINER:=gpu}"
export RUN_NAME TRAINER

source "$(dirname "${BASH_SOURCE[0]}")/_stage_common.sh"

# Unset means "whatever the composed experiment says". physics/ae inherits 200,
# which is not a batch-sized run, so set MAX_EPOCHS for an ad-hoc training; the
# Pareto study's training overlay carries its agreed 30 in the config itself.
: "${MAX_EPOCHS:=}"
# Resume an interrupted run from a checkpoint. Leave empty to start fresh.
: "${CKPT_PATH:=}"

epoch_args=()
if [[ -n "$MAX_EPOCHS" ]]; then
  [[ "$MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
    echo "MAX_EPOCHS must be a positive integer." >&2
    exit 2
  }
  epoch_args=("trainer.max_epochs=$MAX_EPOCHS")
fi

# Gradient clipping is a config value like any other; only override it when the
# caller explicitly asks.
clip_args=()
if [[ -n ${GRAD_CLIP+x} ]]; then
  clip_args=("trainer.gradient_clip_val=$GRAD_CLIP")
fi
if [[ -n "$CKPT_PATH" && ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  exit 2
fi

resume_args=()
if [[ -n "$CKPT_PATH" ]]; then
  echo "Resuming training from: $CKPT_PATH"
  # Keep prior checkpoints and plots when resuming the same run.
  resume_args=("ckpt_path=$CKPT_PATH" "callbacks.clear_ckpts=null")
fi

stage_banner "STAGE 1  TRAIN + VAL EVALUATION  (max_epochs=${MAX_EPOCHS:-from config})"

exec python3 src/train.py \
  "${COMMON_ARGS[@]}" \
  "${ALGO_ARGS[@]}" \
  "${clip_args[@]}" \
  "${epoch_args[@]}" \
  "${resume_args[@]}"

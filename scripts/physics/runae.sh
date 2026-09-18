#!/usr/bin/env bash
# ===========================================================================
# STAGE 1 of 4 -- train the autoencoder
# ===========================================================================
# Fits the model and runs the ordinary AE evaluation callbacks (reconstruction
# plots, anomaly score, Wasserstein, threshold drift). Its scientific output is
# the checkpoint:
#
#   checkpoints/<experiment_name>/<RUN_NAME>/loss_total.ckpt
#
# Nothing here computes a Pareto metric. The analysis is stages 2-4:
#
#   RUN_NAME=$RUN_NAME bash scripts/physics/runprobes.sh    # leakage L
#   RUN_NAME=$RUN_NAME bash scripts/physics/runmetrics.sh   # eff / E / collapse / auroc
#   bash scripts/physics/runcollect.sh                      # Pareto front
#
# Usage:
#   RUN_NAME=AE_30ep MAX_EPOCHS=30 bash scripts/physics/runae.sh
#
# Every knob is an environment variable; see scripts/physics/_stage_common.sh
# for the shared ones (EXPERIMENT, TRAINER, CPU_THREADS, RAW_DATA_DIR and the
# model hyperparameters, which MUST match in stages 2 and 3).

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

stage_banner "STAGE 1/4  TRAIN  (max_epochs=${MAX_EPOCHS:-from config})"

exec python3 src/train.py \
  "${COMMON_ARGS[@]}" \
  "${ALGO_ARGS[@]}" \
  "${clip_args[@]}" \
  "${epoch_args[@]}" \
  "${resume_args[@]}"

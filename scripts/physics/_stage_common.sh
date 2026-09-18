#!/usr/bin/env bash
# Shared settings for the four pipeline stages. Sourced, never executed.
#
#   stage 1  runae.sh       train the AE            -> loss_total.ckpt
#   stage 2  runprobes.sh   four leakage probes     -> leakage_probes.json
#   stage 3  runmetrics.sh  remaining Pareto metrics-> eff/correlation/collapse/auroc
#   stage 4  runcollect.sh  aggregate + Pareto front
#
# Stages 2 and 3 read only the checkpoint stage 1 wrote, so they are independent
# of each other and may run at the same time. Both must be given the SAME
# EXPERIMENT, RUN_NAME and algorithm overrides as the stage-1 run: the checkpoint
# stores weights but not the config, and both loaders use strict=True, so a
# mismatched architecture fails loudly instead of measuring the wrong model.

set -euo pipefail

# --- identity ---------------------------------------------------------------
# RUN_NAME is the only thing that links the four stages together. It is required
# rather than defaulted, because a wrong default would silently analyse another
# run's checkpoint.
: "${RUN_NAME:?Set RUN_NAME to the run you are training or analysing, e.g. RUN_NAME=AE_30ep_gamma0.1}"

# Must match the composed experiment's experiment_name. physics/ae and its
# ae_metrics overlay both use physics_ae_models.
: "${EXPERIMENT:=physics/ae}"

# Overrides the experiment's own experiment_name, which is the directory every
# stage addresses: checkpoints/<experiment_name>/<run_name>. Set it to give one
# study its own directory, e.g. EXPERIMENT_NAME=Pareto_Front_260918. Leave empty
# to keep the experiment's default. Must be identical in every stage of a run.
: "${EXPERIMENT_NAME:=}"

# --- paths ------------------------------------------------------------------
# Default to the local checkout layout used on the laptop; the batch wrappers in
# batch/ override all of these.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${RAW_DATA_DIR:=${REPO_ROOT}/../../03_Data/adl1t_data/parquet_files}"
: "${PROJECT_ROOT:=${REPO_ROOT}}"
# configs/paths/default.yaml reads this from the environment, not from an
# override: output_root: ${oc.env:ADL1T_OUTPUT_ROOT,${paths.root_dir}}. It is
# what decides where checkpoints/<experiment_name>/<run_name> lives, so stages
# 2 and 3 MUST see the same value as the stage 1 that wrote the checkpoint.
: "${ADL1T_OUTPUT_ROOT:=${PROJECT_ROOT}}"

# --- compute ----------------------------------------------------------------
# gpu or cpu. CPU is the cluster default (supervisor decision, 2026-09-18): the
# model is ~20k parameters, so training is bound by host-side data movement, and
# the shared CPU pool is ~5200 slots against ~50 for GPU.
: "${TRAINER:=cpu}"

# Sizes only the awkward->torch conversion.
: "${DATA_WORKERS:=3}"

# Intra-op thread budget. Must track request_cpus on the cluster, NOT
# DATA_WORKERS: nproc reports the machine, not the slot, so it is passed in.
: "${CPU_THREADS:=${DATA_WORKERS}}"

: "${MPLCONFIGDIR:=${PROJECT_ROOT}/.matplotlib}"

# --- validation -------------------------------------------------------------
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "Invalid RUN_NAME: '$RUN_NAME'" >&2
  exit 2
}

export MPLCONFIGDIR PROJECT_ROOT ADL1T_OUTPUT_ROOT
export NUMEXPR_MAX_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
mkdir -p "$MPLCONFIGDIR"

# The trainer config group takes a devices list on gpu and a count on cpu.
if [[ "$TRAINER" == "gpu" ]]; then
  TRAINER_ARGS=(trainer=gpu "trainer.devices=[0]")
else
  TRAINER_ARGS=(trainer=cpu trainer.devices=1)
fi

# Every stage shares these, so an accidental divergence between stage 1 and the
# stages that read its checkpoint is impossible.
COMMON_ARGS=(
  paths.raw_data_dir="$RAW_DATA_DIR"
  experiment="$EXPERIMENT"
  run_name="$RUN_NAME"
  logger=mlflow
  data.data_awkward2torch.workers="$DATA_WORKERS"
  "${TRAINER_ARGS[@]}"
)
if [[ -n "$EXPERIMENT_NAME" ]]; then
  COMMON_ARGS+=(experiment_name="$EXPERIMENT_NAME")
fi

# --- model hyperparameters --------------------------------------------------
# The configs are the source of truth. Nothing is passed on the command line
# unless it is explicitly set in the environment, so an unset variable means
# "whatever the composed experiment says" rather than a literal buried in this
# script silently overriding it.
#
# This matters beyond tidiness. The resolved manifest each run writes is the
# scientific record of what was trained, and Phase 2 validates runs against it;
# a script-level default that disagrees with the config produces a record that
# does not describe the model. Until 2026-09-18 this file did exactly that,
# shadowing five of them:
#
#   lr 0.0019859329798336714 vs 0.0013029941778430407   weight_decay 1e-06 vs 0.001
#   delta 1.0 vs 3.0         input_noise_std 0.0 vs 1e-04   grad clip 5.0 vs 0.0
#
# To change a hyperparameter, change the config. To try one ad hoc, set the
# variable for that invocation -- and set the same one for stages 2 and 3, or
# the run_manifest fingerprint check will stop them.
: "${PARETO_CANDIDATE:=0}"

_maybe() {
  # _maybe VARNAME hydra.key  -> append the override only if VARNAME is set
  local name="$1" key="$2"
  [[ -n ${!name+x} ]] && ALGO_ARGS+=("${key}=${!name}")
  return 0
}

if [[ -n ${MI_NUM_BINS+x} ]]; then
  [[ "$MI_NUM_BINS" =~ ^[1-9][0-9]*$ ]] && (( MI_NUM_BINS >= 2 )) || {
    echo "MI_NUM_BINS must be an integer of at least 2." >&2
    exit 2
  }
fi

ALGO_ARGS=()

if (( PARETO_CANDIDATE )); then
  # A Pareto-study run is parameterised through pareto_study.candidate, NOT
  # through algorithm.*: the study experiment derives algorithm.mi_gamma and the
  # rest FROM the candidate, and it is the candidate that configuration_id is
  # built from. Overriding algorithm.mi_gamma directly would train the right
  # model and then file it under the wrong grid point, which Phase 2 catches only
  # at the very end. These five are the searched parameters, so they are required
  # rather than optional.
  : "${SEED:?PARETO_CANDIDATE=1 requires SEED}"
  : "${MI_GAMMA:?PARETO_CANDIDATE=1 requires MI_GAMMA}"
  : "${MI_NUM_BINS:?PARETO_CANDIDATE=1 requires MI_NUM_BINS}"
  : "${ARCHITECTURE_ID:?PARETO_CANDIDATE=1 requires ARCHITECTURE_ID}"
  : "${ENCODER_NODES:?PARETO_CANDIDATE=1 requires ENCODER_NODES}"
  ALGO_ARGS=(
    pareto_study.candidate.autoencoder_seed="$SEED"
    pareto_study.candidate.mi_gamma="$MI_GAMMA"
    pareto_study.candidate.mi_sensitive_num_bins="$MI_NUM_BINS"
    pareto_study.candidate.architecture_id="$ARCHITECTURE_ID"
    pareto_study.candidate.encoder_nodes="$ENCODER_NODES"
  )
else
  _maybe SEED             seed
  _maybe LR               algorithm.optimizer.lr
  _maybe WEIGHT_DECAY     algorithm.optimizer.weight_decay
  _maybe BETAS            algorithm.optimizer.betas
  _maybe DELTA            algorithm.delta
  _maybe MI_GAMMA         algorithm.mi_gamma
  _maybe MI_TEMPERATURE   algorithm.mi_temperature
  _maybe MI_NUM_BINS      algorithm.mi_sensitive_num_bins
  _maybe ENCODER_NODES    algorithm.encoder.nodes
  _maybe INPUT_NOISE_STD  algorithm.input_noise_std
fi

stage_banner() {
  echo "==============================================================="
  echo " $1"
  echo "   run_name:    $RUN_NAME"
  echo "   experiment:  $EXPERIMENT"
  echo "   trainer:     $TRAINER  (threads=$CPU_THREADS, workers=$DATA_WORKERS)"
  echo "   raw data:    $RAW_DATA_DIR"
  echo "   output root: $ADL1T_OUTPUT_ROOT"
  echo "   checkpoints: $ADL1T_OUTPUT_ROOT/checkpoints/<experiment_name>/$RUN_NAME"
  echo "==============================================================="
}

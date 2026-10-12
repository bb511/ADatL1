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
# Usage (local):
#
#   bash scripts/physics/runae.sh                       # one Pareto grid point
#   MAX_EPOCHS=200 bash scripts/physics/runae.sh        # a full-length run
#   MI_GAMMA=0.3 MI_NUM_BINS=50 bash scripts/physics/runae.sh
#
# Local defaults. Each applies only when the variable is not already set, so the
# cluster (batch/runae_pareto.sh, which sets all of them from
# batch/pareto_runs.txt) never sees one:
#
#   EXPERIMENT_NAME   local-ae                  -> checkpoints/local-ae/
#   MAX_EPOCHS        3                         only when unset; MAX_EPOCHS= (empty)
#                                               means the config's value (200 for
#                                               physics/pareto_fet_train)
#   PARETO_CANDIDATE  1                         parameterise through pareto_study.candidate
#   EXPERIMENT        physics/pareto_fet_train
#   SEED              180524
#   MI_GAMMA          0.1
#   MI_NUM_BINS       40
#   ARCHITECTURE_ID   h64_32
#   ENCODER_NODES     [64,32,8]
#   RUN_NAME          Seed<seed>_Gamma_<gamma>_Bins_<bins>_architecture_<arch>_Run<NN>,
#                     the pattern of runae_pareto_makegrid.sh, with NN one past the
#                     highest RunNN of that grid point already in
#                     checkpoints/<experiment_name>/. A new training clears its own
#                     run directory, so every call gets a new name instead of
#                     overwriting the previous run.
#   TRAINER           gpu
#
# The run name is printed in the banner. Stages 2 and 3 need it and the same
# EXPERIMENT_NAME and grid point:
#
#   EXPERIMENT=physics/pareto_fet EXPERIMENT_NAME=local-ae PARETO_CANDIDATE=1 \
#     SEED=180524 MI_GAMMA=0.1 MI_NUM_BINS=40 ARCHITECTURE_ID=h64_32 \
#     ENCODER_NODES='[64,32,8]' RUN_NAME=<run name> \
#     bash scripts/physics/runae_pareto_runprobes.sh                    # stage 2
#   EXPERIMENT_NAME=local-ae bash scripts/physics/runae_pareto_runcollect.sh  # stage 3
#
# PARETO_CANDIDATE=0 trains the plain physics/ae reference instead (RUN_NAME
# cvar25_t169 unless set); its hyperparameters live in
# configs/experiment/physics/ae.yaml.
#
# Every knob is an environment variable; see scripts/physics/_stage_common.sh
# for the shared ones (TRAINER, CPU_THREADS, RAW_DATA_DIR and the model
# hyperparameters, which MUST match in stage 2 and runae_test.sh).

: "${EXPERIMENT_NAME:=local-ae}"
# No colon: only an UNSET MAX_EPOCHS becomes 3. Empty means "the config's
# epochs"; batch/runae_pareto.sh exports it empty unless the submit sets it, so
# a cluster job never trains for 3 epochs by accident.
: "${MAX_EPOCHS=3}"
: "${PARETO_CANDIDATE:=1}"
if [[ "$PARETO_CANDIDATE" == 1 ]]; then
  : "${EXPERIMENT:=physics/pareto_fet_train}"
  : "${SEED:=180524}"
  : "${MI_GAMMA:=0.1}"
  : "${MI_NUM_BINS:=40}"
  : "${ARCHITECTURE_ID:=h64_32}"
  : "${ENCODER_NODES:=[64,32,8]}"
  if [[ -z "${RUN_NAME:-}" ]]; then
    # Gamma as runae_pareto_makegrid.sh writes it: 1.0 -> 1, 0.10 -> 0.1.
    _gamma="$(python3 -c 'import sys; v = float(sys.argv[1]); print(int(v) if v.is_integer() else repr(v))' "$MI_GAMMA" 2>/dev/null)" || {
      echo "MI_GAMMA must be a number, got '$MI_GAMMA'." >&2
      exit 2
    }
    _base="Seed${SEED}_Gamma_${_gamma}_Bins_${MI_NUM_BINS}_architecture_${ARCHITECTURE_ID}"
    # The same root _stage_common.sh resolves: ADL1T_OUTPUT_ROOT, else
    # PROJECT_ROOT, else the checkout.
    _repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    _ckpts="${ADL1T_OUTPUT_ROOT:-${PROJECT_ROOT:-$_repo}}/checkpoints/${EXPERIMENT_NAME}"
    _last=0
    for _dir in "${_ckpts}/${_base}"_Run*; do
      _num="${_dir##*_Run}"
      if [[ -e "$_dir" && "$_num" =~ ^[0-9]+$ ]] && (( 10#$_num > _last )); then
        _last=$((10#$_num))
      fi
    done
    RUN_NAME="${_base}_Run$(printf '%02d' $((_last + 1)))"
    unset _gamma _base _repo _ckpts _last _dir _num
  fi
fi
: "${RUN_NAME:=cvar25_t169}"
# The batch wrappers export their own TRAINER (cpu on the cluster), which wins.
: "${TRAINER:=gpu}"
export EXPERIMENT_NAME PARETO_CANDIDATE RUN_NAME TRAINER

source "$(dirname "${BASH_SOURCE[0]}")/_stage_common.sh"

# Empty means "whatever the composed experiment says" (200 for physics/ae and
# physics/pareto_fet_train). Unset was already turned into the local 3 above.
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

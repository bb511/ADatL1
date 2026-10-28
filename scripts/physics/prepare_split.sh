#!/usr/bin/env bash
# ===========================================================================
# Build the data cache of a NEW train/valid/test split, once, before any job
# of a study on that split starts (see src/prepare_split.py for the details).
# ===========================================================================
#
#   DATA_SPLIT_SEED=666 bash scripts/physics/prepare_split.sh    # local
#   condor_submit batch/prepare_split.sub                        # lxplus
#
# It writes mlready/<processor>_default_split<seed>/ next to the original
# cache and finishes with mlready/<...>_split<seed>/splits/split_info.yaml
# (status: complete). Every stage started with the same DATA_SPLIT_SEED reads
# that cache and refuses to start without the record. Then, for example:
#
#   condor_submit EXPERIMENT_NAME=Pareto-Front-261015-split666 DATA_SPLIT_SEED=666 batch/runae_pareto.sub
#   condor_submit EXPERIMENT_NAME=Pareto-Front-261015-split666 DATA_SPLIT_SEED=666 batch/runprobes_pareto.sub
#   condor_submit EXPERIMENT_NAME=Pareto-Front-261015-split666 batch/runcollect.sub
#
# Only the data split changes: weight seed, architecture and every other
# hyperparameter stay what the study config says.

: "${DATA_SPLIT_SEED:?Set DATA_SPLIT_SEED to the split seed to prepare, e.g. DATA_SPLIT_SEED=666}"
[[ "$DATA_SPLIT_SEED" =~ ^[0-9]+$ ]] || {
  echo "DATA_SPLIT_SEED must be a non-negative integer, got '$DATA_SPLIT_SEED'." >&2
  exit 2
}
(( 10#$DATA_SPLIT_SEED != 42 )) || {
  echo "Split 42 is the original split; its cache already exists and is not rebuilt." >&2
  exit 2
}

# The study experiment, so the cache holds exactly the features the study
# trains and evaluates on. Its candidate defaults resolve the config; the model
# itself is never built.
: "${EXPERIMENT:=physics/pareto_fet_train}"
PARETO_CANDIDATE=0
RUN_NAME="prepare_split${DATA_SPLIT_SEED}"
# Satisfies the split-name check in _stage_common.sh; nothing is written there.
EXPERIMENT_NAME="prepare-split${DATA_SPLIT_SEED}"
export EXPERIMENT PARETO_CANDIDATE RUN_NAME EXPERIMENT_NAME DATA_SPLIT_SEED

source "$(dirname "${BASH_SOURCE[0]}")/_stage_common.sh"

stage_banner "PREPARE DATA SPLIT  (seed $DATA_SPLIT_SEED)"

exec python3 src/prepare_split.py \
  "${COMMON_ARGS[@]}" \
  task_name=prepare_split

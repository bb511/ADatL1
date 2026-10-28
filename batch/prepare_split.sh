#!/bin/bash
# Build the data cache of one new split on a worker node. The seed arrives via
# the queue statement in batch/prepare_split.sub; see
# scripts/physics/prepare_split.sh and src/prepare_split.py.

STAGE_LABEL="Prepare data split ${DATA_SPLIT_SEED:-?}"
export STAGE_LABEL

: "${DATA_SPLIT_SEED:?Set DATA_SPLIT_SEED via the queue statement in batch/prepare_split.sub}"
export DATA_SPLIT_SEED

# _stage_env.sh requires a RUN_NAME; this job has no run, only a cache.
: "${RUN_NAME:=prepare_split${DATA_SPLIT_SEED}}"
export RUN_NAME

# The executable is copied into the sandbox, so resolve everything through the
# bind-mounted checkout (see batch/runae_pareto.sh).
: "${CODE_DIR:=/eos/user/l/lbehrens/adatl1/ADatL1}"
export CODE_DIR

STAGE_ENV="${CODE_DIR}/batch/_stage_env.sh"
[[ -r "$STAGE_ENV" ]] || {
  echo "FATAL: cannot read $STAGE_ENV" >&2
  exit 2
}
source "$STAGE_ENV"

echo "DATA_SPLIT_SEED:   $DATA_SPLIT_SEED"
echo "Data cache root:   ${PROJECT_ROOT}/data/data_2025E+G/mlready"
echo
exec bash "${CODE_DIR}/scripts/physics/prepare_split.sh"

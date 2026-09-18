#!/bin/bash
# STAGE 1 of the Pareto study -- train ONE grid point.
#
# One job per run; the grid point arrives through the environment, set by the
# queue statement in batch/runae_pareto.sub from batch/pareto_runs.txt.
#
# Everything that is not a searched parameter -- learning rate, weight decay,
# delta, temperature, noise, gradient clipping -- is frozen by the study manifest
# (configs/pareto_study/fet_et.yaml, inherited through physics/pareto_fet) and is
# deliberately NOT overridden here. The resolved manifest is the scientific
# record of what was trained; a command-line override that disagrees with it is
# exactly the divergence that record exists to prevent.

STAGE_LABEL="Pareto stage 1/4: train one grid point"
export STAGE_LABEL

# The study experiment, minus the training callbacks that need the auxiliary
# signal datasets resident during fit. See the config for the measured saving.
: "${EXPERIMENT:=physics/pareto_fet_train}"
: "${EXPERIMENT_NAME:=Pareto_Front_260918}"
export EXPERIMENT EXPERIMENT_NAME

# Parameterise through pareto_study.candidate, not algorithm.*: the candidate is
# what configuration_id is built from, and therefore what pairs the two seeds of
# one grid point together downstream.
: "${PARETO_CANDIDATE:=1}"
export PARETO_CANDIDATE

: "${RUN_NAME:?Set RUN_NAME via the queue statement in batch/runae_pareto.sub}"
: "${SEED:?Set SEED via the queue statement}"
: "${MI_GAMMA:?Set MI_GAMMA via the queue statement}"
: "${MI_NUM_BINS:?Set MI_NUM_BINS via the queue statement}"
: "${ARCHITECTURE_ID:?Set ARCHITECTURE_ID via the queue statement}"
: "${ENCODER_NODES_US:?Set ENCODER_NODES_US via the queue statement}"

# The job list stores the encoder shape as 64_32_8 rather than [64,32,8], because
# HTCondor splits a queue-from-file line on commas and a bracketed list would be
# torn into three fields.
ENCODER_NODES="[${ENCODER_NODES_US//_/,}]"
export SEED MI_GAMMA MI_NUM_BINS ARCHITECTURE_ID ENCODER_NODES

# HTCondor copies the executable into the sandbox and renames it
# condor_exec.exe, so "$(dirname "$0")" is the scratch directory, not the repo.
# Nothing else is transferred, so a relative source silently finds nothing: the
# job then runs on with CODE_DIR unset and dies at the final exec with status
# 127 (cluster 333803, 2026-09-18). Resolve it through CODE_DIR, the
# bind-mounted checkout, exactly as the final exec already does.
: "${CODE_DIR:=/eos/user/l/lbehrens/adatl1/ADatL1}"
export CODE_DIR

STAGE_ENV="${CODE_DIR}/batch/_stage_env.sh"
[[ -r "$STAGE_ENV" ]] || {
  echo "FATAL: cannot read $STAGE_ENV" >&2
  echo "       CODE_DIR=$CODE_DIR -- is the checkout bind-mounted here?" >&2
  exit 2
}
source "$STAGE_ENV"

echo "SEED:              $SEED"
echo "MI_GAMMA:          $MI_GAMMA"
echo "MI_NUM_BINS:       $MI_NUM_BINS"
echo "ARCHITECTURE_ID:   $ARCHITECTURE_ID"
echo "ENCODER_NODES:     $ENCODER_NODES"
echo "max_epochs:        from configs/experiment/physics/pareto_fet_train.yaml"
echo
echo "Running scripts/physics/runae.sh ..."
exec bash "${CODE_DIR}/scripts/physics/runae.sh"

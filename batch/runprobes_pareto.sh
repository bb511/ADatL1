#!/bin/bash
# STAGE 2 of the Pareto study -- one grid point per job.
#
# Takes the SAME job list as stage 1 (batch/pareto_runs.txt) rather than a list
# of run names, because it needs the grid point, not just the directory: the
# algorithm config has to be reproduced exactly for the run_manifest fingerprint
# check to pass, and the candidate values are what reproduce it.
#
# EXPERIMENT is physics/pareto_fet, not the _train overlay stage 1 used. That
# overlay differs only in `data` and `callbacks`, never in `algorithm`, so the
# fingerprint still matches -- while this stage does need the evaluation
# callbacks the training overlay strips.

STAGE_LABEL="Pareto stage 2/4"
export STAGE_LABEL

: "${EXPERIMENT:=physics/pareto_fet}"
: "${PARETO_CANDIDATE:=1}"
export EXPERIMENT PARETO_CANDIDATE

# Read the merged stage-1 tree in place on EOS, not the empty job sandbox.
: "${ADL1T_OUTPUT_ROOT:=/eos/user/l/lbehrens/adatl1/ADatL1/outputs}"
export ADL1T_OUTPUT_ROOT

: "${RUN_NAME:?Set RUN_NAME via the queue statement}"
: "${SEED:?Set SEED via the queue statement}"
: "${MI_GAMMA:?Set MI_GAMMA via the queue statement}"
: "${MI_NUM_BINS:?Set MI_NUM_BINS via the queue statement}"
: "${ARCHITECTURE_ID:?Set ARCHITECTURE_ID via the queue statement}"
: "${ENCODER_NODES_US:?Set ENCODER_NODES_US via the queue statement}"

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

# Resolved from PARETO_EXPERIMENT_NAME in batch/_stage_env.sh, so every stage of
# the study addresses the same checkpoints/<experiment_name>/ directory.
: "${EXPERIMENT_NAME:=$PARETO_EXPERIMENT_NAME}"
export EXPERIMENT_NAME
echo "EXPERIMENT_NAME:   $EXPERIMENT_NAME"

echo "SEED / GAMMA / BINS / ARCH: $SEED / $MI_GAMMA / $MI_NUM_BINS / $ARCHITECTURE_ID"
echo
echo "Running scripts/physics/runprobes.sh ..."
exec bash "${CODE_DIR}/scripts/physics/runprobes.sh"

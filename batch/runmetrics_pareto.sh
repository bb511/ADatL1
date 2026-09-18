#!/bin/bash
# STAGE 3 of the Pareto study -- one grid point per job.
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

STAGE_LABEL="Pareto stage 3/4"
export STAGE_LABEL

: "${EXPERIMENT:=physics/pareto_fet}"
: "${EXPERIMENT_NAME:=Pareto_Front_260918}"
: "${PARETO_CANDIDATE:=1}"
export EXPERIMENT EXPERIMENT_NAME PARETO_CANDIDATE

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

source "$(dirname "$0")/_stage_env.sh"

echo "SEED / GAMMA / BINS / ARCH: $SEED / $MI_GAMMA / $MI_NUM_BINS / $ARCHITECTURE_ID"
echo
echo "Running scripts/physics/runmetrics.sh ..."
exec bash "${CODE_DIR}/scripts/physics/runmetrics.sh"

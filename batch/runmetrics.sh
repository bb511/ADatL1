#!/bin/bash
# STAGE 3 of 4 on HTCondor -- the remaining Pareto metrics.
#
# One job per run. RUN_NAME comes from the queue statement in
# batch/runmetrics.sub. Reads the stage-1 checkpoint and writes eff_summary.json,
# mean_correlations.json, collapse_summary.json and auroc_summary.json beside it.
#
# Independent of stage 2: both read the same checkpoint and write into disjoint
# subdirectories, so the two can be submitted at the same time.
#
# EXPERIMENT defaults to physics/ae_metrics, the overlay that inherits physics/ae
# unchanged and only switches on the four summary callbacks. For a Pareto-study
# run set EXPERIMENT=physics/pareto_fet instead.

STAGE_LABEL="AE evaluation metrics (stage 3/4)"
export STAGE_LABEL

: "${ADL1T_OUTPUT_ROOT:=/eos/user/l/lbehrens/adatl1/ADatL1/outputs}"
: "${EXPERIMENT:=physics/ae_metrics}"
export ADL1T_OUTPUT_ROOT EXPERIMENT

source "$(dirname "$0")/_stage_env.sh"

: "${ARCHITECTURE_ID:=ae_standalone}"
export ARCHITECTURE_ID

echo "ARCHITECTURE_ID:   $ARCHITECTURE_ID"
echo
echo "Running scripts/physics/runmetrics.sh ..."
exec bash "${CODE_DIR}/scripts/physics/runmetrics.sh"

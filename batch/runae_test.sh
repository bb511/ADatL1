#!/bin/bash
# TEST evaluation on HTCondor -- one job per run: loss_total.ckpt on the held-out
# test split, written beside the checkpoint on EOS under plots/test/.
#
# Thin wrapper around scripts/physics/runae_test.sh, which reads everything it
# needs to rebuild the stage-1 config from the run's own run_manifest.yaml and
# resolved_config.yaml. The queue therefore only needs run names, not the grid
# point (unlike batch/runae_pareto.sh and batch/runprobes_pareto.sh).
#
#   condor_submit batch/runae_test.sub
#   condor_submit RUNS=batch/my_runs.txt EXPERIMENT_NAME=Pareto-Front-260928 batch/runae_test.sub

STAGE_LABEL="Test evaluation of loss_total.ckpt"
export STAGE_LABEL

# Read and write the merged tree on EOS in place, like stage 2: the checkpoint is
# already there and the test plots belong beside it.
: "${ADL1T_OUTPUT_ROOT:=/eos/user/l/lbehrens/adatl1/ADatL1/outputs}"
export ADL1T_OUTPUT_ROOT

: "${RUN_NAME:?Set RUN_NAME via the queue statement}"

# HTCondor copies the executable into the sandbox, so resolve everything through
# CODE_DIR, the bind-mounted checkout (see batch/runprobes_pareto.sh).
: "${CODE_DIR:=/eos/user/l/lbehrens/adatl1/ADatL1}"
export CODE_DIR
STAGE_ENV="${CODE_DIR}/batch/_stage_env.sh"
[[ -r "$STAGE_ENV" ]] || { echo "FATAL: cannot read $STAGE_ENV" >&2; exit 2; }
source "$STAGE_ENV"

# The submit file passes EXPERIMENT_NAME through; empty means the study default.
: "${EXPERIMENT_NAME:=$PARETO_EXPERIMENT_NAME}"
export EXPERIMENT_NAME
echo "EXPERIMENT_NAME:   $EXPERIMENT_NAME"

# _stage_env.sh exports EXPERIMENT=physics/ae; runae_test.sh ignores it and takes
# the experiment from the run itself (TEST_EXPERIMENT overrides that).
echo
echo "Running scripts/physics/runae_test.sh ..."
exec bash "${CODE_DIR}/scripts/physics/runae_test.sh"

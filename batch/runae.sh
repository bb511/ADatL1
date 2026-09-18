#!/bin/bash
# STAGE 1 of 4 on HTCondor -- train the autoencoder.
#
# Fits the model and runs the ordinary AE evaluation callbacks. Its output is
# the checkpoint:
#
#   outputs/checkpoints/<experiment_name>/<RUN_NAME>/loss_total.ckpt
#
# The leakage probes used to run in this same job and cost ~27 min per run on
# top of training. They are now stage 2 (batch/runprobes.sub), and the remaining
# Pareto metrics are stage 3 (batch/runmetrics.sub). Both read the checkpoint
# this job produces, so submit them only after these jobs have returned and
# their outputs/ trees have been merged onto EOS.
#
# This job WRITES a new tree, so ADL1T_OUTPUT_ROOT stays in the job scratch dir
# and transfer_output_files brings it home. Stages 2 and 3 instead read the
# merged tree in place, which is why their wrappers point ADL1T_OUTPUT_ROOT at
# EOS rather than at scratch.

STAGE_LABEL="AE training (stage 1/4)"
export STAGE_LABEL

: "${RUN_NAME:=AE_LXPLUS_30ep}"
export RUN_NAME

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

: "${MAX_EPOCHS:=30}"
export MAX_EPOCHS

echo "MAX_EPOCHS:        $MAX_EPOCHS"
echo
echo "Running scripts/physics/runae.sh ..."
exec bash "${CODE_DIR}/scripts/physics/runae.sh"

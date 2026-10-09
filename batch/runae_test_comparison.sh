#!/bin/bash
# The gamma = 0 comparison of tested runs (scripts/physics/runae_test_comparison.sh)
# as an HTCondor job.
#
# It runs on a worker because it reads and writes the checkpoint tree on EOS, and
# an apptainer shell on lxplus could not reach EOS (2026-10-08: "chdir
# /eos/user/l/lbehrens/adatl1/ADatL1: permission denied"). Batch jobs in the same
# container can, as every stage shows. Light: pandas/matplotlib, no torch, no data.
#
#   condor_submit batch/runae_test_comparison.sub                        # batch/test_runs.txt
#   condor_submit RUNS=batch/my_runs.txt batch/runae_test_comparison.sub
#   condor_submit "RUN_NAMES=RunA RunB" batch/runae_test_comparison.sub
#   condor_submit FORCE=1 ... batch/runae_test_comparison.sub            # redraw comparison_gamma0/
#
# EXPERIMENT_NAME defaults to PARETO_EXPERIMENT_NAME in batch/_stage_env.sh.

STAGE_LABEL="gamma = 0 comparison of the test outputs"
export STAGE_LABEL

# Read and write the merged tree on EOS in place, like the test evaluation.
: "${ADL1T_OUTPUT_ROOT:=/eos/user/l/lbehrens/adatl1/ADatL1/outputs}"
export ADL1T_OUTPUT_ROOT

: "${CODE_DIR:=/eos/user/l/lbehrens/adatl1/ADatL1}"
export CODE_DIR

# _stage_env.sh requires a RUN_NAME; this job covers a list of runs instead.
RUN_NAME=gamma0_comparison
export RUN_NAME
STAGE_ENV="${CODE_DIR}/batch/_stage_env.sh"
[[ -r "$STAGE_ENV" ]] || { echo "FATAL: cannot read $STAGE_ENV" >&2; exit 2; }
source "$STAGE_ENV"

: "${EXPERIMENT_NAME:=$PARETO_EXPERIMENT_NAME}"
export EXPERIMENT_NAME
echo "EXPERIMENT_NAME:   $EXPERIMENT_NAME"

# A run list given relative to the checkout (the job itself runs in the sandbox).
if [[ -n "${RUNS:-}" && "$RUNS" != /* ]]; then
  RUNS="${CODE_DIR}/${RUNS}"
fi
export RUNS FORCE

echo
echo "Running scripts/physics/runae_test_comparison.sh $* ..."
exec bash "${CODE_DIR}/scripts/physics/runae_test_comparison.sh" "$@"

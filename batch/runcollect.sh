#!/bin/bash
# STAGE 4 of 4 on HTCondor -- aggregate every run and build the Pareto front.
#
# A whole-study step, not a per-run one: a single job, no RUN_NAME, no queue
# list. Pure pandas/numpy - no torch, no GPU, minutes rather than hours.
#
# It reads the merged study tree on EOS in place and writes phase2/ and phase3/
# back into it, so nothing is staged into the job sandbox and
# transfer_output_files is deliberately absent from the submit file.

set -euo pipefail

echo "========================================"
echo "Pareto collect + select (stage 4/4)"
echo "========================================"
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

if [[ -d /opt/venv/bin ]]; then
  export PATH="/opt/venv/bin:$PATH"
fi

python3 --version

: "${CODE_DIR:=/eos/user/l/lbehrens/adatl1/ADatL1}"
: "${ADL1T_OUTPUT_ROOT:=/eos/user/l/lbehrens/adatl1/outputs}"
: "${STUDY_ID:=fet-et-pareto-v1}"
: "${STUDY_ROOT:=${ADL1T_OUTPUT_ROOT}/pareto_studies/${STUDY_ID}}"

# Set EXPERIMENT_NAME to build the study map from a directory of individually
# trained runs instead of expecting one the study runner declared up front. The
# submit file passes it through the environment.
: "${EXPERIMENT_NAME:=}"
export ADL1T_OUTPUT_ROOT STUDY_ID STUDY_ROOT EXPERIMENT_NAME

echo "CODE_DIR:        $CODE_DIR"
echo "STUDY_ROOT:      $STUDY_ROOT"
echo "EXPERIMENT_NAME: ${EXPERIMENT_NAME:-<using an existing study map>}"
echo

exec bash "${CODE_DIR}/scripts/physics/runcollect.sh"

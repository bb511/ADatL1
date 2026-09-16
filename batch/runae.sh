#!/bin/bash

set -euo pipefail

echo "========================================"
echo "AE training (GPU) on HTCondor"
echo "========================================"

echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "PWD:      $(pwd)"
echo "User:     $(whoami)"

echo
echo "Python:"
command -v python
python --version

echo
echo "GPU:"
nvidia-smi || echo "nvidia-smi unavailable"

# ---------------------------------------------------------------------------
# scripts/physics/runae.sh needs these. Its own defaults point at the NGT
# cluster (/shared/adatl1, /tmp/ADatL1, /scratch/...), which do not exist here,
# and RUN_NAME / ADL1T_OUTPUT_ROOT are mandatory - it aborts if they are unset.
# ---------------------------------------------------------------------------

# Code: same EOS checkout that test_container.sh runs from.
: "${CODE_DIR:=/eos/user/l/lbehrens/adatl1/ADatL1}"

# Data: runae.sh checks for
#   ${PROJECT_ROOT}/data/data_2025E+G/{extracted,processed,mlready}
: "${PROJECT_ROOT:=/eos/user/l/lbehrens/adl1t-stage}"

# Outputs: job scratch dir, transferred back to the submit dir on exit.
SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
: "${ADL1T_OUTPUT_ROOT:=${SCRATCH}/outputs}"
: "${MPLCONFIGDIR:=${SCRATCH}/matplotlib}"

: "${RUN_NAME:=AE_LXPLUS_3ep}"
: "${MAX_EPOCHS:=3}"

export CODE_DIR PROJECT_ROOT ADL1T_OUTPUT_ROOT MPLCONFIGDIR RUN_NAME MAX_EPOCHS

mkdir -p "$ADL1T_OUTPUT_ROOT" "$MPLCONFIGDIR"

echo
echo "CODE_DIR:          $CODE_DIR"
echo "PROJECT_ROOT:      $PROJECT_ROOT"
echo "ADL1T_OUTPUT_ROOT: $ADL1T_OUTPUT_ROOT"
echo "RUN_NAME:          $RUN_NAME"
echo "MAX_EPOCHS:        $MAX_EPOCHS"

echo
echo "Running scripts/physics/runae.sh ..."
exec bash "${CODE_DIR}/scripts/physics/runae.sh"

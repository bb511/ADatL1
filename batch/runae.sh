#!/bin/bash

set -euo pipefail

echo "========================================"
echo "AE training (GPU) on HTCondor"
echo "========================================"

echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "PWD:      $(pwd)"
echo "User:     $(whoami)"

# The dependencies live in a venv inside the image (see enter-container.sh).
# runae.sh invokes python3, which otherwise resolves to the system interpreter.
if [[ -d /opt/venv/bin ]]; then
  export PATH="/opt/venv/bin:$PATH"
fi

echo
echo "Python:"
command -v python3
python3 --version
python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

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

: "${RUN_NAME:=AE_LXPLUS_30ep}"
: "${MAX_EPOCHS:=30}"

# DATA_WORKERS only affects the awkward->torch conversion, not peak RSS
# (measured: 3 workers 14256 MB, 1 worker 14289 MB). Keep it <= request_cpus.
: "${DATA_WORKERS:=3}"

export CODE_DIR PROJECT_ROOT ADL1T_OUTPUT_ROOT MPLCONFIGDIR RUN_NAME MAX_EPOCHS DATA_WORKERS

mkdir -p "$ADL1T_OUTPUT_ROOT" "$MPLCONFIGDIR"

echo
echo "CODE_DIR:          $CODE_DIR"
echo "PROJECT_ROOT:      $PROJECT_ROOT"
echo "ADL1T_OUTPUT_ROOT: $ADL1T_OUTPUT_ROOT"
echo "RUN_NAME:          $RUN_NAME"
echo "MAX_EPOCHS:        $MAX_EPOCHS"
echo "DATA_WORKERS:      $DATA_WORKERS"

# Baseline for sizing future jobs. The run logs [phase] and [mem] lines
# throughout (src/utils/instrumentation.py); grep them out of the .out file:
#   grep -E "\[phase\]|\[mem\]|\[data\]" batch/logs/runae.<cluster>.0.out

echo
echo "Running scripts/physics/runae.sh ..."
exec bash "${CODE_DIR}/scripts/physics/runae.sh"

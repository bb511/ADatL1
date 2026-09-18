#!/bin/bash

set -euo pipefail

echo "========================================"
echo "AE training on HTCondor"
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

# gpu or cpu. The cluster runs CPU-only by default (supervisor decision,
# 2026-09-18) and the workload suits it: the AE is ~20k parameters, so the
# measured 40 s/epoch on an H100 MIG slice sat far below the card's compute
# ceiling - the run is bound by host-side data movement, not by the GPU.
# The practical gain is scheduling: ~5200 shared CPU slots against ~50 GPU
# slots, and no "Hostgroup == gpu" requirement to satisfy.
: "${TRAINER:=cpu}"

if [[ "$TRAINER" == "gpu" ]]; then
  echo
  echo "GPU:"
  nvidia-smi || echo "nvidia-smi unavailable"
fi

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

# Intra-op thread budget for the CPU trainer. Must match request_cpus in the
# submit file: nproc inside the sandbox reports the whole machine, not the
# slot, so a wrong value here either wastes cores or oversubscribes them.
: "${CPU_THREADS:=7}"   # keep equal to request_cpus in batch/runae.sub

export CODE_DIR PROJECT_ROOT ADL1T_OUTPUT_ROOT MPLCONFIGDIR RUN_NAME MAX_EPOCHS
export DATA_WORKERS TRAINER CPU_THREADS

mkdir -p "$ADL1T_OUTPUT_ROOT" "$MPLCONFIGDIR"

echo
echo "CODE_DIR:          $CODE_DIR"
echo "PROJECT_ROOT:      $PROJECT_ROOT"
echo "ADL1T_OUTPUT_ROOT: $ADL1T_OUTPUT_ROOT"
echo "RUN_NAME:          $RUN_NAME"
echo "MAX_EPOCHS:        $MAX_EPOCHS"
echo "DATA_WORKERS:      $DATA_WORKERS"
echo "TRAINER:           $TRAINER"
echo "CPU_THREADS:       $CPU_THREADS"

# Baseline for sizing future jobs. The run logs [phase] and [mem] lines
# throughout (src/utils/instrumentation.py); grep them out of the .out file:
#   grep -E "\[phase\]|\[mem\]|\[data\]" batch/logs/runae.<cluster>.0.out

echo
echo "Running scripts/physics/runae.sh ..."
exec bash "${CODE_DIR}/scripts/physics/runae.sh"

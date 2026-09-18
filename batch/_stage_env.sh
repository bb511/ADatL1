#!/bin/bash
# Shared HTCondor-side environment for the four pipeline stages. Sourced by
# batch/runae.sh, batch/runprobes.sh, batch/runmetrics.sh, batch/runcollect.sh.
#
# Its job is to translate the lxplus/EOS layout into the environment variables
# the scripts/physics/* stage scripts expect, whose own defaults point at the
# NGT cluster (/shared/adatl1, /scratch/...) and do not exist here.

set -euo pipefail

echo "========================================"
echo "${STAGE_LABEL:-ADatL1 stage} on HTCondor"
echo "========================================"
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "PWD:      $(pwd)"
echo "User:     $(whoami)"

# The dependencies live in a venv inside the image (see enter-container.sh).
# The stage scripts invoke python3, which otherwise resolves to the system
# interpreter, which has no torch.
if [[ -d /opt/venv/bin ]]; then
  export PATH="/opt/venv/bin:$PATH"
fi

echo
echo "Python:"
command -v python3
python3 --version

# gpu or cpu. The cluster runs CPU-only by default (supervisor decision,
# 2026-09-18): the AE is ~20k parameters, so the measured 40 s/epoch on an H100
# MIG slice sat far below the card's compute ceiling - the run is bound by
# host-side data movement. The practical gain is scheduling: ~5200 shared CPU
# slots against ~50 GPU slots, and no "Hostgroup == gpu" requirement.
: "${TRAINER:=cpu}"

if [[ "$TRAINER" == "gpu" ]]; then
  echo
  echo "GPU:"
  nvidia-smi || echo "nvidia-smi unavailable"
  python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
fi

# Code: the EOS checkout that test_container.sh runs from.
: "${CODE_DIR:=/eos/user/l/lbehrens/adatl1/ADatL1}"

# Data: the stage scripts read ${PROJECT_ROOT}/data/data_2025E+G/{extracted,processed,mlready}
: "${PROJECT_ROOT:=/eos/user/l/lbehrens/adl1t-stage}"
: "${RAW_DATA_DIR:=${PROJECT_ROOT}/raw/parquet_files}"

# Outputs: job scratch dir, transferred back to the submit dir on exit.
SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
: "${ADL1T_OUTPUT_ROOT:=${SCRATCH}/outputs}"
: "${MPLCONFIGDIR:=${SCRATCH}/matplotlib}"

# DATA_WORKERS only affects the awkward->torch conversion, not peak RSS
# (measured: 3 workers 14256 MB, 1 worker 14289 MB). Keep it <= request_cpus.
: "${DATA_WORKERS:=3}"

# Intra-op thread budget. Must match request_cpus in the submit file: nproc
# inside the sandbox reports the whole machine, not the slot, so a wrong value
# here either wastes cores or oversubscribes them.
: "${CPU_THREADS:=7}"

# Every stage in this pipeline shares one identity and one architecture. If any
# of these differ from the stage-1 run that produced the checkpoint, the strict
# load_state_dict in stage 2 and 3 fails.
: "${EXPERIMENT:=physics/ae}"
: "${RUN_NAME:?Set RUN_NAME, e.g. via the queue statement in the submit file}"

export CODE_DIR PROJECT_ROOT RAW_DATA_DIR ADL1T_OUTPUT_ROOT MPLCONFIGDIR
export DATA_WORKERS TRAINER CPU_THREADS EXPERIMENT RUN_NAME

mkdir -p "$ADL1T_OUTPUT_ROOT" "$MPLCONFIGDIR"

echo
echo "CODE_DIR:          $CODE_DIR"
echo "PROJECT_ROOT:      $PROJECT_ROOT"
echo "ADL1T_OUTPUT_ROOT: $ADL1T_OUTPUT_ROOT"
echo "EXPERIMENT:        $EXPERIMENT"
echo "RUN_NAME:          $RUN_NAME"
echo "DATA_WORKERS:      $DATA_WORKERS"
echo "TRAINER:           $TRAINER"
echo "CPU_THREADS:       $CPU_THREADS"

# Every stage logs [phase], [mem] and [data] lines (src/utils/instrumentation.py).
# To size the next submission from measurement rather than inference:
#   grep -E "\[phase\]|\[mem\]|\[data\]" batch/logs/<stage>.<cluster>.*.out
echo

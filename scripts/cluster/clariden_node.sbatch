#!/bin/bash
#SBATCH --account=a0166
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=pool-node
#SBATCH --output=logs/clariden_node_%j.out
# The pool drivers of one or more run*_search.sh files on one whole clariden node.
#
# Every clariden partition allocates whole nodes, so the per-trial launcher
# (submitit_slurm_clariden) pays for four GH200s per one-GPU trial. Here each
# catalogue's drivers run under Hydra's basic launcher, one after another within
# a driver, spread over the node's four GPUs by submit_search.sh, and the job ends
# with its last driver. Each further catalogue adds one driver per GPU. A node is
# billed for the time it is held, so the limit (the partition's maximum) costs
# nothing unused; a driver still running at the limit is marked dead in its log,
# and scripts/cluster/sweep_status.py prints its relaunch.
#
# Submit from the repository root, which the output path is relative to:
#   sbatch scripts/cluster/clariden_node.sbatch scripts/cifar10/runae_search.sh
#   sbatch scripts/cluster/clariden_node.sbatch scripts/cifar10/runae_search.sh \
#       scripts/cifar10/runsvdd_search.sh logger=none
#   sbatch scripts/cluster/clariden_node.sbatch --only 1,2,3,4 \
#       scripts/physics/runae_search.sh
# An argument ending in .sh is a catalogue; every other one is a hydra override
# appended to all of their drivers. --only, if given, comes first and applies to
# every catalogue.
set -eu
cd "${SLURM_SUBMIT_DIR:?submit this with sbatch}" || exit 1
. scripts/cluster/lib.sh

# The env's activate.d script reads LD_LIBRARY_PATH, unset in a batch shell.
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$CLARIDEN_CONDA/etc/profile.d/conda.sh"
conda activate "$CLARIDEN_ENV"
# Torch starts one thread per core (288) in every driver without a cap.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8}

opts=()
if [ "${1:-}" = --only ]; then opts=(--only "${2:?--only needs a value}"); shift 2; fi
cats=()
overrides=()
for a in "$@"; do
    case "$a" in *.sh) cats+=("$a") ;; *) overrides+=("$a") ;; esac
done
[ "${#cats[@]}" -gt 0 ] || { echo "no run*_search.sh given" >&2; exit 2; }

# slurm kills every process of the job once this shell exits, so at the time limit
# it passes the TERM on and waits for submit_search.sh to mark its drivers stopped.
pids=()
trap 'kill -TERM "${pids[@]}" 2>/dev/null; wait; exit 143' TERM
for c in "${cats[@]}"; do
    bash scripts/cluster/submit_search.sh "${opts[@]}" --gpus 4 --wait "$c" "${overrides[@]}" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=$((failed + 1)); fi
done
echo "$(date '+%F %T') $((${#cats[@]} - failed)) of ${#cats[@]} catalogue(s) ended cleanly"
[ "$failed" -eq 0 ]

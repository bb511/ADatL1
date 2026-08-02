#!/usr/bin/env bash
#SBATCH --account=a0166
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --mem=440G
#SBATCH --time=04:00:00
#SBATCH --array=0-11%8
#SBATCH --job-name=cch-svdd-inv-eval

set -euo pipefail
readonly REPO=/users/vjimenez/adatl1
: "${CCHAMBER_SVDD_INVERSE_ROOT:?Set CCHAMBER_SVDD_INVERSE_ROOT}"
cd "${REPO}"
pids=()
for slot in 0 1 2 3; do
  index=$((SLURM_ARRAY_TASK_ID * 4 + slot))
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=72 --gpus-per-node=1 --mem=110G \
    uv run --frozen --no-sync python scripts/cchamber_svdd_inverse_cap_campaign.py evaluate \
      --root "${CCHAMBER_SVDD_INVERSE_ROOT}" --trajectory-index "${index}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"

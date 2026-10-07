#!/usr/bin/env bash
# ===========================================================================
# gamma = 0 comparison of the TEST outputs (after scripts/physics/runae_test.sh)
# ===========================================================================
# The test-split counterpart of stage 4 phase 4c. For every gamma != 0 run of
# EXPERIMENT_NAME that has test correlation matrices, it writes
#
#   plots/test/loss_total/correlation_matrix/<dataset>/<Method>/comparison_gamma0/
#       |r_reco(run)| - |r_reco(gamma = 0 run)|, 6 PNGs, the gamma = 0 CSV copy and
#       reference.json (FET.Et row green where the run is closer to 0)
#   plots/test/loss_total/correlation_matrix/<dataset>/mean_correlations.json
#       + spaces.reconstruction_gamma0 and "mean increase compared to gamma = 0"
#
# and the MLflow galleries of those folders, exactly as phase 4c does for val
# (src/analysis/correlation_gamma0_comparison.py). Nothing under plots/val is
# touched.
#
# The reference is the gamma = 0 run of the same experiment with the same seed,
# architecture and epochs, and it must have been TESTED as well
# (runae_test.sh). Runs whose gamma = 0 run has no test outputs are reported as
# skipped. Existing comparison plots are kept unless FORCE=1; the
# mean_correlations.json fields are rewritten on every pass.
#
# Usage:
#   EXPERIMENT_NAME=Pareto-Front-261002 bash scripts/physics/runae_test_comparison.sh
#   EXPERIMENT_NAME=... RUN_NAME=<gamma != 0 run> bash scripts/physics/runae_test_comparison.sh
#   EXPERIMENT_NAME=... FORCE=1 bash ...     redraw, e.g. after re-testing the gamma = 0 run
#
# Pure pandas/matplotlib: no torch, no data, seconds per run.

set -euo pipefail

: "${EXPERIMENT_NAME:?Set EXPERIMENT_NAME to the checkpoint folder, e.g. EXPERIMENT_NAME=Pareto-Front-261002}"

# Same defaults as _stage_common.sh.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${CODE_DIR:=$REPO_ROOT}"
: "${PROJECT_ROOT:=${REPO_ROOT}}"
: "${ADL1T_OUTPUT_ROOT:=${PROJECT_ROOT}}"
: "${MLRUNS_ROOT:=${ADL1T_OUTPUT_ROOT}/logs/mlflow/mlruns}"
: "${MPLCONFIGDIR:=${PROJECT_ROOT}/.matplotlib}"
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"

EXPERIMENT_DIR="${ADL1T_OUTPUT_ROOT}/checkpoints/${EXPERIMENT_NAME}"
[[ -d "$EXPERIMENT_DIR" ]] || {
  echo "FATAL: no experiment folder $EXPERIMENT_DIR" >&2
  exit 2
}

args=(--experiment-dir "$EXPERIMENT_DIR" --split test --mlruns-root "$MLRUNS_ROOT")
if [[ -n "${RUN_NAME:-}" ]]; then
  args+=(--run-name "$RUN_NAME")
fi
if [[ "${FORCE:-0}" == 1 ]]; then
  args+=(--force)
fi

cd "$CODE_DIR"
echo "gamma = 0 comparison of the test outputs in $EXPERIMENT_DIR"
exec python3 scripts/plot_correlation_gamma0.py "${args[@]}"

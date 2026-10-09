#!/usr/bin/env bash
# ===========================================================================
# Compare tested runs with the gamma = 0 / 50-bin run (after runae_test.sh)
# ===========================================================================
# For every gamma != 0 run in the list (gamma = 0 entries are skipped: they are the
# reference), two steps on the TEST outputs of scripts/physics/runae_test.sh:
#
# 1. Correlation matrices, as stage 3 phase 4c does for val
#    (src/analysis/correlation_gamma0_comparison.py):
#      plots/test/loss_total/correlation_matrix/<dataset>/<Method>/comparison_gamma0/
#          |r_reco(run)| - |r_reco(gamma = 0)|, 6 PNGs, the gamma = 0 CSV copy and
#          reference.json (FET.Et row green where the run is closer to 0)
#      plots/test/loss_total/correlation_matrix/<dataset>/mean_correlations.json
#          + spaces.reconstruction_gamma0 and "mean increase compared to gamma = 0"
#    plus their MLflow galleries.
#
# 2. Everything else the test evaluation produces
#    (src/analysis/run_vs_gamma0_comparison.py):
#      plots/test/loss_total/comparison/
#          summary.{csv,json,png}   efficiency median/min/mean/CVaR25, threshold
#                                   drift, Wasserstein: run, gamma = 0, difference,
#                                   change in %, better/worse
#          efficiency/              efficiency per signal
#          ascore_operational/      mean anomaly score per dataset
#          reco/<dataset>/          input and both reconstructions in one plot,
#                                   their difference below (+ reco_summary.csv)
#          reference.json
#
# The reference is the gamma = 0 / 50-bin run of the same experiment with the
# same seed, architecture and epochs; it must have been TESTED as well (keep it in
# the run list of runae_test.sh). Nothing under plots/val is touched.
#
# Runs, first match wins:
#   bash scripts/physics/runae_test_comparison.sh RUN [RUN ...]   from the command line
#   RUNS=batch/my_runs.txt bash scripts/physics/runae_test_comparison.sh
#   bash scripts/physics/runae_test_comparison.sh                 batch/test_runs.txt
# A run file holds one run name per line; blank lines and lines starting with #
# are ignored.
#
#   EXPERIMENT_NAME=Pareto-Front-261002   required: checkpoints/<EXPERIMENT_NAME>/
#   FORCE=1     redraw existing comparison_gamma0/ plots (e.g. after re-testing the
#               gamma = 0 run); comparison/ and mean_correlations.json are always
#               rewritten
#   DRY_RUN=1   print the two commands instead of running them
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

# --- the runs ------------------------------------------------------------------
runs=()
if (( $# > 0 )); then
  runs=("$@")
  source_label="command line"
else
  : "${RUNS:=${CODE_DIR}/batch/test_runs.txt}"
  [[ -r "$RUNS" ]] || {
    echo "FATAL: no run list: give run names as arguments or set RUNS (tried $RUNS)" >&2
    exit 2
  }
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"                      # drop comments
    line="${line//[[:space:]]/}"            # and whitespace
    [[ -n "$line" ]] && runs+=("$line")
  done < "$RUNS"
  source_label="$RUNS"
fi
(( ${#runs[@]} > 0 )) || {
  echo "FATAL: the run list is empty ($source_label)" >&2
  exit 2
}

run_args=()
for run in "${runs[@]}"; do
  run_args+=(--run-name "$run")
done

correlation=(python3 scripts/plot_correlation_gamma0.py
  --experiment-dir "$EXPERIMENT_DIR" --split test --mlruns-root "$MLRUNS_ROOT" "${run_args[@]}")
if [[ "${FORCE:-0}" == 1 ]]; then
  correlation+=(--force)
fi
metrics=(python3 scripts/compare_with_gamma0.py
  --experiment-dir "$EXPERIMENT_DIR" --split test --ckpt loss_total "${run_args[@]}")

cd "$CODE_DIR"
echo "==============================================================="
echo " TEST COMPARISON WITH THE GAMMA = 0 / 50-BIN RUN"
echo "   experiment: $EXPERIMENT_DIR"
echo "   runs:       ${#runs[@]} from $source_label"
echo "==============================================================="

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${correlation[@]}"; echo
  printf '%q ' "${metrics[@]}"; echo
  exit 0
fi

echo "--- 1/2 correlation matrices ---"
"${correlation[@]}"
echo
echo "--- 2/2 efficiency, anomaly score, threshold drift, Wasserstein, reconstruction ---"
"${metrics[@]}"

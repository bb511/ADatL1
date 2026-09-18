#!/usr/bin/env bash
# ===========================================================================
# STAGE 4 of 4 -- aggregate every run and build the Pareto front
# ===========================================================================
# Reads the per-run artifacts that stages 2 and 3 wrote, aggregates them over
# paired seeds, applies the feasibility constraints, and selects the front.
#
#   phase2/  pareto_metrics.csv, pareto_metrics.parquet,
#            <configuration_id>/pareto_metrics.json
#   phase3/  pareto_front.csv, pareto_candidates.csv,
#            pareto_selection.json, pareto_projection.png
#
# Unlike stages 1-3 this is a whole-study step, not a per-run one, so it takes no
# RUN_NAME. It is pure pandas/numpy: no torch, no GPU, minutes not hours.
#
# This stage consumes a study map: an explicit list of every configuration and
# seed with its resolved manifest. There are two ways to get one.
#
#   1. The study runner wrote it. scripts/physics/run_pareto_fet_ngt.sh --run
#      declares the whole grid up front, at STUDY_ROOT/study_map.yaml. If that
#      file exists it is used as is.
#
#   2. Built from an experiment directory. Every run trained by stage 1 leaves a
#      run_manifest.yaml beside its checkpoint, so a directory of autoencoders
#      describes itself. Set EXPERIMENT_NAME and the map is assembled from
#      whatever is in checkpoints/<EXPERIMENT_NAME>/. This is the path for
#      autoencoders trained one at a time, whenever and wherever there was
#      capacity, and collected afterwards.
#
# Runs pair into configurations by configuration_id, which excludes the seed, so
# two runs differing only in seed aggregate together automatically.
#
# Usage:
#   EXPERIMENT_NAME=physics_ae_models bash scripts/physics/runcollect.sh
#   STUDY_ROOT=/path/to/pareto_studies/fet-et-pareto-v1 bash scripts/physics/runcollect.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${ADL1T_OUTPUT_ROOT:=${REPO_ROOT}}"
: "${STUDY_ID:=fet-et-pareto-v1}"
: "${STUDY_ROOT:=${ADL1T_OUTPUT_ROOT}/pareto_studies/${STUDY_ID}}"
: "${STUDY_MAP:=${STUDY_ROOT}/study_map.yaml}"
: "${PHASE2_OUTPUT:=${STUDY_ROOT}/phase2}"
: "${PHASE3_OUTPUT:=${STUDY_ROOT}/phase3}"

: "${EXPERIMENT_NAME:=}"
: "${CHECKPOINTS_DIR:=${ADL1T_OUTPUT_ROOT}/checkpoints}"

cd "$REPO_ROOT"

if [[ ! -f "$STUDY_MAP" ]]; then
  [[ -n "$EXPERIMENT_NAME" ]] || {
    echo "Missing study map: $STUDY_MAP" >&2
    echo >&2
    echo "Either point STUDY_ROOT at a study the runner created, or set" >&2
    echo "EXPERIMENT_NAME to build the map from a directory of trained runs:" >&2
    echo "  EXPERIMENT_NAME=physics_ae_models bash scripts/physics/runcollect.sh" >&2
    exit 2
  }

  EXPERIMENT_DIR="${CHECKPOINTS_DIR}/${EXPERIMENT_NAME}"
  [[ -d "$EXPERIMENT_DIR" ]] || {
    echo "No such experiment directory: $EXPERIMENT_DIR" >&2
    exit 2
  }

  echo "--- building the study map from $EXPERIMENT_DIR ---"
  # Reports which configurations are missing an expected seed. Those are
  # rejected by Phase 2 rather than silently dropped, so read that list: it is
  # the to-do list of runs still to train.
  python3 scripts/build_study_map.py \
    --experiment-dir "$EXPERIMENT_DIR" \
    --output "$STUDY_MAP"
  echo
fi

echo "==============================================================="
echo " STAGE 4/4  COLLECT + SELECT"
echo "   study map: $STUDY_MAP"
echo "   phase 2:   $PHASE2_OUTPUT"
echo "   phase 3:   $PHASE3_OUTPUT"
echo "==============================================================="

echo "--- phase 2: aggregating per-run artifacts ---"
python3 scripts/collect_pareto_study.py \
  --study-map "$STUDY_MAP" \
  --output-dir "$PHASE2_OUTPUT"

echo "--- phase 3: selecting the Pareto front ---"
python3 scripts/select_pareto_front.py \
  --input-table "${PHASE2_OUTPUT}/pareto_metrics.parquet" \
  --output-dir "$PHASE3_OUTPUT"

echo
echo "Front:     ${PHASE3_OUTPUT}/pareto_front.csv"
echo "Selection: ${PHASE3_OUTPUT}/pareto_selection.json"
echo "Plot:      ${PHASE3_OUTPUT}/pareto_projection.png"

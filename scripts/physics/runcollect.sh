#!/usr/bin/env bash
# ===========================================================================
# STAGE 4 of 4 -- aggregate every run and build the Pareto front
# ===========================================================================
# Reads the per-run artifacts that stages 2 and 3 wrote (one run per
# configuration), applies the feasibility constraints, and selects the front.
#
#   phase2/  pareto_metrics.csv, pareto_metrics.parquet,
#            <configuration_id>/pareto_metrics.json
#   phase3/  pareto_front.csv, pareto_candidates.csv, pareto_selection.json
#   phase4/  the figures, drawn from the phase3 tables and nothing else
#
# Unlike stages 1-3 this is a whole-study step, not a per-run one, so it takes no
# RUN_NAME. It is pure pandas/numpy: no torch, no GPU, minutes not hours.
#
# This stage consumes a study map: an explicit list of the one run of every
# configuration, with its resolved manifest. There are two ways to get one.
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
# The study is single-seed: a directory holding two runs of one configuration
# is refused, with the duplicates named.
#
# Usage:
#   bash scripts/physics/runcollect.sh    # EXPERIMENT_NAME defaults to Pareto-Front-261002
#   STUDY_ROOT=/path/to/pareto_studies/fet-et-pareto-v1 bash scripts/physics/runcollect.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${ADL1T_OUTPUT_ROOT:=${REPO_ROOT}}"
: "${STUDY_ID:=fet-et-pareto-v1}"
# One study tree per experiment directory, so a new study never picks up (or
# overwrites) the study map and phase outputs of an earlier one.
: "${EXPERIMENT_NAME:=Pareto-Front-261002}"
: "${STUDY_ROOT:=${ADL1T_OUTPUT_ROOT}/pareto_studies/${EXPERIMENT_NAME}}"
: "${STUDY_MAP:=${STUDY_ROOT}/study_map.yaml}"
: "${PHASE2_OUTPUT:=${STUDY_ROOT}/phase2}"
: "${PHASE3_OUTPUT:=${STUDY_ROOT}/phase3}"
: "${PHASE4_OUTPUT:=${STUDY_ROOT}/phase4}"

: "${CHECKPOINTS_DIR:=${ADL1T_OUTPUT_ROOT}/checkpoints}"

cd "$REPO_ROOT"

# Rebuild a map that was built from the checkpoint folder when runs were added
# since: a stale map silently collects only the old runs (2026-10-01: 111 of
# 189 after the refinement grid). A map without a matching checkpoint folder
# (e.g. one written by the NGT runner) is left alone.
EXPERIMENT_DIR="${CHECKPOINTS_DIR}/${EXPERIMENT_NAME}"
if [[ -f "$STUDY_MAP" && -n "$EXPERIMENT_NAME" && -d "$EXPERIMENT_DIR" ]]; then
  if [[ -n "$(find "$EXPERIMENT_DIR" -mindepth 2 -maxdepth 2 -name run_manifest.yaml -newer "$STUDY_MAP" -print -quit)" ]]; then
    stale="${STUDY_MAP%.yaml}.stale-$(date +%Y%m%d-%H%M%S).yaml"
    echo "--- study map is older than some runs; moving it to $stale and rebuilding ---"
    mv "$STUDY_MAP" "$stale"
  fi
fi

if [[ ! -f "$STUDY_MAP" ]]; then
  [[ -n "$EXPERIMENT_NAME" ]] || {
    echo "Missing study map: $STUDY_MAP" >&2
    echo >&2
    echo "Either point STUDY_ROOT at a study the runner created, or set" >&2
    echo "EXPERIMENT_NAME to build the map from a directory of trained runs:" >&2
    echo "  EXPERIMENT_NAME=Pareto-Front-261002 bash scripts/physics/runcollect.sh" >&2
    exit 2
  }

  [[ -d "$EXPERIMENT_DIR" ]] || {
    echo "No such experiment directory: $EXPERIMENT_DIR" >&2
    exit 2
  }

  echo "--- building the study map from $EXPERIMENT_DIR ---"
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
echo "   phase 4:   $PHASE4_OUTPUT"
echo "==============================================================="

echo "--- phase 2: aggregating per-run artifacts ---"
python3 scripts/collect_pareto_study.py \
  --study-map "$STUDY_MAP" \
  --output-dir "$PHASE2_OUTPUT"

echo "--- phase 3: selecting the Pareto front ---"
python3 scripts/select_pareto_front.py \
  --input-table "${PHASE2_OUTPUT}/pareto_metrics.parquet" \
  --output-dir "$PHASE3_OUTPUT"

echo "--- phase 4: drawing the figures ---"
# Reads only the phase 3 tables, so what is drawn is exactly what was selected.
python3 scripts/plot_pareto_study.py \
  --candidates "${PHASE3_OUTPUT}/pareto_candidates.csv" \
  --front "${PHASE3_OUTPUT}/pareto_front.csv" \
  --output-dir "$PHASE4_OUTPUT"

echo
echo "Front:     ${PHASE3_OUTPUT}/pareto_front.csv"
echo "Selection: ${PHASE3_OUTPUT}/pareto_selection.json"
echo "Figures:   ${PHASE4_OUTPUT}/"

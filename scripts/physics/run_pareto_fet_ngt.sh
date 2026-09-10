#!/usr/bin/env bash

# Execute the frozen FET.Et validation Pareto study from one NGT GPU pod.
#
# The script is intentionally sequential: one full ADatL1 run loads substantial
# data into host memory, and the usual NGT interactive pod exposes one GPU.
# It is safe to stop between completed runs and resume with --run or --all.
#
# Required environment (defaults follow scripts/physics/runae.sh):
#   PROJECT_ROOT=/shared/adatl1       persistent data, logs, checkpoints, results
#   CODE_DIR=/path/to/ADatL1          checked-out repository with dependencies ready
#   RAW_DATA_DIR=/.../parquet_files   source parquet directory
#
# Example in an NGT pod:
#   export PROJECT_ROOT=/shared/$USER/adatl1
#   export RAW_DATA_DIR=$PROJECT_ROOT/raw/parquet_files
#   export CODE_DIR=/shared/$USER/ADatL1
#   cd "$CODE_DIR"
#   bash scripts/physics/run_pareto_fet_ngt.sh --plan
#   bash scripts/physics/run_pareto_fet_ngt.sh --all
#
# `--all` runs the complete frozen study: 303 configurations x 3 paired AE
# seeds = 909 uncapped validation runs.  It never enables test data.

set -uo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly DEFAULT_PROJECT_ROOT="/shared/adatl1"
readonly STUDY_ID="fet-et-pareto-v1"
readonly PROTOCOL_VERSION="fet-et-pareto-v1"
readonly EXPERIMENT_NAME="physics_pareto_fet_v1"

ACTION=""
RERUN_INCOMPLETE=0

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [--plan | --run | --collect | --all] [--rerun-incomplete]

Actions:
  --plan              Print the frozen grid size and NGT storage plan; write nothing.
  --run               Create the study map and run every incomplete grid member.
  --collect           Aggregate an existing completed/partial study and select its front.
  --all               Equivalent to --run followed by --collect.

Options:
  --rerun-incomplete  Permit overwriting a run directory that exists but lacks a
                       complete, reportable artifact set.  This is explicit because
                       the training callback clears the checkpoint directory on start.
  -h, --help           Show this help.

Environment variables:
  PROJECT_ROOT         Persistent study/data root (default: $DEFAULT_PROJECT_ROOT).
  CODE_DIR             ADatL1 checkout (default: repository containing this script).
  RAW_DATA_DIR         Raw parquet root (default: \$PROJECT_ROOT/raw/parquet_files).
  DATA_WORKERS         Loader and BLAS thread count (default: 3).
  MPLCONFIGDIR         Matplotlib cache, preferably on /scratch.
  ALLOW_DIRTY_GIT=1    Allow a study from a checkout with uncommitted changes.

Results are stored under:
  \$PROJECT_ROOT/pareto_studies/$STUDY_ID/
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

note() {
  echo "[$SCRIPT_NAME] $*"
}

while (($#)); do
  case "$1" in
    --plan|--run|--collect|--all)
      [[ -z "$ACTION" ]] || die "Choose exactly one action."
      ACTION="${1#--}"
      ;;
    --rerun-incomplete)
      RERUN_INCOMPLETE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

[[ -n "$ACTION" ]] || {
  usage >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_ROOT:=$DEFAULT_PROJECT_ROOT}"
: "${CODE_DIR:=$(cd "$SCRIPT_DIR/../.." && pwd)}"
: "${RAW_DATA_DIR:=$PROJECT_ROOT/raw/parquet_files}"
: "${DATA_WORKERS:=3}"
: "${MPLCONFIGDIR:=/scratch/adatl1/matplotlib}"

readonly PROJECT_ROOT CODE_DIR RAW_DATA_DIR DATA_WORKERS MPLCONFIGDIR
export PROJECT_ROOT RAW_DATA_DIR MPLCONFIGDIR
export NUMEXPR_MAX_THREADS="$DATA_WORKERS"
export NUMEXPR_NUM_THREADS="$DATA_WORKERS"
export OMP_NUM_THREADS="$DATA_WORKERS"
export MKL_NUM_THREADS="$DATA_WORKERS"
export OPENBLAS_NUM_THREADS="$DATA_WORKERS"
readonly STUDY_ROOT="$PROJECT_ROOT/pareto_studies/$STUDY_ID"
readonly MANIFEST_ROOT="$STUDY_ROOT/manifests"
readonly STUDY_MAP="$STUDY_ROOT/study_map.yaml"
readonly RUN_PLAN="$STUDY_ROOT/run_plan.tsv"
readonly RUN_STATUS="$STUDY_ROOT/run_status.tsv"
readonly STUDY_METADATA="$STUDY_ROOT/study_metadata"
readonly PHASE2_OUTPUT="$STUDY_ROOT/phase2"
readonly PHASE3_OUTPUT="$STUDY_ROOT/phase3"

readonly -a PAIRED_SEEDS=(123 456 789)
readonly -a ARCHITECTURE_IDS=(h64_32 h128_64 h64_64_32)
readonly -a ARCHITECTURE_NODES=('[64,32,8]' '[128,64,8]' '[64,64,32,8]')
readonly -a REGULARIZED_GAMMAS=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)
readonly -a TRAINING_BINS=(10 20 30 40 50 60 70 80 90 100)

configuration_id() {
  local gamma="$1"
  local bins="$2"
  local architecture_id="$3"
  printf '%s__gamma-%s__bins-%s__arch-%s' \
    "$STUDY_ID" "$gamma" "$bins" "$architecture_id"
}

run_name() {
  local configuration="$1"
  local seed="$2"
  printf '%s__seed-%s' "$configuration" "$seed"
}

manifest_dir_for_run() {
  local run="$1"
  printf '%s/%s' "$MANIFEST_ROOT" "$run"
}

checkpoint_dir_for_run() {
  local run="$1"
  printf '%s/checkpoints/%s/%s' "$PROJECT_ROOT" "$EXPERIMENT_NAME" "$run"
}

for_each_run() {
  # Invoke the supplied callback with:
  # configuration_id seed gamma bins architecture_id encoder_nodes run_name
  local callback="$1"
  local architecture_index architecture_id nodes seed gamma bins configuration run

  for architecture_index in "${!ARCHITECTURE_IDS[@]}"; do
    architecture_id="${ARCHITECTURE_IDS[$architecture_index]}"
    nodes="${ARCHITECTURE_NODES[$architecture_index]}"

    # A single canonical gamma-zero baseline is required per architecture.
    configuration="$(configuration_id 0.0 50 "$architecture_id")"
    for seed in "${PAIRED_SEEDS[@]}"; do
      run="$(run_name "$configuration" "$seed")"
      "$callback" "$configuration" "$seed" 0.0 50 "$architecture_id" "$nodes" "$run"
    done

    for gamma in "${REGULARIZED_GAMMAS[@]}"; do
      for bins in "${TRAINING_BINS[@]}"; do
        configuration="$(configuration_id "$gamma" "$bins" "$architecture_id")"
        for seed in "${PAIRED_SEEDS[@]}"; do
          run="$(run_name "$configuration" "$seed")"
          "$callback" "$configuration" "$seed" "$gamma" "$bins" \
            "$architecture_id" "$nodes" "$run"
        done
      done
    done
  done
}

print_plan() {
  local regularized_count=$(( ${#ARCHITECTURE_IDS[@]} * ${#REGULARIZED_GAMMAS[@]} * ${#TRAINING_BINS[@]} ))
  local baseline_count=${#ARCHITECTURE_IDS[@]}
  local configuration_count=$((regularized_count + baseline_count))
  local run_count=$((configuration_count * ${#PAIRED_SEEDS[@]}))

  cat <<EOF
Frozen FET.Et Pareto study
  study ID:        $STUDY_ID
  protocol:        $PROTOCOL_VERSION
  configurations:  $configuration_count ($baseline_count gamma-zero + $regularized_count regularized)
  paired seeds:    ${PAIRED_SEEDS[*]}
  total runs:      $run_count, sequentially on the pod's GPU
  study root:      $STUDY_ROOT
  source checkout: $CODE_DIR
  raw data:        $RAW_DATA_DIR

The runner writes a deterministic study map before training. It skips only runs
with all five required reportable artifacts. Existing incomplete runs require
--rerun-incomplete, because rerunning clears their checkpoint directories.
EOF
}

check_ngt_environment() {
  [[ -f "$CODE_DIR/src/train.py" ]] || die "Missing $CODE_DIR/src/train.py. Set CODE_DIR."
  [[ -f "$CODE_DIR/scripts/collect_pareto_study.py" ]] || die "Missing Pareto collector in $CODE_DIR."
  [[ -f "$CODE_DIR/scripts/select_pareto_front.py" ]] || die "Missing Pareto selector in $CODE_DIR."
  [[ "$DATA_WORKERS" =~ ^[1-9][0-9]*$ ]] || die "DATA_WORKERS must be a positive integer."
  [[ -d "$RAW_DATA_DIR" ]] || die "Missing RAW_DATA_DIR: $RAW_DATA_DIR"

  local data_dir
  for data_dir in extracted processed mlready; do
    [[ -d "$PROJECT_ROOT/data/data_2025E+G/$data_dir" ]] || {
      die "Missing staged data directory: $PROJECT_ROOT/data/data_2025E+G/$data_dir"
    }
  done

  command -v python3 >/dev/null || die "python3 is not available."
  python3 -c 'import hydra, omegaconf, pandas, pyarrow, sklearn, torch' || \
    die "The active image lacks one or more ADatL1 dependencies."
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable; request an NVIDIA GPU pod."
  nvidia-smi >/dev/null || die "The requested GPU is not usable in this pod."

  if [[ "${ALLOW_DIRTY_GIT:-0}" != "1" ]] && [[ -n "$(git -C "$CODE_DIR" status --porcelain)" ]]; then
    die "The checkout is dirty. Commit/stash it, or explicitly set ALLOW_DIRTY_GIT=1."
  fi
}

archive_study_metadata() {
  mkdir -p "$STUDY_METADATA" "$MANIFEST_ROOT" "$PHASE2_OUTPUT" "$PHASE3_OUTPUT"

  printf '%s\n' "$STUDY_ID" > "$STUDY_METADATA/study_id.txt"
  printf '%s\n' "$PROTOCOL_VERSION" > "$STUDY_METADATA/protocol_version.txt"
  if [[ ! -f "$STUDY_METADATA/started_at.txt" ]]; then
    date --iso-8601=seconds > "$STUDY_METADATA/started_at.txt"
  fi
  date --iso-8601=seconds > "$STUDY_METADATA/last_invocation_at.txt"
  git -C "$CODE_DIR" rev-parse HEAD > "$STUDY_METADATA/git_commit.txt"
  git -C "$CODE_DIR" status --short > "$STUDY_METADATA/git_status.txt"
  python3 --version > "$STUDY_METADATA/python_version.txt"
  python3 -m pip freeze > "$STUDY_METADATA/pip_freeze.txt"
  nvidia-smi > "$STUDY_METADATA/nvidia_smi.txt"
  cp "$CODE_DIR/configs/experiment/physics/pareto_fet.yaml" \
    "$STUDY_METADATA/pareto_fet.yaml"
  cp "$CODE_DIR/poetry.lock" "$STUDY_METADATA/poetry.lock"
}

write_map_entry() {
  local configuration="$1"
  local seed="$2"
  local _gamma="$3"
  local _bins="$4"
  local _architecture_id="$5"
  local _nodes="$6"
  local run="$7"
  local manifest_dir checkpoint_dir
  manifest_dir="$(manifest_dir_for_run "$run")"
  checkpoint_dir="$(checkpoint_dir_for_run "$run")"

  cat <<EOF
  - configuration_id: $configuration
    autoencoder_seed: $seed
    manifest_path: $manifest_dir/pareto_manifest.resolved.yaml
    checkpoint_run_dir: $checkpoint_dir
EOF
}

write_plan_entry() {
  local configuration="$1"
  local seed="$2"
  local gamma="$3"
  local bins="$4"
  local architecture_id="$5"
  local nodes="$6"
  local run="$7"
  local manifest_dir checkpoint_dir
  manifest_dir="$(manifest_dir_for_run "$run")"
  checkpoint_dir="$(checkpoint_dir_for_run "$run")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$configuration" "$seed" "$gamma" "$bins" "$architecture_id" "$nodes" \
    "$run" "$manifest_dir" "$checkpoint_dir"
}

write_study_map_and_plan() {
  mkdir -p "$STUDY_ROOT"
  local map_tmp="$STUDY_MAP.tmp"
  local plan_tmp="$RUN_PLAN.tmp"

  cat > "$map_tmp" <<EOF
schema_version: 1
study_id: $STUDY_ID
protocol_version: $PROTOCOL_VERSION
expected_autoencoder_seeds: [123, 456, 789]
runs:
EOF
  for_each_run write_map_entry >> "$map_tmp"

  printf 'configuration_id\tautoencoder_seed\tmi_gamma\tmi_sensitive_num_bins\tarchitecture_id\tencoder_nodes\trun_name\tmanifest_dir\tcheckpoint_run_dir\n' > "$plan_tmp"
  for_each_run write_plan_entry >> "$plan_tmp"

  mv "$map_tmp" "$STUDY_MAP"
  mv "$plan_tmp" "$RUN_PLAN"
  note "Wrote $STUDY_MAP and $RUN_PLAN."
}

is_complete_reportable_run() {
  local run="$1"
  local manifest_dir checkpoint_dir probe_json
  manifest_dir="$(manifest_dir_for_run "$run")"
  checkpoint_dir="$(checkpoint_dir_for_run "$run")"
  probe_json="$checkpoint_dir/plots/val/loss_total/probes/leakage_probes.json"

  [[ -f "$manifest_dir/pareto_manifest.resolved.yaml" ]] || return 1
  [[ -f "$probe_json" ]] || return 1
  grep -Eq '"probe_valid"[[:space:]]*:[[:space:]]*true' "$probe_json" || return 1
  [[ -f "$checkpoint_dir/plots/val/loss_total/eff/eff_summary.json" ]] || return 1
  [[ -f "$checkpoint_dir/plots/val/loss_total/correlation_matrix/normal/mean_correlations.json" ]] || return 1
  [[ -f "$checkpoint_dir/plots/val/loss_total/latent_collapse/collapse_summary.json" ]] || return 1
  [[ -f "$checkpoint_dir/plots/val/loss_total/auroc/auroc_summary.json" ]]
}

run_one() {
  local configuration="$1"
  local seed="$2"
  local gamma="$3"
  local bins="$4"
  local architecture_id="$5"
  local nodes="$6"
  local run="$7"
  local manifest_dir checkpoint_dir log_file status timestamp
  manifest_dir="$(manifest_dir_for_run "$run")"
  checkpoint_dir="$(checkpoint_dir_for_run "$run")"
  log_file="$STUDY_ROOT/run_logs/$run.log"

  if is_complete_reportable_run "$run"; then
    note "SKIP complete $run"
    printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$run" skipped complete >> "$RUN_STATUS"
    return 0
  fi

  if [[ -e "$manifest_dir" || -e "$checkpoint_dir" ]]; then
    if (( ! RERUN_INCOMPLETE )); then
      note "INCOMPLETE $run; refusing to overwrite. Re-run with --rerun-incomplete."
      printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$run" blocked incomplete_existing_run >> "$RUN_STATUS"
      return 1
    fi
    note "RERUN incomplete $run (the training callback will clear its checkpoint directory)."
  fi

  mkdir -p "$manifest_dir" "$(dirname "$log_file")"
  note "START $run"
  timestamp="$(date --iso-8601=seconds)"
  printf '%s\t%s\t%s\t%s\n' "$timestamp" "$run" started "$log_file" >> "$RUN_STATUS"

  local -a command=(
    python3 src/train.py
    "paths.root_dir=$PROJECT_ROOT"
    "paths.raw_data_dir=$RAW_DATA_DIR"
    experiment=physics/pareto_fet
    "run_name=$run"
    "pareto_study.candidate.autoencoder_seed=$seed"
    "pareto_study.candidate.mi_gamma=$gamma"
    "pareto_study.candidate.mi_sensitive_num_bins=$bins"
    "pareto_study.candidate.architecture_id=$architecture_id"
    "pareto_study.candidate.encoder_nodes=$nodes"
    "data.data_awkward2torch.workers=$DATA_WORKERS"
    trainer=gpu
    'trainer.devices=[0]'
    "hydra.run.dir=$manifest_dir"
  )

  if HYDRA_FULL_ERROR=1 "${command[@]}" 2>&1 | tee "$log_file"; then
    if is_complete_reportable_run "$run"; then
      note "DONE $run"
      printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$run" completed "$log_file" >> "$RUN_STATUS"
      return 0
    fi
    note "INCOMPLETE $run after a zero exit status; inspect $log_file"
    printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$run" incomplete_after_success "$log_file" >> "$RUN_STATUS"
    return 1
  else
    # This assignment must be the first command in `else`; later logging
    # commands would overwrite the failed pipeline status in `$?`.
    status=$?
  fi

  note "FAILED $run (exit $status); continuing with remaining runs."
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$run" "failed_exit_$status" "$log_file" >> "$RUN_STATUS"
  return 1
}

run_grid() {
  mkdir -p "$STUDY_ROOT/run_logs"
  if [[ ! -f "$RUN_STATUS" ]]; then
    printf 'timestamp\trun_name\tstatus\tdetail\n' > "$RUN_STATUS"
  fi

  local failures=0
  run_one_counted() {
    if ! run_one "$@"; then
      failures=$((failures + 1))
    fi
  }
  for_each_run run_one_counted
  unset -f run_one_counted

  if (( failures )); then
    note "$failures run(s) were not complete. The collector will retain them as invalid."
    return 1
  fi
  note "All grid runs have reportable artifact sets."
}

collect_and_select() {
  [[ -f "$STUDY_MAP" ]] || die "Missing $STUDY_MAP. Run --run first."
  note "Collecting paired-seed metrics."
  (
    cd "$CODE_DIR"
    python3 scripts/collect_pareto_study.py \
      --study-map "$STUDY_MAP" \
      --output-dir "$PHASE2_OUTPUT"
  ) || return $?
  note "Selecting the validation Pareto front."
  (
    cd "$CODE_DIR"
    python3 scripts/select_pareto_front.py \
      --input-table "$PHASE2_OUTPUT/pareto_metrics.parquet" \
      --output-dir "$PHASE3_OUTPUT"
  ) || return $?
  note "Study outputs are in $PHASE2_OUTPUT and $PHASE3_OUTPUT."
}

case "$ACTION" in
  plan)
    print_plan
    ;;
  run)
    check_ngt_environment
    archive_study_metadata
    write_study_map_and_plan
    cd "$CODE_DIR"
    run_grid
    ;;
  collect)
    check_ngt_environment
    archive_study_metadata
    collect_and_select
    ;;
  all)
    check_ngt_environment
    archive_study_metadata
    write_study_map_and_plan
    cd "$CODE_DIR"
    run_exit_code=0
    run_grid || run_exit_code=$?
    collect_exit_code=0
    collect_and_select || collect_exit_code=$?
    if (( run_exit_code || collect_exit_code )); then
      exit 1
    fi
    ;;
esac

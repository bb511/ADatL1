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
SHARD_INDEX=1
SHARD_COUNT=1

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [--plan | --run | --collect | --all] [--rerun-incomplete]

Actions:
  --plan              Print the frozen grid size and NGT storage plan; write nothing.
  --run               Create the study map and run every incomplete grid member.
  --collect           Aggregate an existing completed/partial study and select its front.
  --all               Equivalent to --run followed by --collect.

Options:
  --shard <i>/<n>     Run only every n-th grid member starting at i (1-based).
                      Interleaved, so each shard gets a mix of architectures and
                      gammas rather than one shard taking all of one arm. Use it
                      to spread the study over n batch jobs. Collection is a
                      separate step once every shard has finished.
  --rerun-incomplete  Permit overwriting a run directory that exists but lacks a
                       complete, reportable artifact set.  This is explicit because
                       the training callback clears the checkpoint directory on start.
  -h, --help           Show this help.

Environment variables:
  PROJECT_ROOT         Persistent study/data root (default: $DEFAULT_PROJECT_ROOT).
  CODE_DIR             ADatL1 checkout (default: repository containing this script).
  RAW_DATA_DIR         Raw parquet root (default: \$PROJECT_ROOT/raw/parquet_files).
  DATA_WORKERS         Loader and BLAS thread count (default: 3).
  ADL1T_OUTPUT_ROOT    Where the study writes (default: \$PROJECT_ROOT). Point this
                       at the job sandbox on a batch worker.
  PARETO_ACCELERATOR   gpu (default) or cpu.
  MAX_EPOCHS           Override trainer.max_epochs for every run (default: the
                       manifest's inherited value).
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
    --shard)
      [[ "${2:-}" =~ ^([1-9][0-9]*)/([1-9][0-9]*)$ ]] || die "--shard needs <i>/<n>, got '\''${2:-}'\''."
      SHARD_INDEX="${BASH_REMATCH[1]}"
      SHARD_COUNT="${BASH_REMATCH[2]}"
      ((SHARD_INDEX <= SHARD_COUNT)) || die "--shard index $SHARD_INDEX exceeds count $SHARD_COUNT."
      shift
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
: "${MAX_EPOCHS:=}"
# gpu or cpu. The model is ~20k parameters and the measured 40 s/epoch on an
# H100 MIG slice is ~60x its compute ceiling, so training is bound by host-side
# data movement rather than the GPU. CPU is therefore competitive, and the CPU
# pool on a shared batch system is far larger than the GPU pool.
: "${PARETO_ACCELERATOR:=cpu}"
: "${MPLCONFIGDIR:=/scratch/adatl1/matplotlib}"
# Where the study writes. configs/paths/default.yaml resolves
# paths.output_root to ${oc.env:ADL1T_OUTPUT_ROOT,${paths.root_dir}}, so this
# must be the same value or the checkpoint paths in the study map will not be
# where training actually wrote. Defaults to PROJECT_ROOT (unchanged on NGT);
# on a batch worker point it at the job sandbox.
: "${ADL1T_OUTPUT_ROOT:=$PROJECT_ROOT}"

readonly PROJECT_ROOT CODE_DIR RAW_DATA_DIR DATA_WORKERS MPLCONFIGDIR ADL1T_OUTPUT_ROOT
export PROJECT_ROOT RAW_DATA_DIR MPLCONFIGDIR ADL1T_OUTPUT_ROOT
export NUMEXPR_MAX_THREADS="$DATA_WORKERS"
export NUMEXPR_NUM_THREADS="$DATA_WORKERS"
export OMP_NUM_THREADS="$DATA_WORKERS"
export MKL_NUM_THREADS="$DATA_WORKERS"
export OPENBLAS_NUM_THREADS="$DATA_WORKERS"
readonly STUDY_ROOT="$ADL1T_OUTPUT_ROOT/pareto_studies/$STUDY_ID"
readonly MANIFEST_ROOT="$STUDY_ROOT/manifests"
readonly STUDY_MAP="$STUDY_ROOT/study_map.yaml"
readonly RUN_PLAN="$STUDY_ROOT/run_plan.tsv"
readonly RUN_STATUS="$STUDY_ROOT/run_status.tsv"
readonly STUDY_METADATA="$STUDY_ROOT/study_metadata"
readonly PHASE2_OUTPUT="$STUDY_ROOT/phase2"
readonly PHASE3_OUTPUT="$STUDY_ROOT/phase3"

readonly PARETO_MANIFEST="$CODE_DIR/configs/experiment/physics/pareto_fet.yaml"

# The grid is read from the experiment manifest so there is exactly one
# predeclared source of truth. Hardcoding it here would drift: a seed list that
# disagrees with the manifest makes the collector reject every run.
read_grid_from_manifest() {
  [[ -f "$PARETO_MANIFEST" ]] || die "Missing Pareto manifest: $PARETO_MANIFEST"
  python3 - "$PARETO_MANIFEST" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as handle:
    study = yaml.safe_load(handle)["pareto_study"]
space = study["search_space"]
reg, base = space["regularized"], space["gamma_zero_baseline"]
if list(reg["architectures"]) != list(base["architectures"]):
    raise SystemExit("regularized and gamma_zero_baseline architectures differ")
nodes = lambda n: "[" + ",".join(str(v) for v in n) + "]"
emit = lambda name, vals: print(f"{name}=({' '.join(vals)})")
emit("PAIRED_SEEDS", [str(s) for s in study["paired_autoencoder_seeds"]])
emit("ARCHITECTURE_IDS", list(reg["architectures"]))
emit("ARCHITECTURE_NODES", [f"'{nodes(v)}'" for v in reg["architectures"].values()])
emit("REGULARIZED_GAMMAS", [str(g) for g in reg["mi_gamma"]])
emit("TRAINING_BINS", [str(b) for b in reg["mi_sensitive_num_bins"]])
print(f"BASELINE_GAMMA={base['mi_gamma']}")
print(f"BASELINE_BINS={base['mi_sensitive_num_bins']}")
PYEOF
}

grid_definition="$(read_grid_from_manifest)" || die "Could not read the grid from $PARETO_MANIFEST"
eval "$grid_definition"
unset grid_definition
readonly -a PAIRED_SEEDS ARCHITECTURE_IDS ARCHITECTURE_NODES REGULARIZED_GAMMAS TRAINING_BINS
readonly BASELINE_GAMMA BASELINE_BINS

((${#PAIRED_SEEDS[@]} >= 2)) || die "The manifest declares ${#PAIRED_SEEDS[@]} paired seed(s); aggregation requires at least two."

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
  # Mirrors paths.checkpoints_dir = ${paths.output_root}/checkpoints/.
  printf '%s/checkpoints/%s/%s' "$ADL1T_OUTPUT_ROOT" "$EXPERIMENT_NAME" "$run"
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
    configuration="$(configuration_id "$BASELINE_GAMMA" "$BASELINE_BINS" "$architecture_id")"
    for seed in "${PAIRED_SEEDS[@]}"; do
      run="$(run_name "$configuration" "$seed")"
      "$callback" "$configuration" "$seed" "$BASELINE_GAMMA" "$BASELINE_BINS" \
        "$architecture_id" "$nodes" "$run"
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
  # Raw parquet is only read when a dataset is absent from the extracted cache.
  # On the EOS deployment the raw tree was never staged, and batch/runae.sh runs
  # the same pipeline without it, so a missing RAW_DATA_DIR is not by itself a
  # reason to refuse the study. The real precondition is the staged cache
  # checked immediately below; if that is complete, extraction short-circuits
  # and paths.raw_data_dir is never dereferenced. Warn so the cause is still
  # visible in the job log if a cache miss does occur later.
  [[ -d "$RAW_DATA_DIR" ]] || \
    note "RAW_DATA_DIR does not exist: $RAW_DATA_DIR. Continuing, because the extracted cache below is what training reads. A cache miss will fail at data-load time and point back here."

  local data_dir
  for data_dir in extracted processed mlready; do
    [[ -d "$PROJECT_ROOT/data/data_2025E+G/$data_dir" ]] || {
      die "Missing staged data directory: $PROJECT_ROOT/data/data_2025E+G/$data_dir"
    }
  done

  command -v python3 >/dev/null || die "python3 is not available."
  python3 -c 'import hydra, omegaconf, pandas, pyarrow, sklearn, torch' || \
    die "The active image lacks one or more ADatL1 dependencies."
  if [[ "$PARETO_ACCELERATOR" == "gpu" ]]; then
    command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable; request an NVIDIA GPU pod."
    nvidia-smi >/dev/null || die "The requested GPU is not usable in this pod."
  fi

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
  if [[ "$PARETO_ACCELERATOR" == "gpu" ]]; then
    nvidia-smi > "$STUDY_METADATA/nvidia_smi.txt"
  fi
  printf '%s\n' "$PARETO_ACCELERATOR" > "$STUDY_METADATA/accelerator.txt"
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
expected_autoencoder_seeds: [$(IFS=,; echo "${PAIRED_SEEDS[*]}")]
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
    "hydra.run.dir=$manifest_dir"
  )

  case "$PARETO_ACCELERATOR" in
    gpu) command+=(trainer=gpu 'trainer.devices=[0]') ;;
    cpu) command+=(trainer=cpu 'trainer.devices=1') ;;
    *)   die "PARETO_ACCELERATOR must be gpu or cpu, got '\''$PARETO_ACCELERATOR'\''." ;;
  esac

  # Unset keeps the manifest's inherited trainer.max_epochs.
  if [[ -n "$MAX_EPOCHS" ]]; then
    command+=("trainer.max_epochs=$MAX_EPOCHS")
  fi

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
  local position=0
  local selected=0
  run_one_counted() {
    # Interleaved stride so every shard sees a mix of the grid.
    if (( position % SHARD_COUNT != SHARD_INDEX - 1 )); then
      position=$((position + 1))
      return 0
    fi
    position=$((position + 1))
    selected=$((selected + 1))
    if ! run_one "$@"; then
      failures=$((failures + 1))
    fi
  }
  if (( SHARD_COUNT > 1 )); then
    note "Shard $SHARD_INDEX/$SHARD_COUNT of this grid."
  fi
  for_each_run run_one_counted
  unset -f run_one_counted
  note "This invocation handled $selected run(s)."

  if (( failures )); then
    note "$failures run(s) were not complete. The collector will retain them as invalid."
    return 1
  fi
  note "All grid runs have reportable artifact sets."
}

collect_and_select() {
  # Rewrite rather than require. The map is fully determined by the manifest and
  # ADL1T_OUTPUT_ROOT, and a map written inside a batch sandbox records paths
  # that no longer exist once the outputs have been transferred elsewhere.
  write_study_map_and_plan
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
    (( SHARD_COUNT == 1 )) || die "--shard applies to --run, not --collect."
    # Phase 2 and 3 are pandas over the per-run JSON artifacts and never build a
    # model. Requiring a GPU here would make it impossible to run the analysis
    # on lxplus, which is the only place all shards' outputs meet.
    PARETO_ACCELERATOR=cpu
    check_ngt_environment
    archive_study_metadata
    collect_and_select
    ;;
  all)
    (( SHARD_COUNT == 1 )) || die "--shard needs --run; collect separately once all shards finish."
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

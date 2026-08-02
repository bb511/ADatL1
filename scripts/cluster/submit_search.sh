#!/bin/bash
# Submit the hyperparameter-search drivers of a run*_search.sh file on clariden.
#
# This is the sibling of submit_pareto.sh, and the difference matters. A
# run*_pareto.sh file holds bare single-training commands, so submit_pareto.sh
# has to prepend '-m hydra/launcher=... timeout_min=...'. A run*_search.sh block
# already carries '-m' and its own launcher settings. So this script runs each
# block as written and adds only what the file cannot know: the raw data path
# and a private sweep directory.
#
# Which launcher a block carries is a property of the block, not of this script.
# The physics ae/dsae/dsvae/realnvp/vae searches (both tiers) were run with
# hydra/launcher=submitit_local on olqti and still say so; only dte/svdd and the
# whole of cifar10/robustad carry submitit_slurm_clariden + n_jobs=6. Submitting
# a submitit_local block on clariden starts a local driver that never reaches
# slurm -- check the block with --list/--dry-run before launching it there.
#
# Each block is one long-lived Optuna driver: it keeps n_jobs trials in the
# queue and stays alive until its study reaches n_trials. Drivers are therefore
# nohup'd and this command returns as soon as they are all up.
#
# Usage (from the repository root, on clariden):
#   bash scripts/cluster/submit_search.sh --list    scripts/physics/runsvdd_search.sh
#   bash scripts/cluster/submit_search.sh --dry-run scripts/physics/runsvdd_search.sh
#   bash scripts/cluster/submit_search.sh           scripts/physics/runsvdd_search.sh
#   bash scripts/cluster/submit_search.sh --only cap,consistency scripts/physics/rundte_search.sh
#
#   --list              list the drivers in the file (with their launcher) and exit
#   --dry-run           print the full commands without running anything; works
#                       off the cluster, so it needs neither the env nor the data
#   --only <sel>        comma-separated study names (substring) or 1-based
#                       indices; default is every driver in the file
#   --raw-data-dir <p>  value for paths.raw_data_dir; defaults to
#                       $SCRATCH/adl1t_data/parquet_files when $SCRATCH is set.
#                       Only applied to files that reference it (the physics ones).
#
# Any extra hydra overrides after the file name are appended to every driver.
# Hydra takes the last of a duplicated override, so appending wins over the
# value baked into the file.
set -eu

# shellcheck source=scripts/cluster/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

only=""
dry=0
list=0
# Empty when neither is set, so the "no raw data path given" branch below is
# reachable -- '${SCRATCH:-}/adl1t_data/...' would silently yield '/adl1t_data/...'.
raw_data_dir="${RAW_DATA_DIR:-${SCRATCH:+$SCRATCH/adl1t_data/parquet_files}}"

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry=1; shift ;;
        --list) list=1; shift ;;
        --only) only="${2:?--only needs a value}"; shift 2 ;;
        --raw-data-dir) raw_data_dir="${2:?--raw-data-dir needs a path}"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

usage="usage: submit_search.sh [--list|--dry-run] [--only sel] [--raw-data-dir p]"
usage="$usage <run*_search.sh> [overrides...]"
file=${1:?$usage}
shift || true
extra="$*"

require_repo_root
[ -f "$file" ] || { echo "no such file: $file" >&2; exit 1; }

# Parsed once; every loop below reads from this list.
cmds=$(extract_cmds "$file")

# Optuna storage a driver will use: an explicit hydra.sweeper.storage override
# if the block carries one, else the value in the hparams_search config it
# selects. No block carries one today, so in practice it is always the config.
storage_of() {
    local s hp
    s=$(printf "%s" "$1" | grep -oE "hydra\.sweeper\.storage='[^']+'" | head -1 | sed "s/.*='//;s/'\$//")
    if [ -z "$s" ]; then
        hp=$(field "$1" 'hparams_search')
        if [ -n "$hp" ] && [ -f "configs/hparams_search/${hp}.yaml" ]; then
            s=$(grep -oE "^[[:space:]]*storage:[[:space:]]*\S+" "configs/hparams_search/${hp}.yaml" |
                head -1 | awk '{print $2}')
        fi
    fi
    printf "%s" "$s"
}

n_total=$(printf "%s\n" "$cmds" | grep -c . || true)
if [ "$n_total" -eq 0 ]; then
    echo "No search commands found in $file" >&2
    exit 1
fi

if [ "$list" -eq 1 ]; then
    printf "%-3s %-26s %-34s %-24s %s\n" "#" "study" "experiment_name" "launcher" "n_trials"
    i=0
    while IFS= read -r cmd; do
        i=$((i + 1))
        printf "%-3s %-26s %-34s %-24s %s\n" "$i" \
            "$(field "$cmd" 'hydra\.sweeper\.study_name')" \
            "$(field "$cmd" 'experiment_name')" \
            "$(field "$cmd" 'hydra/launcher')" \
            "$(field "$cmd" 'hydra\.sweeper\.n_trials')"
    done <<EOF
$cmds
EOF
    exit 0
fi

# Domain-prefixed stem: the three domains share file stems, and a bare stem
# makes concurrent submissions clobber each other's driver logs.
stem="$(basename "$(dirname "$file")")_$(basename "$file" .sh)"
logdir="logs/searches/$stem"
sweep_ts=$(date +%Y-%m-%d_%H-%M-%S)
[ "$dry" -eq 1 ] || mkdir -p "$logdir"

command -v sbatch >/dev/null 2>&1 || echo "warning: sbatch not on PATH - are you on clariden?" >&2

# Override with PYTHON=/path/to/env/bin/python3 to skip activation entirely.
# Skipped for --dry-run, which is meant to be usable off the cluster.
PYTHON=${PYTHON:-python3}
[ "$dry" -eq 1 ] || require_imports "$PYTHON" "hydra, optuna" \
    "conda activate $CLARIDEN_ENV" \
    "(or re-run with PYTHON=$CLARIDEN_ENV/bin/python3)"

# Create each sqlite study database, serially, before any driver starts.
# Drivers launched together against a not-yet-existing file all run optuna's
# alembic bootstrap at once and lose the race -- observed as
# 'UNIQUE constraint failed: alembic_version.version_num' and 'database is
# locked', which kills the driver seconds in. Doing it once here removes it.
# Idempotent: on an existing database this only opens and closes it.
if [ "$dry" -eq 0 ]; then
    printf "%s\n" "$cmds" | while IFS= read -r c; do storage_of "$c"; echo; done |
        sort -u | grep -v '^$' | while IFS= read -r url; do
        case "$url" in
            sqlite:///*)
                path=${url#sqlite:///}
                # The URLs carry a '?timeout=60' busy timeout (sqlite raises
                # 'database is locked' immediately without one, which kills the
                # driver). SQLAlchemy strips the query itself; this check needs
                # the bare path or it would never find an existing database.
                path=${path%%\?*}
                if [ ! -f "$path" ]; then
                    mkdir -p "$(dirname "$path")"
                    echo "initialising study database: $path"
                    "$PYTHON" -c 'import optuna,sys; optuna.storages.RDBStorage(sys.argv[1])' "$url"
                fi ;;
        esac
    done
fi

i=0
n_sub=0
while IFS= read -r cmd; do
    i=$((i + 1))
    study=$(field "$cmd" 'hydra\.sweeper\.study_name')
    selected "$i" "$study" || continue

    # Only files that already reference the raw data path need it rewritten; the
    # cifar10/robustad searches read data/ relative to the repo root. Rewritten
    # in place rather than appended, so the command carries exactly one value and
    # the placeholder check below stays meaningful.
    if printf "%s" "$cmd" | grep -q 'paths\.raw_data_dir='; then
        if [ -z "$raw_data_dir" ]; then
            echo "error: $file needs paths.raw_data_dir but none was given." >&2
            echo "       pass --raw-data-dir <path> (or set RAW_DATA_DIR/SCRATCH)" >&2
            exit 1
        fi
        # Existence is enforced only for a real run: --dry-run is for inspecting
        # the commands, and is routinely done off the machine that holds the data.
        if [ ! -d "$raw_data_dir" ]; then
            if [ "$dry" -eq 1 ]; then
                echo "note: '$raw_data_dir' not present here (fine for --dry-run)" >&2
            else
                echo "error: raw data dir '$raw_data_dir' does not exist." >&2
                exit 1
            fi
        fi
        cmd=$(printf "%s" "$cmd" |
              sed -E "s#paths\.raw_data_dir=[^ ]+#paths.raw_data_dir=${raw_data_dir}#")
    fi

    # Private sweep dir per driver: hydra's default second-resolution timestamp
    # collides when several drivers start together, merging their outputs (and
    # the clariden launcher nests submitit_folder under hydra.sweep.dir).
    sweep=" hydra.sweep.dir=logs/train/multiruns/${sweep_ts}_${stem}_j$i"
    full="$PYTHON src/train.py$cmd$sweep${extra:+ $extra}"

    assert_no_placeholder "$full" "driver $i ($study)" \
        "pass --raw-data-dir <real path>."

    if [ "$dry" -eq 1 ]; then
        echo "[$i/$n_total] $full"
        echo
        n_sub=$((n_sub + 1))
        continue
    fi
    echo "[$i/$n_total] starting driver: $study"
    # Keep the previous log: relaunching a driver that died is the main reason
    # to re-run this script, and truncating would destroy the evidence of why.
    log="$logdir/${i}_${study}.log"
    if [ -f "$log" ]; then mv "$log" "$log.$(date +%Y%m%d-%H%M%S)"; fi
    nohup bash -c "$full" > "$log" 2>&1 &
    n_sub=$((n_sub + 1))
    # Drivers sharing one sqlite file still contend while creating their study
    # rows; stagger them rather than starting six in the same second.
    sleep 5
done <<EOF
$cmds
EOF

if [ "$n_sub" -eq 0 ]; then
    echo "nothing matched --only '$only'" >&2
    exit 1
fi

if [ "$dry" -eq 0 ]; then
    echo
    echo "$n_sub driver(s) running detached (logs: $logdir)."
    echo "Trials run as slurm jobs; drivers are nohup'd, safe to log out."
    echo "Queue:   squeue --me"
    echo "Drivers: pgrep -af 'src/train.py' | grep $stem"
fi

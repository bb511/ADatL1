#!/bin/bash
# Submit the hyperparameter-search drivers of a run*_search.sh file on clariden.
#
# This is the sibling of submit_pareto.sh, and the difference matters. A
# run*_pareto.sh file holds bare single-training commands, so submit_pareto.sh
# has to prepend '-m hydra/launcher=... timeout_min=...'. A run*_search.sh block
# already carries '-m', hydra/launcher=submitit_slurm_clariden,
# hydra.launcher.timeout_min and hydra.sweeper.n_jobs. So this script runs each
# block as written and adds only what the file cannot know: the raw data path
# and a private sweep directory.
#
# Each block is one long-lived Optuna driver: it keeps n_jobs trials in the
# queue and stays alive until its study reaches n_trials. Drivers are therefore
# nohup'd and this command returns as soon as they are all up.
#
# Usage (from the repository root, on clariden):
#   bash scripts/submit_search.sh --list    scripts/physics/runsvdd_search.sh
#   bash scripts/submit_search.sh --dry-run scripts/physics/runsvdd_search.sh
#   bash scripts/submit_search.sh           scripts/physics/runsvdd_search.sh
#   bash scripts/submit_search.sh --only cap,consistency scripts/physics/rundte_search.sh
#
#   --list              list the drivers in the file and exit
#   --dry-run           print the full commands without running anything
#   --only <sel>        comma-separated study names (substring) or 1-based
#                       indices; default is every driver in the file
#   --raw-data-dir <p>  value for paths.raw_data_dir; defaults to
#                       $SCRATCH/adl1t_data/parquet_files. Only applied to files
#                       that actually reference it (the physics ones).
#
# Any extra hydra overrides after the file name are appended to every driver.
# Hydra takes the last of a duplicated override, so appending wins over the
# value baked into the file.
set -eu

only=""
dry=0
list=0
raw_data_dir="${RAW_DATA_DIR:-${SCRATCH:-}/adl1t_data/parquet_files}"

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

file=${1:?usage: submit_search.sh [--list|--dry-run] [--only sel] [--raw-data-dir p] <run*_search.sh> [overrides...]}
shift || true
extra="$*"

[ -f .project-root ] || { echo "run this from the repository root" >&2; exit 1; }
[ -f "$file" ] || { echo "no such file: $file" >&2; exit 1; }

# Same block parser as submit_pareto.sh: a block opens at the commented
# 'python3 src/train.py' line and runs to the first line without a continuation.
extract_cmds() {
    awk '
        /^# python3 src\/train\.py/ { collecting = 1; cmd = ""; next }
        collecting {
            line = $0
            sub(/^# ?/, "", line)
            gsub(/^[ \t]+|[ \t]+$/, "", line)
            ends = (line ~ /\\$/)
            sub(/[ \t]*\\$/, "", line)
            if (line != "") cmd = cmd " " line
            if (!ends) { collecting = 0; print cmd }
        }
    ' "$1"
}

field() { printf "%s" "$1" | grep -oE "$2=[^ ]+" | head -1 | cut -d= -f2- ; }

# Optuna storage a driver will use: an explicit hydra.sweeper.storage override
# if the block carries one (cifar10/robustad), else the value in the
# hparams_search config it selects (physics).
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

n_total=$(extract_cmds "$file" | wc -l | tr -d " ")
[ "$n_total" -eq 0 ] && { echo "No search commands found in $file" >&2; exit 1; }

if [ "$list" -eq 1 ]; then
    printf "%-3s %-26s %-34s %s\n" "#" "study" "experiment_name" "n_trials"
    i=0
    extract_cmds "$file" | while IFS= read -r cmd; do
        i=$((i + 1))
        printf "%-3s %-26s %-34s %s\n" "$i" \
            "$(field "$cmd" 'hydra\.sweeper\.study_name')" \
            "$(field "$cmd" 'experiment_name')" \
            "$(field "$cmd" 'hydra\.sweeper\.n_trials')"
    done
    exit 0
fi

# $1 = index, $2 = study name. Written with explicit ifs rather than
# 'test && action': under 'set -e' a bare failing && list aborts the script.
selected() {
    if [ -z "$only" ]; then return 0; fi
    local idx="$1" name="$2" s
    local IFS=','
    for s in $only; do
        if [ -n "$s" ]; then
            case "$s" in
                # An all-digit selector is an index only. Matching it as a
                # substring too would make '--only 1' also pick cvar10eff_*.
                *[!0-9]*) case "$name" in *"$s"*) return 0 ;; esac ;;
                *) if [ "$s" = "$idx" ]; then return 0; fi ;;
            esac
        fi
    done
    return 1
}

# Domain-prefixed stem: the three domains share file stems, and a bare stem
# makes concurrent submissions clobber each other's driver logs.
stem="$(basename "$(dirname "$file")")_$(basename "$file" .sh)"
logdir="logs/searches/$stem"
sweep_ts=$(date +%Y-%m-%d_%H-%M-%S)
[ "$dry" -eq 1 ] || mkdir -p "$logdir"

command -v sbatch >/dev/null 2>&1 || echo "warning: sbatch not on PATH - are you on clariden?" >&2

# Create each sqlite study database, serially, before any driver starts.
# Drivers launched together against a not-yet-existing file all run optuna's
# alembic bootstrap at once and lose the race -- observed as
# 'UNIQUE constraint failed: alembic_version.version_num' and 'database is
# locked', which kills the driver seconds in. Doing it once here removes it.
# Idempotent: on an existing database this only opens and closes it.
if [ "$dry" -eq 0 ]; then
    extract_cmds "$file" | while IFS= read -r c; do storage_of "$c"; echo; done |
        sort -u | grep -v '^$' | while IFS= read -r url; do
        case "$url" in
            sqlite:///*)
                path=${url#sqlite:///}
                if [ ! -f "$path" ]; then
                    mkdir -p "$(dirname "$path")"
                    echo "initialising study database: $path"
                    python3 -c 'import optuna,sys; optuna.storages.RDBStorage(sys.argv[1])' "$url"
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
    full="python3 src/train.py$cmd$sweep${extra:+ $extra}"

    case "$full" in
        *"/path/to/"*)
            echo "error: driver $i still contains a '/path/to/...' placeholder." >&2
            exit 1 ;;
    esac

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
$(extract_cmds "$file")
EOF

[ "$n_sub" -eq 0 ] && { echo "nothing matched --only '$only'" >&2; exit 1; }

if [ "$dry" -eq 0 ]; then
    echo
    echo "$n_sub driver(s) running detached (logs: $logdir)."
    echo "Trials run as slurm jobs; drivers are nohup'd, safe to log out."
    echo "Queue:   squeue --me"
    echo "Drivers: pgrep -af 'src/train.py' | grep $stem"
fi

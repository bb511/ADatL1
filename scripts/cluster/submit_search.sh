#!/bin/bash
# Submit the pool drivers of a run*_search.sh file.
#
# A run*_search.sh block already carries '-m', so this script runs each block as
# written and adds only what the file cannot know: a private sweep directory.
#
# The pool blocks name no launcher: the venue appends one after the file name,
# as each catalogue's header shows (submitit_slurm_clariden + n_jobs=6 on
# clariden; nothing, or submitit_local, on a GPU host). Without one, Hydra's
# basic launcher runs the trials inside the driver, which on clariden means on
# the login node and never reaches slurm -- check with --list/--dry-run before
# launching there.
#
# Each block is one long-lived Optuna driver: it keeps n_jobs trials in the
# queue and stays alive until it has run its own n_trials, while the live
# drivers of a catalogue (four, or eight in the two waves of a physics file)
# fill one shared study. Drivers are therefore nohup'd and this command returns
# as soon as they are all up.
#
# Usage (from the repository root):
#   bash scripts/cluster/submit_search.sh --list    scripts/physics/runsvdd_search.sh
#   bash scripts/cluster/submit_search.sh --dry-run scripts/physics/runsvdd_search.sh
#   bash scripts/cluster/submit_search.sh           scripts/physics/runsvdd_search.sh
#   bash scripts/cluster/submit_search.sh --only 1,3 scripts/physics/rundte_search.sh
#
#   --list              list the drivers in the file (with their launcher) and exit
#   --dry-run           print the full commands without running anything; works
#                       off the cluster, since it needs no training env
#   --only <sel>        comma-separated study names (substring) or 1-based
#                       indices; default is every driver in the file
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

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry=1; shift ;;
        --list) list=1; shift ;;
        --only) only="${2:?--only needs a value}"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

usage="usage: submit_search.sh [--list|--dry-run] [--only sel]"
usage="$usage <run*_search.sh> [overrides...]"
file=${1:?$usage}
shift || true
extra="$*"

require_repo_root
[ -f "$file" ] || { echo "no such file: $file" >&2; exit 1; }

# Parsed once; every loop below reads from this list.
cmds=$(extract_cmds "$file")

# $1 = command, $2 = key. The value of a sweeper key in the hparams_search
# config the command selects, e.g. 'storage' or 'study_name'.
from_config() {
    local hp
    hp=$(field "$1" 'hparams_search')
    if [ -n "$hp" ] && [ -f "configs/hparams_search/${hp}.yaml" ]; then
        grep -oE "^[[:space:]]*$2:[[:space:]]*\S+" "configs/hparams_search/${hp}.yaml" |
            head -1 | awk '{print $2}'
    fi
}

# Optuna storage a driver will use: an explicit hydra.sweeper.storage override
# if the block carries one, else the value in the hparams_search config it
# selects. No block carries one today, so in practice it is always the config.
storage_of() {
    local s
    s=$(printf "%s" "$1" | grep -oE "hydra\.sweeper\.storage='[^']+'" | head -1 | sed "s/.*='//;s/'\$//")
    [ -n "$s" ] || s=$(from_config "$1" storage)
    printf "%s" "$s"
}

# The study a driver fills, found the same way. The live blocks leave it to the
# config, which names one study for all of them; the dormant blocks of a lattice
# catalogue name their own.
study_of() {
    local s
    s=$(field "$1" 'hydra\.sweeper\.study_name')
    [ -n "$s" ] || s=$(from_config "$1" study_name)
    printf "%s" "$s"
}

# The launcher a driver runs under. Hydra takes the last hydra/launcher
# override, so an appended one wins over the block's; with neither, Hydra's
# basic launcher runs the trials in-process.
launcher_of() {
    local l
    l=$(printf "%s %s" "$1" "$extra" | grep -oE "hydra/launcher=[^ ]+" |
        tail -1 | cut -d= -f2-)
    printf "%s" "${l:-basic}"
}

n_total=$(printf "%s\n" "$cmds" | grep -c . || true)
if [ "$n_total" -eq 0 ]; then
    echo "No search commands found in $file" >&2
    exit 1
fi

if [ "$list" -eq 1 ]; then
    printf "%-3s %-10s %-20s %-6s %-24s %s\n" "#" "study" "experiment" "seed" "launcher" "n_trials"
    i=0
    while IFS= read -r cmd; do
        i=$((i + 1))
        printf "%-3s %-10s %-20s %-6s %-24s %s\n" "$i" \
            "$(study_of "$cmd")" \
            "$(field "$cmd" 'experiment')" \
            "$(field "$cmd" 'hydra\.sweeper\.sampler\.seed')" \
            "$(launcher_of "$cmd")" \
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
    study=$(study_of "$cmd")
    selected "$i" "$study" || continue

    # Private sweep dir per driver: hydra's default second-resolution timestamp
    # collides when several drivers start together, merging their outputs (and
    # the clariden launcher nests submitit_folder under hydra.sweep.dir).
    sweep=" hydra.sweep.dir=logs/train/multiruns/${sweep_ts}_${stem}_j$i"
    full="$PYTHON src/train.py$cmd$sweep${extra:+ $extra}"

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

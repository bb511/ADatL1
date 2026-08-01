#!/bin/bash
# Submit training commands of a run*_pareto.sh file to slurm on clariden,
# one job per command, via the hydra submitit launcher.
#
# Parses the commented 'python3 src/train.py' blocks of the given file, strips
# the local taskset/device pinning (each slurm job gets its own GPU), prepends
# '-m hydra/launcher=submitit_slurm_clariden hydra.launcher.timeout_min=...'
# and launches each submission driver detached (nohup), so this command returns
# after all jobs are submitted. Driver logs land in logs/submit/<file-stem>/.
#
# Usage (from the repository root, on clariden):
#   bash scripts/submit_pareto.sh scripts/physics/runae_pareto.sh \
#       paths.raw_data_dir=/path/to/adl1t_data/parquet_files
#   bash scripts/submit_pareto.sh --only consistency scripts/physics/runae_pareto.sh \
#       paths.raw_data_dir=/path/to/adl1t_data/parquet_files
#
#   --dry-run       print the full commands without submitting anything
#   --only <sel>    comma-separated run_name substrings (e.g. 'consistency') or
#                   1-based indices; default is every command in the file
#
# Any extra hydra overrides after the file name are appended to every job
# (later overrides win, so e.g. paths.raw_data_dir replaces the in-file one).
#
# A pareto file holds every strategy's points, so submitting one whole file is
# usually not what you want -- at 12 h per job, re-running the four strategies
# that are already trained is an expensive accident. Hence --only.
set -eu

TIMEOUT_MIN=720  # slurm time limit per job; the launcher default (60) is too short

dry=0
only=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry=1; shift ;;
        --only) only="${2:?--only needs a value}"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

file=${1:?usage: submit_pareto.sh [--dry-run] [--only sel] <run*_pareto.sh> [overrides...]}
shift || true
extra="$*"

[ -f "$file" ] || { echo "no such file: $file" >&2; exit 1; }

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

n_total=$(extract_cmds "$file" | wc -l | tr -d " ")
if [ "$n_total" -eq 0 ]; then
    echo "No training commands found in $file" >&2
    exit 1
fi

# $1 = index, $2 = run_name. Written with explicit ifs rather than
# 'test && action': under 'set -e' a bare failing && list aborts the script.
selected() {
    if [ -z "$only" ]; then return 0; fi
    local idx="$1" name="$2" s
    local IFS=','
    for s in $only; do
        if [ -n "$s" ]; then
            case "$s" in
                # An all-digit selector is an index only. Matching it as a
                # substring too would make '--only 1' also pick *_t1, *_t10, ...
                *[!0-9]*) case "$name" in *"$s"*) return 0 ;; esac ;;
                *) if [ "$s" = "$idx" ]; then return 0; fi ;;
            esac
        fi
    done
    return 1
}

# Preflight: the drivers are nohup'd, so an unimportable dependency shows up only
# as a tiny log per driver and nothing ever reaches slurm. Forgetting to activate
# the env once cost 24 silent failures on the search side, so check it here too.
# Override with PYTHON=/path/to/env/bin/python3 to skip activation entirely.
PYTHON=${PYTHON:-python3}
if [ "$dry" -eq 0 ] && ! "$PYTHON" -c 'import hydra' >/dev/null 2>&1; then
    echo "error: '$PYTHON' cannot import hydra - the adl1t env is not active." >&2
    echo "       conda activate /users/podagiu/.conda/envs/adl1t" >&2
    echo "       (or re-run with PYTHON=/users/podagiu/.conda/envs/adl1t/bin/python3)" >&2
    exit 1
fi

command -v sbatch >/dev/null 2>&1 || echo "warning: sbatch not on PATH - are you on clariden?" >&2

# Every generated block carries a literal
# 'paths.raw_data_dir=/path/to/adl1t_data/parquet_files' placeholder. Rewriting
# it in place (rather than relying on the appended override winning) is what
# makes the placeholder check below meaningful.
rdd=$(printf "%s" "$extra" | grep -oE 'paths\.raw_data_dir=[^ ]+' | tail -1 | cut -d= -f2- || true)

# Prefix with the domain directory: physics/cifar10/robustad share file stems,
# and a bare stem makes concurrent submissions clobber each other's logs.
stem="$(basename "$(dirname "$file")")_$(basename "$file" .sh)"
logdir="logs/submit/$stem"
[ "$dry" -eq 1 ] || mkdir -p "$logdir"
sweep_ts=$(date +%Y-%m-%d_%H-%M-%S)

i=0
n_sub=0
while IFS= read -r cmd; do
    i=$((i + 1))
    run_name=$(printf "%s" "$cmd" | grep -oE 'run_name=[^ ]+' | head -1 | cut -d= -f2- || true)
    selected "$i" "$run_name" || continue

    # Each slurm job sees a single GPU.
    cmd=$(printf "%s" "$cmd" | sed -E "s/trainer\.devices=\[[0-9]+\]/trainer.devices=[0]/")
    if [ -n "$rdd" ]; then
        cmd=$(printf "%s" "$cmd" | sed -E "s#paths\.raw_data_dir=[^ ]+#paths.raw_data_dir=${rdd}#")
    fi
    # Unique sweep dir per job: hydra's default second-resolution timestamp
    # collides when several drivers start simultaneously, merging job outputs.
    launcher="-m hydra/launcher=submitit_slurm_clariden hydra.launcher.timeout_min=$TIMEOUT_MIN hydra.sweep.dir=logs/train/multiruns/${sweep_ts}_${stem}_j$i"
    full="$PYTHON src/train.py $launcher$cmd $extra"

    case "$full" in
        *"/path/to/"*)
            echo "error: job $i ($run_name) still contains a '/path/to/...' placeholder." >&2
            echo "       pass paths.raw_data_dir=<real path> after the file name." >&2
            exit 1 ;;
    esac

    n_sub=$((n_sub + 1))
    if [ "$dry" -eq 1 ]; then
        echo "[$i/$n_total] $full"
        continue
    fi
    echo "[$i/$n_total] submitting: $run_name"
    nohup bash -c "$full" > "$logdir/job_${i}_${run_name}.log" 2>&1 &
    sleep 1
done <<EOF
$(extract_cmds "$file")
EOF

if [ "$n_sub" -eq 0 ]; then
    echo "nothing matched --only '$only' in $file" >&2
    exit 1
fi

if [ "$dry" -eq 0 ]; then
    echo
    echo "$n_sub of $n_total job(s) submitted (logs: $logdir)."
    echo "Jobs run under slurm; drivers are nohup'd, safe to log out."
    echo "Check queue with: squeue --me"
else
    echo
    echo "$n_sub of $n_total job(s) would be submitted."
fi

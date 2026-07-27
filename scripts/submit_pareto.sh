#!/bin/bash
# Submit every training command of a run*_pareto.sh file to slurm on clariden,
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
#
# Any extra hydra overrides after the file name are appended to every job
# (later overrides win, so e.g. paths.raw_data_dir replaces the in-file one).
# Pass --dry-run as the first argument to print the commands without running.
set -eu

TIMEOUT_MIN=720  # slurm time limit per job; the launcher default (60) is too short

dry=0
if [ "${1:-}" = "--dry-run" ]; then
    dry=1
    shift
fi
file=${1:?usage: submit_pareto.sh [--dry-run] <run*_pareto.sh> [overrides...]}
shift
extra="$*"

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
echo "Submitting $n_total jobs from $file"

# Prefix with the domain directory: physics/cifar10/robustad share file stems,
# and a bare stem makes concurrent submissions clobber each other's logs.
stem="$(basename "$(dirname "$file")")_$(basename "$file" .sh)"
logdir="logs/submit/$stem"
[ "$dry" -eq 1 ] || mkdir -p "$logdir"
sweep_ts=$(date +%Y-%m-%d_%H-%M-%S)

i=0
extract_cmds "$file" | while IFS= read -r cmd; do
    i=$((i + 1))
    # Each slurm job sees a single GPU.
    cmd=$(printf "%s" "$cmd" | sed -E "s/trainer\.devices=\[[0-9]+\]/trainer.devices=[0]/")
    # Unique sweep dir per job: hydra's default second-resolution timestamp
    # collides when several drivers start simultaneously, merging job outputs.
    launcher="-m hydra/launcher=submitit_slurm_clariden hydra.launcher.timeout_min=$TIMEOUT_MIN hydra.sweep.dir=logs/train/multiruns/${sweep_ts}_${stem}_j$i"
    full="python3 src/train.py $launcher$cmd $extra"
    if [ "$dry" -eq 1 ]; then
        echo "[$i/$n_total] $full"
        continue
    fi
    echo "[$i/$n_total] submitting: $(printf "%s" "$cmd" | grep -oE "run_name=[^ ]+")"
    nohup bash -c "$full" > "$logdir/job_$i.log" 2>&1 &
    sleep 1
done

if [ "$dry" -eq 0 ]; then
    echo "All submission drivers launched (logs: $logdir)."
    echo "Jobs run under slurm; drivers are nohup'd, safe to log out."
    echo "Check queue with: squeue --me"
fi

#!/bin/bash
# Run training commands of run*_pareto.sh files locally on an NGT session.
#
# The NGT sessions are interactive kubernetes pods with no batch scheduler, so
# unlike scripts/submit_pareto.sh (which hands every point to slurm through the
# submitit launcher) this script *is* the scheduler: it keeps a fixed number of
# jobs running, each pinned to one GPU and a 3-CPU range, and starts the next
# queued point as soon as a slot frees.
#
# Each Pareto point is a single run with its own hyperparameters, not a sweep,
# so there is nothing for a hydra launcher to schedule -- the plain
# 'taskset -c a-b' + 'CUDA_VISIBLE_DEVICES=g' form that the generated blocks
# were written for (see the commented local variants in any run*_pareto.sh) is
# what we use.
#
# Usage (from the repository root, inside an NGT session):
#
#   # L40s session: 2 full cards, 3 jobs each
#   bash scripts/submit_pareto_ngt.sh --only consistency --shard 1/6 \
#       --gpus 0,1 --per-gpu 3 --tag session-1 scripts/*/run*_pareto.sh
#
#   # H100 session: 4 MIG slices of 12 GB, capped at 6 concurrent (2/2/1/1)
#   bash scripts/submit_pareto_ngt.sh --only consistency --shard 2/6 \
#       --gpus 0,1,2,3 --per-gpu 2 --max-jobs 6 --tag session-2 \
#       scripts/*/run*_pareto.sh
#
#   --only <sel>        comma-separated run_name substrings (e.g. 'consistency')
#                       or 1-based indices; default is every command in the files
#   --shard <i>/<n>     take every n-th selected job starting at i (1-based).
#                       Interleaved, so each shard gets a mix of domains/models
#                       rather than one shard taking all of robustad.
#   --gpus <list>       comma-separated GPU indices to use, e.g. 0,1 or 0,1,2,3
#   --per-gpu <n>       how many jobs may share one GPU (default 1)
#   --max-jobs <n>      cap on concurrent jobs; default is gpus*per-gpu. Use it
#                       to spread fewer jobs over more GPUs: 6 over 4 slices
#                       lands 2/2/1/1 because slots take GPUs round-robin.
#   --raw-data-dir <p>  rewrite paths.raw_data_dir in every block
#   --tag <name>        log subdirectory name (default: hostname)
#   --emit-joblist <f>  write the selected commands to <f> and exit
#   --dry-run           print what would run, launch nothing
#
# This script blocks until every one of its jobs is done. The pod survives a
# dropped 'kubectl exec' but the exec'd process tree does not, so launch it
# detached:
#
#   setsid nohup bash scripts/submit_pareto_ngt.sh ... > run.log 2>&1 &
set -eu

PYTHON=${PYTHON:-/opt/venv/bin/python3}
CPUS_PER_JOB=3
POLL_SECONDS=5
DEFAULT_RAW_DATA_DIR=/shared/deodagiu/adl1t_data/parquet_files

dry=0
only=""
shard_i=1
shard_n=1
gpus="0"
per_gpu=1
max_jobs=0
raw_data_dir="$DEFAULT_RAW_DATA_DIR"
tag=""
emit=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry=1; shift ;;
        --only) only="${2:?--only needs a value}"; shift 2 ;;
        --shard)
            sel="${2:?--shard needs i/n}"
            shard_i=${sel%%/*}
            shard_n=${sel##*/}
            case "$shard_i$shard_n" in *[!0-9]*) echo "bad --shard '$sel'" >&2; exit 2 ;; esac
            if [ "$shard_i" -lt 1 ] || [ "$shard_i" -gt "$shard_n" ]; then
                echo "bad --shard '$sel': need 1 <= i <= n" >&2; exit 2
            fi
            shift 2 ;;
        --gpus) gpus="${2:?--gpus needs a value}"; shift 2 ;;
        --per-gpu) per_gpu="${2:?--per-gpu needs a value}"; shift 2 ;;
        --max-jobs) max_jobs="${2:?--max-jobs needs a value}"; shift 2 ;;
        --raw-data-dir) raw_data_dir="${2:?--raw-data-dir needs a value}"; shift 2 ;;
        --tag) tag="${2:?--tag needs a value}"; shift 2 ;;
        --emit-joblist) emit="${2:?--emit-joblist needs a value}"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

[ $# -gt 0 ] || { echo "usage: submit_pareto_ngt.sh [options] <run*_pareto.sh>..." >&2; exit 2; }
[ -n "$tag" ] || tag=$(hostname)

# GPU pool -> array.
IFS=',' read -r -a gpu_arr <<< "$gpus"
n_gpus=${#gpu_arr[@]}
[ "$n_gpus" -gt 0 ] || { echo "no GPUs given" >&2; exit 2; }

slots=$((n_gpus * per_gpu))
if [ "$max_jobs" -gt 0 ] && [ "$max_jobs" -lt "$slots" ]; then
    slots=$max_jobs
fi

# Same parser as scripts/submit_pareto.sh:49-62 -- the block format is identical,
# so this stays byte-for-byte in step with the generator.
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

# The pods are cgroup-restricted to a CPU subset that is neither 0-based nor
# contiguous -- an L40s session reports '64-94,96-126,192-222,224-254', so a
# naive 'taskset -c 0-2' dies with "failed to set affinity: Invalid argument"
# and the job never starts. Expand the allowed list and hand out explicit CPU
# ids instead of ranges, which also sidesteps the holes at 95 and 223.
cpus_allowed() {
    local list item start end c
    list=$(grep Cpus_allowed_list /proc/self/status 2>/dev/null | awk '{print $2}')
    if [ -z "$list" ]; then
        # Not linux (e.g. a --dry-run on a laptop): fall back to 0..nproc-1.
        c=$( (nproc 2>/dev/null) || echo 4 )
        seq 0 $((c - 1))
        return
    fi
    local IFS=','
    for item in $list; do
        case "$item" in
            *-*) start=${item%%-*}; end=${item##*-}
                 for ((c = start; c <= end; c++)); do echo "$c"; done ;;
            *)   echo "$item" ;;
        esac
    done
}

# $1 = index, $2 = run_name. Explicit ifs rather than 'test && action': under
# 'set -e' a bare failing && list aborts the script.
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

# ---------------------------------------------------------------------------
# Build the selected, sharded job list.
# ---------------------------------------------------------------------------
declare -a JOB_CMD JOB_NAME
idx=0        # index within the --only selection, before sharding
n_seen=0

for file in "$@"; do
    [ -f "$file" ] || { echo "no such file: $file" >&2; exit 1; }
    while IFS= read -r cmd; do
        [ -n "$cmd" ] || continue
        n_seen=$((n_seen + 1))
        run_name=$(printf "%s" "$cmd" | grep -oE 'run_name=[^ ]+' | head -1 | cut -d= -f2- || true)
        selected "$n_seen" "$run_name" || continue

        # Interleaved shard: index 0 -> shard 1, index 1 -> shard 2, ...
        if [ "$shard_n" -gt 1 ] && [ $((idx % shard_n)) -ne $((shard_i - 1)) ]; then
            idx=$((idx + 1))
            continue
        fi
        idx=$((idx + 1))

        # One GPU per job. CUDA_VISIBLE_DEVICES picks the physical card, so the
        # process always sees it as device 0.
        cmd=$(printf "%s" "$cmd" | sed -E "s/trainer\.devices=\[[0-9]+\]/trainer.devices=[0]/")
        cmd=$(printf "%s" "$cmd" | sed -E "s#paths\.raw_data_dir=[^ ]+#paths.raw_data_dir=${raw_data_dir}#")

        full="$PYTHON src/train.py$cmd"
        case "$full" in
            *"/path/to/"*)
                echo "error: '$run_name' still contains a '/path/to/...' placeholder." >&2
                echo "       pass --raw-data-dir <real path>." >&2
                exit 1 ;;
        esac

        JOB_CMD+=("$full")
        JOB_NAME+=("$run_name")
    done <<EOF
$(extract_cmds "$file")
EOF
done

n_jobs=${#JOB_CMD[@]}
if [ "$n_jobs" -eq 0 ]; then
    echo "nothing matched --only '$only' (shard $shard_i/$shard_n) in the given files" >&2
    exit 1
fi

if [ -n "$emit" ]; then
    mkdir -p "$(dirname "$emit")"
    : > "$emit"
    for j in $(seq 0 $((n_jobs - 1))); do printf "%s\n" "${JOB_CMD[$j]}" >> "$emit"; done
    echo "wrote $n_jobs job(s) to $emit"
    exit 0
fi

# Preflight: without it a missing env yields N tiny logs and no training at all.
if [ "$dry" -eq 0 ] && ! "$PYTHON" -c 'import hydra' >/dev/null 2>&1; then
    echo "error: '$PYTHON' cannot import hydra." >&2
    echo "       re-run with PYTHON=/opt/venv/bin/python3" >&2
    exit 1
fi

logdir="logs/submit_ngt/$tag"
[ "$dry" -eq 1 ] || mkdir -p "$logdir"

echo "tag=$tag  shard=$shard_i/$shard_n  jobs=$n_jobs  slots=$slots  gpus=[$gpus] per_gpu=$per_gpu"
echo "logs -> $logdir"
echo

# ---------------------------------------------------------------------------
# Slot table. Slot s always runs on gpu_arr[s % n_gpus] and CPUs
# [s*3, s*3+2], so a cap of 6 over 4 GPUs loads them 2/2/1/1.
# ---------------------------------------------------------------------------
ALLOWED=()
while IFS= read -r c; do ALLOWED+=("$c"); done < <(cpus_allowed)
n_allowed=${#ALLOWED[@]}
need=$((slots * CPUS_PER_JOB))
if [ "$n_allowed" -lt "$need" ]; then
    if [ "$dry" -eq 1 ]; then
        # A --dry-run on a laptop has no cgroup list; wrap so the table still prints.
        echo "warning: only $n_allowed CPUs available, need $need - slot lists below wrap" >&2
    else
        echo "error: need $need CPUs for $slots slots but the cgroup allows only $n_allowed" >&2
        exit 1
    fi
fi

declare -a slot_pid slot_gpu slot_cpus
for s in $(seq 0 $((slots - 1))); do
    slot_pid[$s]=0
    slot_gpu[$s]=${gpu_arr[$((s % n_gpus))]}
    cl=""
    for k in $(seq 0 $((CPUS_PER_JOB - 1))); do
        cl="$cl,${ALLOWED[$(((s * CPUS_PER_JOB + k) % n_allowed))]}"
    done
    slot_cpus[$s]="${cl#,}"
done

if [ "$dry" -eq 1 ]; then
    for j in $(seq 0 $((n_jobs - 1))); do
        s=$((j % slots))
        echo "[$((j + 1))/$n_jobs] slot $s gpu ${slot_gpu[$s]} cpus ${slot_cpus[$s]}: ${JOB_NAME[$j]}"
        echo "    CUDA_VISIBLE_DEVICES=${slot_gpu[$s]} taskset -c ${slot_cpus[$s]} ${JOB_CMD[$j]}"
    done
    echo
    echo "$n_jobs job(s) would run, $slots at a time."
    exit 0
fi

launched=0
j=0
while [ "$j" -lt "$n_jobs" ]; do
    free=-1
    for s in $(seq 0 $((slots - 1))); do
        pid=${slot_pid[$s]}
        if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then free=$s; break; fi
    done
    if [ "$free" -lt 0 ]; then
        sleep "$POLL_SECONDS"
        continue
    fi

    gpu=${slot_gpu[$free]}
    cpus=${slot_cpus[$free]}
    name=${JOB_NAME[$j]}
    log="$logdir/job_$((j + 1))_${name}.log"

    echo "[$((j + 1))/$n_jobs] slot $free gpu $gpu cpus $cpus -> $name"
    CUDA_VISIBLE_DEVICES="$gpu" nohup taskset -c "$cpus" \
        bash -c "${JOB_CMD[$j]}" > "$log" 2>&1 &
    slot_pid[$free]=$!
    launched=$((launched + 1))
    j=$((j + 1))
    sleep 1
done

wait
echo
echo "$launched job(s) finished (logs: $logdir)."

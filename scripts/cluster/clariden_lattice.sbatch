#!/bin/bash
#SBATCH --account=a0166
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=12:00:00
# Without it clariden gives a job 450 GiB; its MaxMemPerNode is 850000 MB, so --mem=0
# (the node's 870 GB) is refused (27 Oct).
#SBATCH --mem=820G
#SBATCH --job-name=lattice-node
#SBATCH --output=logs/clariden_lattice_%j.out
# The lattices of one or more dump-regime run*_search.sh files on one whole clariden
# node: each catalogue's pool is trained, checked and replayed by src/eval.py.
#
# Each catalogue runs as a chain in the background. Its drivers run as in
# clariden_node.sbatch, one per GPU per catalogue. Its lattice passes the check when
# the Optuna study holds the target number of COMPLETE trials and the lattice
# directory as many complete runs (meta.pt, initial.pt and one epoch_*.pt per epoch)
# and nothing else; a run directory short of that is moved to
# lattices/<domain>/_partial/<lattice>/ before every check, since src/eval.py would
# replay it. A study left short is topped up once, by the relaunch that
# scripts/cluster/sweep_status.py prints, and checked again. A lattice that passes is
# replayed under a lock, one lattice at a time across the chains, because each replay
# holds REPLAY_WORKERS processes of about 6.7 GB. A lattice directory that already
# holds runs is not trained again, and a lattice whose selection tables are written
# is not replayed again, so that a resubmission after a time limit only tops up and
# replays what is left. The job ends with its last chain.
#
# The target is the sum of the live blocks' n_trials, or the number of drivers times
# an appended hydra.sweeper.n_trials; the epochs are the blocks' trainer.max_epochs
# or an appended one; the lattice is the experiment_name of the experiment config,
# unless a block or an appended override sets it; the domain is the catalogue's
# directory.
#
# Submit from the repository root, which the output path is relative to:
#   sbatch scripts/cluster/clariden_lattice.sbatch scripts/cifar10/runae_search.sh \
#       scripts/cifar10/runrealnvp_search.sh
#   REPLAY=0 sbatch scripts/cluster/clariden_lattice.sbatch \
#       scripts/cifar10/runsvdd_search.sh hydra.sweeper.n_trials=1
#   SLURM_SUBMIT_DIR=$PWD bash scripts/cluster/clariden_lattice.sbatch --dry-run \
#       scripts/cifar10/runae_search.sh
# An argument ending in .sh is a catalogue; every other one is a hydra override
# appended to all of their drivers. --dry-run, if given, comes first: it prints each
# catalogue's driver commands, target, epochs, lattice and replay, runs nothing and
# needs no env. From the environment: REPLAY=0 ends each chain after its check and
# prints the replay instead of running it, TOPUP=0 does the same for the relaunch of a
# short study, REPLAY_WORKERS sets the replay's workers, REPLAY_THREADS the threads of
# each (1, the reference, by default), REPLAY_LIMIT its time limit (timeout's syntax)
# and MONITOR_S the interval, in seconds, of the load samples in
# logs/clariden_lattice_<job>/monitor.log.
set -eu
cd "${SLURM_SUBMIT_DIR:?submit this with sbatch}" || exit 1
. scripts/cluster/lib.sh

# One worker per run of a 90-run lattice, about 600 GB beside the drivers of the
# chains still training (856 GB per node); the smoke measures both and sets it.
REPLAY_WORKERS=${REPLAY_WORKERS:-90}
# A replay whose worker died stays blocked in the pool (the smoke of 27 Oct held its node
# 45 min that way), so each replay is stopped at a limit well above its expected time.
REPLAY_LIMIT=${REPLAY_LIMIT:-4h}
MONITOR_S=${MONITOR_S:-300}
# The torch, OpenMP and BLAS threads of every replay worker (worker_threads of
# src/eval.py; more than one may change float32 results in the last bits).
REPLAY_THREADS=${REPLAY_THREADS:-1}
# The malloc settings spare the replay workers the trim and mmap churn of their many
# short-lived tensors (much faster on an NGT CPU pod, tables bitwise unchanged).
REPLAY_ENV=(OMP_NUM_THREADS=$REPLAY_THREADS MKL_NUM_THREADS=$REPLAY_THREADS
    OPENBLAS_NUM_THREADS=$REPLAY_THREADS
    MALLOC_MMAP_THRESHOLD_=33554432 MALLOC_TRIM_THRESHOLD_=17179869184
    MALLOC_TOP_PAD_=268435456)

say() { echo "$(date '+%F %T') $*"; }
die() { echo "$*" >&2; exit 2; }

# The last value of override key $1 (a regex) among the appended ones, as Hydra
# takes the last.
appended() {
    printf "%s\n" "${overrides[@]}" | grep -oE "^$1=.+" | tail -1 | cut -d= -f2- |
        tr -d "'" || true
}

# The value of key $2 in yaml file $1, read as text.
yaml_value() {
    grep -oE "^[[:space:]]*$2:[[:space:]]*\S+" "$1" 2>/dev/null | head -1 |
        awk '{print $2}' || true
}

# The first of: an appended override $1, the first block's, yaml key $3 of file $2.
setting() {
    local v
    v=$(appended "$1")
    v=${v:-$(field "$first" "$1" | tr -d "'")}
    if [ -z "$v" ] && [ -n "${2:-}" ]; then v=$(yaml_value "$2" "$3"); fi
    printf "%s" "$v"
}

# The globals of catalogue $1 that a chain and the dry run use.
catalogue() {
    local cmds per hp
    file=$1
    cmds=$(extract_cmds "$file")
    first=$(printf "%s\n" "$cmds" | head -1)
    n_drivers=$(printf "%s\n" "$cmds" | grep -c . || true)
    [ "$n_drivers" -gt 0 ] || die "$file: no live block"
    domain=$(basename "$(dirname "$file")")
    stem="${domain}_$(basename "$file" .sh)"
    per=$(appended 'hydra\.sweeper\.n_trials')
    if [ -n "$per" ]; then
        target=$((n_drivers * per))
    else
        target=$(printf "%s\n" "$cmds" | grep -oE 'hydra\.sweeper\.n_trials=[0-9]+' |
            cut -d= -f2 | awk '{s += $1} END {print s + 0}')
    fi
    epochs=$(setting 'trainer\.max_epochs')
    lattice=$(setting experiment_name \
        "configs/experiment/$(field "$first" experiment).yaml" experiment_name)
    hp="configs/hparams_search/$(field "$first" hparams_search).yaml"
    study=$(setting 'hydra\.sweeper\.study_name' "$hp" study_name)
    db=$(setting 'hydra\.sweeper\.storage' "$hp" storage)
    db=${db#sqlite:///}
    db=${db%%\?*}
    seed0=$(field "$first" 'hydra\.sweeper\.sampler\.seed')
    ldir="lattices/$domain/$lattice"
    for v in target epochs lattice study db seed0; do
        [ -n "${!v}" ] || die "$file: no $v found"
    done
}

# Runs "$@" in the background and waits for it, so that a TERM reaching the chain
# reaches the command too: bash runs a trap only once its foreground command returns.
run_step() {
    local rc=0
    "$@" &
    step=$!
    wait "$step" || rc=$?
    step=""
    return "$rc"
}

is_complete() {
    [ -f "$1/meta.pt" ] && [ -f "$1/initial.pt" ] &&
        [ "$(find "$1" -maxdepth 1 -name 'epoch_*.pt' | wc -l)" -eq "$epochs" ]
}

set_aside() {
    local d partial="lattices/$domain/_partial/$lattice"
    for d in "$ldir"/*/; do
        if [ -d "$d" ] && ! is_complete "$d"; then
            mkdir -p "$partial"
            mv "${d%/}" "$partial/"
            say "$stem: incomplete run $(basename "$d") moved to $partial/"
        fi
    done
}

count_runs() {
    local d n=0
    for d in "$ldir"/*/; do
        if [ -d "$d" ] && is_complete "$d"; then n=$((n + 1)); fi
    done
    echo "$n"
}

complete_trials() {
    python - "$db" "$study" <<'EOF'
import sys

sys.path.insert(0, "scripts/cluster")
from sweep_status import complete_of

print(complete_of(sys.argv[1], sys.argv[2]))
EOF
}

# Sets the incomplete runs aside; true when study and lattice both hold the target
# and the lattice nothing else.
settle() {
    local runs entries
    set_aside
    done_n=$(complete_trials) || { say "$stem: cannot read $db"; return 1; }
    runs=$(count_runs)
    entries=$(find "$ldir" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
    say "$stem: $done_n/$target COMPLETE trials, $runs/$target complete runs" \
        "($entries entries) in $ldir"
    [ "$done_n" -eq "$target" ] && [ "$runs" -eq "$target" ] &&
        [ "$entries" -eq "$runs" ]
}

# The relaunch of sweep_status.py's launch_command: the first driver, for the
# shortfall, with the fresh sampler seed 1000 * its seed + the COMPLETE count.
top_up() {
    local n=$((target - done_n))
    if [ "$n" -le 0 ]; then say "$stem: no top-up, the study is not short"; return 1; fi
    local cmd=(bash scripts/cluster/submit_search.sh --only 1 --gpus 4 --wait "$file"
        "${overrides[@]}" "hydra.sweeper.n_trials=$n"
        "hydra.sweeper.sampler.seed=$((1000 * seed0 + done_n))")
    if [ "${TOPUP:-1}" = 0 ]; then
        say "$stem: TOPUP=0, not run: ${cmd[*]}"
        return 1
    fi
    say "$stem: relaunching the first driver for $n trial(s)"
    run_step "${cmd[@]}" >> "$clog" 2>&1 || say "$stem: the relaunched driver failed"
}

replay() {
    local cmd=(python src/eval.py "eval=$domain" "lattice=$lattice"
        "workers=$REPLAY_WORKERS" "worker_threads=$REPLAY_THREADS")
    if [ "${REPLAY:-1}" = 0 ]; then
        say "$stem: REPLAY=0, not run: ${cmd[*]}"
        return
    fi
    # The selection writes this table last.
    if [ -f "logs/eval/$lattice/${lattice}_eval.csv" ]; then
        say "$stem: logs/eval/$lattice/ already holds the selection, not replayed again"
        return
    fi
    say "$stem: queued for the replay lock"
    if ! run_step flock "$logdir/replay.lock" timeout -k 60 "$REPLAY_LIMIT" \
        env "${REPLAY_ENV[@]}" "${cmd[@]}" \
        > "$logdir/replay_$lattice.log" 2>&1; then
        say "$stem: replay failed, see $logdir/replay_$lattice.log"
        return 1
    fi
    say "$stem: replayed into logs/eval/$lattice/"
}

chain() {
    trap 'kill -TERM ${step:-} 2>/dev/null || true; wait ${step:-} || true; exit 143' \
        TERM
    catalogue "$1"
    clog="$logdir/$stem.log"
    if [ -d "$ldir" ] && [ -n "$(ls -A "$ldir")" ]; then
        say "$stem: $ldir already holds runs, not trained again"
    else
        say "$stem: training $target trial(s) of $epochs epochs into $ldir"
        run_step bash scripts/cluster/submit_search.sh --gpus 4 --wait "$file" \
            "${overrides[@]}" >> "$clog" 2>&1 || say "$stem: a driver failed"
    fi
    if ! settle; then
        top_up && settle || { say "$stem: check failed, replay skipped"; return 1; }
    fi
    replay
}

dry_run() {
    catalogue "$1"
    echo "== $1: $target trial(s) over $n_drivers drivers, $epochs epochs"
    echo "   study '$study' in $db; lattice $ldir"
    bash scripts/cluster/submit_search.sh --dry-run --gpus 4 "$1" "${overrides[@]}"
    echo "   if short: submit_search.sh --only 1 --gpus 4 --wait $1 ..." \
        "hydra.sweeper.n_trials=<$target - COMPLETE>" \
        "hydra.sweeper.sampler.seed=<1000 * $seed0 + COMPLETE>"
    echo "   replay: python src/eval.py eval=$domain lattice=$lattice" \
        "workers=$REPLAY_WORKERS worker_threads=$REPLAY_THREADS"
    echo
}

# Creates each MLflow experiment before any driver starts: FileStore checks for a
# name, then creates it, so drivers starting together can make two of one name.
create_experiments() {
    python - "file:$PWD/logs/mlflow/mlruns" "$@" <<'EOF'
import sys

from mlflow.tracking import MlflowClient

client = MlflowClient(sys.argv[1])
for name in sys.argv[2:]:
    if client.get_experiment_by_name(name) is None:
        client.create_experiment(name)
EOF
}

sample() {
    date '+%F %T'
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    free -g
    # The replay's workers are spawned, so their command line names multiprocessing.
    ps -eo rss=,args= | awk '
        $2 !~ /python/ { next }
        /src\/train\.py/ { t += $1; n++; if ($1 > tm) tm = $1 }
        /src\/eval\.py|multiprocessing/ { e += $1; m++; if ($1 > em) em = $1 }
        END { g = 1048576; printf "rss GiB: train %.1f over %d (max %.1f), " \
            "eval %.1f over %d (max %.1f)\n", t / g, n, tm / g, e / g, m, em / g }'
}

monitor() {
    trap 'kill ${nap:-} 2>/dev/null || true; exit 0' TERM
    while :; do
        sample >> "$logdir/monitor.log" 2>&1
        sleep "$MONITOR_S" &
        nap=$!
        wait "$nap"
    done
}

dry=0
if [ "${1:-}" = --dry-run ]; then dry=1; shift; fi
cats=()
overrides=()
for a in "$@"; do
    case "$a" in *.sh) cats+=("$a") ;; *) overrides+=("$a") ;; esac
done
[ "${#cats[@]}" -gt 0 ] || die "no run*_search.sh given"
if [ "$dry" -eq 1 ]; then
    for c in "${cats[@]}"; do dry_run "$c"; done
    exit 0
fi

# The env's activate.d script reads LD_LIBRARY_PATH, unset in a batch shell.
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$CLARIDEN_CONDA/etc/profile.d/conda.sh"
conda activate "$CLARIDEN_ENV"
# Torch starts one thread per core (288) in every driver without a cap.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
# The compute nodes read CIFAR-10 from the pushed arrow cache.
export HF_DATASETS_OFFLINE=1
# Absolute, for the replay's spawned workers and every run's configuration; the
# .env of setup.sh holds ".".
export PROJECT_ROOT=$PWD

logdir="logs/clariden_lattice_${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$logdir"
lattices=()
for c in "${cats[@]}"; do
    catalogue "$c"
    lattices+=("$lattice")
done
create_experiments "${lattices[@]}"

monitor &
mon=$!
# slurm kills every process of the job once this shell exits, so at the time limit
# it passes the TERM on and waits for submit_search.sh to mark its drivers stopped.
pids=()
trap 'kill -TERM "${pids[@]}" "$mon" 2>/dev/null || true; wait || true; exit 143' TERM
for c in "${cats[@]}"; do
    chain "$c" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=$((failed + 1)); fi
done
kill "$mon" 2>/dev/null || true
wait "$mon" || true
say "$((${#cats[@]} - failed)) of ${#cats[@]} chain(s) passed (logs: $logdir)"
[ "$failed" -eq 0 ]

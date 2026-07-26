#!/bin/bash
# Smoke test for the clariden pareto submission (physics AE).
#
# Extracts TWO commands from scripts/physics/runae_pareto.sh -- the first
# semi-supervised cvar25 point and the first agnostic cap point, covering both
# command templates -- and submits them through the same
# scripts/submit_pareto.sh used for the real campaign, but shortened
# (2 epochs, few batches, 30 min slurm limit) and under a throwaway
# experiment_name (debug_pareto_smoke), so nothing collides with real runs.
#
# Usage (from the repository root, on clariden):
#   bash scripts/smoketest_pareto.sh paths.raw_data_dir=/path/to/parquet_files
#
# Any extra hydra overrides are appended to both jobs. After the jobs finish
# (squeue --me), verify with the printed check command.
set -eu

src=scripts/physics/runae_pareto.sh

# Build a mini pareto file holding the first block of the CVAR25 and CAP
# sections of the real script (kept in the same commented format).
tmp=$(mktemp /tmp/pareto_smoke_XXXXXX)
awk '
    /^# CVAR25 TRAINING/ { want = 1 }
    /^# CAP TRAINING/    { want = 1 }
    /^# taskset/ && want { grab = 1 }
    grab {
        print
        if ($0 !~ /\\$/) { grab = 0; want = 0; print "" }
    }
' "$src" > "$tmp"

n=$(grep -c "python3 src/train.py" "$tmp")
echo "smoke file: $tmp ($n commands)"
[ "$n" -eq 2 ] || { echo "expected 2 commands, got $n -- aborting"; exit 1; }
grep -oE "run_name=[^ \\\\]+" "$tmp"

case "$*" in
    *paths.raw_data_dir=*) ;;
    *) echo "WARNING: no paths.raw_data_dir override given; jobs will use the" \
            "placeholder path from the script and fail at data loading." ;;
esac

echo
echo "--- dry run ---"
bash scripts/submit_pareto.sh --dry-run "$tmp" \
    experiment_name=debug_pareto_smoke \
    trainer.max_epochs=2 \
    +trainer.limit_train_batches=10 \
    +trainer.limit_val_batches=4 \
    +trainer.limit_test_batches=4 \
    "~evaluation.evaluator.ckpts.single.ascore_operational" \
    hydra.launcher.timeout_min=30 \
    "$@"

echo
echo "--- submitting ---"
bash scripts/submit_pareto.sh "$tmp" \
    experiment_name=debug_pareto_smoke \
    trainer.max_epochs=2 \
    +trainer.limit_train_batches=10 \
    +trainer.limit_val_batches=4 \
    +trainer.limit_test_batches=4 \
    "~evaluation.evaluator.ckpts.single.ascore_operational" \
    hydra.launcher.timeout_min=30 \
    "$@"

echo
echo "Watch the queue with:  squeue --me"
echo "Driver logs:           logs/submit/$(basename "$tmp" .sh)/"
echo
echo "Once both jobs have finished, verify the runs logged their eval metrics:"
echo "  python -c \""
echo "from mlflow.tracking import MlflowClient"
echo "c = MlflowClient('file:logs/mlflow/mlruns')"
echo "e = c.get_experiment_by_name('debug_pareto_smoke')"
echo "for r in c.search_runs([e.experiment_id]):"
echo "    ek = [k for k in r.data.metrics if k.startswith('eval/')]"
echo "    print(r.info.run_name, len(ek), 'eval metrics,',"
echo "          'optuna pair:', r.data.metrics.get('eval/val/optimized_main'))\""
echo
echo "Expect 2 runs (cvar25_*, cap_*), each with >0 eval metrics and a"
echo "non-None optuna pair. The smoke experiment/checkpoints can be deleted"
echo "afterwards (experiment_name=debug_pareto_smoke, checkpoints/debug_pareto_smoke)."

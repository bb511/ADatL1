# Clariden restart canary

Run this isolated gate after creating the new immutable execution worktree and before
submitting any production search or retraining allocation. It tests the exact Slurm
shape used by retraining: an ordinary `USR1` reaches the Lightning process launched by
`srun`, Lightning writes an HPC checkpoint, and the restarted allocation archives it
as scheduler evidence instead of restoring it. Training rolls back to an authenticated
end-of-epoch `last.ckpt` whose epoch matches contiguous MLflow history. State in the
production efficiency/CAP/W1/drift/stable callbacks survives that exact restore,
retained stable files remain present, and one explicitly tagged MLflow run spans both
allocations. Nonzero-learning-rate terminal parameters must equal an uninterrupted
same-allocation reference run with identical logger/callback topology exactly. The
checkpoint is accepted only when Lightning's ready, started, and processed epoch counts
all equal `epoch + 1`; validation-end pseudo-boundaries are rejected.
The same run must contain exactly one `epoch_idx` record for each completed epoch,
with both steps and values equal to `0,1,2,3,4,5`; this catches a restart that silently
duplicates or skips an epoch even when it reaches the nominal final epoch.

The canary uses synthetic data under a dedicated root. It never reads or writes a
scientific campaign. The first allocation deliberately lasts until the scheduler signal;
the requeued allocation removes that delay and should finish in seconds. The debug job
normally consumes about three to five minutes of GPU allocation plus queue time.

Choose a fresh root and substitute the immutable execution worktree/commit:

```bash
export CANARY_REPO=/iopsstor/scratch/cscs/vjimenez/adatl1/worktrees/<worktree>
export CANARY_COMMIT=$(git -C "${CANARY_REPO}" rev-parse HEAD)
export CANARY_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1/canaries/retrain_restart_canary_${CANARY_COMMIT:0:12}_$(date +%Y%m%dT%H%M%S)

sbatch --parsable \
  --export=ALL,PHYSICS_BGPAIR_REPOSITORY_ROOT="${CANARY_REPO}",PHYSICS_BGPAIR_RESTART_CANARY_ROOT="${CANARY_ROOT}",PHYSICS_BGPAIR_EXPECTED_COMMIT="${CANARY_COMMIT}" \
  "${CANARY_REPO}/scripts/clariden/physics_background_pairing_restart_canary.sbatch"
```

Do not resubmit after the first allocation: Lightning itself must requeue the same job.
Check it without modifying scheduler state:

```bash
squeue -j <job-id> -o '%.18i %.30j %.8T %.10M %.10l %R'
sacct -j <job-id> -X --format=JobID,JobName,State,Elapsed,Timelimit,RestartCnt,ExitCode
jq . "${CANARY_ROOT}/COMPLETE.json"
```

Once the allocation is terminal, make the result a campaign prerequisite through the
control plane:

```bash
uv run python scripts/physics_background_pairing_operations.py \
  --root "${BGPAIR_ROOT}" --execution-repository "${CANARY_REPO}" \
  validate-restart-canary --canary-root "${CANARY_ROOT}" --job-id "<job-id>"
```

This creates `state/restart_canary_validation.json` idempotently only after independently
checking the frozen commit, every referenced artifact/hash, the exact MLflow histories
and tags, append-preserved scheduler log, and terminal Slurm restart accounting.

Success means the job ends `COMPLETED` with a nonzero restart count and `COMPLETE.json`
exists. That artifact authenticates the archived, explicitly non-resumable HPC evidence;
the exact completed-epoch resume checkpoint and its MLflow history; restored callback
state; stable checkpoint files; one MLflow run; equality with the uninterrupted
parameter oracle; and append-preserved scheduler-log sentinels from both allocations.
Preserve the whole canary root and scheduler log as campaign preflight evidence. Any
missing or invalid item makes the script fail closed and forbids production submission.

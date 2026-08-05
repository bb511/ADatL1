# Clariden restart canary

Run this isolated gate after creating the new immutable execution worktree and before
submitting any production search or retraining allocation. It tests the exact Slurm
shape used by retraining: an ordinary `USR1` reaches the Lightning process launched by
`srun`, Lightning writes and restores an HPC checkpoint, state in the production
efficiency/CAP/W1/drift/stable callbacks survives, retained stable files remain present, and one
explicitly tagged MLflow run spans both allocations.
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

Success means the job ends `COMPLETED` with a nonzero restart count and `COMPLETE.json`
exists. That artifact authenticates the input HPC checkpoint, restored callback state,
stable checkpoint files, MLflow run/tags/allocation history, deterministic terminal model
hash, the exact contiguous epoch history, and the append-preserved scheduler-log sentinels
from both allocations. Preserve the whole canary root and scheduler log as campaign
preflight evidence. Any missing or invalid item makes the script fail closed and forbids
production submission.

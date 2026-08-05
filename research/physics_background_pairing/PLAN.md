# Autonomous physics background-pairing research plan

## 0. Mission, current state, and non-negotiable outcome

This document is the authoritative runbook for completing the physics auto-research
campaign. A future coding agent should be able to start from this file, inspect the
current machine state, and execute the campaign to completion without reconstructing
the scientific intent from conversation history.

The goal is to repeat the paper's signal-agnostic physics model-selection experiment
while changing only the requested validation/model-selection inputs and score views:

1. Keep the paper training data and ordinary combined `normal` validation/test data.
2. Replace the model-selection comparison between recorded ZeroBias and
   `SingleNeutrino_E-10-gun` with held-out source-specific samples from two recorded
   ZeroBias eras:
   - background0: `ZB_run396102` (2025E)
   - background1: `ZB_run398183` (2025G)
3. Evaluate CAP with four persisted event mappings and an empirical-CDF control.
4. Evaluate Wasserstein-1 and threshold drift on the same two source samples.
5. For AE and VAE, retain the paper-native score and add residual Mahalanobis/OAS as a
   parallel score view. Do not replace the native score.
6. Produce all search, Pareto, retraining, downstream, provenance, and reporting
   artifacts needed to understand and reproduce the result.
7. Finish with a self-contained `REPORT.md` and portable CSV/JSON result package.

The six-study implementation was committed at
`e1322058bc821f6c612ced6f9b78b0ce580b74d1`. This is the implementation baseline, not
an instruction to hard-code that commit forever: execution must use the final commit
that contains every code/config/reporting change required by this runbook. The campaign
design pins the exact commit and rejects a different checkout.

At the time this runbook was written:

- the six-study training/evaluation implementation, explicit search-trial bindings,
  reverse-drift diagnostics, and portable reporting utility are implemented;
- all eight persisted mapping tables have been authenticated;
- isolated AE/VAE OAS, CAP, CDF, and W1 pilots passed on an older commit;
- the new unified all-metric pilot has not run;
- no schema-2 six-study scratch campaign, Optuna database, full search, Pareto freeze,
  200-epoch retraining, final aggregation, or final report exists;
- older scratch campaigns are superseded schema-1/56-cell campaigns and must not be
  resumed or mixed with this campaign.

The agent must keep the following outcome in mind before every mutation or job
submission:

> Exactly six shared Optuna studies must yield 600 usable 50-epoch trials each; 56
> logical two-objective fronts must then be frozen; every unique front configuration
> must be retrained once for 200 epochs; and every one of the 56 reported winners must
> be selected by mean downstream performance and represented in the final portable
> results and `REPORT.md`.

Do not launch 56 independent 600-trial searches. The user explicitly chose six shared
model studies so that every validation strategy for a model sees the same sampled
hyperparameter configurations.

## 1. Scientific contract

### 1.1 Relationship to the paper protocol

The paper workflow, adapted to the approved six shared studies, is:

1. Run 600 usable Optuna trials for each of the six models. Each trial evaluates every
   relevant dataset/score/validation strategy for that model.
2. Train every search trial for exactly 50 epochs.
3. For each logical strategy within each trial, configure three checkpoint candidate
   families in this order:
   - last epoch;
   - best EMA checkpoint for that strategy's primary criterion;
   - stable native operational-anomaly-score checkpoint.
   The paper prose calls these three retained candidates, but the `dev/patrick`
   implementation creates the stable checkpoint only if a 10-epoch window has less
   than 5% variation. When that condition never occurs, its configured directory is
   empty and the evaluator compares the two available candidates; it does not invent a
   fallback or fail the trial. Preserve this reference behavior and audit availability.
4. Choose one checkpoint per logical strategy using the paper's fixed relative-change
   comparison, `1.25 * relative_change(Q') + relative_change(Q'') > 0`, with metric
   directions respected.
5. After 600 complete finite objective vectors exist for every model, reconstruct the
   strategy-specific two-dimensional Pareto fronts. Ignore the high-dimensional global
   Optuna front.
6. Freeze all fronts before looking at downstream anomaly performance.
7. Retrain every unique Pareto-front configuration exactly once for 200 epochs using
   seed 123. A trial shared by several fronts is retrained once and referenced by each
   front.
8. Reapply the per-strategy checkpoint rule to the retrained run.
9. Within each logical front, choose the reported retrained model using mean downstream
   anomaly efficiency over the 20 held-out anomaly datasets.

The shared-pool design gives each model one joint multi-objective M-TPE sampling history
rather than independent two-objective M-TPE histories. This is an intentional,
user-approved difference from a literal independent-search replication and must be
reported as a methodological limitation.

### 1.2 Data invariants and leakage boundary

- Training is unchanged: use the ordinary combined ZeroBias training split.
- The standard combined `normal` validation and test loaders remain unchanged.
- Model-selection callbacks additionally receive ordered source-specific validation or
  test streams named `ZB_run396102` and `ZB_run398183`.
- Search validation contains only `normal` and those two background streams.
  `validation_aux_datasets: []` prevents all downstream anomalies from entering search.
- The 20 anomaly datasets are restored only after all Pareto fronts have been frozen.
- The source-specific search streams use held-out validation events, not training rows.
- Validation CAP tables are used during search/retrained validation. Test CAP tables
  are used only in held-out test evaluation.
- The paper operating point is unchanged: target rate 0.25 kHz from a base rate of
  28,608.8064 kHz. Preprocessing, normalization, feature definitions, search spaces,
  and downstream datasets are otherwise unchanged.
- Do not mix source IDs, table splits, or different prefix lengths. Mapping
  authentication must fail closed.

The two source samples are from different data-taking eras. That is intentional: this
campaign studies robustness across recorded-background era shift. It is not a
data-versus-simulation agreement study and must not be described as one.

### 1.3 Pairing inputs

The authoritative pairing root is:

```text
/iopsstor/scratch/cscs/vjimenez/adatl1/data/data_2025E+G/pairing/ZB_run396102_to_ZB_run398183
```

Each source contributes exactly 163,840 validation events and 163,840 test events,
corresponding to 10 batches of 16,384. Four authenticated, deterministic one-to-one
mapping strategies exist for each split:

- `flat_physical`
- `physics_summary`
- `typed_sliced_wasserstein`
- `jetclr`

The eight production files are:

```text
{validate,test}_{flat_physical,physics_summary,typed_sliced_wasserstein,jetclr}_cap_n163840.pt
```

`research/physics_background_pairing/preflight.json` records every table's path,
SHA-256, source names, source tensor hashes, split, and pair count. The empirical-CDF
CAP control is table-free: it pairs by anomaly-score rank at evaluation time.

Pairing changes CAP only. W1 depends only on the two score marginals and is invariant to
a bijective reordering of the same events. Drift is directional and uses background0
calibration to background1 evaluation.

### 1.4 Models and anomaly scores

| Model | Paper-native score used here | Additional score |
|---|---|---|
| AE | masked reconstruction MSE | masked residual OAS/Mahalanobis |
| VAE | raw latent KL (`kl_raw`) | masked residual OAS/Mahalanobis |
| DSAE | reconstruction error | none |
| DSVAE | latent KL/variational score | none |
| DeepSVDD (`svdd`) | distance from learned center | none |
| RealNVP | likelihood/density score | none |

AE training continues to minimize its paper Huber reconstruction loss; MSE is the
native anomaly score, not the training loss. VAE's native control remains latent KL.

OAS is fit only from a deterministic bounded clean training-normal residual sample
(`max_samples: 163840`, `sample_seed: 123`). Physics padding is treated as missing:

- feature means use observed entries only;
- missing residuals are mean-imputed only for covariance fitting;
- structurally never-observed coordinates are excluded from the fitted active subspace
  and receive zero precision rows/columns;
- event Mahalanobis energy includes and normalizes over observed features only;
- OAS state is checkpointed so every retained checkpoint is evaluated with its own
  consistent residual state.

Native and OAS downstream operating thresholds must be estimated independently from
validation data. Never reuse a native-score threshold for OAS.

## 2. Exact experiment matrix and objective semantics

### 2.1 Six shared studies

The persistent Optuna studies are exactly:

| Study/model ID | Scores evaluated | Logical fronts | Scalar objectives/trial |
|---|---:|---:|---:|
| `ae` | native, residual OAS | 14 | 28 |
| `vae` | native, residual OAS | 14 | 28 |
| `dsae` | native | 7 | 14 |
| `dsvae` | native | 7 | 14 |
| `svdd` | native | 7 | 14 |
| `realnvp` | native | 7 | 14 |

Total: six databases, 3,600 usable trained search trials, and 56 logical Pareto fronts.
The DTE model is intentionally excluded because it was not one of the six physics
models in the paper experiment being replicated.

### 2.2 Ordered logical objectives

For each available score, the objective order is fixed:

| Logical objective ID | Q' | Q' direction | Q'' | Q'' direction |
|---|---|---|---|---|
| `<score>__cap__flat_physical` | CAP | maximize | native operational score | minimize |
| `<score>__cap__physics_summary` | CAP | maximize | native operational score | minimize |
| `<score>__cap__typed_sliced_wasserstein` | CAP | maximize | native operational score | minimize |
| `<score>__cap__jetclr` | CAP | maximize | native operational score | minimize |
| `<score>__cap__cdf` | CAP | maximize | native operational score | minimize |
| `<score>__wasserstein` | W1(E,G) | minimize | native operational score | minimize |
| `<score>__drift` | threshold drift E->G | minimize | native operational score | minimize |

AE/VAE flatten all seven native pairs first, followed by all seven residual-OAS pairs.
The other models flatten the seven native pairs. Objective insertion order in the YAML,
the Hydra sweeper direction list, `optimized_metric.json`, `design.json`, and Optuna
must match exactly. A mismatch is a campaign-stopping error.

For OAS fronts, OAS supplies Q' but Q'' remains the stable native anomaly-score
criterion, exactly as requested. Each logical objective is allowed to compare only its
own primary EMA checkpoint, `last`, and `single/ascore_operational/stable`.

### 2.3 Downstream result

The retraining suites evaluate these 20 held-out anomaly datasets:

```text
GluGluHTo2B_Par-MH-125
GluGluHto2G_Par-MH-125
GluGluHto2G_Par-MH-90
GluGluHto2Tau_Par-MH-125
GluGlutoHHto2B2WtoLNu2Q_Par-c2-0-kl-1-kt-1
HHHto4B2Tau_Par-c3-0-d4-0
HHHto6B_Par-c3-0-d4-0
SUSYGluGluToBBHTo2B_Par-M-1200
SUSYGluGluToBBHToBB_Par-M-120
SUSYGluGluToBBHToBB_Par-M-350
SUSYGluGluToBBHToBB_Par-M-600
TTHTo2C_Par-MH-125
TTHto2B_Par-MH-125
VBFHTo2C_Par-MH-125
VBFHto2B_Par-MH-125
VBFHto2Tau_Par-MH-125
WtoTauto3Mu
ggH-suep-decay
smj-case-A
haa-4b-ma15
```

For a front member, the downstream oracle statistic is:

```text
mean_downstream_efficiency = sum(efficiency_operational over 20 anomalies) / 20
```

The winner is the front member with the largest mean. This step uses downstream labels
and must always be called an oracle/reporting selection, never a signal-agnostic model
selection step. No downstream result may be read before the front manifest is frozen.

## 3. Implementation map: files and responsibilities

### 3.1 Configurations used by the six studies

The only experiment wrappers for this campaign are:

```text
configs/experiment/physics/ae_background_all.yaml
configs/experiment/physics/vae_background_all.yaml
configs/experiment/physics/dsae_background_all.yaml
configs/experiment/physics/dsvae_background_all.yaml
configs/experiment/physics/svdd_background_all.yaml
configs/experiment/physics/realnvp_background_all.yaml
```

They inherit the paper-aligned `physics/*_agnostic.yaml` model/trainer/callback setup,
switch data to `basis`, expose the two source streams, remove auxiliary anomaly
validation datasets, and attach the shared selection suite. AE installs
`ResidualOASStateCallback` and uses native `mse`; VAE installs
`VAEResidualScoreStateCallback` and uses native `kl_raw`.

The inherited model, trainer, data, and HPO settings were audited against
`dev/patrick` at reference commit `9c2c4ec`. Treat that branch as provenance/reference,
not as the execution branch. The later posterior-consistency callback is intentionally
outside this campaign because the requested primary metrics are CAP, W1, and drift.

Pairing and selection configuration:

```text
configs/pairing/physics.yaml
configs/selection_suite/physics_background_native.yaml
configs/selection_suite/physics_background_native_oas.yaml
```

- `pairing/physics.yaml` defines source names, 163,840 events, scratch artifact root,
  and schema-2 optimized-metric output.
- `physics_background_native.yaml` defines the seven native training callbacks, EMA
  checkpoints, evaluation callbacks, primary/secondary directions, and the three
  checkpoint families.
- `physics_background_native_oas.yaml` composes the native suite and appends the seven
  residual-OAS callbacks/checkpoints/objectives for AE/VAE.
- Every training callback logs its epoch metric to MLflow. The evaluator logs selected
  objective scalars and optimized-checkpoint tags at the end of each run.

Optuna wrappers:

```text
configs/hparams_search/ae_shared_optuna.yaml
configs/hparams_search/vae_shared_optuna.yaml
configs/hparams_search/dsae_shared_optuna.yaml
configs/hparams_search/dsvae_shared_optuna.yaml
configs/hparams_search/svdd_shared_optuna.yaml
configs/hparams_search/realnvp_shared_optuna.yaml
```

Each inherits the corresponding paper HPO file (`*_optuna.yaml`), including its search
space and M-TPE settings, then sets 600 trials, the shared study name, and the exact 28
or 14 direction vector.

Retraining/downstream overlays:

```text
configs/retrain_suite/physics_background_native.yaml
configs/retrain_suite/physics_background_native_oas.yaml
```

The native overlay restores all 20 anomalies and native efficiency evaluation. The OAS
overlay extends it with `eff_residual_oas` and a separate `residual_oas` threshold
namespace.

Do not use these older files for the primary campaign:

```text
configs/experiment/physics/*_background_pairing.yaml
configs/selection_metric/*
configs/pairing/physics_cdf.yaml
```

They support earlier one-cell/56-independent-study experiments and are retained only
for compatibility and isolated diagnostics. The shared selection suite already
contains all four mappings and CDF simultaneously.

### 3.2 Runtime code

- `src/train.py`: composes training and evaluation, returns the flattened objective
  tuple to Hydra/Optuna, and writes schema-2 `optimized_metric.json`.
- `src/evaluation/evaluator.py`: evaluates retained checkpoints, activates only the
  callbacks required for each candidate, applies the fixed relative-change rule per
  logical objective, and records the selected checkpoint/objective pair.
- `src/data/L1AD_datamodule.py`: exposes ordered source-specific validation/test loaders
  and filters auxiliary datasets.
- `src/callbacks/cap.py`: training-time CAP plus masked active-subspace residual OAS
  state fitting/scoring.
- `src/callbacks/efficiency.py`: validation-derived operational thresholds with optional
  score namespaces.
- `src/evaluation/callbacks/{cap,wasserstein,thres_drift,efficiency}.py`: checkpoint
  evaluation and artifact generation.

### 3.3 Orchestration

- `scripts/physics_background_pairing_campaign.py` is the campaign state machine. Its
  commands are `init`, `status`, `pilot`, `sweep`, `submit-next`, `freeze`, `retrain`,
  `resolve-retrain-index`, and `aggregate`.
- `scripts/physics_background_pairing_operations.py` is the mutable control plane for
  an immutable campaign. It submits an exact frozen-worktree search chunk, refuses
  scheduler or Optuna overlap, and holds an interprocess campaign lock across live
  scheduler checks, `sbatch`, and ledger registration. Reconciliation seals terminal
  logs and validates every successful packed chunk's exact bindings, objective
  contract, run provenance, complete MLflow histories, and required retained
  checkpoint/threshold state before sealing `chunk_validation.json`. Successful
  retrain array tasks likewise seal an immutable, task-bound segment/final receipt. It creates
  consistent quiescent SQLite backups, applies a fail-closed terminal canary gate, and
  provides explicit backed-up, scheduler-proven `repair-stale` recovery and a
  quiescent, independently validated `aggregate` transition.
  Search submission enforces the 720-minute partition limit and refuses to exceed 600
  potentially usable trials, including the final partial chunk. It never imports or
  changes the scientific runner used by an active campaign.
- `scripts/clariden/physics_background_pairing_pilot.sbatch` runs one unified one-epoch
  model pilot on the debug partition.
- `scripts/clariden/physics_background_pairing_sweep.sbatch` is a 12-hour packed
  `normal` allocation: one Hydra/Optuna controller launches four independent one-GPU
  `srun` steps per GH200 node.
- `scripts/clariden/physics_background_pairing_retrain.sbatch` retrains one manifest
  index on one GPU through an authenticated bounded local-task map.
- `scripts/clariden/physics_background_pairing_restart_canary.sbatch` and
  `scripts/physics_background_pairing_restart_canary.py` force a real Slurm USR1
  requeue on synthetic data and authenticate callback/HPC/MLflow/log restoration before
  any full campaign is allowed.
- `research/physics_background_pairing/preflight.json` authenticates mapping inputs.
- `research/physics_background_pairing/RUN_LEDGER.md` is the concise human status
  ledger. Update it after every phase transition or failure classification.
- `scripts/physics_background_pairing_report.py` provides deterministic `snapshot` and
  fail-closed `finalize` exports. Search chunks write `trial_bindings.json`, and every
  optimized-metric artifact records its Hydra, checkpoint, and MLflow identities.

The campaign manager already implements:

- commit/preflight pinning;
- six persistent SQLite studies;
- usable-trial accounting based on complete finite full-length vectors;
- overlap refusal for a single model database;
- mixed-direction two-dimensional front construction;
- first-600-usable freezing;
- unique retraining deduplication;
- hash-checked retraining markers;
- score-aware downstream aggregation.

## 4. Scratch layout and provenance isolation

### 4.1 Never reuse the superseded campaign

The default names embedded in the Slurm scripts were placeholders. At plan-writing
time, the new default six-study worktree/campaign did not exist, while older campaign
directories contained the superseded schema. Do not delete, mutate, or reuse those
directories.

Create a fresh campaign ID from the final execution commit, for example:

```bash
export BGPAIR_COMMIT="$(git rev-parse HEAD)"
export BGPAIR_ID="physics_background_shared_${BGPAIR_COMMIT:0:12}"
export BGPAIR_SCRATCH="/iopsstor/scratch/cscs/vjimenez/adatl1"
export BGPAIR_REPO="${BGPAIR_SCRATCH}/worktrees/${BGPAIR_ID}"
export BGPAIR_ROOT="${BGPAIR_SCRATCH}/campaigns/${BGPAIR_ID}"
export BGPAIR_UV="${BGPAIR_SCRATCH}/tools/uv-0.11.32/uv"
```

Before creating either path, resolve it and verify it does not already contain a
campaign. Never recursively remove a broad scratch/worktree/campaign path. If the path
exists with a different commit or design, choose a new campaign ID.

Create a detached worktree pinned to the final commit:

```bash
git worktree add --detach "${BGPAIR_REPO}" "${BGPAIR_COMMIT}"
git -C "${BGPAIR_REPO}" status --short
git -C "${BGPAIR_REPO}" rev-parse HEAD
```

The status must be clean and the SHA must equal `BGPAIR_COMMIT`. Run production only
from this frozen worktree. Do code changes and analysis development in the ordinary
`research/main` checkout.

### 4.2 Worktree environment

Create a worktree-local ignored `.env` with these semantics:

```text
PROJECT_ROOT=<BGPAIR_REPO>
DATA_DIR=/iopsstor/scratch/cscs/podagiu/data
RAW_DATA_DIR=/iopsstor/scratch/cscs/vjimenez/adatl1/raw
PHYSICS_PAIRING_DIR=/iopsstor/scratch/cscs/vjimenez/adatl1/data/data_2025E+G/pairing/ZB_run396102_to_ZB_run398183
LOG_DIR=<BGPAIR_ROOT>/runtime/logs
OUTPUT_DIR=<BGPAIR_ROOT>/runtime/outputs
CHECKPOINT_DIR=<BGPAIR_ROOT>/runtime/checkpoints
HYDRA_FULL_ERROR=1
WANDB_MODE=offline
```

Do not blindly symlink the mutable checkout's `.env`: its `PROJECT_ROOT`, `LOG_DIR`,
and `CHECKPOINT_DIR` may mix runs from different campaigns. Keeping runtime outputs
inside `BGPAIR_ROOT` prevents retrain names such as `ae__trial0001` from colliding with
older campaigns and makes the final campaign auditable.

The Slurm scripts use the shared Clariden environment:

```text
UV_PROJECT_ENVIRONMENT=/iopsstor/scratch/cscs/vjimenez/adatl1/.venv-clariden
UV_CACHE_DIR=/iopsstor/scratch/cscs/vjimenez/adatl1/uv-cache-clariden-arm64
UV_PYTHON_INSTALL_DIR=/iopsstor/scratch/cscs/vjimenez/adatl1/python
```

Pass the worktree and campaign root explicitly on every `sbatch` invocation:

```text
PHYSICS_BGPAIR_REPOSITORY_ROOT=<BGPAIR_REPO>
PHYSICS_BGPAIR_CAMPAIGN_ROOT=<BGPAIR_ROOT>
```

This avoids relying on the scripts' historical default directory names.

### 4.3 Authoritative campaign tree

All durable campaign artifacts should end under `BGPAIR_ROOT`, except the immutable
input pair tables and the global Slurm output stream:

```text
<BGPAIR_ROOT>/
  design.json
  pilots/<model>/
    optimized_metric.json
    success.json
  studies/<model>.db
  sweeps/<model>/<UTC timestamp>/
  selection/pareto_retrain_manifest.json
  retraining/hydra/<retrain-index>/optimized_metric.json
  retraining/markers/<retrain-index>.json
  runtime/
    checkpoints/
    logs/mlflow/mlruns/
    outputs/
  state/
    experiment_matrix.csv
    job_history.csv
    failures.jsonl
    latest_status.json
  results/
    REPORT.md
    README.md
    provenance.json
    study_status.csv
    trials.csv
    front_memberships.csv
    retrains.csv
    checkpoint_selections.csv
    checkpoint_candidates.csv
    checkpoint_candidate_availability.csv
    primary_metrics.csv
    downstream_efficiencies.csv
    winners.csv
    mlflow_runs.csv
    figures/
    logs/
    manifest.sha256
```

The `state/` and complete portable `results/` exports are generated by
`scripts/physics_background_pairing_report.py`. Validate it against real pilot/canary
artifacts before production, then use it throughout the campaign. Never place Optuna
databases, checkpoints, MLflow stores, or raw data in git.

## 5. Pre-execution gates

No full search may start until all gates below are green and recorded in
`RUN_LEDGER.md`.

### Gate A: final code and environment

1. Ensure `research/main` contains every needed code/config/reporting change.
2. Run `git status --short`; unrelated or uncommitted changes must not enter the frozen
   worktree.
3. Run the targeted tests:

```bash
uv run pytest -q \
  tests/test_shared_model_selection.py \
  tests/test_optimized_metric_artifact.py \
  tests/test_physics_background_pairing_campaign.py \
  tests/test_physics_background_pairing_configs.py \
  tests/test_l1_datamodule_eval_cap.py \
  tests/test_cap_metric.py \
  tests/test_residual_oas_cap.py
```

4. Run `uv run pre-commit run -a` and `git diff --check`. The current legacy baseline
   has unrelated whitespace/shebang debt, so `pre-commit -a` rewrites or rejects files
   outside this campaign. Restore only those hook-induced unrelated changes, record the
   baseline failure in the ledger, and require every hook plus a direct Black check to
   pass on all campaign-changed files. Do not absorb the 82-file legacy cleanup into
   the frozen scientific commit.
5. Compose all six experiment/HPO configurations. The configuration tests are the
   preferred reproducible check; do not hand-compare only the YAML direction counts.
6. Check scratch capacity/quota for Optuna databases, Hydra outputs, MLflow, retained
   checkpoints, and the eventual retraining set. Record free space in the ledger.
7. Commit the exact green state, then create the detached worktree and isolated `.env`.

### Gate B: input authentication

From the frozen worktree, verify every tracked table checksum:

```bash
jq -r '.tables[] | "\(.sha256)  \(.path)"' \
  research/physics_background_pairing/preflight.json | sha256sum -c -
```

Confirm that all eight rows report `OK`, `events_per_source` is 163,840, the source
names are correct, and every `authenticated` value is true. If an input checksum or
ordered tensor hash fails, stop. Do not bypass table authentication or regenerate a
mapping silently.

### Gate C: initialize immutable design

From `BGPAIR_REPO`:

```bash
"${BGPAIR_UV}" run --frozen --no-sync python \
  scripts/physics_background_pairing_campaign.py \
  --root "${BGPAIR_ROOT}" init
```

Inspect `BGPAIR_ROOT/design.json`. Required properties:

- `schema_version == 2`;
- `code_commit == BGPAIR_COMMIT`;
- `preflight_sha256` equals the tracked file;
- six studies with unique model IDs and database names;
- target 600, search epochs 50, retrain epochs 200;
- AE/VAE have 14 logical objectives each; the other models have seven each;
- objective names, directions, and indices match the HPO YAML.

Once `design.json` exists, do not replace or edit it. A code/preflight change requires a
new worktree and new campaign root.

Generate `state/experiment_matrix.csv` immediately from `design.json`: one row for each
of the 56 expected logical fronts, with model, score, metric, strategy, directions, and
phase/status columns. This is the master completeness checklist throughout the run.

### Gate C.5: forced Slurm restart canary

Before any production search or retraining allocation, run the isolated procedure in
`research/physics_background_pairing/RESTART_CANARY.md` from the exact immutable
execution worktree. It deliberately holds the first debug allocation until Slurm sends
ordinary `USR1`, requires Lightning to write an HPC checkpoint and requeue the same
job, then authenticates restoration in the production efficiency/CAP/W1/drift/stable
callbacks. It also requires one explicitly tagged MLflow run with exact `epoch_idx`
steps and values `0..5`, retained stable files, deterministic terminal model weights,
append-preserved scheduler sentinels from both allocations, and a hash-bearing
`COMPLETE.json` under a fresh non-campaign canary root. A completed job without a
nonzero restart count does not pass. Preserve the complete canary root and scheduler
log as preflight evidence; do not submit full-scale work after any failure.

### Gate D: unified GPU pilots

Run one unified pilot for every model. At minimum, AE and VAE are mandatory because
their OAS state paths differ, but all six should pass before spending the full budget.
Each pilot uses one epoch, two training batches, the full 163,840-event validation
streams, every model-relevant CAP/W1/drift callback, and only one inner CAP optimizer
epoch.

Submit at most two debug jobs at a time because the debug QoS permits only two
submitted jobs for this user:

```bash
sbatch --parsable \
  --export=ALL,PHYSICS_BGPAIR_REPOSITORY_ROOT="${BGPAIR_REPO}",PHYSICS_BGPAIR_CAMPAIGN_ROOT="${BGPAIR_ROOT}",PILOT_MODEL=ae \
  "${BGPAIR_REPO}/scripts/clariden/physics_background_pairing_pilot.sbatch"
```

Repeat for `vae`, `dsae`, `dsvae`, `svdd`, and `realnvp`, respecting the two-job cap.
Record job ID, queue delay, runtime, peak memory if available, exit state, and artifact
paths.

A pilot succeeds only if all of these hold:

- Slurm state is `COMPLETED` with exit code 0;
- `<BGPAIR_ROOT>/pilots/<model>/success.json` exists and names the frozen commit;
- `optimized_metric.json` has schema 2 and exactly the expected objective order;
- every logical selection has a non-null checkpoint and finite objective pair;
- every selected checkpoint contains a finite `thres_operational` buffer and AE/VAE
  checkpoints also contain an independent finite
  `thres_operational__residual_oas` buffer;
- all four mapping tables and CDF were actually exercised;
- AE/VAE native and residual-OAS views both appear;
- MLflow contains per-epoch CAP/W1/drift and score-specific metric names;
- no anomaly validation loader was used.

The search suites instantiate threshold-calibration callbacks even though downstream
efficiency is not a search objective. They see only the normal validation stream because
`data.validation_aux_datasets: []`; this makes checkpoint state complete without exposing
any downstream anomaly sample during model selection.

Earlier isolated 8-10 minute pilots do not satisfy this gate. The unified pilot's
runtime is not yet known and may approach the current 90-minute debug limit.

### Gate E: one-trial production canaries

The unified pilot does not exercise Optuna, SQLite, Hydra multirun, or the packed
launcher. First run `physics_background_pairing_packed_pilot.sbatch`; it asks four
temporary Optuna trials and proves ordered Hydra job/result binding across all four
GPUs on one debug node. Then launch four real 50-epoch trials per model through the
production `sweep` path before large chunks. These are valid production trials and
count toward 600 if usable.

Use `--nodes=1`, `--trials 4 --jobs 4 --timeout-min 720`. Confirm the database direction
vector, four full finite objective vectors, ordered Optuna-to-Hydra bindings, distinct
MLflow links/checkpoints, and wall time.
Measure AE and VAE especially carefully because they evaluate 28 objectives. If a
50-epoch canary cannot comfortably finish within 12 hours, do not launch bulk normal
jobs. Account `a0166` cannot use the 24-hour `low` QOS, so report the scheduler limit
before considering any scientific change.

After all six jobs terminate, authenticate the complete gate from the mutable control
checkout:

```bash
uv run python scripts/physics_background_pairing_operations.py \
  --root "${BGPAIR_ROOT}" \
  --execution-repository "${BGPAIR_REPO}" validate-canaries
```

This must produce `state/production_canary_validation.json` and otherwise fail closed.
It requires six successful hash-sealed Slurm jobs, 24 usable 50-epoch Optuna trials,
exact Optuna/Hydra/MLflow bindings, all finite contiguous metric histories, 224 logical
checkpoint selections, and all 272 configured candidate slots. The 248 required
last/primary candidates must exist with finite native thresholds plus independent OAS
thresholds for AE/VAE; each of the 24 stable slots records whether its strict criterion
produced a checkpoint. It also reauthenticates the resolved E/G-only search
configuration. Do not submit a bulk chunk without this file.

## 6. Search execution: six studies, 600 usable trials each

### 6.1 Submission model

`physics_background_pairing_sweep.sbatch` requests twelve ARM `normal` nodes by default.
Normal allocations are node-exclusive and each node has four GH200 GPUs, so the custom
Hydra launcher dispatches 48 ordered jobs as 48 concurrent resource-exact `srun` steps:
four per node, each with one GPU, 16 CPUs, and 96 GB RAM. One controller retains the
original Optuna ask/tell sequence, Hydra job numbering, SQLite writer, and result order;
there are no nested Submitit allocations and no idle controller node. The `xfer` nodes
are x86-64 and cannot execute the pinned ARM environment, while account `a0166` has only
the `normal` QOS.

Use the operational helper from a clean mutable/reporting checkout for normal chunk
submission. It requires all three external inputs explicitly and records the returned
Slurm allocation in the campaign ledger atomically:

```bash
uv run python scripts/physics_background_pairing_operations.py \
  --root "${BGPAIR_ROOT}" \
  --execution-repository "${BGPAIR_REPO}" \
  submit-search --model ae --trials 48 --jobs 48 --timeout-min 720 \
  --data-dir "${DATA_DIR}" --raw-data-dir "${RAW_DATA_DIR}" \
  --pairing-dir "${PHYSICS_PAIRING_DIR}"
```

The helper calculates `--nodes=ceil(jobs/4)`, authenticates the frozen commit and clean
worktree, exports campaign-local runtime paths, queries user-scoped live Slurm jobs,
and rejects an active or unregistered packed allocation, active ledger row, or
`RUNNING`/`WAITING` Optuna trial for the same model. The check, submission, and ledger
write share one interprocess lock. It additionally requires the authenticated six-model
terminal canary artifact, rejects timeouts above 720 minutes, and refuses
`usable_complete + trials` above 600; therefore the last request must use the exact
remaining count. The direct `sbatch` equivalent is a recovery/debug fallback only:

```bash
sbatch --parsable \
  --nodes=6 \
  --export=ALL,PHYSICS_BGPAIR_REPOSITORY_ROOT="${BGPAIR_REPO}",PHYSICS_BGPAIR_CAMPAIGN_ROOT="${BGPAIR_ROOT}",STUDY_MODEL=ae,CHUNK_TRIALS=48,CHUNK_JOBS=24,TIMEOUT_MIN=720 \
  "${BGPAIR_REPO}/scripts/clariden/physics_background_pairing_sweep.sbatch"
```

Different models use different SQLite databases and may run concurrently if account
quota allows. Never overlap two chunks for the same model. The campaign refuses a new
chunk when that study has `RUNNING` or `WAITING` trials, but scheduler state must still
be checked before submission.

Set `--nodes=ceil(CHUNK_JOBS/4)` whenever overriding the 48-job default. Start with a
one-node, four-trial production canary after the packed debug pilot.
Increase only if scheduler quota, SQLite locking, filesystem load, and GPU throughput
are healthy. `CHUNK_TRIALS=48` is a convenient accounting unit, not a scientific
requirement. From an empty study, twelve full chunks plus a final 24 produce 600; after
the four production-canary trials, twelve full chunks plus a final 20 produce 600.
Failed/unusable trials require replacements based on the exact remaining usable count.

### 6.2 Completion semantics

After every chunk, run:

```bash
uv run python scripts/physics_background_pairing_operations.py \
  --root "${BGPAIR_ROOT}" \
  --execution-repository "${BGPAIR_REPO}" reconcile

"${BGPAIR_UV}" run --frozen --no-sync python \
  scripts/physics_background_pairing_campaign.py \
  --root "${BGPAIR_ROOT}" status
```

Persist the output atomically to `state/latest_status.json`, append a timestamped row to
`state/job_history.csv`, and update `RUN_LEDGER.md`.

For a successful packed job, `reconcile` writes `chunk_validation.json` beside
`chunk.json` and records its SHA-256 as the ledger's `success_sha256`; the separate
`bindings_sha256` seals `trial_bindings.json`. A chunk is not sealed merely because
Optuna contains finite vectors. Every chunk-local binding must be `COMPLETE`, match the
persistent Optuna trial and schema-2 artifact, have exact objective order/provenance and
one finished 50-epoch MLflow run with complete finite histories, and retain all required
last/primary checkpoints with the correct finite native/OAS threshold namespace.

Only `usable_complete` counts toward 600. A usable trial is:

- Optuna state `COMPLETE`;
- exactly 28 values for AE/VAE or 14 values for the other models;
- every value finite.

Failed, pruned, timed-out, incomplete, or nonfinite trials do not count. The campaign
requests replacements from `600 - usable_complete`. `freeze` uses the first 600 usable
trials ordered by Optuna trial number, so later extra usable trials are not silently
mixed into the analysis.

Back up each SQLite database after a completed chunk when no writer is active. Store
the backup under `<BGPAIR_ROOT>/studies/backups/<model>/<timestamp>.db` and record its
SHA-256. Use the consistent SQLite backup API through:

```bash
uv run python scripts/physics_background_pairing_operations.py \
  --root "${BGPAIR_ROOT}" \
  --execution-repository "${BGPAIR_REPO}" \
  backup-study --model ae
```

Do not copy a live SQLite database directly.

### 6.3 Per-trial outputs that must exist

Every search trial must provide:

- the full Optuna parameter dictionary and ordered objective vector in the model DB;
- a schema-2 `optimized_metric.json` containing:
  - `objective_order`;
  - the flattened vector;
  - `selections[logical_objective].optimized_ckpt_name`;
  - `selections[logical_objective].optimized_metric`;
- the configured last, primary-EMA, and stable-native checkpoint families required by
  each logical objective; last and primary must contain a checkpoint, while an empty
  stable family is valid only when the reference 10-epoch/5% criterion never fired;
- MLflow epoch series for every relevant native/OAS CAP, rank correlation, W1, drift,
  and native secondary score;
- Hydra resolved config and run logs;
- a clear relationship among model, Optuna trial number, Hydra run directory, run name,
  MLflow run ID, and checkpoint root.

The report/export utility must eventually index these relationships in
`mlflow_runs.csv` and `checkpoint_selections.csv`. Do not assume the global MLflow store
alone will be easy to reconstruct after thousands of runs.

## 7. Autonomous monitoring, sub-agents, and token-efficient waiting

### 7.1 Main-agent responsibility

The main agent owns the scientific contract, mutations, job submissions/cancellations,
campaign phase transitions, freeze, downstream unblinding, aggregation, interpretation,
and final report. It must keep `design.json`, `experiment_matrix.csv`, job history, and
`RUN_LEDGER.md` synchronized.

Use sub-agents for bounded read-only investigation tasks that would otherwise bloat the
main context, for example:

- reading/tail-summarizing many Slurm logs;
- matching failed array indices to error signatures;
- checking `squeue`/`sacct` state and extracting elapsed/max-RSS information;
- auditing one model's Optuna counts/objective lengths;
- verifying artifact/hash completeness;
- extracting one section of MLflow metrics;
- independently checking tables, figures, or quantitative claims for `REPORT.md`.

Give each sub-agent an exact scope, paths/job IDs, and required concise output schema.
Sub-agents should not submit/cancel jobs or edit shared campaign files unless explicitly
delegated. Avoid concurrent edits to the same file. The main agent validates the
sub-agent's evidence before acting and records only the necessary summary and paths.

### 7.2 Predictive waiting instead of polling

Before waiting on the scheduler, estimate both queue and runtime blocking time:

1. Query `squeue` for current state/reason and `squeue --start -j <job>` when available.
2. Query `sacct` for completed jobs of the same phase/model and compute a conservative
   recent median or upper-quartile runtime.
3. Estimate:
   - predicted start = scheduler estimate, or a conservative queue estimate when none
     exists;
   - predicted completion = predicted start + measured phase/model runtime + buffer.
4. Record the prediction and evidence in `job_history.csv`/`RUN_LEDGER.md`.
5. Keep doing independent useful work while jobs are queued/running: reporting code,
   artifact indexing, another completed-model audit, or failure classification.
6. When queued scheduler progress is genuinely the only blocker, tell the user with the
   job IDs, queue reason, and prediction, then stop polling. Do not create a timer and do
   not run a blocking shell `sleep`.
7. Resume with one scheduler/accounting check after the user returns or supplies new
   evidence that the jobs advanced.

Suggested check timing:

- `PENDING (Priority/Resources)`: next check near `squeue --start` plus 5-10 minutes;
  if no estimate exists, use a conservative interval informed by recent queue history,
  normally 1-2 hours rather than a few minutes.
- `RUNNING`: next check near the measured expected completion, or at roughly 75-85% of
  the job time limit for an uncalibrated phase.
- `COMPLETING`: check in 5-15 minutes.
- failure, nonzero exit, OOM, node failure, or missing output: investigate immediately.

If the platform cannot create a timer/recurring monitor, tell the user that queue/run
completion is the only blocker, provide the predicted next useful check time, and yield
control. Never pretend a timer was scheduled.

### 7.3 Scheduler commands and evidence

Use concise read-only checks such as:

```bash
squeue -u "$USER" -o '%.18i %.9P %.30j %.8T %.10M %.10l %R'
squeue --start -j <job-id>
sacct -j <job-id> --format=JobID,JobName,Partition,State,Elapsed,Timelimit,MaxRSS,ExitCode
```

Do not cancel or modify unrelated jobs. A scheduler queue is a blocker only when all
other useful in-scope work is complete.

## 8. Failure taxonomy and recovery rules

Classify every failure before retrying and append a machine-readable event to
`state/failures.jsonl` with timestamp, phase, model/index, job ID, error signature,
classification, action, and outcome.

### 8.1 Infrastructure-only failures

Examples: node failure, preemption, transient filesystem error, scheduler cancellation,
or controller death while valid worker state remains.

- Preserve logs and artifacts.
- Verify actual worker and Optuna state.
- Retry only the missing/failed work with unchanged scientific config.
- Infrastructure failures do not count toward 600 unless the DB contains a full finite
  `COMPLETE` trial.

### 8.2 Time limits

Normal-partition packed search allocations have an absolute 720-minute limit. The 200-epoch
retraining script also requests 12 hours, and downstream evaluation of many checkpoint
candidates may be the larger timeout risk.

- Use production canary runtimes to predict whether the limit is safe.
- Retraining is deliberately divided at cumulative epoch boundaries 40, 80, 120, 160,
  and 200. Each normal boundary persists an authenticated continuation checkpoint and
  resolved config, exits successfully, and is reconciled against that exact ledger
  task. A later `submit-retrains` invocation maps the still-missing frozen index into a
  fresh array. This avoids Clariden's `MaxBatchRequeue=5` limit. Epoch-200 training also
  exits after committing its boundary; one additional fresh task runs evaluation only
  (`train=false`) from the immutable checkpoint families, so evaluator signals/HPC
  files can never become fit-resume state. This changes no epoch, batch, callback,
  hyperparameter, seed, or objective semantics.
- Lightning receives an emergency `SIGUSR1` 15 minutes before the allocation limit. The
  Slurm job uses an ordinary job-step signal and `srun`; never add the `B:` prefix,
  because Slurm documents that `B:` signals only the batch shell and excludes Python.
  Emergency HPC checkpoints live in the retrain-index-specific
  `trainer.default_root_dir`, are authenticated on restart, and are archived after the
  interrupted segment reaches its deterministic boundary.
- `--requeue` is reserved for Lightning's emergency USR1 path; normal boundaries never
  call `scontrol requeue`. `REQUEUED` remains an active ledger state. Use Slurm append
  mode for `%A_%a` output so an emergency requeue cannot truncate prior-attempt evidence.
- Do not pass a timeout greater than the partition maximum.
- Record the partition/time-limit change as operational, not scientific.
- The controller and all packed steps share one 12-hour allocation, so queue time is
  paid before the controller starts and cannot consume training wall time.

### 8.3 SQLite contention or stale Optuna trials

Forty-eight workers may contend on one SQLite database despite `timeout=600`.

- Never run two controllers for the same model.
- Reduce parallel jobs when lock errors occur.
- Back up the quiescent database before repair.
- A hard kill may leave Optuna trials `RUNNING`. First prove with `squeue` and `sacct`
  that no corresponding worker exists. Then invoke the explicit mutable-control command
  with every exact stale trial number, for example:

```bash
uv run python scripts/physics_background_pairing_operations.py \
  --root "${BGPAIR_ROOT}" --execution-repository "${BGPAIR_REPO}" \
  repair-stale --model ae --trial 48 --trial 49
```

  The command reconciles the ledger under the campaign lock, refuses a live same-model
  or unregistered packed job, requires every requested trial to be `RUNNING`/`WAITING`,
  creates and hashes a consistent database backup, then marks only those trial numbers
  `FAIL` through Optuna's API. It updates any explicit binding state and writes both a
  `state/stale_repairs/*.json` audit artifact and a `failures.jsonl` success/refusal
  event. Never edit SQLite manually or repair an inferred range of trial numbers.

### 8.4 Scientific/code/data failures

Examples: wrong objective order, missing score, table authentication failure,
non-deterministic loader prefix, padding bug, anomaly leakage, systematic nonfinite
objective, or config mismatch.

- Stop all new submissions.
- Diagnose and fix in the mutable checkout.
- Add a regression test and rerun all gates/pilots.
- Commit the fix and initialize a new campaign root pinned to the new commit.
- Do not mix pre-fix and post-fix usable trials in one scientific campaign.

Do not work around OOM by silently changing the scientific batch size or number of
validation events. First determine whether resources, implementation, or science is at
fault.

### 8.5 Partial retraining artifacts

Successful retrain indices have hash-bearing marker files. Every successful Slurm task
also has an immutable `state/retrain_receipts/<array>_<task>.json` receipt that seals
the exact task map, frozen row, completed boundary snapshot, and immutable segment
artifacts (or its exact final marker). This prevents later restart-state mutation from
silently rebinding an earlier ledger row. `retrain` returns early if a marker already
exists and the operations-level aggregate gate rejects missing/hash-changed artifacts.

- Never fabricate or hand-edit a success marker.
- For a corrupt/partial index, preserve the failed marker/artifacts in an archive,
  document the reason, and rerun only that index after ensuring no job is active. Never
  delete or replace a sealed task receipt; a changed receipt or referenced artifact is
  a provenance failure requiring a new campaign.
- Split large arrays and lower array concurrency when filesystem or queue pressure is
  high.

## 9. Pareto freeze and 200-epoch retraining

### 9.1 Freeze

Freeze only after all six studies report exactly 600 usable trials and both the
scheduler and Optuna databases are quiescent:

```bash
uv run python scripts/physics_background_pairing_operations.py \
  --root "${BGPAIR_ROOT}" --execution-repository "${BGPAIR_REPO}" freeze
```

The operations-level gate holds the campaign lock while it reconciles Slurm, rejects
every live search/retrain allocation and RUNNING/WAITING Optuna trial, authenticates
the exact first 600 usable trials per model and their Hydra/MLflow/optimized-metric
bindings, and hashes every required retained checkpoint candidate. It then makes a
transactionally consistent hashed backup of all six quiescent SQLite databases,
invokes the commit-pinned campaign runner, reconstructs all fronts independently, and
writes `state/freeze_validation.json`. A pre-existing Pareto manifest is accepted only
after the same validation; file existence alone is never a resume condition.

The command writes:

```text
<BGPAIR_ROOT>/selection/pareto_retrain_manifest.json
```

Required manifest checks:

- schema 2 and frozen commit match `design.json`;
- all 56 logical objective IDs appear;
- every front is nonempty;
- every membership references one of the first 600 usable trials for its model;
- mixed maximize/minimize dominance is correct;
- `selection_uses_downstream_anomalies` is false;
- deduplicated `retrain_runs` contains one row per unique `(model, trial_number)`;
- each row has seed 123 and 200 epochs;
- membership-to-retrain indices are complete.
- `state/freeze_validation.json` authenticates 3,600 selected trials, 56 nonempty
  fronts, all required checkpoint candidates, six database backups, and the immutable
  manifest hash.

Once written, treat the manifest as immutable. Do not inspect downstream efficiencies
and then rebuild a more favorable front.

Export `front_memberships.csv` and `retrains.csv` immediately and record file hashes.

### 9.2 Retraining submission

Let `N` be:

```bash
jq '.retrain_runs | length' \
  "${BGPAIR_ROOT}/selection/pareto_retrain_manifest.json"
```

Before submission, estimate GPU-hours and scratch growth from the measured 200-epoch
canary/pilot behavior and `N`. If `N` is unexpectedly close to the theoretical search
pool size, independently audit the front construction before committing that budget;
do not change or prune a correct front merely because it is expensive.

Submit an indexed array, initially with conservative concurrency:

```bash
uv run python scripts/physics_background_pairing_operations.py \
  --root "${BGPAIR_ROOT}" \
  --execution-repository "${BGPAIR_REPO}" \
  submit-retrains --concurrency 24 \
  --data-dir "${DATA_DIR}" --raw-data-dir "${RAW_DATA_DIR}" \
  --pairing-dir "${PHYSICS_PAIRING_DIR}"
```

The helper validates every existing marker and referenced hash and computes the exact
missing frozen-index set. Clariden permits array task IDs only through 1000
(`MaxArraySize=1001`), while the frozen retrain union can contain as many as 3,600
indices. Each invocation therefore submits only the first at-most-1,001 missing indices
as local task IDs `0..N-1`. It writes a content-addressed map under
`state/retrain_task_maps/<sha256>.json`; the map authenticates the campaign, frozen
manifest, and every local-to-frozen row. Its hash is exported to Slurm and recorded in
each retrain ledger row's `bindings_sha256`. The batch script, campaign runner, and
reconciler independently verify local task, actual index, map path/hash, and manifest
row. Missing, modified, oversized, non-contiguous, or inconsistent maps fail closed.
Legacy ledger rows without a map hash retain their former direct-index interpretation.

Only one retrain batch may be live at once. After it is terminal and reconciled, invoke
`submit-retrains` again for the next bounded batch. A failed actual index may receive a
different local task ID in a later recovery batch: the scientific restart identity is
stable, while every attempt records its authenticated submission map and resumes from
the existing boundary checkpoint. The helper refuses overlaps using both user-scoped
Slurm state and the ledger, and scheduler check, `sbatch`, and registration share the
campaign operation lock. Use `missing-retrains` before and after each array. Direct
`sbatch --array` is a recovery fallback only and must reproduce the map contract.

Each index must:

- reconstruct the exact frozen Optuna hyperparameters;
- train exactly 200 cumulative epochs with seed 123 across five authenticated 40-epoch
  segments, restoring optimizer, scheduler, RNG, loop, model, and callback state;
- reuse one persisted MLflow run ID across every allocation and fail closed if its
  identity tag, frozen row, manifest hash, or restart state differs;
- log immutable hyperparameters on the first segment only; pre-create and authenticate
  the source-commit, run-name, campaign-commit, and retrain-index MLflow tags, with
  experiment creation serialized across concurrent array tasks;
- recompute the full relevant selection suite;
- select a checkpoint separately for every logical objective;
- evaluate all 20 anomalies for the score(s) supported by the model;
- derive native and OAS operating thresholds independently;
- write schema-2 `optimized_metric.json` and a hash-bearing marker.

The per-index restart record is
`retraining/restart_state/<index>/state.json`. It records every Slurm attempt, resume
source, phase, cumulative boundary, copied boundary checkpoint/config and hash, emergency HPC
evidence, the frozen row/manifest/commit identity, and the single MLflow run ID. A
repeated attempt uses an HPC checkpoint, a committed boundary, or a live `last.ckpt`
whose epoch agrees exactly with contiguous MLflow `epoch_idx` history. Scratch retry is
allowed only when both checkpoint and history are empty; one-sided or inconsistent
state fails closed. Segment files are atomically published, and an interrupted publish
is adopted only when both copies match the live checkpoint/resolved config exactly.
The final completion transition occurs only after evaluation artifacts pass their full
audit; a crash during marker finalization enters an idempotent finalization phase.
Completion
markers are accepted only after the final resolved config proves 200 epochs, seed and
Optuna parameters match the manifest, selected checkpoints remain confined to the
campaign and contain the exact finite native/OAS thresholds, and every logical
selection has exactly 20 held-out downstream efficiencies.

After every terminal retrain array, run `reconcile` before submitting its continuation.
Reconciliation writes one immutable receipt per successful local task. Earlier
40-epoch receipts remain valid after later attempts mutate `state.json`; their ledger
`success_sha256` always names the receipt, never the mutable restart state.

The completion count must equal `N`:

```bash
find "${BGPAIR_ROOT}/retraining/markers" -name '*.json' | wc -l
```

Compare actual marker indices with `0..N-1`; a raw count alone does not detect duplicate
or missing indices. Resubmit only missing/failed indices. Do not aggregate until every
marker and referenced CSV/hash validates.

## 10. Aggregation, portable export, and analysis implementation

### 10.1 Existing aggregation

After all retraining markers are valid, use the locked operations transition (not the
campaign runner directly):

```bash
uv run python scripts/physics_background_pairing_operations.py \
  --root "${BGPAIR_ROOT}" --execution-repository "${BGPAIR_REPO}" aggregate
```

This reconciles Slurm, rejects every live/active retrain, authenticates the freeze
receipt, exact marker set, and every marker, invokes the commit-pinned campaign runner,
then independently checks all 56 ordered winners, every frozen membership, tie-break,
and 20 finite downstream efficiencies. It writes an idempotent
`state/aggregate_validation.json`; reruns authenticate its manifest/marker/oracle hashes
without rewriting the oracle.

This writes schema-2:

```text
<BGPAIR_ROOT>/results/oracle_winners.json
```

It contains all front memberships and exactly one downstream-oracle winner per logical
front, including parameters, selected checkpoint, retrained objective pair, mean
efficiency, and 20 per-anomaly efficiencies.

### 10.2 Reporting/export utility

The deterministic reporting utility and synthetic-fixture tests are:

```text
scripts/physics_background_pairing_report.py
tests/test_physics_background_pairing_report.py
```

It has two idempotent modes:

1. `snapshot`: update `state/latest_status.json`, `study_status.csv`, trial exports,
   MLflow/run/checkpoint indices, and completeness diagnostics without reading
   downstream anomalies before freeze.
2. `finalize`: require a frozen manifest, all retrain markers, and
   `oracle_winners.json`; generate every final CSV/JSON/figure, `REPORT.md`, and the
   package hash manifest.

The utility must:

- read but never modify Optuna databases, MLflow stores, checkpoints, or frozen
  manifests;
- export all 3,600 selected search trials to portable CSV/JSON, including parameters,
  ordered objectives, state, duration, and trial number;
- index every search/retrain Hydra directory, MLflow run ID, selected checkpoint, and
  source artifact;
- export all 56 fronts and every membership;
- export every retrain and its completion/hash state;
- independently authenticate every retrain restart identity, five 40-epoch segment
  boundaries, owning attempts, continuation/config hashes, terminal evaluation, and
  MLflow run identity;
- export long-form downstream efficiency rows;
- export CAP, rank correlation, W1, forward drift, and native-secondary metrics for
  reported checkpoints;
- export the persisted reverse G->E held-out drift as a clearly labelled post-hoc
  diagnostic; it is not part of the primary Optuna objective and cannot alter fronts;
- use paths relative to the portable result package where possible;
- include code commit, dirty-state assertion, `uv.lock` hash, preflight hash,
  `design.json` hash, database hashes, and source result hashes in `provenance.json`;
- copy the frozen plan, preflight, design, front manifest, oracle JSON, resolved config
  summaries, relevant Slurm logs, and selected compact plots into `results/`;
- write `manifest.sha256` for every included result-package file;
- fail loudly on missing rows, duplicate logical winners, nonfinite values, hash
  mismatch, or absolute-path-only references that make the package unusable;
- be unit-tested with small synthetic Optuna/manifest/marker fixtures.

The existing evaluator artifact conventions are useful when implementing extraction:

- schema-2 `optimized_metric.json` is authoritative for the chosen checkpoint and Q'/Q''
  pair;
- CAP `<callback>_summary/summary.pkl` contains two sequential trusted-local pickle
  objects: first the CAP checkpoint dictionary, then the rank-correlation dictionary;
- W1 summary `summary.pkl` files contain one checkpoint dictionary;
- drift summary `summary.pkl` files contain two sequential trusted-local pickle
  objects: the primary forward E->G checkpoint dictionary, followed by the post-hoc
  reverse G->E diagnostic dictionary;
- efficiency `values.csv` has `checkpoint,intervention,metric,value`, and aggregation
  uses rows with `metric == efficiency_operational`;
- MLflow is a cross-check and source for epoch histories, not a substitute for the
  frozen schema-2 selection artifact.

Only load these pickle artifacts from the trusted campaign worktree/scratch outputs;
never accept arbitrary external pickle input.

Do not copy raw data, full checkpoints, whole MLflow stores, or SQLite backups into the
portable `results/` directory. Reference their authoritative scratch paths and hashes.

### 10.3 Required portable files

At minimum, produce:

| File | Required contents |
|---|---|
| `provenance.json` | commit, environment/lock hashes, design/preflight hashes, roots, budgets, seeds |
| `study_status.csv` | six rows with complete/failed/pruned/running/waiting/usable counts and durations |
| `trials.csv` | one row per Optuna trial with model, number, state, duration, params, objective vector |
| `front_memberships.csv` | logical front, model, score, strategy, trial, two objective values, retrain index |
| `retrains.csv` | unique retrain index, model/trial, status, job, runtime, marker/hash, shared fronts |
| `retrain_restart_evidence.csv` | five rows per retrain with boundary, owning attempt/Slurm identity, continuation/config paths and hashes |
| `checkpoint_selections.csv` | search/retrain run, logical objective, chosen candidate/name, Q', Q'' |
| `checkpoint_candidates.csv` | all 40,800 configured physical search-candidate slots, availability, path/hash, thresholds |
| `checkpoint_candidate_availability.csv` | per-model/family configured, required, available, and absent counts |
| `primary_metrics.csv` | reported checkpoint CAP/rankcorr/W1/drift/native-secondary and directions |
| `downstream_efficiencies.csv` | long form: front/winner/model/score/anomaly/efficiency |
| `winners.csv` | exactly 56 rows with winner identity, objectives, mean/median/worst/CVaR efficiencies |
| `mlflow_runs.csv` | model/trial/retrain, MLflow experiment/run ID, artifact path/status |
| `job_history.csv` | submissions, predictions, states, durations, exit codes, retries |
| `failures.jsonl` | classified failures and resolutions |
| `REPORT.md` | final human-readable scientific report |
| `manifest.sha256` | hash of every portable package file |

Mirror or copy the final compact package to the repository-ignored convenience path:

```text
results/physics_background_pairing/<BGPAIR_ID>/
```

The authoritative copy remains `<BGPAIR_ROOT>/results/`. A future user should be able
to download that single directory and understand the campaign without mounting MLflow
or opening SQLite.

## 11. Analysis and interpretation rules

### 11.1 Completeness before science

Before interpreting results, prove:

- exactly six study IDs;
- at least 600 usable trials/model and exactly the first 600 used/model;
- 50 epochs/search trial and 200 epochs/retrain;
- exactly 56 unique logical fronts and 56 reported winner rows;
- every front nonempty;
- every retrain marker present and hash-valid;
- every winner has 20 finite downstream efficiencies;
- AE/VAE each have native and OAS rows; other models do not;
- native/OAS threshold namespaces differ;
- no downstream anomaly was present during search/front construction;
- all mappings/test splits authenticate;
- the frozen commit/preflight match every phase.

If any completeness check fails, `REPORT.md` remains draft and must not state final
scientific conclusions.

### 11.2 Comparisons to report

1. **Pairing strategy:** within each model/score, compare the four mapping-based CAP
   winners and CDF control using mean, median, minimum, and bottom-quartile/CVaR
   downstream efficiency. Raw CAP magnitudes may not be directly comparable across
   pairing definitions; downstream outcomes and within-strategy ranks are primary.
2. **Model-selection metric:** compare CAP variants against W1 and forward drift.
   Pairing labels do not apply to W1 or drift.
3. **Anomaly score:** for AE and VAE, compare native vs OAS for the same Q' metric and
   strategy. Report mean deltas, per-anomaly deltas, winner/front overlap, and whether
   OAS helps consistently or only on particular anomaly families.
4. **Model family:** compare the six models without conflating native score semantics.
5. **Robustness:** report mean, median, worst anomaly, bottom-quartile/CVaR, and the
   dispersion across the 20 anomalies. These are across-anomaly summaries, not
   training-seed uncertainty.
6. **Checkpoint behavior:** report stable-candidate availability and how often `last`,
   primary EMA, and stable native win per metric/model/score. This validates the paper
   checkpoint rule's practical effect without hiding strict-stability non-availability.
7. **Proxy association:** analyze how validation Q'/Q'' values and front location
   associate with downstream mean performance. Distinguish descriptive correlation
   from causal or unbiased model-selection claims.
8. **Era directionality:** report E->G drift as primary and G->E as post-hoc diagnostic.

Do not invent uncertainty bars across training seeds: the approved retraining protocol
uses one fixed seed. It is valid to summarize variation across anomaly datasets, but
label it correctly.

### 11.3 Interpretation guardrails

- The downstream winner is an oracle chosen with anomaly labels; it measures the best
  member available on a front, not deployable signal-free selection performance.
- The front itself is signal-agnostic and must be frozen before oracle evaluation.
- Six shared studies improve paired strategy comparability but couple sampling through
  a high-dimensional M-TPE history.
- W1 cannot test pairing quality because it ignores event correspondence.
- Different CAP pairings encode different notions of event similarity; a higher raw
  CAP under one mapping is not automatically superior to a lower raw CAP under another.
- OAS is a residual metric with estimated covariance; its apparent gains may reflect
  correlated feature scaling and should be discussed alongside mask/active-feature
  diagnostics.
- The E/G samples are recorded backgrounds from different eras. Conclusions are about
  this era pair, preprocessing, event prefix, and target operating point.

## 12. Final `REPORT.md` contract

`<BGPAIR_ROOT>/results/REPORT.md` is required. Also copy the final human-readable report
to `research/physics_background_pairing/REPORT.md` if it is suitable for version
control; keep large tables/plots in scratch and link them through the portable package.

The report must contain:

1. **Executive summary:** the scientific question, whether the campaign completed, and
   the most important findings with exact table references.
2. **Provenance and reproducibility:** commit, preflight/design hashes, environment,
   roots, dates, scheduler resources, seeds, and links to package files.
3. **Data:** training/normal invariants, E/G source definitions, counts/splits, anomaly
   leakage boundary, mappings, and CDF control.
4. **Models and scores:** six models, native scores, OAS definition and padding mask.
5. **Protocol:** six shared studies, budgets, objective order/directions, three
   checkpoints, relative-change rule, fronts, deduplicated retraining, oracle winner.
6. **Execution/completeness:** study and job counts, failures/retries, wall-clock and GPU
   usage where available, 56-front/retrain validation.
7. **Main results:** a 56-row machine table plus readable grouped tables/figures.
8. **Pairing comparison:** four mappings vs CDF for CAP.
9. **Metric comparison:** CAP vs W1 vs drift.
10. **AE/VAE score comparison:** native vs OAS overall and per anomaly.
11. **Model comparison and robustness:** mean/median/worst/CVaR and anomaly-family
    behavior.
12. **Checkpoint and proxy diagnostics:** candidate selection frequencies, rank
    correlation, front/downstream association, and directional drift.
13. **Limitations:** oracle use of downstream labels, shared high-dimensional sampler,
    fixed retraining seed, scheduler failures/missingness, and scope of E/G conclusions.
14. **Conclusion:** concise evidence-backed answer to whether and how pairing and OAS
    alter useful model selection.
15. **Artifact index:** every CSV/JSON/figure/log bundle and its role.

Every quantitative statement must be reproducible from a named CSV/JSON file. The
report must distinguish planned, partial, and complete results and must never silently
drop failed models/strategies.

## 13. Phase checklist and autonomous stopping conditions

The agent should work through these phases in order and update both
`state/experiment_matrix.csv` and `RUN_LEDGER.md`:

- [ ] Final code/reporting utility committed and tests green.
- [ ] Fresh detached worktree and isolated campaign root created.
- [ ] Worktree `.env`, quota, pair input, and shared UV environment verified.
- [ ] `design.json` schema 2 initialized and audited.
- [ ] Six unified model pilots completed successfully.
- [ ] Six one-trial production canaries completed and runtime/resource plan calibrated.
- [ ] AE has 600 usable 50-epoch trials.
- [ ] VAE has 600 usable 50-epoch trials.
- [ ] DSAE has 600 usable 50-epoch trials.
- [ ] DSVAE has 600 usable 50-epoch trials.
- [ ] SVDD has 600 usable 50-epoch trials.
- [ ] RealNVP has 600 usable 50-epoch trials.
- [ ] Search trial/MLflow/checkpoint indices exported and audited.
- [ ] Exactly 56 two-dimensional fronts frozen before downstream access.
- [ ] All unique 200-epoch retraining indices completed and hash-validated.
- [ ] `oracle_winners.json` contains 56 winners and complete per-anomaly data.
- [ ] CAP/rankcorr/W1/drift/reverse-drift/checkpoint diagnostics exported.
- [ ] Portable CSV/JSON/figure/log package generated and hash-manifested.
- [ ] Independent sub-agent audit finds no missing strategy/model/artifact.
- [ ] Final `REPORT.md` written, cross-checked, and copied to the requested locations.

The auto-research task is complete only when every box is satisfied. A queued/running
job is not a terminal blocker while other useful work remains: predict its completion
and continue that work. Do not create a timer or poll automatically. A genuine
scientific/configuration failure requires fixing, retesting, a new immutable
commit/campaign if trials have begun, and continued execution. Stop and tell the user
when new authority is required, the scheduler/external system is the sole blocker, or a
scientific choice outside this contract would materially change the experiment.

# Physics background-pairing confirmation: autonomous research plan

## 0. Mission and terminal outcome

This is the authoritative runbook for the next phase of the physics
background0/background1 model-selection study. It supersedes the exhaustive
346-Pareto-configuration retraining described in `PLAN.md`; that file remains the
historical discovery-campaign record.

The next phase must answer two primary questions reproducibly:

1. Does CAP select models with better downstream anomaly efficiency at 250 Hz and
   better AUPRC than Wasserstein-1 (W1) and threshold drift?
2. Does event pairing improve CAP relative to its empirical-CDF control, and which
   pairing definitions improve which endpoint?

The required outcome is a three-seed, 200-epoch confirmation of every native-score
selection strategy, using all five CAP variants plus W1 and drift. Exact duplicate
hyperparameter choices are trained once and reused. A seeded random pairing is added
as a negative-control diagnostic, never as an Optuna objective. The phase ends only
when all runs and evaluations are hash-valid, portable CSV/JSON artifacts exist, and
the repository-root `REPORT.md` answers both questions with uncertainty estimates.

Do not start another Optuna search. Do not retrain 421 checkpoint files or all 346
Pareto configurations. Those numbers belong to the completed discovery evaluation.

## 1. Frozen discovery evidence

The completed discovery campaign is:

```text
/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/physics_background_nsgaii_05b649c
```

It contains:

- six successful NSGA-II studies;
- 600 usable 50-epoch trials per model, 3,600 successful training configurations;
- 56 logical two-objective fronts;
- 346 unique model/trial configurations in the union of those fronts;
- 421 physical checkpoint files selected from last, primary EMA, and stable-native
  checkpoint families;
- 591 front memberships and 601 checkpoint/score evaluations;
- complete downstream efficiency and AUPRC over 20 anomaly datasets.

The joined endpoint evidence is under:

```text
<DISCOVERY_ROOT>/results/endpoint_analysis/
```

The principal files are `primary_front_summary.csv`,
`controlled_selector_comparison.csv`, `fixed_pairing_vs_cdf.csv`, `oas_panel.csv`,
`combined_long.csv`, and `endpoint_analysis.json`. The authenticated AUPRC source is
`<DISCOVERY_ROOT>/auprc_evaluation/aggregate.json`, whose SHA-256 is:

```text
ffaa4e66c67408cbed0975ec50363789db655010bb66251c87826ea69b9b0225
```

The discovery report is the repository-root `REPORT.md`. Treat its results as
screening evidence, not independent confirmation: the same 20 downstream anomalies
were used to choose the reported 50-epoch front winners.

## 2. Data, models, scores, and leakage boundary

Training data and ordinary validation/test data remain unchanged. Model-selection
callbacks compare two held-out recorded-background sources:

- background0: `ZB_run396102` (2025E);
- background1: `ZB_run398183` (2025G).

The five CAP variants are:

1. `flat_physical`: slot-level physical quantities with presence bits and periodic
   phi encoding;
2. `physics_summary`: permutation- and global-rotation-invariant event summaries;
3. `typed_sliced_wasserstein`: energy-weighted typed point-cloud projection
   quantiles;
4. `jetclr`: frozen, L2-normalized learned JetCLR event embeddings;
5. `cdf`: empirical anomaly-score-rank pairing, the prespecified control.

W1 and drift use the same background sources but do not consume event pairings. W1
compares score marginals; drift compares the 250 Hz operating threshold between
background0 and background1.

Use paper-native anomaly scores as the primary analysis:

| Model | Native score |
|---|---|
| AE | masked reconstruction MSE |
| VAE | raw latent KL |
| DSAE | reconstruction error |
| DSVAE | latent KL/variational score |
| DeepSVDD | distance from learned center |
| RealNVP | likelihood/density score |

OAS must not replace AE MSE or VAE KL in the 42-cell primary confirmation matrix. On
the discovery checkpoints, OAS reduced the best AE result from 4.77%/0.628 to
3.09%/0.501 and the best VAE result from
5.36%/0.634 to 4.25%/0.544. During confirmation, masked OAS evaluation on every
sealed AE/VAE checkpoint is mandatory, but it does not add OAS-selected
hyperparameter configurations to the 28-config primary matrix. Persist CAP, W1, and
drift values derived from both the native and OAS score streams wherever the model
supports OAS. State explicitly that this is a same-checkpoint score comparison on
native-selected trajectories, not an independently optimized OAS selector. The OAS
implementation must apply the physics presence/padding mask before covariance fitting
and scoring so nonexistent object slots contribute neither residuals nor covariance.

Downstream anomaly labels must never affect training, checkpoint selection, early
stopping, retries, or adaptive seed decisions. Evaluate them only after a run's
validation-selected checkpoints and completion receipt are sealed.

## 3. Exact confirmation matrix

### 3.1 Logical cells and exact deduplication

There are 42 primary logical cells:

```text
6 models x (5 CAP variants + W1 + drift) = 42
```

The discovery efficiency-selection rule chose the Optuna trials below. Equal trial
numbers within a model are exactly the same hyperparameter configuration and must be
trained only once per seed.

| Model | Flat CAP | Summary CAP | Typed CAP | JetCLR CAP | CDF CAP | W1 | Drift | Unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AE | 154 | 154 | 154 | 352 | 369 | 418 | 154 | 4 |
| VAE | 590 | 507 | 590 | 507 | 634 | 393 | 516 | 5 |
| DSAE | 478 | 40 | 40 | 40 | 478 | 478 | 40 | 2 |
| DSVAE | 328 | 456 | 580 | 436 | 368 | 421 | 318 | 7 |
| DeepSVDD | 177 | 400 | 400 | 400 | 102 | 447 | 142 | 5 |
| RealNVP | 457 | 377 | 539 | 357 | 457 | 645 | 645 | 5 |
| **Total** |  |  |  |  |  |  |  | **28** |

The confirmation manifest must extract the complete `params_json` for each of these
28 `(model, trial_number)` keys from `<DISCOVERY_ROOT>/results/trials.csv` and verify
it against the corresponding frozen Optuna SQLite database. Never copy parameters
manually from this table, mix parameters from several trials, or round floating-point
values. The manifest must contain the exact parameter dictionary, source database and
trial identity, source CSV/database hashes, every logical cell using the configuration,
and a content hash.

The same training trajectory can serve several logical cells because all selection
callbacks are evaluated together. Each cell may still choose a different checkpoint
from that trajectory; deduplicate training, not checkpoint semantics.

### 3.2 Seeds and run count

Use three new, fixed seeds that were not the discovery training seed. They are frozen
before submission as `271828`, `314159`, and `161803`; reserve seeds four and five are
`141421` and `173205`. Do not choose or replace seeds based on pilot performance.

```text
28 configurations x 3 seeds = 84 independent 200-epoch runs
```

The variance-based extension is predeclared as follows: after all three primary seeds
complete for one model, add both reserve seeds to every configuration of that model
only if any multiplicity-adjusted primary-efficiency contrast interval spans both
`-0.005` and `+0.005`. Never add seeds because a preferred method is losing, and never
use secondary AUPRC to trigger the extension. The five-seed maximum is 140 runs. The
manifest, analysis, and report code must support mixed seed counts across models: a
model that triggers the rule has five seeds while an untriggered model remains at
three. A triggered extension is part of the required campaign, not an optional
follow-up. The final-report command must refuse final status until both reserve seeds
have completed for every triggered model.

### 3.3 Why approximate hyperparameter merging is forbidden

The selected configurations often look similar, but only exact duplicates may be
merged. Small differences can accumulate over 200 constant-learning-rate epochs:

- VAE W1 and drift differ mainly in KL warmup (0.2 versus 0.3);
- the two DSAE regimes share architecture and LR but differ in Adam beta2 and weight
  decay;
- DeepSVDD CAP candidates share `[24,8,4]` architecture and sometimes LR but differ
  in network decay and clipping;
- DSVAE candidates share a narrow LR range but differ in KL scale, warmup, rho
  architecture, log-variance clamp, and clipping.

Creating an averaged or consensus configuration would introduce an untested 29th
configuration and break provenance. If compute must be reduced, reduce seeds under a
predeclared rule; do not silently merge near configurations.

## 4. Hyperparameter evidence and interpretation

An exploratory tree-importance analysis over all 600 usable trials per model was
persisted as:

```text
<DISCOVERY_ROOT>/results/endpoint_analysis/search_parameter_importance.csv
```

Mean importance over the seven native primary objectives was approximately:

| Model | Main associations |
|---|---|
| AE | encoder nodes 76%, LR 10% |
| VAE | encoder nodes 35%, LR 23%, KL scale 18% |
| DSAE | pooling 47%, rho nodes 31%, LR 9% |
| DSVAE | LR 20%, KL scale 19%, rho nodes 13% |
| DeepSVDD | encoder nodes 59%, LR 22%, network decay 10% |
| RealNVP | LR 21%; remaining importance spread across clipping, flows, optimizer and depth |

These values are descriptive and model-dependent, not causal importances; NSGA-II
samples are adaptive and impurity-based importance can favor high-cardinality
parameters. Their legitimate conclusion is limited: learning rate is important, but
it is not the sole training driver.

RealNVP deserves an explicit diagnostic because physics-summary selected LR
`1.663e-3`, while flat/CDF selected `1.903e-4`, alongside a hidden-width change. If
the 200-epoch physics-summary advantage disappears, inspect learning curves and
stability before attributing the discovery gain to pairing. Do not alter either
configuration during the confirmation.

## 5. Random-pairing negative control

### 5.0 Frozen decision

Include random pairing in the campaign. It is computationally cheap because CAP is
evaluated on already trained, sealed checkpoints; it does not require another Optuna
study or another 200-epoch training configuration. Use the same random tables for
every eligible checkpoint so comparisons are paired and reproducible. This control
must appear beside flat-physical, physics-summary, typed-sliced, JetCLR, and CDF in
the diagnostic result tables, but it remains outside the seven primary selection
strategies and therefore does not change the 28 configurations, 42 logical cells, or
84 primary runs.

### 5.1 Scientific purpose

Random pairing tests whether structured correspondence contributes information beyond
an arbitrary bijection. It is a diagnostic, not a model-selection strategy:

- do not add it to Optuna;
- do not create a random-CAP Pareto front;
- do not let it select hyperparameters or checkpoints;
- do not count it as an eighth selector in the 42-cell matrix.

If structured CAP separates useful from poor checkpoints more reliably than random
CAP, the pairing contains relevant structure. If random and structured CAP behave
similarly, CAP may be driven mainly by marginal distribution effects.

### 5.2 Mapping contract

Generate at least three and preferably five fixed random bijections for validation and
test. Each must:

- map the exact ordered 163,840-event background0 prefix to the exact ordered
  163,840-event background1 prefix;
- be sampled without replacement with an explicit frozen seed;
- be distinct between validation and test through separate frozen seeds;
- use the repository's normal pair-table schema and strict source/split metadata;
- include source tensor hashes, permutation hash, generator/version, seed, split,
  event count, and file SHA-256;
- pass uniqueness, bounds, total-coverage, split, source-name, and source-hash tests;
- never be regenerated after results are inspected.

Do not resample pairings per epoch. Do not use a many-to-one random nearest-neighbor
table. The negative control is a uniform random permutation.

### 5.3 Evaluation schedule

First evaluate random CAP retrospectively on the frozen discovery checkpoint universe.
This is cheaper than retraining and can detect implementation or interpretation
problems. Preserve the existing 421/601 efficiency/AUPRC artifacts unchanged and write
random-control results to an isolated namespace.

During 200-epoch confirmation, evaluate random CAP offline on each sealed reported
checkpoint, not as an every-epoch callback. Multiple inner CAP fits per random seed
would materially increase epoch cost while providing no checkpoint-selection value.
Persist every seed-specific CAP value plus its mean and variance. Structured CAP and
CDF remain the only CAP callbacks used for validation checkpoint selection.

Retrospective random-CAP results are conditioned on a checkpoint universe selected by
other metrics. Report them as diagnostic association, never as evidence of how random
CAP would have optimized or selected a model.

The interpretation is deliberately narrow:

- structured CAP outperforming random CAP as a checkpoint-ranking diagnostic is
  evidence that the correspondence contains useful structure;
- structured and random CAP behaving similarly suggests that CAP is responding
  mainly to score marginals rather than the proposed event correspondence;
- random CAP outperforming a structured method is a warning that the structured
  mapping may inject harmful or irrelevant geometry;
- none of these outcomes, by itself, is an unbiased downstream efficiency/AUPRC
  comparison for a random-CAP-selected model, because random CAP did not define the
  Optuna search or the frozen confirmation configurations.

## 6. Implementation work before production

Use small, config-driven changes and preserve all discovery artifacts.

1. **Confirmation manifest builder**
   - Read the frozen endpoint/front results and the six Optuna databases.
   - Materialize the exact 28 configurations, three seeds, 42 logical-cell mappings,
     source hashes, and run identities.
   - Reject missing/duplicate keys, parameter disagreement, dirty execution commit,
     or any cardinality other than 28/42/84.
2. **Random-table producer**
   - Extend the pairing utilities with a deterministic uniform-bijection mode or add
     a narrowly scoped producer that emits the existing pair-table schema.
   - Add unit tests for reproducibility, differing seeds, bijection, source binding,
     split separation, and hash validation.
3. **Random-CAP evaluator**
   - Evaluate one or more frozen checkpoints over every random table.
   - Store seed-level and summarized results under a separate
     `random_cap_evaluation/` namespace.
   - Never register random CAP in `optimized_metric_configs`.
4. **Multi-seed segmented retrainer**
   - Key identity by `(model, trial_number, seed)`.
   - Train exactly 200 cumulative epochs from scratch, preserving optimizer, callback,
     RNG, dataloader-shuffler, MLflow, and exact-epoch state across allocations.
   - Evaluate all five structured/CDF CAP callbacks, W1, and drift on the same
     trajectory.
   - For AE/VAE, persist mandatory masked-OAS CAP/W1/drift diagnostics with
     independently calibrated OAS thresholds, without adding OAS-selected
     hyperparameter configurations.
5. **Endpoint aggregator**
   - Join efficiency and AUPRC on exact model/trial/seed/checkpoint-hash/anomaly keys.
   - Preserve logical-cell multiplicity when one physical run serves several cells.
   - Emit seed-level and anomaly-level long tables plus paired comparisons.
6. **Tests and pilots**
   - Add synthetic manifest/deduplication, random-control, restart, hash, aggregation,
     and incomplete-matrix tests.
   - Run focused CPU tests and pre-commit hooks on every changed file.
   - Run one short real-data GPU canary per model before production. Exercise restart
     and final offline random CAP at least once.

Do not mutate the completed discovery root except to add explicitly namespaced,
hash-manifested diagnostic results. Prefer a new confirmation campaign root and a
detached clean worktree pinned to the final implementation commit.

### 6.1 Code and configuration inventory

The implementation is intentionally split between a small control plane, immutable
experiment configs, and pure aggregation code. A future agent must inspect these
files before changing campaign semantics:

| Path | Responsibility |
|---|---|
| `scripts/physics_background_pairing_confirmation_campaign.py` | Prepare, authenticate, submit, reconcile, and report status for the fixed confirmation matrix; it must never launch Optuna. |
| `src/evaluation/physics_background_pairing_confirmation_matrix.py` | Reconstruct the exact 28 unique configurations, 42 logical selector cells, fixed seeds, hashes, and run IDs from discovery evidence. |
| `scripts/clariden/physics_background_pairing_confirmation_canary.sbatch` | One short real-data GPU canary per model family before production. |
| `scripts/clariden/physics_background_pairing_restart_canary.sbatch` | Forced interruption/resume test for exact epoch, optimizer, callback, RNG, and dataloader state. |
| `scripts/clariden/physics_background_pairing_retrain.sbatch` | Restart-safe 40-epoch production segments up to 200 cumulative epochs. |
| `scripts/physics_background_pairing_confirmation_endpoints.py` | Resolve the exact physical selected checkpoint and extract the sealed 20-anomaly efficiency and AUPRC rows. |
| `src/evaluation/physics_background_pairing_confirmation_analysis.py` and its script wrapper | Produce per-model and macro paired comparisons, hierarchical-bootstrap intervals, and multiplicity-aware tables. |
| `src/utils/pairing/random_tables.py` and `scripts/physics_background_pairing_random_tables.py` | Freeze and validate the prespecified random bijections without touching structured pairing tables. |
| `src/evaluation/callbacks/random_cap.py` | Compute CAP for frozen random tables without registering it as an optimization objective. |
| `configs/diagnostic_suite/physics_background_random_cap_native.yaml` | Offline random-CAP evaluation using each model's native anomaly score. |
| `configs/diagnostic_suite/physics_background_random_cap_native_oas.yaml` | Separate masked-OAS diagnostic for AE/VAE only. |
| `scripts/physics_background_pairing_random_cap_evaluation.py` | Prepare authenticated discovery/confirmation checkpoint tasks, evaluate them, report status, and aggregate random-CAP outputs. |
| `scripts/clariden/physics_background_pairing_random_cap.sbatch` | GPU array wrapper for offline random-CAP tasks. |
| `src/evaluation/physics_background_pairing_random_cap_analysis.py` and its script wrapper | Reauthenticate random-CAP aggregates, reproduce five-bijection mean/variance, and compute split/score-specific structured-versus-random rank diagnostics without pooled panels. |

The paper-reference experiment configs under `configs/experiment/physics/` remain the
source of model/data/trainer behavior. New confirmation tooling overlays exact trial
parameters and run identity; it must not rewrite those reference configs. The five
structured/CDF CAP callbacks plus W1 and drift execute together on each trajectory,
which is what permits 42 logical comparisons from only 28 unique configurations per
seed. Random CAP remains offline because it neither selects a checkpoint nor changes
training.

### 6.2 Ordered execution procedure

Use placeholders below rather than embedding machine-specific paths in committed
configuration. `EXECUTION_REPOSITORY` must be a clean detached worktree at the tested
commit, not a mutable development checkout.

```bash
# 1. Freeze the authenticated matrix in a new root.
uv run python scripts/physics_background_pairing_confirmation_campaign.py \
  --root "$CONFIRM_ROOT" \
  --execution-repository "$EXECUTION_REPOSITORY" \
  prepare --discovery-root "$DISCOVERY_ROOT"

# 2. Freeze and validate the random-bijection controls before looking at results.
uv run python scripts/physics_background_pairing_random_tables.py \
  --root "$CONFIRM_ROOT" freeze
uv run python scripts/physics_background_pairing_random_tables.py \
  --root "$CONFIRM_ROOT" validate

# 3. Submit all six model canaries and separately exercise forced restart.
uv run python scripts/physics_background_pairing_confirmation_campaign.py \
  --root "$CONFIRM_ROOT" --execution-repository "$EXECUTION_REPOSITORY" \
  submit-canaries --concurrency 6 --data-dir "$DATA_DIR" \
  --raw-data-dir "$RAW_DATA_DIR" --pairing-dir "$PAIRING_DIR"

# 4. Only after all canary/restart receipts validate, submit production segments.
uv run python scripts/physics_background_pairing_confirmation_campaign.py \
  --root "$CONFIRM_ROOT" --execution-repository "$EXECUTION_REPOSITORY" \
  submit --concurrency "$SAFE_CONCURRENCY" --data-dir "$DATA_DIR" \
  --raw-data-dir "$RAW_DATA_DIR" --pairing-dir "$PAIRING_DIR"

# 5. Reconcile after every Slurm wave, inspect status, and resubmit only missing work.
uv run python scripts/physics_background_pairing_confirmation_campaign.py \
  --root "$CONFIRM_ROOT" --execution-repository "$EXECUTION_REPOSITORY" reconcile
uv run python scripts/physics_background_pairing_confirmation_campaign.py \
  --root "$CONFIRM_ROOT" --execution-repository "$EXECUTION_REPOSITORY" status

# 6. Extract exact checkpoint-bound endpoint rows, then perform paired analysis.
uv run python scripts/physics_background_pairing_confirmation_endpoints.py \
  --root "$CONFIRM_ROOT" --output-dir "$CONFIRM_ROOT/results/endpoints"
uv run python scripts/physics_background_pairing_confirmation_analysis.py \
  --matrix "$CONFIRM_ROOT/selection/native_28_manifest.json" \
  --endpoint-rows "$CONFIRM_ROOT/results/endpoints/endpoints_long.csv" \
  --endpoint-receipt "$CONFIRM_ROOT/results/endpoints/endpoint_extraction.json" \
  --oas-rows "$CONFIRM_ROOT/results/endpoints/oas_diagnostics.csv" \
  --output-dir "$CONFIRM_ROOT/results/analysis"
```

The control-plane commands must be treated as gates: a failed hash, count, canary, or
restart validation blocks production submission. Do not bypass a gate by editing a
receipt. Exact CLI arguments may be rechecked with `--help`; record every actual
command and Slurm job ID in campaign state.

Random CAP has two phases. First prepare/evaluate the sealed discovery universe for a
cheap retrospective diagnostic. After confirmation endpoints are sealed, prepare the
confirmation tasks using `checkpoint_selections.csv` and its endpoint receipt, submit
the random-CAP array, then run `status` and `aggregate`. The confirmation preparation
must deduplicate physical checkpoints while retaining all logical selector
memberships. Never run `prepare-confirmation` before endpoint extraction has bound
every row to a checkpoint hash.

The runtime-bound retrospective discovery manifest is authoritative and must be
reused rather than regenerated:

```text
<DISCOVERY_ROOT>/random_cap_evaluation/discovery_5293a29/manifest.json
```

It contains 346 physical tasks and is bound to execution commit `5293a29`, the exact
immutable runtime data cache, and the versioned random-table namespace
`random_pairing_runtime_5293a29`. Its manifest SHA-256 is
`cd755201c36b036a2b5b9b83fffd28dbb0153614de42b3d7e695b6b137fd463d`; the random-table
manifest content hash is
`d767bff5f5fb21eba668455f85fc02504159783426a91e3d00b68cf382a5d67a`.
Always set `PHYSICS_BGPAIR_RANDOM_CAP_NAMESPACE=discovery_5293a29` when invoking the
discovery evaluator control plane. Three earlier namespaces are failed pilot evidence
and must not be resumed or aggregated: the unversioned `discovery`,
`discovery_5f2a9df`, and `discovery_1f037a0`. Validate one task's terminal receipt
against the authoritative namespace before submitting tasks 1--345. Do not overwrite
or reprepare the manifest to accommodate a runtime mismatch; fix the implementation,
create a new commit/worktree/namespace, and preserve the failed evidence.

The authenticated random-CAP analysis must verify the prepared-manifest hash, task
list hashes, table/source hashes, task membership identity, unique keys, dynamic
cardinalities, and recomputed seed means/sample variances before producing any claim.
For discovery, quantify checkpoint-ranking association only within model, logical
objective, split, and anomaly-score stream. Compare the structured optimized CAP
value with the mean over the five random bijections using tie-aware Spearman
correlation; also persist each random seed's correlation and their range. Never pool
models, splits, or native/OAS streams to manufacture a larger sample.

## 7. Training and checkpoint protocol

For every `(model, trial, seed)` run:

1. Reconstruct the exact frozen hyperparameters.
2. Initialize from scratch; do not fine-tune the 50-epoch checkpoint.
3. Train exactly 200 epochs on unchanged paper training data.
4. Retain last, strategy-primary EMA, and stable native anomaly-score checkpoint
   candidates.
5. Apply the fixed `1.25 Q' + Q''` relative-change checkpoint rule independently for
   every logical cell served by the run.
6. Seal checkpoint paths, hashes, thresholds, metric values, epoch, seed, config, and
   MLflow identity before downstream evaluation.
7. Evaluate all 20 anomalies on each cell's validation-selected checkpoint and record
   both 250 Hz efficiency and AUPRC.
8. Evaluate the random-CAP controls offline on the same sealed checkpoint.

A single run can yield several different reported checkpoints because different
logical selectors monitor different validation metrics. Never force all cells sharing
hyperparameters to use one checkpoint.

Use authenticated 40-epoch boundaries at 40, 80, 120, 160, and 200 to remain within
Clariden's 12-hour allocation limit. Emergency HPC checkpoints are scheduler evidence,
not normal resume boundaries. Resume only exact end-of-epoch state with consistent
epoch history and RNG/dataloader state.

## 8. Runtime and scheduler plan

Measured median 50-epoch discovery runtimes imply:

| Model | Runs | Estimated 200-epoch time/run | GPU-hours |
|---|---:|---:|---:|
| AE | 12 | 36.3 h | 436 |
| VAE | 15 | 35.4 h | 531 |
| DSAE | 6 | 18.6 h | 112 |
| DSVAE | 21 | 18.6 h | 390 |
| DeepSVDD | 15 | 18.2 h | 274 |
| RealNVP | 15 | 18.0 h | 269 |
| **Total** | **84** |  | **about 2,013 GPU-hours** |

Ideal compute-only wall time is about 10.5 days at eight concurrent GPUs, 5.2 days at
16, 3.5 days at 24, or 2.6 days at 32. With queueing and segment transitions, budget
roughly 5-8 days when 16-24 GPUs can be kept active. Re-estimate after canaries; do not
promise ideal time as scheduler time.

Keep as many independent run segments active as account limits permit, but never run
two segments for the same `(model, trial, seed)` concurrently. Prioritize a balanced
mixture of models so an early systematic failure does not consume the entire budget
for one family.

After each submission or completion:

- reconcile Slurm state with the task manifest and completion receipts;
- update durable `job_history.csv`, failure logs, and matrix status;
- estimate queue delay and runtime from `squeue --start` and recent `sacct` evidence;
- continue useful independent work while jobs run;
- when the scheduler is the sole blocker, report job IDs/reasons and predicted next
  useful check time, then use the environment's supported wait/monitor mechanism if
  available; never burn tokens with tight polling or a blocking shell sleep.

Use bounded read-only sub-agents for log classification, scheduler summaries, per-model
artifact audits, and independent result checks when this avoids bloating the main
context. The main agent retains authority for code changes, submissions, recovery,
scientific interpretation, and final completeness.

## 9. Statistical analysis contract

### 9.1 Primary and secondary endpoints

- Primary: mean downstream signal efficiency at the fixed 250 Hz operating point.
- Secondary: mean downstream AUPRC/average precision.

Never select a checkpoint, seed, or configuration using either downstream endpoint.
Attach both endpoints to the same validation-selected checkpoint.

### 9.2 Controlled comparisons

For each model:

1. Compare CDF-CAP with W1 and drift. This is the controlled answer to whether CAP
   beats the other model-selection metrics.
2. Compare each of flat-physical, physics-summary, typed-sliced, and JetCLR CAP
   separately with CDF-CAP.
3. Do not use “best of four learned pairings” as the primary test. It has a selection
   advantage. It may appear only as a clearly labelled exploratory system result.
4. Compare structured CAP metric/checkpoint rankings with random-CAP rankings and
   random-seed variability. Random CAP has no downstream winner of its own.

Use paired seeds and paired anomaly datasets. Report raw seed results, mean delta,
confidence/compatibility interval, sign consistency, and practical effect size. Use a
hierarchical bootstrap that resamples seeds and anomaly datasets, or an equivalent
predeclared paired model. Correct multiplicity across the four learned-pairing versus
CDF tests within each model. Do not treat 20 anomalies as 20 independent training
replicates.

Report mean, median, worst anomaly, and bottom-quartile/CVaR for both endpoints. First
average the available seeds within each anomaly and logical cell; compute worst-anomaly
and bottom-five-anomaly/CVaR summaries over those 20 anomaly means. Do not flatten the
seed-by-anomaly grid, because that incorrectly treats seeds as additional anomaly
types. AUPRC is prevalence-sensitive; preserve anomaly-specific prevalence and do not
compare its absolute magnitude with the paper unless an exactly matching paper
endpoint exists.

### 9.3 Claims and limitations

The discovery evidence suggests:

- CDF-CAP has the strongest controlled macro mean and wins AUPRC for five of six
  native models;
- learned pairing can improve 250 Hz efficiency for VAE, DSVAE, and RealNVP;
- only DSVAE and DeepSVDD showed a clear AUPRC gain over CDF;
- OAS is not competitive with native AE/VAE scores.

These are hypotheses to confirm, not guaranteed outcomes. The confirmation report must
state when 200-epoch or multi-seed results reverse them.

Because all 20 existing anomaly datasets were inspected during discovery, new held-out
anomaly datasets would be required for fully independent external confirmation. If no
new anomalies are available, this phase establishes retraining/seed stability and must
not be presented as an untouched holdout test.

## 10. Durable artifact layout

Create a new confirmation root, for example:

```text
/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/physics_background_confirmation_<commit>
```

Required layout:

```text
<CONFIRM_ROOT>/
  design.json
  selection/native_28_manifest.json
  random_pairing/
    tables/
    manifest.json
    diagnostics.json
  random_cap_evaluation/
    manifest.json
    aggregate.json
  retraining/
    task_maps/
    restart_state/
    boundaries/
    markers/
  runtime/
    checkpoints/
    logs/
    mlflow/
    outputs/
  state/
    experiment_matrix.csv
    job_history.csv
    failures.jsonl
    latest_status.json
  results/
    REPORT.md
    provenance.json
    configurations.csv
    logical_cells.csv
    runs.csv
    checkpoint_selections.csv
    endpoints_long.csv
    endpoints_summary.csv
    selector_comparison.csv
    pairing_vs_cdf.csv
    random_cap.csv
    oas_diagnostics.csv
    paper_comparison.csv
    figures/
    logs/
    manifest.sha256
```

All result tables must include model, trial, seed, score, selector, pairing,
checkpoint path/hash, anomaly, endpoint, and value where applicable. Persist large
runtime artifacts only in scratch. Mirror the compact result package under the
repository-ignored `results/` directory for download. Keep the human-readable final
report at repository-root `REPORT.md` as requested.

Every phase transition must be represented by an immutable JSON receipt containing
input hashes, output hashes, execution commit, timestamp, command identity, and exact
cardinalities. File existence alone is never proof of completion.

## 11. Failure and recovery rules

- Classify infrastructure, scheduler, code/data, numerical, and provenance failures
  separately before retrying.
- Retry infrastructure failures with unchanged scientific configuration.
- Preserve failed logs and partial artifacts; never overwrite evidence silently.
- A code, data, mapping, score, or selection-semantics fix requires a new execution
  commit and, once production results exist, a new confirmation root.
- Never change batch size, event prefix, pairing seed, hyperparameters, epochs, or
  endpoint definition merely to make a failed run pass.
- Resume only from authenticated completed-epoch boundaries.
- Never fabricate success markers or edit Optuna/manifest/receipt databases manually.
- Do not cancel or mutate unrelated user jobs.

## 12. Final report requirements

The final `REPORT.md` must be concise but information-dense and contain:

1. exact completion/provenance and whether all 84 runs succeeded;
2. a complete table with efficiency and AUPRC in every campaign cell, highlighting
   the winner for each endpoint independently;
3. controlled CDF-CAP versus W1 and versus drift comparisons per model and in macro
   average, reported separately for efficiency and AUPRC rather than collapsed into a
   combined win count;
4. every learned pairing versus CDF per model and in macro average, with paired-seed
   uncertainty separately for efficiency and AUPRC;
5. random-CAP values, between-random-seed variability, and checkpoint-ranking
   association with every structured CAP variant as a negative control;
6. a direct AE/VAE native-versus-masked-OAS conclusion on identical checkpoint hashes,
   clearly separated from native-selected primary results;
7. paper efficiency comparisons, with no invented paper AUPRC;
8. training curves and discussion of whether 50-epoch findings persist at 200 epochs;
9. failures, missingness, multiplicity treatment, and discovery-selection limitation;
10. a direct conclusion stating where pairing improves efficiency, AUPRC, both, or
    neither, and whether CAP beats W1 and drift separately;
11. the per-model reserve-seed decision and evidence, including completion of every
    triggered extension;
12. discovery hypotheses that reversed at 200 epochs or across seeds, not only those
    that survived.

Use “beats” or “improves” only when the prespecified multiplicity-adjusted interval
supports that conclusion. Label a favorable point estimate whose interval crosses
zero as directional or inconclusive. Efficiency differences are percentage-point
deltas; AUPRC differences are absolute-score deltas, and every table must print those
units. Paper comparisons are cross-system benchmarks because the validation/model-
selection sources differ; they cannot by themselves isolate a causal effect of
background0/background1 or pairing.

Every quantitative statement must be reproducible from a named CSV/JSON artifact.
Copy the compact report package to the requested locations and verify
`manifest.sha256` before declaring completion.

## 13. Autonomous execution checklist

- [ ] Freeze final implementation commit and clean detached worktree.
- [ ] Build and authenticate the 28-configuration/42-cell/three-seed manifest.
- [ ] Generate and authenticate validation/test random bijections.
- [ ] Implement and test offline random-CAP evaluation.
- [ ] Run retrospective random CAP on the frozen discovery checkpoint universe.
- [ ] Implement/test segmented multi-seed confirmation runner and aggregation.
- [ ] Complete one real-data canary for all six models.
- [ ] Launch and track all 84 runs while respecting per-run exclusivity.
- [ ] Authenticate all five 40-epoch boundaries and terminal markers per run.
- [ ] Seal validation-selected checkpoints before downstream evaluation.
- [ ] Evaluate efficiency and AUPRC on all 20 anomalies for all 42 cells and seeds.
- [ ] Evaluate random CAP on every sealed reported checkpoint.
- [ ] Persist mandatory masked-OAS AE/VAE diagnostics without changing the primary
      matrix.
- [ ] Produce paired uncertainty, multiplicity-corrected comparisons, and robustness
      summaries.
- [ ] Independently audit counts, hashes, joins, and claims, using sub-agents for
      bounded read-only checks where useful.
- [ ] Write and validate final scratch and repository-root reports.

The task is complete only when every required box is satisfied or a genuine external
blocker is reported with exact evidence. A queued job is not a reason to abandon the
campaign; continue independent analysis and implementation until scheduler completion
is the only remaining dependency.

## 14. Live handoff state (2026-08-17 09:29 Europe/Zurich)

This section records the current execution identity. Treat durable campaign receipts
and Slurm as authoritative if this snapshot becomes stale.

```text
Training implementation commit: ddfcfbb46d7fed7308b2066328b7b898a6316cc8
Current analysis/random-control commit: 5293a29
Confirmation root:
  /iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/physics_background_confirmation_ddfcfbb
Clean execution worktree:
  /iopsstor/scratch/cscs/vjimenez/adatl1/worktrees/physics_background_confirmation_ddfcfbb
Random-CAP execution worktree:
  /iopsstor/scratch/cscs/vjimenez/adatl1/worktrees/physics_background_random_cap_5293a29
Reserve-extension execution worktree:
  /iopsstor/scratch/cscs/vjimenez/adatl1/worktrees/physics_background_confirmation_reserve_runtime
Reserve-extension execution commit: 0aefbe8a5ead152edacf301cb8ac63dacb95c534
Discovery root:
  /iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/physics_background_nsgaii_05b649c
```

The confirmation root has authenticated 28 configurations, 42 native logical cells,
84 primary runs, 126 run-to-cell memberships, the three frozen primary seeds, and ten
frozen random tables. The forced-restart canary job `3098320` passed after exactly one
requeue and has a sealed validation receipt. All six independent real-data model
canaries passed: AE `3098330`, VAE `3098337`, DSAE `3098343`, DSVAE `3098346`,
DeepSVDD `3098356`, and RealNVP `3098364`. Production array `3098367` contains all 84
primary runs with concurrency 32. Fourteen runs have authenticated epoch-40
boundaries: indices 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, and 21. Array
`3099316` is their exact disjoint 40--80 continuation. Eighteen other first segments
remain running and indices 32--83 have not started. This is not 84 complete 200-epoch
allocations. Reconcile and submit the 40--80, 80--120, 120--160, and 160--200 segments
only after the preceding authenticated boundary for each run.

Retrospective random-CAP pilot `3098386_0` passed against the authoritative
`discovery_5293a29` namespace and produced the expected native/OAS validation/test
rows over all five frozen random bijections. At this snapshot, 307 of 346 tasks are
complete with no invalid artifacts. Packed debug job `3099042` has completed tasks
304--306 and is processing 307 before 308--309; normal array `3098406` retains exact
non-overlapping coverage of tasks 310--345. Preserve that coverage when moving work
between queues: cancel an exact pending task before resubmitting it elsewhere, and
never duplicate a running or completed task. Failed pilots `3098368`, `3098378`, and
`3098383` remain diagnostic evidence and must not be counted as completed work.

The reserve-extension worktree descends directly from the immutable training commit
and changes only the authenticated control-plane and its tests. Its Git-diff SHA-256
is `b8f266228e0cf5d0c2d88bb22f010b60e49be7b525da3e83d23199b7dda6a094`; 115
focused tests passed in that exact worktree. Use it only after the three-primary-seed
analysis has frozen the per-model extension decisions. Offline random-CAP evaluation
uses its own clean, manifest-recorded execution commit because it neither trains nor
selects the sealed confirmation checkpoints.

During execution, update durable state under `<CONFIRM_ROOT>/state/` rather than
continually rewriting this section. Use bounded sub-agents for independent log,
artifact, or statistical audits when useful, but require the primary agent to verify
their conclusions against receipts before submissions or scientific claims. When
jobs or the scheduler are the only blocker, predict the next useful check from recent
runtime and queue evidence, use the supported timer/wait mechanism, and resume until
all required artifacts and the final report are complete.

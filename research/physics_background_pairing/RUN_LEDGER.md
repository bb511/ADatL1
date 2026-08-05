# Run ledger

Last updated: 2026-08-05 (Europe/Zurich)

## Contract and inputs

- Plan: `research/physics_background_pairing/PLAN.md`
- Pair artifacts:
  `/iopsstor/scratch/cscs/vjimenez/adatl1/data/data_2025E+G/pairing/ZB_run396102_to_ZB_run398183`
- Validation/test prefix per source: 163,840 events (10 x 16,384)
- Pairings found for both splits: `flat_physical`, `physics_summary`,
  `typed_sliced_wasserstein`, `jetclr`
- Frozen debug campaign commit: `6d10cd5601ffa86111a12415b521c4e104c261e6`.
- Superseded pilot campaigns at `ebe2ce1` and `bb0f981` are archived in scratch;
  neither is eligible for production submission.
- Superseded packed execution commit: `59b47b0c41bcb2e9f233d0695bb2ba005c4e9c8f`.
- Superseded packed campaign root:
  `/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/physics_background_shared_59b47b0c41bc`.
- The earlier `d7fd395485d8` reference and all `59b47b0c41bc` canaries were cancelled
  after a checkpoint audit proved that dynamically registered operating-threshold
  buffers were absent from retained checkpoints.
- Campaign `6edb3f7` is also superseded: pilot `3008187` proved that the shared
  search suite still set `callbacks.anomaly_eff: null`, so the new buffer lifecycle
  code was never invoked; queued VAE pilot `3008189` was cancelled with it.
- Current replacement execution commit: `cbeb3fad2500b11de6bb5c5e1571ec46000fa673`.
- Current replacement campaign root:
  `/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/physics_background_shared_cbeb3fa`.

## Status

| Phase | State | Evidence / next action |
|---|---|---|
| Scientific contract | complete | `PLAN.md` |
| Implementation audit | complete | Model/data/HPO settings match `dev/patrick`; requested metric scope, score routing, CDF, retraining, and padding semantics audited |
| Code and configs | fix in validation | Threshold buffers are registered before checkpointing and the shared search suites now instantiate native plus AE/VAE OAS calibration callbacks on the normal validation stream only |
| Tests and preflight | complete | 36 selection/config/campaign tests and 54 reporting/analysis/artifact-audit tests pass; all six configs compose, all 8 pair tables authenticate, and scratch capacity is sufficient |
| Autonomous runbook | complete | `PLAN.md` now specifies the full six-study state machine, artifact contract, scheduler/timer policy, failure recovery, portable exports, analysis, and final `REPORT.md` |
| Provenance/report hardening | complete | Added persistent Optuna-to-Hydra bindings, run/checkpoint/MLflow identities, reverse G-to-E drift, fail-closed artifact audits, and deterministic scientific-analysis tables in the final portable package |
| Pilots | complete | Corrected unified jobs AE `3008281`, VAE `3008282`, DSAE `3008321`, DSVAE `3008344`, DeepSVDD `3008351`, and RealNVP `3008360` passed; four-GPU temporary AE/Optuna pilot `3008377` passed all launcher, binding, objective, and threshold gates |
| Production canaries | running | Four-trial x 50-epoch jobs AE `3008470`, VAE `3008471`, DSAE `3008472`, DSVAE `3008473`, DeepSVDD `3008474`, and RealNVP `3008475` are running concurrently on six one-node allocations |
| Full searches | superseded before completion | Jobs AE `3007292`, VAE `3007311`, DSAE `3007310`, DSVAE `3007318`, DeepSVDD `3007340`, RealNVP `3007345`, and legacy `3005821`/`3005823` were cancelled; no usable trial completed and no bulk chunk was submitted |
| Retraining/evaluation | pending | Offline 56-front freeze and unique 200-epoch retraining remain implemented; corrected checkpoint-specific native/OAS threshold restoration requires GPU validation |
| Aggregation/report | pending | Score-aware downstream oracle aggregation is implemented; awaits validation and results |

The `59b47b0c41bc` and `6edb3f7` campaigns are superseded and must never be resumed.
No bulk search chunk was submitted. `cbeb3fa` is debug-only until its focused checkpoint,
unified GPU, packed-launcher, and 50-epoch canary gates pass.

Repository-wide `pre-commit -a` is not a usable clean gate on the current baseline: it
rewrites 82 unrelated legacy files and separately reports pre-existing executable shell
scripts without shebangs. Those hook-induced edits were restored. All campaign-scoped
hooks (including direct Black, isort, docformatter, interrogate, flake8, Bandit, and
codespell) pass; the unrelated baseline debt must not be mixed into this campaign.

## Decisions

- The ordinary combined validation data remains available and unchanged; source-specific
  loaders are additional model-selection views.
- W1 is run once per model because it is invariant to a bijective reordering of the same
  two score samples.
- Primary drift direction is 2025E to 2025G. The reverse direction is a diagnostic.
- AE residual OAS state must be checkpointed and fitted from a deterministic bounded
  training-normal sample so it is practical during every validation epoch.
- AE and VAE each retain the paper-native anomaly score as a direct control: AE MSE
  versus residual OAS, and VAE latent KL versus residual OAS.
- All logical strategies for one model share the same 600 sampled configurations. This
  gives six trained studies rather than 56 and enables direct strategy comparisons;
  logical two-dimensional fronts are reconstructed offline.
- Normal jobs are node-exclusive. One Hydra/Optuna controller therefore packs four
  ordered, resource-exact one-GPU `srun` steps per node instead of wasting three GPUs
  or running competing controllers against one study.

## Verification log

- 2026-08-05: packed launcher pilot `3008377` completed in 6m25s. Its temporary
  Optuna study contains trials 0-3 in `COMPLETE` state, exactly 28 finite values
  per trial, ordered Hydra bindings 0-3, and four threshold audits with distinct
  finite native/OAS thresholds. Scheduler accounting proves four one-GPU steps,
  each with 16 CPUs and 96 GB, and worker logs prove CUDA devices 0-3. Six real
  four-trial x 50-epoch canaries were then submitted from the same frozen commit:
  AE `3008470`, VAE `3008471`, DSAE `3008472`, DSVAE `3008473`, DeepSVDD
  `3008474`, and RealNVP `3008475`; all six entered `RUNNING` immediately.

- 2026-08-05: all six corrected unified pilots on `cbeb3fa` completed in 2m37s
  to 4m54s with the exact 14/7 logical-objective contracts and finite threshold
  audits. AE and VAE checkpoints contain distinct native/OAS thresholds. An
  independent MLflow audit of AE/VAE proved exactly the normal, background0, and
  background1 validation streams, all 25 required finite one-epoch histories, all
  four authenticated mapping strategies plus CDF, and no anomaly loader exposure.
  The packed launcher pilot uses a temporary database and does not touch any
  production study.

- 2026-08-05: replacement AE pilot `3008187` on `6edb3f7` produced retained
  checkpoints but the live state-dict audit still found no threshold keys. Resolved
  Hydra configuration and callback logs proved the cause: the shared selection suite
  explicitly disabled `anomaly_eff`, so the fixed buffer registration code never ran.
  The job and queued VAE pilot `3008189` were cancelled. Commit `cbeb3fa` now enables
  native calibration for all six models and a separately namespaced OAS calibration
  for AE/VAE while keeping `validation_aux_datasets: []`; 36 focused tests pass.
  Replacement pilots `3008281`/`3008282` were submitted from a fresh campaign.

- 2026-08-05: a direct audit of completed pilot and live canary Lightning files proved
  that `residual_oas_location`, `residual_oas_precision`, and `residual_oas_ready` were
  checkpointed, but neither `thres_operational` nor
  `thres_operational__residual_oas` appeared in any retained state dict. Because the
  evaluator cycles checkpoint candidates on the same in-memory module, downstream
  test evaluation would otherwise reuse the final epoch's thresholds. All six packed
  canaries and the isolated legacy reference were cancelled and preserved. The active
  campaign failure ledger records this as a scientific-code failure. Threshold buffers
  are now declared as persistent NaN sentinels before checkpointing and updated
  independently by native/OAS validation; 23 focused tests pass. A real GPU checkpoint
  proof is required before creating the replacement production campaign.

- 2026-08-05: all six four-trial production canaries were accepted on one node each:
  AE `3007292`, VAE `3007311`, DSAE `3007310`, DSVAE `3007318`, DeepSVDD `3007340`,
  and RealNVP `3007345`. Every job asked four Optuna trials with exactly 50 epochs and
  launched one worker on each of GPUs 0-3. Each model has four distinct MLflow runs;
  AE/VAE studies have 28 scalar directions and the other studies have 14. DSAE and
  DSVAE, DeepSVDD, and RealNVP completed epoch 0 for all workers at step 764 with seven
  finite summary metrics each; all AE/VAE workers completed epoch 0 with all 14 finite
  native/OAS summary metrics. First-epoch projections are 8.5 hours for AE/VAE and
  4.4-4.7 hours for the other models, inside the 12-hour limit. No traceback, OOM,
  nonfinite metric, or failed step is present. Active-job, latest-study, and startup
  audit snapshots are persisted under campaign `state/`.

- 2026-08-05: scheduler accounting showed that a nominal one-GPU normal job reserves
  an entire four-GPU GH200 node. A custom packed launcher now runs four ordered Hydra
  jobs per node under one Optuna controller. Packed pilot `3006163` proved distinct
  GPUs 0-3 and exposed a concurrent file-backed MLflow experiment-creation race;
  three workers completed and one returned a failed Hydra result. The controller now
  pre-creates the experiment. Retry `3006210` then failed before launching workers
  because the real Optuna plugin supplies tuple overrides; the launcher now normalizes
  that API boundary to a list, with a regression test. Neither failed pilot touched a
  production study. Candidate pilot `3006215` then completed all four workers on
  distinct GPUs from frozen commit `59b47b0c41bc` in 4m24s. Optuna trials 0-3 are all
  `COMPLETE`, each has 28 finite values, and the ordered trial/Hydra bindings and
  scheduler/log hashes are persisted under the campaign `state/` directory.

- 2026-08-05: final execution campaign `d7fd395485d8` passed all six unified pilots:
  AE `3005586`, VAE `3005587`, DSAE `3005633`, DSVAE `3005661`, DeepSVDD `3005772`,
  and RealNVP `3005779`. The cross-pilot audit proved 56 logical selections, exact
  finite 28/14-value vectors, matching objective order, MLflow/run/checkpoint
  provenance, and two AE/VAE or one native-only reverse-drift artifacts. AE production
  controller `3005821` launched worker `3005823`; Optuna created the exact 28-direction
  study and epoch 0 logged every native/OAS objective at 13:00 CEST. At 8.2 minutes per
  epoch, predicted trial completion is about 19:50 CEST, inside the 12-hour limit.

- 2026-08-05: the first 50-epoch AE canary gate found orchestration-only defects
  before any Optuna trial was created. Clariden `xfer` nodes are x86-64 and cannot run
  the pinned ARM environment. Account `a0166` also lacks the `slowdown` QOS required
  by the 24-hour `low` partition, so the controller uses a 12-hour ARM `normal` node.
  Hydra also requires the SQLite URL containing `?timeout=600` to be quoted as one
  override value. The exact override now composes successfully, 38 focused campaign
  tests pass, and all changed-file hooks pass.

- 2026-08-05: active campaign `fec1619f9321` passed all six unified GPU pilots:
  AE `3005314` (3m59s, 7.03 GiB peak), VAE `3005446` (4m12s, 7.07 GiB), DSAE
  `3005499` (2m31s, 5.71 GiB), DSVAE `3005510` (2m40s, 6.13 GiB), DeepSVDD
  `3005522` (2m24s, 11.29 GiB), and RealNVP `3005531` (2m12s, 6.06 GiB).
  A cross-pilot audit proved 56 logical selections, exact 28/14-value vectors,
  finite pairs, commit/run/checkpoint/MLflow provenance, 24 AE/VAE or 12 native-only
  metric families, forward/reverse drift plots, and an empty anomaly-validation
  allowlist. Scheduler/log hashes and success artifacts are in campaign `state/`.

- 2026-08-05: schema-2 campaign `bcdffe3eb8c1` authenticated all eight pair tables and
  froze the correct six-study/56-front design, but unified AE pilot job `3004987`
  failed before evaluation because the evaluator detached unresolved
  `${physics_selection.native_secondary}` interpolations from the Hydra root. No
  Optuna trial was submitted. The failed campaign is superseded and its log hash,
  scheduler record, and classified failure are persisted under its `state/` directory.
  The evaluator now receives fully resolved objective dictionaries, covered by a
  runtime-instantiation regression test; a new commit-derived campaign is required.
- 2026-08-05: replacement campaign `483da88a2bd5` passed evaluator construction and
  reached held-out E/G validation, then unified AE pilot job `3005037` exposed that
  the new reverse-drift plot directory was not created before rendering. No search
  trial was submitted. The campaign is superseded with the job/log/failure evidence
  preserved. The reverse directory is now created explicitly and the callback test
  asserts every plot destination exists. Final reporting was also completed with four
  compact figures plus winner-resolved Hydra configs and deterministic compressed
  search/retrain/Slurm log indices.
- 2026-08-05: campaign `e184adeb8676` unified AE pilot job `3005118` successfully
  trained and evaluated the last, all 14 primary-EMA candidates, and stable-native
  candidate, including both reverse drift plots. It failed only when provenance tried
  to read mandatory-but-unset `hydra.job.num` in a Hydra single run. The campaign is
  superseded before any search trial, with complete scheduler/log/failure evidence.
  Single-run provenance now serializes this field as null while preserving integer
  multirun indices; both cases are covered.

- 2026-08-04: implemented the six shared 600-trial model studies. Every trial now
  returns all native/OAS CAP, W1, drift, and native-secondary objective pairs in a
  frozen order; the 56 logical two-dimensional fronts are reconstructed offline.
- 2026-08-04: added independent native/OAS validation thresholds for downstream AE/VAE
  evaluation, restricted search validation to normal plus the two source backgrounds,
  and kept all 20 anomaly datasets held out until 200-epoch retraining.
- 2026-08-04: the new shared evaluator, campaign, score thresholds, configurations,
  pairing paths, and masked OAS contracts passed 45 targeted tests. All six final
  configs recomposed after formatting, and every changed-file pre-commit hook passed.

- 2026-08-04: model, trainer, data, and HPO settings match the `dev/patrick` physics
  setup at `9c2c4ec`; the later posterior-consistency callback is intentionally absent
  because this campaign's requested primary metrics are CAP, W1, and drift. The six
  HPO spaces otherwise match, with the SQLite lock timeout supplied by the campaign.
- 2026-08-04: added paper-native/OAS score factoring for AE and VAE, an empirical-CDF
  CAP control, and preserved the selected pairing during Pareto retraining.
- 2026-08-04: OAS fitting and scoring now treat padded coordinates as missing, using
  observed-only locations, neutral mean imputation for covariance fitting, and
  observed-feature-only event energies.
- 2026-08-04: expanded targeted suite passed (40 tests), including composition of all
  56 primary search cells; pre-commit passed on every changed file.
- 2026-08-04: merged the newer `origin/research/main` JetCLR and mapping-table work,
  preserved the CDF/native/OAS campaign contract, and passed the expanded post-merge
  suite (49 tests).
- 2026-08-04: a real-data pilot exposed one structurally absent physics coordinate.
  Residual OAS now fits only the globally observed feature subspace, embeds inactive
  precision rows/columns as zero, and retains observed-feature event normalization.
  The expanded targeted suite passed 52 tests and all pre-commit hooks.
- 2026-08-04: fresh frozen-commit debug pilots passed end to end: AE residual-OAS CAP
  with the authenticated `physics_summary` table (`3000424`, 9m59s), VAE
  residual-OAS CAP with the table-free CDF control (`3000425`, 10m12s), and AE
  residual-OAS W1 (`3000500`, 7m58s). Each produced its expected selected checkpoint,
  evaluator artifacts, optimized-metric artifact, and success marker.

- 2026-08-03: all six paper-model configurations composed with the JetCLR variant.
- 2026-08-03: CAP, Wasserstein, and drift search-only overlays composed independently.
- 2026-08-03: targeted callback, AE-score, pairing, configuration, provenance, and
  campaign suite passed (25 tests).
- 2026-08-03: pre-commit passed on every changed file.
- 2026-08-03: validation and test tables for all four strategies authenticated against
  the exact ordered 163,840-event source-loader tensors. See `preflight.json`.
- 2026-08-03: initial pilot submission was rejected with
  `QOSMaxSubmitJobPerUserLimit`; no job was created and no unrelated queued work was
  cancelled. Pilot and chunked search submission scripts are ready for the first free
  scheduler slots.
- 2026-08-03: paper protocol rechecked from the supplied PDF. Search evaluation now
  retains all three required checkpoint candidates. The campaign freezes every Pareto
  trial, retrains it for 200 epochs, and performs the paper's explicitly labelled
  downstream-oracle selection only after the Pareto manifest is immutable.

# Run ledger

Last updated: 2026-08-16 (Europe/Zurich)

- 2026-08-16: all six frozen NSGA-II screening studies reached exactly 600 usable
  COMPLETE trials (3,600 total, 50 epochs each) with no RUNNING or WAITING trials.
  The authenticated freeze contains 56 signal-blind logical fronts, 591 front
  memberships, and 346 unique `(model, trial)` configurations to retrain for 200
  epochs. `state/freeze_validation.json` seals the six database backups, 3,600 MLflow
  runs, 33,600 checkpoint selections, and 40,800 configured checkpoint slots.
  `results/SCREENING_AUDIT.md` and `results/analysis/` now persist generation-wise
  convergence diagnostics plus authenticated paper 250 Hz/q99 baselines; downstream
  improvement remains explicitly provisional until retraining and evaluation finish.
  Retraining array `3093119` was submitted from the frozen commit with 346 mapped
  tasks, a 24-task concurrency limit, and the campaign-owned immutable data snapshot;
  its first 24 AE tasks entered RUNNING immediately.

- 2026-08-16: the first strict freeze attempt exposed a control-plane-only receipt
  bug: sealed-chunk auditing rewrote `chunk_validation.json` with a fresh
  `validated_at` before comparing its full-file SHA. Job `3037884` was the sole
  affected sealed receipt; its binding, chunk, Slurm log, all 48 objective artifacts,
  retained checkpoints, and MLflow histories independently revalidated unchanged.
  Commit `2535d9abd76b3bcbca87e50935773f869b149b60` makes receipt validation idempotent
  and rejects semantic changes. The old/new receipt hashes, unchanged evidence,
  pre-repair ledger backup, and explicit limitation are preserved under
  `state/validation_seal_repairs/20260816T140100Z__3037884.json`. The subsequent full
  sealed audit and freeze passed.

- 2026-08-10: sustained six-pipeline screening execution is healthy. AE and VAE have
  each sealed generations 0-2 (144 usable trials/model) and have generation 3 queued
  or running. DSAE, DSVAE, DeepSVDD, and RealNVP have each sealed generations 0-4
  (240 usable trials/model) and have generation 5 queued or running. Every terminal
  chunk has been reconciled, fully artifact-authenticated, and followed by a
  transactionally consistent study backup before the next generation. No scientific
  trial failure or nonfinite objective vector has occurred. One mutable control-plane
  bug omitted the parent sampler payload during chunk-local validation; persisted
  Optuna generations, seeds, resolved samplers, and 48 unique parameter sets were
  independently verified correct. Commit `24eeed397aae93a636f9a21c2184537f57058ac1`
  fixes only that validator and adds a regression test; no scientific artifact or
  frozen execution code changed.

- 2026-08-08: clean NSGA-II execution campaign
  `physics_background_nsgaii_05b649c` is active at frozen commit
  `05b649c4439db6c7a3bfc5196addfc27b03dcc05`. The isolated forced-restart canary
  (`3036360`), all six unified one-epoch pilots (`3036377`, `3036378`, `3036391`,
  `3036392`, `3036403`, `3036404`), and all six four-trial packed one-epoch pilots
  (`3036441`, `3036442`, `3036458`, `3036459`, `3036490`, `3036491`) completed.
  Six disposable four-trial, 50-epoch production canaries are running as jobs
  `3036509`-`3036514`; after about 54 minutes their MLflow histories show AE at
  epochs 4-5, VAE at epoch 5, and DSAE/DSVAE/DeepSVDD/RealNVP at epochs 9-11, with
  no failure signature. The six final study databases remain pristine. Once the
  terminal canary artifact authenticates, submit the first 48-member generation for
  every model through the operations helper.

- 2026-08-09: all six disposable 4-trial x 50-epoch production canaries completed
  and passed the terminal authentication gate. Runtimes were AE 8:27:44, VAE
  7:34:38, DSAE 4:23:45, DSVAE 3:58:26, DeepSVDD 3:42:19, and RealNVP 3:47:00.
  `state/production_canary_validation.json` authenticates all 24 trial bindings, 224
  logical checkpoint selections, all 248 required last/primary candidates, 16 of 24
  optional strict-stable candidates, exact MLflow histories and score thresholds,
  and pristine final studies. Generation-0 search jobs were then submitted with 48
  trials on 12 nodes each: AE `3037884`, VAE `3037885`, DSAE `3037886`, DSVAE
  `3037888`, DeepSVDD `3037889`, and RealNVP `3037890`. All six are currently
  `PENDING (Priority)`; only AE initially has a scheduler estimate (`05:28:42` local).

- 2026-08-08: the search protocol was corrected before the next clean campaign. Stage
  1 is now six shared NSGA-II screening studies, not six high-dimensional M-TPE
  studies. The frozen base sampler contract is `NSGAIISampler(seed=123,
  population_size=48, mutation_prob=None, crossover_prob=0.9,
  swapping_prob=0.5, constraints_func=None)`. Every trial must persist
  `nsga2:generation`; portable reporting authenticates the resolved Hydra sampler,
  hashes the design mapping and seed policy, and exports the generation per sampled
  trial. A generation-less `FAIL` is accepted only with explicit binding evidence that
  it failed before sampling; stale repairs additionally require their hashed repair
  artifact. No generation is backfilled.
  Canaries use disposable databases and never enter the final studies. A healthy final
  study runs twelve complete 48-trial generations and a final 24 trials: exactly 600.
  Each sampler process receives a unique authenticated seed given exactly by
  `123 + model_offsets[model] + starting_optuna_trial_number`, with model offsets
  `ae=0`, `vae=100000`, `dsae=200000`, `dsvae=300000`, `svdd=400000`, and
  `realnvp=500000`. Allocated Optuna trial numbers are never reused during recovery.
  Stage 2
  will freeze a target manifest from screening and run independent, two-objective
  paper-style M-TPE studies only for the selected targets. Because the targets follow
  inspection of the all-20-anomaly oracle, this is exploratory, target-selection-biased
  follow-up rather than unbiased confirmation. The stages must use separate
  databases/artifact namespaces and must not be pooled.

- 2026-08-08: all campaigns and scheduler evidence listed below predate the NSGA-II
  correction. They remain useful implementation/runtime evidence but are scientifically
  superseded and operationally read-only: do not submit, repair, retrain, aggregate, or
  finalize into those roots. No clean NSGA-II campaign
  root, accepted NSGA-II production canary, full screening generation, or targeted
  confirmation study is recorded in this ledger yet.

- 2026-08-05: implementation commit
  `ed19c503538ef71ba61d9c6a230afb55872bf28e` closes the retraining control-plane
  audit: bounded retrain maps, fresh 40-epoch continuation jobs, separate epoch-200
  evaluation, exact MLflow/checkpoint recovery, immutable task receipts, independently
  gated aggregation, restart-provenance exports, and the forced USR1 canary. The 184-test
  campaign regression suite and every changed-file hook pass. Its detached worktree is
  `/iopsstor/scratch/cscs/vjimenez/adatl1/worktrees/physics_background_shared_ed19c50`;
  the fresh empty campaign is
  `/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/physics_background_shared_ed19c50`.
  Forced restart debug job `3010505` is running from that exact commit. No search or
  retraining job has been submitted to the new campaign.

- 2026-08-05: froze the next execution checkout at commit
  `1e0b94089e13ae64251387eb2141c9480537a17c` in
  `/iopsstor/scratch/cscs/vjimenez/adatl1/worktrees/physics_background_shared_1e0b940`
  and initialized its empty six-study campaign at
  `/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/physics_background_shared_1e0b940`.
  The design authenticates the existing pair-table preflight and targets exactly 600
  usable 50-epoch trials for each model. No jobs have been submitted to this campaign;
  it was superseded empty by `ed19c50` after the final retraining audit found scheduler
  array/requeue, immutable receipt, aggregate-gate, and report-provenance gaps.

- 2026-08-05: added a fail-closed operations-level Pareto freeze transition. The gate
  requires exact 600-trial quiescence for every study, authenticates all selected
  binding/metric/MLflow/checkpoint evidence, creates and hashes consistent backups of
  all six SQLite databases, invokes the commit-pinned front builder, independently
  reconstructs the 56 fronts and deduplicated retrain union, and seals
  `state/freeze_validation.json`. Existing manifests must pass the same checks.

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
| Code and configs | complete locally | Commit `ed19c50` contains the audited shared search, callback restart, bounded retrain, aggregate, and reporting paths; it is not pushed |
| Tests and preflight | complete | 184 campaign tests pass; every changed-file hook, shell syntax check, Python compilation, all six config compositions, and all 8 pair-table authentications pass |
| Autonomous runbook | complete | `PLAN.md` now specifies the full six-study state machine, artifact contract, scheduler/timer policy, failure recovery, portable exports, analysis, and final `REPORT.md` |
| Provenance/report hardening | complete | Added persistent Optuna-to-Hydra bindings, run/checkpoint/MLflow identities, reverse G-to-E drift, fail-closed artifact audits, and deterministic scientific-analysis tables in the final portable package |
| Operational control | complete | Separate mutable helper locks scheduler checks through submission/ledger registration, rejects live or unregistered search/retrain allocations, enforces the 720-minute limit, audits complete per-chunk Optuna/Hydra/MLflow/checkpoint state before sealing success, and provides backed-up scheduler-proven exact stale-trial repair |
| Pilots | complete | Corrected unified jobs AE `3008281`, VAE `3008282`, DSAE `3008321`, DSVAE `3008344`, DeepSVDD `3008351`, and RealNVP `3008360` passed; four-GPU temporary AE/Optuna pilot `3008377` passed all launcher, binding, objective, and threshold gates |
| Forced restart canary | running | Exact-commit debug job `3010505` must requeue once and emit authenticated `COMPLETE.json` before any new campaign search submission |
| Production canaries | evidence-only, partly terminal | Older-commit four-trial jobs AE `3008470`, VAE `3008471`, DSAE `3008472`, and RealNVP `3008475` are running; DSVAE `3008473` and DeepSVDD `3008474` completed cleanly. All six remain runtime evidence, not the `ed19c50` terminal campaign gate |
| Full searches | superseded before completion | Jobs AE `3007292`, VAE `3007311`, DSAE `3007310`, DSVAE `3007318`, DeepSVDD `3007340`, RealNVP `3007345`, and legacy `3005821`/`3005823` were cancelled; no usable trial completed and no bulk chunk was submitted |
| Retraining/evaluation | implemented, pending data | Five authenticated 40-epoch segments, separate evaluation, emergency requeue recovery, bounded arrays, and immutable per-task receipts pass local tests; real forced-requeue evidence is pending |
| Aggregation/report | implemented, pending data | Locked exact-marker aggregation writes `aggregate_validation.json`; final reporting independently validates restart provenance and exports `retrain_restart_evidence.csv`; campaign results are not yet available |

The `59b47b0c41bc`, `6edb3f7`, `cbeb3fa`, and `1e0b940` campaigns are superseded and
must never be resumed for the final study. No bulk search chunk was submitted. Only the
fresh `ed19c50` campaign can proceed after its forced restart and terminal six-model
canary gates pass.

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
- In screening, all logical strategies for one model share the same 600 NSGA-II sampled
  configurations. This gives six screening studies rather than 56 and enables direct
  strategy comparisons; logical two-dimensional fronts are reconstructed offline.
- Targeted confirmation is separate: each frozen selected target receives an
  independent two-objective M-TPE history. Confirmation trials never enter the six
  screening databases or retroactively alter their fronts.
- Normal jobs are node-exclusive. One Hydra/Optuna controller therefore packs four
  ordered, resource-exact one-GPU `srun` steps per node instead of wasting three GPUs
  or running competing controllers against one study.

## Verification log

- 2026-08-05: retrain submission now respects Clariden's `MaxArraySize=1001` even when
  the frozen union reaches 3,600 runs. Each non-overlapping submission uses local IDs
  `0..N-1` for at most 1,001 missing frozen indices and persists a content-addressed,
  manifest/row-authenticated mapping used by the batch runner, campaign, ledger, and
  reconciliation. Later recovery batches may remap an actual index without changing
  its scientific restart identity; submission identity is retained per attempt. Legacy
  direct-index ledger rows remain readable. Normal 40-epoch boundaries exit and resume
  through fresh mapped submissions because `MaxBatchRequeue=5`; only emergency USR1
  recovery self-requeues, with append-mode logs and `REQUEUED` treated as active.
  Epoch-200 training and downstream evaluation are separate tasks. Live-last/MLflow
  epoch recovery, atomic segment publication/orphan adoption, and idempotent audited
  marker finalization cover allocation crashes without silently restarting. The
  persistent MLflow run logs parameters only on its first segment, carries the required
  source-commit/run identity tags, and serializes concurrent experiment creation.

- 2026-08-05: hardened the mutable control plane before bulk transition. Search and
  retrain submissions now share an interprocess lock across user-scoped live Slurm
  checks, `sbatch`, and ledger registration; search timeouts cannot exceed 720 minutes.
  Successful packed reconciliation validates only that chunk's COMPLETE bindings,
  schema-2 objective/provenance contract, finished 50-epoch MLflow histories, and
  required candidate/threshold state before hashing `chunk_validation.json` as
  `success_sha256`. The new explicit `repair-stale` path requires exact trial IDs,
  terminal scheduler proof, and an authenticated pre-mutation SQLite backup before
  using Optuna's API, and persists repair/refusal evidence. This mutable-checkout change
  does not alter the frozen `cbeb3fa` execution worktree or campaign science. All 26
  focused operational-control cases and all changed-file hooks pass.

- 2026-08-05: final reporting now authenticates all 40,800 configured physical search
  checkpoint slots across the authoritative first 600 usable trials per model. It
  requires and hashes all 37,200 last/primary candidates, records optional
  stable-native availability, validates native/OAS threshold namespaces, rejects
  escaped or shared run roots, and exports candidate-level plus grouped portable CSVs.
  Seventy-three focused reporting, operations, analysis, and artifact-audit tests pass;
  all changed-file hooks pass. Local reporting commit: `d274aba`.

- 2026-08-05: successful 200-epoch retrain array tasks now seal their validated result
  marker in the scheduler ledger. This complements selective missing-index recovery and
  makes terminal task provenance self-authenticating. Eighteen operational tests pass.

- 2026-08-05: successful packed search/canary reconciliation now seals the terminal
  chunk manifest and exact Optuna-to-Hydra binding file in addition to the Slurm log.
  The canary gate requires all three hashes. Seventeen operational-control tests pass.

- 2026-08-05: reconciled the paper's three-candidate prose with its `dev/patrick`
  implementation. Stable-native is configured for every trial but is absent when no
  10-epoch window reaches less than 5% variation; the reference evaluator then compares
  last and primary without a fallback. The terminal gate now audits 272 configured
  slots, requires all 248 last/primary checkpoints, and records stable availability.

- 2026-08-05: hardened bulk submission so no individual model can start until the
  authenticated six-model canary artifact exists, and so a final chunk cannot exceed
  the 600-usable-trial target. Sixteen focused operational-control tests now pass.

- 2026-08-05: added a terminal production-canary validator to the mutable control
  plane. It requires six successful hash-sealed jobs, 24 usable 50-epoch trials, exact
  bindings and finite histories, 224 checkpoint selections, and 272 retained candidate
  checkpoints with score-specific threshold state before bulk submission is allowed.
  Thirteen focused operational-control tests and all changed-file hooks pass.

- 2026-08-05: persisted a per-worker runtime projection from actual MLflow summary
  timestamps after 2-6 completed epochs. Native-only canaries project to finish between
  19:53 and 20:30 CEST, AE around 23:56 CEST, and VAE around 00:54 CEST. Even the
  slowest VAE projection retains more than three hours before its 12-hour deadline.
  These are monitoring estimates only; bulk submission remains gated on terminal
  50-epoch authentication.

- 2026-08-05: added the reporting-side operational helper without changing the frozen
  scientific campaign. Five focused tests and every changed-file hook pass. Its live
  reconciliation path authenticated the exact `cbeb3fa` execution worktree, updated
  all registered jobs from Slurm JSON with unambiguous UTC timestamps, sealed completed
  logs by SHA-256, and left active logs unhashed. A real attempted overlapping AE
  submission was rejected before `sbatch`, as required. The plan now reflects the
  user's latest instruction not to create timers when the queue is the sole blocker.

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

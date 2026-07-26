# Causal Chamber Production Campaign Handoff

Last updated: 2026-07-26 on Clariden.

## Live Clariden Update — 2026-07-26

This section supersedes the older execution snapshot below.

- Search job `2898701` completed all 65 tasks: exactly 1,300 authenticated
  200-epoch trajectories, 1,300 unique FINISHED MLflow runs, and 16,900 finite
  metric histories.
- Selection completed with 20 winners and an exact 200-row retrain manifest.
  Intervention labels were not used. The retrain-manifest SHA-256 is
  `5768cdb818d1163827c398d98337213fbad0d84936f9507e0dfde8474d892f55`.
- Safe retraining completed all 200 model/strategy/reporting-seed identities.
  Every marker authenticates the frozen training commit `63b941a`, its selected
  checkpoint hash, and a unique FINISHED MLflow run.
- Outcome-independent physical characterization completed all 58 interventions,
  638 readout effects, and 58 expected-descendant summaries under
  `audits/cchamber_real_20260725_63b941a_physical_shift_6cd53fe`.
  Zero-variance and bootstrap-degenerate standardized effects are explicitly
  marked undefined; no infinity was clipped or replaced by a finite value.
- The final sidecar deployment is
  `/iopsstor/scratch/cscs/vjimenez/adatl1/deployments/cchamber_20260726_d1a3017`
  at commit `d1a30173fe8250fa3a681f393aaa4645d894862b`.
- Threshold calibration passed one real GH200 canary for each of AE, VAE, Deep
  SVDD, and RealNVP. Production jobs calibration `2901192`, freeze `2901209`,
  evaluation `2901210`, and collection `2901211` all completed with exit code
  zero. The threshold root contains exactly 200 validation-derived thresholds
  and 23,200 finite, uniquely keyed result rows.
- The replacement candidate-rank workflow—canary `2901177`, timing `2901178`,
  training `2901179`, freeze `2901180`, evaluation `2901181`, collection
  `2901182`, and analysis `2901183`—completed with exit code zero. It contains
  192 shared trajectories, 960 frozen checkpoint branches, 111,360 sealed
  outcome rows, and 10,000-permutation/10,000-bootstrap rank inference.
- Five candidate-rank MLflow experiments were created sequentially before the
  production array to prevent filesystem-store experiment-creation races.
- Frozen paper analysis job `2901407` completed on the debug partition as a CPU
  job. Its bundle authenticated the threshold, selection, candidate-rank, and
  background-acceptance chains before producing seed-first inference.
- Final information-dense report:
  `/users/vjimenez/adatl1/results/reports/cchamber_real_20260726_d1a3017/FINAL_REPORT.md`.
  It uses three main figures: architecture/criterion performance,
  candidate-rank validity, and physical-shift/detectability/CAP-gain synthesis.
- Primary conclusion: CAP is strongly supported for RealNVP, partially and
  endpoint-dependently supported for AE/VAE, and contradicted for SVDD. Do not
  replace this architecture-specific conclusion with a universal CAP claim.
- The post-confirmatory linear-Gaussian bridge is complete under
  `results/reports/cchamber_real_20260726_d1a3017/gaussian_bridge_final`. It
  rejects a full 11-readout Gaussian model and retains only an explicitly
  approximate two-PC optical subspace. `uniform_red_mid` is nearly aligned with
  the CAP direction and reaches AUPRC `.974` versus oracle `.976`; the
  matched-strength near-orthogonal `uniform_blue_mid` reaches only `.536`
  versus oracle `.977`. This is a mechanistic alignment stress test, not a
  confirmatory result, and it does not revise the completed campaign.

## Post-Handoff Updates

The following updates supersede the working-tree snapshot later in this file:

- `c4f675a`: provenance-safe candidate-rank audit committed after independent
  review, 20 focused/regression tests, and all hooks.
- `7207044`: threshold-safe MLflow/Slurm evaluation committed after three review
  rounds, 24 focused/adversarial tests, and all hooks.
- Immutable sidecar deployment:
  `/iopsstor/scratch/cscs/vjimenez/adatl1/deployments/cchamber_20260725_e8d35a2`
  at commit `e8d35a25410148457fb4e080096cb9e80bba1c2a`; exact-deployment
  compilation and 33 combined threshold/candidate-audit tests passed.
- That exact deployment also passed the synthetic cloud gate: Python 3.10,
  frozen uv lock, clean Git, writable scratch paths, all 76 paper configurations,
  all 76 generated Slurm scripts, and the L1-shaped synthetic data contract.
- The worktree was clean immediately after `7207044`.
- The threshold-safe command path is code-ready, but inventory/canary execution
  remains intentionally gated until search selection and all 200 retrains exist.
- At 2026-07-25 22:26 CEST, the main search had at least 315/1,300 markers, eight
  running tasks, and zero retry markers. Always refresh this live count.

This file is the durable handoff for the active, real Causal Chamber paper
campaign. Do not reduce this work to a smoke test. The objective is to complete
the full experiment, collect the sealed intervention results, investigate them,
produce plots and statistical conclusions, and explain honestly how they support
the CAP and L1 narrative.

## Ultimate Research Goal

Establish whether CAP is a useful signal-agnostic model-selection criterion on a
controlled physical system:

1. Compare CAP with metadata-nearest, encoder-nearest, and random pairing against
   threshold-drift and Wasserstein validation.
2. Report the result independently for AE, VAE, Deep SVDD, and RealNVP.
3. Exploit the known Causal Chamber interventions to characterize controlled
   physical shifts by intervention target, family, and strength.
4. Determine whether physically meaningful pairing makes CAP more predictive of
   downstream anomaly detection than random pairing.
5. Relate this to L1 anomaly detection as a controlled physical analogue of
   background-only selection, without claiming that Causal Chamber is equivalent
   to collision data.

The primary Causal Chamber endpoint is intervention-weighted mean AUPRC over ten
paired reporting seeds. Efficiency at a prespecified 1% false-positive rate
(`q=0.99`) is secondary.

## Source-Accurate L1 Interpretation

`28031_Informative_Model_Valida.pdf` does **not** use 1% as its main L1
operating point:

- Main L1: 250 Hz, approximately `q=0.999991`.
- L1 Appendix D: `q=0.99` is a deliberately looser sensitivity study; method
  separation becomes less sharp.
- CIFAR-10: `q=0.99` is used because it is selective but statistically stable
  for a small benchmark.
- Causal Chamber: `base_rate=null`, so 1% is a sample-level false-positive rate,
  not a physical trigger rate in Hz.

Allowed language: “a controlled physical, L1-like fixed-background-budget
experiment.” Do not call the Causal Chamber endpoint 250 Hz, an extreme-tail L1
validation, or a deployed trigger-rate measurement.

Theory gives monotonic true-positive-rate results at any fixed false-positive
rate only under its alignment assumptions. It does not justify extrapolating a
Causal Chamber result at 1% to the L1 extreme tail.

## Frozen Main Campaign

Repository:

```text
/users/vjimenez/adatl1
branch: research/main
```

Immutable production deployment used for pairing, calibration, search, selection,
and retraining:

```text
/iopsstor/scratch/cscs/vjimenez/adatl1/deployments/cchamber_20260725_63b941a
commit: 63b941a287c48c84e2537d0cfbd07c2240435c0e
```

Campaign root:

```text
/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/cchamber_real_20260725_63b941a
campaign.json SHA-256:
70a43bc5555a9c5d3bda5c35cfffe7387619e89f83e4876774c2bff7c1d3b441
```

MLflow:

```text
file:/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/cchamber_real_20260725_63b941a/logs/mlflow/mlruns
```

Data:

- Real public `lt_interventions_standard_v1`, not synthetic data.
- 11 readout features are visible to anomaly detectors.
- Intervention knobs and labels are not model inputs.
- 58 intervention CSVs plus `uniform_reference.csv`.
- Dataset archive MD5: `476664d024f88e8b7640998bb5e9ee33`.
- Fixed data seed: `314159`.

Pairing:

- Five independently trained pairing encoders: seeds
  `123, 456, 789, 101112, 131415`.
- Seed 123 is the prespecified primary encoder.
- Primary validation table SHA-256:
  `c56265652d7ed4b8df588cd4ad56d9209967d73c8f206f8d55918d31d2add497`.
- Primary test table SHA-256:
  `3fcc5af28ed03c3d7eac5ea23229dcf9835c8a7e0e6c5a413b9ae7784d3fe6fb`.
- Validation and test each contain 1,000 distinct, fully covered pairs.

## Frozen Experimental Design

- Models: AE, VAE, Deep SVDD, RealNVP.
- Candidate pool: checked-in baseline `000` plus 64 scrambled Sobol candidates
  per architecture.
- Development seeds: `101, 202, 303, 404, 505`.
- Search fits: `4 × 65 × 5 = 1,300`.
- Every 200-epoch trajectory logs the same five primary label-free criteria:
  CAP metadata, CAP encoder, CAP random, drift, and Wasserstein.
- It also logs four additional random-pair and four additional encoder-pair CAP
  sensitivity histories, for 13 histories total.
- Selection: mean within-development-seed rank, with direction-aware ranking,
  from one shared surviving candidate pool.
- Reporting: 10 independent seeds for each of 4 models × 5 selection strategies.
- Final retrains: `4 × 5 × 10 = 200`.
- Final sealed table: `200 × 58 interventions × 2 metrics = 23,200` rows.

Primary inference:

- Metadata CAP and encoder CAP are separately compared with random CAP using
  exact paired sign-flip tests and Holm correction.
- Metadata versus encoder CAP equivalence uses paired TOST with an absolute AUPRC
  margin of 0.02 and Holm correction.
- Seed is the unit of paired inference; do not pool events as independent.

## Live Execution Snapshot

Slurm account is `a0166`. Use `debug` for GPU canaries and `normal` for
production. Never train or run heavy tests on a login node.

Completed:

- Pairing job `2898641`: complete and audited.
- Full-fidelity four-model calibration job `2898658`: complete.
- First eight search array tasks: complete with exit code 0.

Active:

```text
search array job: 2898701
snapshot at 2026-07-25 21:52 CEST: 236 / 1,300 result markers
8 tasks RUNNING, remaining tasks PENDING behind the array concurrency limit
retry/failure markers: 0
```

Refresh the state instead of trusting the snapshot:

```bash
CAMPAIGN_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/cchamber_real_20260725_63b941a
squeue -j 2898701
sacct -j 2898701 --starttime 2026-07-25 -X \
  -o JobIDRaw,State,ExitCode,Elapsed
find "$CAMPAIGN_ROOT/search_results" -type f -name 'seed_*.json' | wc -l
find "$CAMPAIGN_ROOT/search_attempts" -type f 2>/dev/null | wc -l
```

A completed marker is acceptable only if it is unique, has MLflow status
`FINISHED`, and contains 200 finite points for all five primary plus eight
sensitivity histories. Completion requires exactly 1,300 valid markers and all
65 array tasks completing successfully.

## Exact Main-Campaign Next Steps

Do not collect or select until all 1,300 search fits are validated.

Then use the immutable `63b941a` deployment:

```bash
DEPLOYMENT=/iopsstor/scratch/cscs/vjimenez/adatl1/deployments/cchamber_20260725_63b941a
CAMPAIGN_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/cchamber_real_20260725_63b941a

cd "$DEPLOYMENT"
uv run --frozen --no-sync python scripts/cchamber_campaign.py \
  collect-candidates --root "$CAMPAIGN_ROOT"
uv run --frozen --no-sync python scripts/cchamber_campaign.py \
  select --root "$CAMPAIGN_ROOT"
```

Validate and hash:

```text
selection/candidate_metrics.csv
selection/candidate_metrics_provenance.json
selection/selected_trials.csv
selection/retrain_manifest.json
selection/selection_provenance.json
```

Only after exact selection validation, submit:

```bash
sbatch "$CAMPAIGN_ROOT/slurm/retrain.sbatch"
```

This creates 200 selected reporting checkpoints. Retraining at `63b941a` is
scientifically valid even though its strategy configs disable the legacy anomaly
efficiency callback; the threshold is calibrated afterward from
`validation.normal`.

### Critical prohibition

Do **not** submit or use:

```text
$CAMPAIGN_ROOT/slurm/evaluate.sbatch
scripts/cchamber_campaign.py run-evaluate
scripts/cchamber_campaign.py collect-final
```

from commit `63b941a`. That legacy path assumes a checkpoint threshold that the
selected retrains generally do not contain. It cannot prove that normal
acceptance and intervention efficiency share the same threshold.

Use the new threshold-safe sidecar described below.

## Threshold-Safe Final Evaluation

The sidecar under development is
`scripts/cchamber_operating_point_audit.py`, with callback support in
`src/evaluation/callbacks/efficiency.py`.

Required workflow after all 200 retrains:

1. Authenticate campaign, pairing, selection, retrain manifest, all 200 retrain
   markers, and all checkpoint hashes.
2. Run 200 validation-only calibrations. Each uses exactly
   `validation.normal`, 1,000 rows, `q=0.99`, interpolation `higher`, and
   comparator `score >= threshold`.
3. Record sample/tensor/pair-table fingerprints, threshold rank, tie count,
   validation triggered count, exact float32 threshold bytes, and their SHA.
4. Freeze a complete 200-record threshold manifest before reading any test or
   intervention outcome.
5. Strictly load each original checkpoint, inject its frozen threshold only
   after loading, and run one test pass containing `test.normal` and all 58
   interventions.
6. Emit normal-FPR diagnostics, 58 AUPRC values, and 58 intervention efficiencies
   carrying the identical checkpoint and threshold hashes.
7. Collect only exact `200 × 58 × 2 = 23,200` coverage.

No test-normal acceptance may change a threshold, exclude a model, revise
selection, or trigger recalibration.

At the handoff timestamp, this implementation is uncommitted and undergoing
focused tests. Do not deploy it until tests, independent review, a clean commit,
and a debug GPU canary pass.

## Candidate-Rank Audit

Purpose: determine whether each label-free proxy actually ranks candidate models
in the same order as sealed downstream performance, rather than judging only the
single selected winner.

Frozen design artifacts:

```text
design/candidate_audit_panel.json
SHA-256 bebeb486c3c55e32ee4e78d4ef2396c6f45212982cf550c8564564f8dd3f4f3d

design/candidate_audit_execution_contract_v1.json
SHA-256 b48d7b80745651223b87bca6516ca89ee35bbe98fea56d6c2a1cc3f493b209e3
```

Design:

- Candidates:
  `000,001,006,010,015,019,024,028,033,037,042,046,051,055,060,064`.
- Four models × 16 candidates × reporting seeds `1001,1002,1003` =
  192 shared 200-epoch trajectories.
- Five simultaneous checkpoint branches per trajectory = 960 checkpoints.
- All 960 hashes must freeze before any intervention evaluation.
- Sealed output: `192 × 5 × 58 × 2 = 111,360` rows.
- Rank association: Spearman primary, Kendall robustness, per-seed Spearman,
  top-4 overlap/enrichment, branch regret, hierarchical paired bootstrap,
  candidate-label permutation, and separate 20-test Holm families for AUPRC and
  1% efficiency.

Implementation files:

```text
scripts/cchamber_candidate_rank_audit.py
src/callbacks/audit.py
configs/experiment/cchamber/*candidate_rank_audit*.yaml
tests/test_cchamber_candidate_rank_audit.py
```

The implementation now authenticates the production candidate provenance,
per-model pool hashes, common surviving pool, seed-123 pairing manifest,
pairing-encoder checkpoint, pair-table metadata, branch manifests, and every
sealed row against the frozen checkpoint.

It remains intentionally blocked until main search collection creates
`selection/candidate_metrics_provenance.json`.

After committing and creating a clean sidecar deployment, first create the audit
design from the authenticated selection and pairing artifacts. The generated
debug canary command is:

```bash
sbatch <AUDIT_ROOT>/slurm/debug_fingerprint_canary.sh
```

The generated request is account `a0166`, partition `debug`, one GPU, 110 GB, and
30 minutes. It must prove that five checkpoint callbacks do not alter the shared
training trajectory before the normal-partition arrays may be submitted.

## Other Frozen Outcome-Blind Analysis Artifacts

All are under the main campaign’s `design/` directory and must not be edited.
Amendments must be new versioned artifacts.

```text
candidate_audit_panel.json
  bebeb486c3c55e32ee4e78d4ef2396c6f45212982cf550c8564564f8dd3f4f3d
paper_analysis_plan_v1.json
  db2a56864d8de21d0b42d4ab3ddd46e56de4ee66d82d5ab19489f67605086633
physical_intervention_catalog_v1.json
  824c81636279a29f6bca8c5216183bc3d490f92ed2db060db3fdfb5af13e2518
physical_shift_estimand_v1.json
  eb071c9ba5052d3d7830813358baa7dbed25a4c612b8c9f31a1069e6a45a9594
paper_analysis_execution_plan_v1.json
  c481cf18d8c747a0542f08fa454940821bf75017f4027ab3d30a3008b8e8a34a
paper_analysis_taxonomy_v1.csv
  218d336a1f95f4cfeb779a62fbc2369e0712c0e1df8e4bea99a1226e8e1ba22d
postselection_analysis_freeze_manifest_v1.json
  c4d6fbdba7070ef4d64b2321014ac5b6acff686c108b2facede3dd9f5c61ad12
candidate_audit_execution_contract_v1.json
  b48d7b80745651223b87bca6516ca89ee35bbe98fea56d6c2a1cc3f493b209e3
```

Sidecar analysis tools already committed on `research/main`:

```text
ea5b81b  corrected paper analysis
75f8712  physical shift characterization
1994bb1  operating-point audit foundation
45d627f  source-accurate 1%/250-Hz documentation
```

Run physical-shift characterization only after selection is frozen. It may read
physical intervention readout distributions but must not change selection.

## Working Tree At Handoff

Current committed HEAD:

```text
45d627f1ffb9d1dbb2fcbab95673b668db7a3eb8
```

Expected uncommitted agent work at the snapshot:

```text
modified:
  scripts/cchamber_operating_point_audit.py
  src/evaluation/callbacks/efficiency.py
  tests/test_cchamber_operating_point_audit.py

untracked:
  tests/test_cchamber_threshold_handoff.py
  scripts/cchamber_candidate_rank_audit.py
  src/callbacks/audit.py
  tests/test_cchamber_candidate_rank_audit.py
  configs/experiment/cchamber/*candidate_rank_audit*.yaml
```

These are intentional concurrent agent changes. Do not discard, reset, or
overwrite them. Review and commit the candidate audit and threshold handoff as
separate logical commits after their focused tests finish.

## Completion Definition

The campaign is not complete until all of the following are proven:

- 1,300 valid MLflow search fits.
- Authenticated candidate collection and frozen selection.
- 200 valid independent reporting retrains.
- 200 immutable validation-derived thresholds frozen before testing.
- Exact test-normal and 58-intervention evaluation with shared threshold hashes.
- Exact 23,200-row primary result table.
- Candidate-rank, pairing-stability, physical-shift, and operating-point audits.
- Prespecified paired inference and multiplicity corrections.
- Architecture-specific and intervention-family/strength plots.
- A written report stating positive, null, and contradictory evidence without
  forcing the desired narrative.
- A source-accurate L1 comparison distinguishing the Causal Chamber 1%
  false-positive rate from the main 250 Hz L1 endpoint.

Do not mark the research goal complete merely because jobs finish. Inspect the
results, validate every provenance/coverage contract, generate the deliverables,
and determine what the evidence actually supports.

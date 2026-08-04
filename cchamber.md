# Causal Chamber Experiments

## Role In The Paper

Causal Chamber is the bridge between the analytical experiments and the L1AD physics
data. Like the analytical setting, the experiment labels tell us which intervention
was performed. Like the physics setting, the anomaly models see low-dimensional real
sensor readouts with interpretable coordinates.

The code must always use the real Causal Chamber CSVs. There is no generated or
synthetic Causal Chamber mode.

## Dataset

The default config is `configs/data/causal_chamber.yaml`. It downloads
`lt_interventions_standard_v1` into `data/causal_chamber` when needed.

The main file is `uniform_reference.csv`. It is split into:

- `train`: normal real rows used to fit the anomaly detector;
- `normal`: held-out real rows used as the first validation/test domain;
- `reference_normal`: disjoint held-out real rows used as the paired reference domain.

The intervention CSVs, for example `uniform_red_mid` or `uniform_osr_angle_1_strong`,
are used as signal/anomaly domains for reporting.

The anomaly model input is readout-only by default:

```text
current, angle_1, angle_2,
ir_1, vis_1, ir_2, vis_2, ir_3, vis_3,
v_board, v_reg
```

Metadata and non-readout numeric columns are retained by the dataset builder for
pairing and diagnostics. The model does not receive them unless `feature_set` is
changed.

## Pairing

Validation/test CAP compares `normal` and `reference_normal`. The datasets are
already emitted in paired order, so the Causal Chamber CAP callback uses
`pairing_type=none` except for the encoder-fixed condition.

Supported real-data pairing strategies:

- `metadata_nearest`: split held-out `uniform_reference` rows into two disjoint
  pools, standardize the non-readout pairing columns, and greedily match one-to-one
  nearest neighbors. FAISS is used when installed, with a torch fallback.
- `random`: use the same disjoint real pools, but randomly permute the reference
  pool. This is the negative control for pairing quality.
- `encoder_nearest`: train a larger AE on real `uniform_reference` training rows,
  freeze it, build fixed nearest-neighbor pair tables in AE latent space, and run
  CAP with `pairing_type=mapping`.
- `cdf`: pair the two validation score samples by deterministic empirical-CDF
  rank. This assignment is recomputed from the current model scores and has no
  pairing seed or learned pairing table.

The intended paper comparison is:

```text
metadata_nearest ~= encoder_nearest > random
```

where the ordering refers to how useful/stable CAP should be as a model-selection
proxy.

## Encoder-Fixed Pair Tables

The pairing encoder is not a new algorithm. It is the existing `ae` algorithm with a
larger Causal Chamber experiment config:

```bash
uv run python src/train.py experiment=cchamber/ae_pairing seed=123
```

Build validation and test pair tables from the selected AE checkpoint:

```bash
uv run python -m src.utils.pairing.build_pair_table \
  --ckpt /path/to/cchamber_ae_encoder.ckpt \
  --out data/causal_chamber/pairs/valid_ae_pairs.pt \
  --stage validate \
  --dataset-1 normal \
  --dataset-2 reference_normal \
  --pairing-mode one_to_one_nearest \
  --k 0 \
  --no-caliper \
  experiment=cchamber/ae_pairing \
  data.pairing_strategy=metadata_nearest \
  data.max_val_batches=-1 \
  'data.signal_experiments=[]'
```

```bash
uv run python -m src.utils.pairing.build_pair_table \
  --ckpt /path/to/cchamber_ae_encoder.ckpt \
  --out data/causal_chamber/pairs/test_ae_pairs.pt \
  --stage test \
  --dataset-1 normal \
  --dataset-2 reference_normal \
  --pairing-mode one_to_one_nearest \
  --k 0 \
  --no-caliper \
  experiment=cchamber/ae_pairing \
  data.pairing_strategy=metadata_nearest \
  data.max_val_batches=-1 \
  'data.signal_experiments=[]'
```

`--k 0` lets the one-to-one matcher expand the FAISS/torch neighbor search until
coverage saturates. The campaign deliberately constructs all three CAP conditions
from the same canonical metadata-ordered source pools. The encoder table builder
then rematches every row from scratch in latent space: it does not consume pair IDs
or metadata distances. This keeps source-tensor fingerprints directly comparable
without making the encoder assignment depend on metadata matching.

Pair-table files are versioned and tied to the encoder checkpoint, split, ordered
source tensors, source sizes, dataset names, and data seed with SHA-256
fingerprints. Reusing a table after any of those inputs changes now fails instead
of silently filtering indices.

## Paper Experiment Matrix

Run these for each anomaly detector: AE, VAE, SVDD, RealNVP.

```text
cchamber_<model>_cap_metadata_nearest
cchamber_<model>_cap_encoder_nearest
cchamber_<model>_cap_cdf
cchamber_<model>_cap_random
cchamber_<model>_drift
cchamber_<model>_wasserstein
```

The four CAP conditions are the main Causal Chamber result. Drift and Wasserstein are
included as non-CAP validation baselines.

For the 2026-08-02 SVDD rerun, residual-OAS/CDF selects AE candidate `060` using
validation-normal information only. Its `[24, 8]` encoder weights initialize a
bias-free SVDD of the same topology; only the incompatible AE bias tensors are
dropped. Every transferred weight remains trainable. Reporting seed controls
`data.train_seed` independently of the fixed split seed, and the freeze gate
requires three distinct minibatch-order and final-state fingerprints per
candidate. The outcome-gated score audit retains the native radial SVDD score
but reverses CAP's direction: SVDD minimizes normal-view CAP because invariance,
not capacity, aligns with its one-class objective.

## Generating Runs

List the configured Causal Chamber experiments:

```bash
uv run python scripts/generation.py list --dataset cchamber
```

Generate Optuna sweep scripts with several reporting seeds:

```bash
uv run python scripts/generation.py generate \
  --dataset cchamber \
  --stage sweep \
  --n-trials 100 \
  --seeds 123,456,789
```

Run metadata-nearest or random CAP directly:

```bash
bash scripts/generated/cchamber_ae_cap_metadata_nearest/sweep.sh
bash scripts/generated/cchamber_ae_cap_random/sweep.sh
```

Run encoder-nearest CAP by exporting the fixed pair tables:

```bash
CCHAMBER_VALID_PAIR_TABLE=data/causal_chamber/pairs/valid_ae_pairs.pt \
CCHAMBER_TEST_PAIR_TABLE=data/causal_chamber/pairs/test_ae_pairs.pt \
bash scripts/generated/cchamber_ae_cap_encoder_nearest/sweep.sh
```

Use `--model ae`, `--model vae`, `--model svdd`, or `--model realnvp` to restrict
generation to one model family.

## Hyperparameter Tuning And Statistics

The generator records tuned parameters from the selected `hparams_search` YAML and
keeps fixed factors in each manifest. For the paper, run sweeps, select the best
trial overrides, retrain the selected configurations across multiple seeds, and
evaluate the resulting checkpoints.

Retrain from selected trial overrides:

```bash
uv run python scripts/generation.py generate \
  --name cchamber_ae_cap_metadata_nearest \
  --stage retrain \
  --selected-overrides scripts/selected/cchamber_ae_cap_metadata_nearest.json
```

Evaluate fixed checkpoints:

```bash
uv run python scripts/generation.py generate \
  --name cchamber_ae_cap_metadata_nearest \
  --stage evaluate \
  --ckpt-manifest scripts/checkpoints/cchamber_ae_cap_metadata_nearest.json
```

For encoder pairing, repeat the AE-pairing training for several model seeds and
compare the resulting validation pair tables. Keep the source-data seed fixed:

```bash
uv run python -m src.utils.pairing.compare_pair_tables \
  --tables data/causal_chamber/pairs/seed*/valid_ae_pairs.pt \
  --out outputs/cchamber_pair_table_seed_comparison.json
```

Report per-intervention anomaly metrics over `data.signal_experiments`, aggregate
over seeds, and include pair-table metadata such as coverage and latent distances
for the encoder-nearest condition.

The repository now enforces this handoff with `scripts/paper_pipeline.py`:

```bash
# Select with label-free validation metrics only. Every strategy must replay
# the same candidate IDs.
uv run python scripts/paper_pipeline.py select \
  --candidate-metrics results/cchamber/candidate_metrics.csv \
  --output-dir results/cchamber/selection

# Generate one retraining script per selected dataset/model/strategy.
uv run python scripts/generation.py generate \
  --dataset cchamber \
  --stage retrain \
  --selected-overrides results/cchamber/selection/retrain_manifest.json \
  --seeds 123,456,789

# Resolve completed retraining runs and generate final evaluation scripts.
uv run python scripts/paper_pipeline.py checkpoints \
  --selected-trials results/cchamber/selection/selected_trials.csv \
  --checkpoints-dir checkpoints \
  --output results/cchamber/checkpoints.json
uv run python scripts/generation.py generate \
  --dataset cchamber \
  --stage evaluate \
  --ckpt-manifest results/cchamber/checkpoints.json

# Annotate callback values with a small manifest, then aggregate.
uv run python scripts/paper_pipeline.py collect \
  --manifest results/cchamber/callback_manifest.csv \
  --output results/cchamber/results.csv
uv run python scripts/paper_pipeline.py aggregate \
  --results results/cchamber/results.csv \
  --output-dir results/cchamber/paper
```

`candidate_metrics.csv` has columns
`dataset,model,seed,candidate_id,strategy,value,params_json`. The collection
manifest has columns `path,dataset,model,strategy,seed` and optional `pairing`;
`path` points to an evaluator callback `values.csv`. Selection rejects downstream
metrics and mismatched candidate pools. Aggregation rejects incomplete paired
seed/intervention coverage.

## Production Campaign

The paper campaign uses `scripts/cchamber_campaign.py`, not independent
strategy-specific Optuna studies. Its frozen design is:

- 64 scrambled Sobol configurations per detector plus the separately labelled
  checked-in baseline;
- AE, VAE, SVDD, and RealNVP;
- five development model seeds with one fixed data split seed;
- all six label-free criteria recorded from the same training trajectory;
- five prespecified random-pairing seeds and five independently trained encoder
  pair tables recorded on every trajectory for pairing-proxy sensitivity;
- one candidate per detector/criterion selected by mean within-seed rank;
- ten independent reporting model seeds per selected configuration; SVDD keeps
  the pretrained initialization and data split fixed while the reporting seed
  controls the fine-tuning minibatch order;
- five pairing-encoder seeds, with seed 123 prespecified as primary and the
  others used only for pairing-stability analysis;
- all 58 interventions sealed during search and retraining, then evaluated once
  at the final checkpoint stage for both AUPRC and secondary anomaly efficiency
  at the prespecified 1% false-positive rate (\(q=0.99\)).

The 1% target is the direct sample-level background acceptance
(\(q=0.99\), with no physical base rate) used for the paper's statistically
stable small-benchmark setting. It is not the main LHC L1 operating point:
the main L1 study reports efficiency at 250 Hz, while \(q=0.99\) appears only
as a looser L1 sensitivity study in Appendix D of
`28031_Informative_Model_Valida.pdf`. Causal Chamber results may therefore be
described as an L1-like fixed-background-budget experiment, but not as a
measurement at the physical L1 trigger rate.

This is 1,300 search fits and 240 final retrains. Clariden allocates a complete
four-GPU node even for a one-GPU request, so generated campaign scripts pack four
independent fits per node. Every fit uses MLflow and carries the campaign, commit,
candidate-pool, seed, checkpoint, and pair-table fingerprints needed to reproduce
the handoff.

Create the immutable design only from a clean deployment worktree:

```bash
uv run --frozen --no-sync python scripts/cchamber_campaign.py design \
  --root "$CAMPAIGN_ROOT" \
  --campaign-id "$CAMPAIGN_ID" \
  --n-candidates 65
```

The resulting `slurm/` folder contains separate pairing, calibration, search,
retrain, and final-evaluation launchers. Gates are sequential: pairing tables
must pass the cross-seed audit before calibration; the MLflow calibration must
pass before the array search; candidate collection and selection must be
complete before retraining; and final aggregation refuses anything other than
the complete 4 × 5 × 10 × 58 × 2-metric result contract.

The primary inferential estimand is the intervention-weighted mean AUPRC within
each of ten paired reporting seeds. Metadata-nearest and encoder-nearest are
tested separately for superiority over random pairing with exact paired sign-flip
tests and Holm correction. Metadata-nearest versus encoder-nearest equivalence is
prespecified at an absolute AUPRC margin of 0.02 using paired TOST with Holm
correction across detector families. Equal-family weighting, intervention family,
target, strength, and family-by-strength summaries are reported as complementary
descriptive analyses. Conclusions remain conditional on the fixed public dataset
split and the prespecified Sobol candidate pool.

## Revised 2026-08-02 AE and SVDD Results

The residual-OAS AE campaign and the independently seeded inverse-CAP SVDD
campaign each contain 16 candidates, three reporting seeds, six checkpoint
branches, and all 58 interventions: 48 trajectories, 288 frozen checkpoints,
and 33,408 complete intervention-metric rows per architecture. VAE and RealNVP
remain unchanged from campaign `cchamber_real_20260801_3789655`.

Values average interventions within reporting seed before averaging seeds.
`Eff.` is efficiency at a threshold calibrated to 1% validation-normal
acceptance. New AE/SVDD intervals use three retrains; unchanged VAE/RealNVP
intervals use ten.

| Architecture | Selection criterion | AUPRC | Eff. | Retrains |
|---|---|---:|---:|---:|
| SVDD | CAP encoder-nearest | 0.5229 | 0.2608 | 3 |
| SVDD | CAP CDF ranks | 0.4320 | 0.0459 | 3 |
| SVDD | Marginal drift | 0.5110 | 0.2507 | 3 |
| SVDD | Wasserstein | 0.5256 | 0.2626 | 3 |
| AE | CAP encoder-nearest | 0.7238 | 0.3813 | 3 |
| AE | CAP CDF ranks | 0.7355 | 0.4609 | 3 |
| AE | Marginal drift | 0.7296 | 0.4407 | 3 |
| AE | Wasserstein | 0.7347 | 0.4370 | 3 |
| VAE | CAP encoder-nearest | 0.2851 | 0.0786 | 10 |
| VAE | CAP CDF ranks | 0.2855 | 0.0815 | 10 |
| VAE | Marginal drift | 0.3710 | 0.0319 | 10 |
| VAE | Wasserstein | 0.3853 | 0.0681 | 10 |
| RealNVP | CAP encoder-nearest | 0.7554 | 0.5678 | 10 |
| RealNVP | CAP CDF ranks | 0.7722 | 0.5615 | 10 |
| RealNVP | Marginal drift | 0.6473 | 0.3322 | 10 |
| RealNVP | Wasserstein | 0.4678 | 0.1074 | 10 |

### Main findings

- Residual-OAS fixes the AE score-geometry mismatch. CDF selection reaches
  0.7355 AUPRC and 0.4609 efficiency, versus 0.5697 and 0.2463 for the old
  scalar-MSE CDF cell. CDF ranks all 16 candidates positively for AUPRC
  (`rho = 0.600`, `p_Holm = 0.0450`) and efficiency (`rho = 0.809`,
  `p_Holm = 0.0006`).
- The outcome-gated SVDD score audit rejected the preregistered center-OAS/CDF
  guess. The strongest complete-grid result preserves radial distance but
  minimizes encoder-nearest CAP: old-panel AUPRC rank `rho = 0.982`,
  `p_Holm = 0.0036`, with validation/test proxy `rho = 1.000`.
- The fresh SVDD rerun confirms that rule. Encoder-nearest CAP ranks AUPRC
  (`rho = 0.947`, `p_Holm = 0.0006`) and efficiency (`rho = 0.876`,
  `p_Holm = 0.0006`), selecting 0.5229 AUPRC and 0.2608 efficiency. Relative
  to the previous corrected encoder-SVDD cell, the gains are 0.0537 and 0.0518.
- Wasserstein has a marginally larger SVDD mean, 0.5256 / 0.2626. The claim is
  therefore ranking validity and competitive deterministic CAP selection, not
  that CAP has the single largest observed mean.
- Every SVDD branch fine-tunes the transferred AE weights. The freeze gate
  verifies changed tensors and three distinct minibatch-order/final-state
  fingerprints for every candidate, eliminating both prior degeneracies.

### Report artifacts

- [Rewritten appendix](results/reports/extra_cchamber.tex)
- [Selected-checkpoint performance](results/reports/extra/cchamber_selected_checkpoint_performance.png)
- [Candidate-ranking validity](results/reports/extra/cchamber_candidate_rank_validity.png)
- [Controlled physical-shift synthesis](results/reports/extra/cchamber_theorem_bridge.png)
- [Authoritative Markdown report](/iopsstor/scratch/cscs/vjimenez/adatl1/reports/cchamber_ae_svdd_oas_20260802_0ea01bb/final_report/results_summary.md)

## Smoke Tests

```bash
make pairing-smoke
make cchamber-pairing-smoke
uv run python tests/train.py experiment=demo/cchamber_ae
uv run python tests/train.py experiment=demo/cchamber_vae
uv run python tests/train.py experiment=demo/cchamber_svdd
uv run python tests/train.py experiment=demo/cchamber_realnvp
uv run pytest tests/test_causal_chamber_datamodule.py tests/test_generation.py
make preflight-local
```

`make pairing-smoke` verifies the exact fixed-table CAP handoff on controlled
57-feature events. `make cchamber-pairing-smoke` repeats the producer/consumer
handoff on all 1,000 held-out rows per reference domain from the real CSVs, with
one real intervention canary. The four model demos exercise their model-specific
tensor paths. Before cluster launch, export both final Causal Chamber pair-table
paths and run `make preflight-cloud`.

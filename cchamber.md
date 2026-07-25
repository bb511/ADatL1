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
  CAP with `pairing_type=precomputed`.

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
  data.pairing_strategy=random \
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
  data.pairing_strategy=random \
  data.max_val_batches=-1 \
  'data.signal_experiments=[]'
```

`--k 0` lets the one-to-one matcher expand the FAISS/torch neighbor search until
coverage saturates. `data.pairing_strategy=random` prevents the metadata pairing
from leaking into the encoder-pair table construction.

Pair-table files are versioned and tied to the encoder checkpoint, split, ordered
source tensors, source sizes, dataset names, and data seed with SHA-256
fingerprints. Reusing a table after any of those inputs changes now fails instead
of silently filtering indices.

## Paper Experiment Matrix

Run these for each anomaly detector: AE, VAE, SVDD, RealNVP.

```text
cchamber_<model>_cap_metadata_nearest
cchamber_<model>_cap_encoder_nearest
cchamber_<model>_cap_random
cchamber_<model>_drift
cchamber_<model>_wasserstein
```

The CAP triplet is the main Causal Chamber result. Drift and Wasserstein are
included as non-CAP validation baselines.

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

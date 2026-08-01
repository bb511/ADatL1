# Pairing Encoder Workflow

This document explains how to train and select the frozen JetCLR pairing encoder used
to build fixed event pair tables for CAP. The pairing encoder is an auxiliary model:
it must not use anomaly-model scores, and it should be selected only from
background/typical-data pairing diagnostics.

## Goal

Train a frozen encoder that maps L1 events into a latent space where two background
validation datasets can be paired by mutual nearest neighbors. The resulting pair
table is then reused by CAP for all anomaly-detector checkpoints.

The intended contract is:

1. train/tune a JetCLR encoder on typical L1 data,
2. select one checkpoint using pairing diagnostics,
3. generate fixed pair tables for each split/dataset pair,
4. run CAP with `pairing_type: precomputed`.

## Setup

From the repository root:

```bash
uv sync --group dev
bash scripts/setup.sh
```

Make sure `.env` or your cluster job exports the usual paths:

```bash
PROJECT_ROOT=/path/to/adatl1
DATA_DIR=/path/to/data
RAW_DATA_DIR=/path/to/raw_l1_inputs
LOG_DIR=/path/to/logs
OUTPUT_DIR=/path/to/outputs
CHECKPOINT_DIR=/path/to/checkpoints
WANDB_MODE=offline
```

Physics runs require the raw L1 files. For a local smoke test, use the synthetic
demo commands below.

## Smoke Test

Before using cluster time, verify the full producer/consumer path on controlled
data:

```bash
make pairing-smoke
```

This trains the JetCLR demo, runs deterministic stress diagnostics, builds
different validation and test tables, and consumes those exact tables through
training-time CAP plus evaluator validation and test. Expected artifacts:

- `results/pairing-smoke/summary.json`
- `results/pairing-smoke/stress/stress_metrics.json`
- `results/pairing-smoke/valid_pairs.pt`
- `results/pairing-smoke/test_pairs.pt`

The smoke deliberately uses the same 57-feature event representation, split
sizes, ordering, callbacks, and evaluator API on both sides of the pair-table
handoff. It does not claim that the synthetic pairs have physics meaning.

Also run the public-data canary:

```bash
make cchamber-pairing-smoke
```

This trains the actual 11-readout Causal Chamber pairing AE, creates 1,000
one-to-one validation pairs and 1,000 separate test pairs from the real CSVs, and
consumes them through a real AE anomaly run with the `uniform_red_mid`
intervention.

## Train One Physics Encoder

Run one baseline JetCLR encoder:

```bash
uv run python src/train.py experiment=physics/jetclr_pairing
```

The config uses:

- algorithm: `configs/algorithm/jetclr.yaml`
- experiment: `configs/experiment/physics/jetclr_pairing.yaml`
- diagnostic callback: `src/evaluation/callbacks/pairing_diagnostics.py`

The default physics diagnostic compares:

- `dataset_1: normal`
- `dataset_2: SingleNeutrino_E-10-gun`
- closure dataset: `normal`

The checkpoint is saved under:

```text
${CHECKPOINT_DIR}/physics_pairing_encoder/<run_name>/last.ckpt
```

The diagnostic JSON is written by the evaluator under the run output/log artifact
location. If using MLflow, also check the run artifacts.

## Hyperparameter Tuning

Use the Optuna config for a small search:

```bash
uv run python src/train.py \
  experiment=physics/jetclr_pairing \
  hparams_search=jetclr_pairing_optuna
```

The HPO objective is the pairing diagnostic selection score:

```text
closure_recall_at_10 * mnn_coverage / (1 + smd_after_mean)
```

This intentionally avoids downstream signal efficiency. Signal efficiency can be
reported later, but should not choose the pairing encoder.

Tune only a modest number of trials first, for example:

```bash
uv run python src/train.py \
  experiment=physics/jetclr_pairing \
  hparams_search=jetclr_pairing_optuna \
  hydra.sweeper.n_trials=20
```

For seed sensitivity, rerun the best region with several seeds:

```bash
uv run python src/train.py experiment=physics/jetclr_pairing seed=101 data.seed=123
uv run python src/train.py experiment=physics/jetclr_pairing seed=102 data.seed=123
uv run python src/train.py experiment=physics/jetclr_pairing seed=103 data.seed=123
```

Keep `data.seed` fixed while varying the model seed. Otherwise the pair-table
comparison confounds encoder sensitivity with different source events.

## What To Inspect

For each candidate checkpoint, inspect the pairing diagnostics JSON. The important
fields are:

- `closure_recall_at_1`, `closure_recall_at_10`: retrieval of two augmented views
  of the same event. Higher is better.
- `closure_median_rank`: median rank of the true augmented partner. Lower is better.
- `mnn_pairs`, `mnn_coverage`: how many cross-dataset pairs survive MNN matching.
  Higher is better, but not at the cost of bad balance.
- `pair_distance_mean`, `pair_distance_p95`: latent distance of accepted pairs.
  Lower is better.
- `smd_before_mean`, `smd_after_mean`: feature balance before and after matching.
  `smd_after_mean` should improve or at least not degrade materially.
- `smd_after_max`: catches a few badly balanced features.
- `selection_score`: compact HPO score.

Recommended acceptance criteria for a usable encoder:

```text
closure_recall_at_10 clearly above random
closure_median_rank close to 1
mnn_coverage reasonably high for the dataset sizes used
smd_after_mean <= smd_before_mean or only mildly worse
pair distances stable across seeds
pair-table overlap stable across seeds
```

Do not select an encoder solely because it gives the largest CAP on anomaly models.
That would make the pairing encoder part of the model-selection target.

## Stress-Test A Candidate

After choosing a checkpoint candidate:

```bash
uv run python -m src.utils.pairing.stress_test_encoder \
  --ckpt /path/to/pairing_encoder.ckpt \
  --out-dir /path/to/pairing_stress/run_name_valid \
  --stage validate \
  --dataset-1 normal \
  --dataset-2 SingleNeutrino_E-10-gun \
  experiment=physics/jetclr_pairing
```

By default, this uses a caliper derived from closure distances. If this rejects too
many pairs during early debugging, add `--no-caliper`; for final results, prefer a
caliper and report the coverage.

The stress test writes:

- `stress_metrics.json`
- `pair_table.pt`

Run the same command for multiple encoder seeds, then compare pair tables:

```bash
uv run python -m src.utils.pairing.compare_pair_tables \
  --tables /path/to/seed101/pair_table.pt /path/to/seed102/pair_table.pt /path/to/seed103/pair_table.pt \
  --out /path/to/pair_table_seed_comparison.json
```

Inspect:

- `mean_jaccard`
- `mean_overlap_min`

These quantify encoder-seed sensitivity of the actual pair table.
The comparison tool intentionally rejects tables from different splits, datasets,
source sizes, source fingerprints, or data seeds.

## Generate Final Pair Tables

Once a checkpoint is selected, generate fixed pair tables for each split and dataset
pair needed by CAP.

Validation table:

```bash
uv run python -m src.utils.pairing.build_pair_table \
  --ckpt /path/to/selected_pairing_encoder.ckpt \
  --out /path/to/pair_tables/valid_normal_vs_singleneutrino.pt \
  --stage validate \
  --dataset-1 normal \
  --dataset-2 SingleNeutrino_E-10-gun \
  --k 20 \
  experiment=physics/jetclr_pairing
```

Test table:

```bash
uv run python -m src.utils.pairing.build_pair_table \
  --ckpt /path/to/selected_pairing_encoder.ckpt \
  --out /path/to/pair_tables/test_normal_vs_singleneutrino.pt \
  --stage test \
  --dataset-1 normal \
  --dataset-2 SingleNeutrino_E-10-gun \
  --k 20 \
  experiment=physics/jetclr_pairing
```

Use `--max-events N` if you need to build a smaller table for quick debugging.

The pair table contains:

```text
schema_version, idx_1, idx_2, distance, rank_1_to_2, rank_2_to_1,
dataset_1, dataset_2, split, encoder_ckpt, metadata
```

Metadata includes SHA-256 fingerprints of the encoder checkpoint and both ordered
source tensors. Loading rejects stale schemas, duplicate/out-of-range pairs,
wrong dataset names or splits, size mismatches, and different source content.
Writers refuse to overwrite an existing artifact unless `--overwrite` is supplied.
Keep these pair tables as experiment artifacts.

## Use Pair Tables In CAP

Replace score-based CAP pairing with precomputed pairing.

Training-time callback:

```yaml
callbacks:
  cap_sn_zb:
    _target_: src.callbacks.cap.CAPCallback
    output_name: 'ascore/full'
    dataset_1: normal
    dataset_2: SingleNeutrino_E-10-gun
    pairing_type: precomputed
    pairing_index_path: /path/to/pair_tables/valid_normal_vs_singleneutrino.pt
    cap_metric_config:
      beta0: 1.0
      normalization_type: sigmoid
      normalization_params: null
      energy_type: adaptive
      energy_params:
        scale: 0.5
      regularization_type: none
      regularization_params: null
      binary: true
      lr: 0.01
      n_epochs: 20
      batch_size: 8192
      normalize_gradients: true
      process_group: null
      dist_sync_fn: null
```

Evaluation must receive both tables because one evaluator run performs validation
and then test:

```yaml
evaluation:
  callbacks:
    cap_sn_zb:
      _target_: src.evaluation.callbacks.cap.CAP
      output_name: 'ascore/full'
      dataset_1: normal
      dataset_2: SingleNeutrino_E-10-gun
      pairing_type: precomputed
      pairing_index_path: /path/to/pair_tables/valid_normal_vs_singleneutrino.pt
      pairing_test_index_path: /path/to/pair_tables/test_normal_vs_singleneutrino.pt
      cap_metric_config: ${callbacks.cap_sn_zb.cap_metric_config}
```

Using the test table as `pairing_index_path` is data leakage during evaluator
validation and is now prevented by split validation.

## Deterministic Physics Controls

The learned JetCLR result should be compared to three controls produced without
anomaly scores or learned parameters:

- `flat_physical` is the deliberately simple data-space baseline. It undoes the
  preprocessing normalization, converts hardware coordinates to GeV/eta/radians,
  keeps object slots and presence bits, and embeds phi as sine/cosine.
- `physics_summary` is the primary interpretable control. Its typed event summary
  contains multiplicity, scalar and leading Et, eta moments, circular energy flow,
  and FET-relative recoil. It is invariant to permutations within an object family
  and to a global azimuthal rotation.
- `typed_sliced_wasserstein` is the stronger transport control. It represents each
  object family as an Et-weighted measure in eta and FET-relative phi, then stores
  deterministic projected weighted quantiles. This is a scalable linearized
  sliced-W2 approximation, not an exact event-by-event EMD solve.

All metric scales are fitted on a deterministic 200,000-event subset of ZeroBias
training data. No target simulation, anomaly-model output, or test label is used.
Each target queries the full ZeroBias reference split. Candidate edges are assigned
globally in `(distance, target_index, reference_index)` order, making ties and
one-to-one conflicts reproducible. A q99 distance caliper is derived only from a
separate ZeroBias-train to ZeroBias-reference closure sample.

The producer reads the ordered tensor caches directly and emits two contracts:

```text
<split>_<strategy>_full.pt
  target_to_reference: LongTensor[N_target]
  reference_to_target: LongTensor[N_reference]
  distance: FloatTensor[N_target]
  valid: BoolTensor[N_target]
  caliper_valid: BoolTensor[N_target]
  candidate_rank: LongTensor[N_target]

<split>_<strategy>_cap[_n<runtime-size>].pt
  strict compact pair table consumed by pairing_type=precomputed
```

For example, generate a full validation map and the usual 81,920/163,840-event
runtime tables with:

```bash
uv run python -m src.utils.pairing.physics_tables \
  --stage validate \
  --strategy physics_summary \
  --out-dir results/physics-pairing-control/production \
  --max-target-events 1199981 \
  --cap-prefix-events 81920 163840 \
  --device cuda:0 \
  --backend torch
```

Repeat with `--stage test` and the other strategies. The torch search is exact and
chunked; FAISS `IndexFlatL2` is used when available on CPU but is not required.
Every artifact records the descriptor state, schema signature, source paths and
SHA-256 fingerprints, search parameters, caliper, coverage, and balance diagnostics.

Use the prefix matching the actual number of auxiliary events collected by CAP.
With `batch_size=8192,max_val_batches=10`, choose `_n81920.pt`; with
`batch_size=16384,max_val_batches=10`, choose `_n163840.pt`. Strict source-size and
source-hash checks deliberately reject the wrong choice.

## Final Checklist

Before using a pairing encoder in the paper:

1. synthetic demo ran successfully,
2. physics JetCLR checkpoint selected from pairing diagnostics,
3. stress metrics saved for the selected checkpoint,
4. seed-sensitivity pair-table comparison saved,
5. final validation/test pair tables generated,
6. CAP configs use `pairing_type: precomputed`,
7. paper tables report both CAP results and pairing diagnostics.
8. `make preflight-cloud` passes from a clean commit on the target environment.
9. physics-control full maps, runtime CAP tables, and diagnostic JSON files are archived.

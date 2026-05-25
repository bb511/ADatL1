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
PROJECT_ROOT=/path/to/codex-adatl1-pairing
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

Before using cluster time, verify the full path on synthetic data:

```bash
uv run python tests/train.py experiment=demo/l1_jetclr logger=none
```

Then stress-test the saved demo checkpoint:

```bash
uv run python -m src.utils.pairing.stress_test_encoder \
  --ckpt checkpoints/demo/l1_jetclr/last.ckpt \
  --out-dir outputs/demo_l1_jetclr_stress \
  --stage validate \
  --dataset-1 normal \
  --dataset-2 reference_normal \
  --no-caliper \
  experiment=demo/l1_jetclr logger=none
```

Expected artifacts:

- `outputs/demo_l1_jetclr_stress/stress_metrics.json`
- `outputs/demo_l1_jetclr_stress/pair_table.pt`

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
uv run python src/train.py experiment=physics/jetclr_pairing seed=101
uv run python src/train.py experiment=physics/jetclr_pairing seed=102
uv run python src/train.py experiment=physics/jetclr_pairing seed=103
```

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
idx_1, idx_2, distance, rank_1_to_2, rank_2_to_1,
dataset_1, dataset_2, split, encoder_ckpt, metadata
```

Keep these pair tables as experiment artifacts. CAP results should always record the
exact pair-table path and encoder checkpoint.

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

Evaluation callback uses the same keys:

```yaml
evaluation:
  callbacks:
    cap_sn_zb:
      _target_: src.evaluation.callbacks.cap.CAP
      output_name: 'ascore/full'
      dataset_1: normal
      dataset_2: SingleNeutrino_E-10-gun
      pairing_type: precomputed
      pairing_index_path: /path/to/pair_tables/test_normal_vs_singleneutrino.pt
      cap_metric_config: ${callbacks.cap_sn_zb.cap_metric_config}
```

## Final Checklist

Before using a pairing encoder in the paper:

1. synthetic demo ran successfully,
2. physics JetCLR checkpoint selected from pairing diagnostics,
3. stress metrics saved for the selected checkpoint,
4. seed-sensitivity pair-table comparison saved,
5. final validation/test pair tables generated,
6. CAP configs use `pairing_type: precomputed`,
7. paper tables report both CAP results and pairing diagnostics.


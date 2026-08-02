# Physics pairing control for CAP

The production control pairs the two named 2025 ZeroBias streams:

- dataset 0: `ZB_run396102` (2025E)
- dataset 1: `ZB_run398183` (2025G)

Training still uses their ordinary combined 60/20/20 split. For CAP,
`L1ADDataModule` reconstructs each run from a compact source-ID sidecar and exposes
two additional, ordered, non-shuffled evaluation loaders. The sidecars use the same
seed (`42`), concatenation order, and split boundaries as `L1DataMLReady`.

## Pairing controls

Four deterministic representations are available:

- `flat_physical`: slot-preserving physical data-space distance with periodic phi.
- `physics_summary`: permutation- and rotation-invariant event summaries.
- `typed_sliced_wasserstein`: typed energy-flow optimal-transport summaries.
- `jetclr`: cosine-equivalent distance between normalized frozen-encoder embeddings.

Each artifact contains a dense tensor with the contract:

```python
map_0_to_1[i] == j
```

Thus row `i` from `ZB_run396102` is paired with row `j` from `ZB_run398183`.
The table also stores the equivalent `idx_1` and `idx_2`, source hashes, dataset
names, split, sizes, descriptor state hash, and diagnostics.

## Generate a table

The cache and source metadata paths below are shown explicitly for reproducibility.
In experiments, the Hydra configuration derives artifact paths from
`${paths.data_dir}` and therefore uses the repository's existing `DATA_DIR` setup.

```bash
uv run python -m src.utils.pairing.physics_tables \
  --cache-root /path/to/data_2025E+G/mlready/eminimalTauFET_pdefaultTauFET_default/robust \
  --source-metadata-dir /path/to/data_2025E+G/pairing/ZB_run396102_to_ZB_run398183/sources \
  --out-dir /path/to/data_2025E+G/pairing/ZB_run396102_to_ZB_run398183 \
  --stage validate \
  --strategy physics_summary \
  --events 163840 \
  --backend faiss_hnsw \
  --device cpu
```

Run the same command with `--stage test`. Repeat for `flat_physical` and
`typed_sliced_wasserstein`. Writers refuse accidental replacement unless
`--overwrite` is supplied.

For the learned representation, use the same command and row count with
`--strategy jetclr --jetclr-checkpoint /path/to/selected.ckpt`. The encoder runs
directly on the exact ordered CAP cache rows; it does not use a full-run lookup.

`events` must equal the rows CAP collects from each named stream. With the supplied
experiment this is `data.batch_size * data.max_val_batches = 16384 * 10`.

## Run CAP

Use the ready-made experiment:

```bash
uv run python src/train.py experiment=physics/vae_background_pairing
```

Choose the control in `configs/pairing/physics.yaml`:

```yaml
physics_pairing:
  strategy: physics_summary
  # strategy: flat_physical
  # strategy: typed_sliced_wasserstein
  # strategy: jetclr
```

The callback uses:

```yaml
dataset_1: ZB_run396102
dataset_2: ZB_run398183
pairing_type: mapping
pairing_index_path: ${physics_pairing.validation_table}
pairing_test_index_path: ${physics_pairing.test_table}
```

Before CAP accepts a table, it verifies the dataset names, split, collected row
counts, index uniqueness and bounds, and SHA-256 hashes of both ordered input
tensors. Any shuffle, different prefix, stale split, or preprocessing change fails
immediately instead of silently applying the wrong map.

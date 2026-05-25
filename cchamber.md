# Causal Chamber Experiments

This note documents the Causal Chamber dataset setup used between the analytical
experiments in `analytical.md` and the L1 physics experiments. The purpose of
this dataset is to keep the benefits of a controlled intervention benchmark
while moving to real, low-dimensional, interpretable detector-like readouts.

No Causal Chamber experiment in this repository should generate synthetic
readouts. All training, validation, model selection, and intervention evaluation
use rows from the public Causal Chamber CSV files.

## Dataset Role

The Causal Chamber light-tunnel dataset contains repeated measurements from a
controlled physical setup. Each CSV corresponds to a reference condition or to a
known intervention. This gives us two properties that are useful for the paper:

- Like the analytical setting, the intervention identity is known. We can group
  evaluation results by intervention target and strength.
- Like the L1 physics setting, the model observes real low-dimensional features
  with meaningful names rather than arbitrary synthetic coordinates.

The default dataset is `lt_interventions_standard_v1`, downloaded under
`data/causal_chamber` by `CausalChamberDataModule` when needed.

## Features and Metadata

The default model input is readout-only:

```text
current, angle_1, angle_2,
ir_1, vis_1, ir_2, vis_2, ir_3, vis_3,
v_board, v_reg
```

These are the features seen by the anomaly-detection model.

The raw CSVs also contain metadata and control/intervention columns such as:

```text
timestamp, config, counter, flag, intervention,
red, green, blue, osr_c, v_c,
pol_1, pol_2,
osr_angle_1, osr_angle_2,
v_angle_1, v_angle_2,
l_11, l_12, ..., diode_*, t_*, camera
```

These columns are not model inputs in the default experiment. They are retained
for dataset construction, intervention bookkeeping, and real-data pairing
diagnostics. This avoids the trivial leakage that would happen if the model were
given the intervention knobs directly.

The relevant config is:

```yaml
data=causal_chamber
data.feature_set=readouts
data.pairing_strategy=nearest
data.pairing_columns=null
```

With `pairing_columns=null`, the datamodule uses all numeric non-readout
columns as the fixed matching descriptor. To run a leakage/sanity ablation, use
`data.feature_set=all_numeric_no_meta`, but this should not be the paper's main
configuration.

## Pairing Strategies

Pairing is part of the Causal Chamber experiment, not an incidental dataloader
detail. The paper should compare at least three ways of constructing the
`normal/reference_normal` validation pairs.

| Pairing | Status | Description | Purpose |
| --- | --- | --- | --- |
| `metadata_nearest` / `nearest` | implemented | Match real reference rows by nearest neighbor in experimental settings. | Main Causal Chamber pairing. |
| `random` | implemented | Use the same two real reference pools, but pair them randomly. | Negative control for CAP. |
| `encoder_nearest` | planned | Train a separate fixed encoder, embed real reference rows, and match nearest neighbors in encoder space. | Test whether learned pairings recover the metadata pairing signal. |

The current `nearest` implementation pairs samples using experimental settings,
not readout similarity. It uses numeric non-readout columns such as light
settings, voltages, polarizer settings, diode settings, and camera. The
downstream anomaly detector does not see these columns.

The `random` pairing is already available:

```bash
uv run python src/train.py \
    experiment=cchamber/ae_paired_agnostic \
    data.pairing_strategy=random
```

This changes only the CAP pairing quality. Drift and Wasserstein compare
marginal score distributions, so they should be essentially insensitive to the
ordering of `normal` and `reference_normal`.

The intended pairing ablation is:

```text
metadata_nearest  ≈  encoder_nearest  >  random
```

where the comparison is made in terms of validation CAP/rank-correlation and
downstream intervention efficiency after model selection. This is a direct
test that CAP benefits from meaningful fixed pairs rather than arbitrary sample
ordering.

## Splits

Training uses only real rows from `uniform_reference`.

The default split fractions are:

```yaml
train_fraction: 0.6
val_fraction: 0.2
test_fraction: 0.2   # implicit remainder
reference_fraction: 0.5
signal_val_fraction: 0.6
```

For validation and test, the datamodule creates two real normal domains:

- `normal`: real rows from a held-out part of `uniform_reference`;
- `reference_normal`: disjoint real rows from the same held-out part, matched
  to `normal` by nearest neighbor in the metadata/control descriptor.

The two domains are emitted in aligned order with a shared `pair_id`. CAP is
configured with `pairing_type: none` because the datamodule has already
constructed the comparison domains and no extra CDF or synthetic pairing should
be applied.

Technically, `pairing_type: none` in the CAP callback means "pair by order":
sample 0 from `normal` is compared with sample 0 from `reference_normal`, sample
1 with sample 1, and so on. Therefore the datamodule owns the fixed pairing.
One implementation improvement is to add an explicit `pairing_type: fixed` or
`pairing_type: pair_id` mode to CAP so that the callback verifies and uses the
`pair_id` field rather than relying only on dataloader order.

Each intervention CSV listed in `data.signal_experiments` is exposed as a held
out signal dataset, for example `uniform_red_mid`, `uniform_v_c_strong`, or
`uniform_t_ir_1_weak`.

## Dataset Contract

After `setup()`, the datamodule exposes a `contract` dictionary with:

- `model_features`: columns given to the model;
- `pairing_features`: columns used to match `normal` and `reference_normal`;
- `excluded_columns`: columns deliberately not used as model inputs;
- `splits`: train/validation/test sizes;
- `intervention_catalog`: parsed intervention names, families, strengths, flags,
  and row counts;
- `pairing.diagnostics`: number of pairs and matching distances.

Inspect it with:

```bash
uv run python - <<'PY'
from src.data.CausalChamber_datamodule import CausalChamberDataModule

dm = CausalChamberDataModule(
    data_dir="data/causal_chamber",
    batch_size=512,
    max_val_batches=1,
    seed=123,
)
dm.setup("fit")
print(dm.contract)
PY
```

These diagnostics should be reported once in the paper or appendix so that the
reader can see which readouts were modeled and which metadata were used only for
matching.

## Model Matrix

The Causal Chamber paper matrix uses the signal-agnostic strategies only:

| Model | Experiment config | Search config |
| --- | --- | --- |
| AE | `cchamber/ae_paired_agnostic` | `ae_optuna` |
| VAE | `cchamber/vae_paired_agnostic` | `vae_optuna` |
| SVDD | `cchamber/svdd_paired_agnostic` | `svdd_optuna` |
| RealNVP | `cchamber/realnvp_paired_agnostic` | `realnvp_optuna` |

The model-selection strategies are:

| Strategy | Selection metric | Direction | Validation domains |
| --- | --- | --- | --- |
| CAP | `cap_ema_normal_vs_reference_normal` | maximize | `normal` vs `reference_normal` |
| Drift | `operational_drift_ema` | minimize | `normal` vs `reference_normal` |
| Wasserstein | `w1dist_ema_normal_vs_reference_normal` | minimize | `normal` vs `reference_normal` |

Semi-supervised CVaR selection is intentionally not part of the Causal Chamber
matrix. The intervention datasets are held out for evaluation, not used to pick
hyperparameters.

List the current matrix with:

```bash
uv run python scripts/generation.py list --dataset cchamber
```

## Main Paper Experiments

The main result should compare CAP, threshold drift, and Wasserstein model
selection on the same real Causal Chamber intervention benchmark.

For each model and strategy:

1. Tune hyperparameters using only the agnostic validation metrics on
   `normal/reference_normal`.
2. Select a trial from the Optuna study using the strategy's main metric and the
   model-native secondary objective.
3. Retrain the selected configuration with reporting seeds.
4. Evaluate the selected checkpoint on all intervention datasets.
5. Report anomaly efficiency by intervention dataset, and aggregate by
   intervention family/strength.

Recommended paper tables/figures:

- overall mean and median anomaly efficiency across interventions;
- per-family breakdown, e.g. color, diode, voltage, polarizer, OSR;
- strength breakdown where available: weak, mid, strong;
- pairing diagnostics for the real `normal/reference_normal` comparison;
- pairing ablation: metadata-nearest vs encoder-nearest vs random;
- optional leakage sanity check comparing readout-only inputs to
  `all_numeric_no_meta`.

## Theory Diagnostic

To connect to `theory.tex`, add an offline analysis that estimates empirical
reproducible signal and nuisance structure from real data.

For an intervention dataset `A`:

1. Fit the same normalization on training `uniform_reference` rows.
2. Let `mu_A - mu_0` be the normalized mean shift between intervention `A` and
   held-out reference rows. Use the normalized direction
   `w_A = (mu_A - mu_0) / ||mu_A - mu_0||` as the empirical anomaly direction.
3. Estimate the reproducible covariance `Sigma_R` from the matched real
   `normal/reference_normal` pairs.
4. Estimate an intervention-aligned reliability proxy:

   ```text
   lambda_Z(A) = w_A^T Sigma_R w_A
   ```

5. Estimate the strongest orthogonal nuisance reliability:

   ```text
   lambda_U(A) = largest eigenvalue of P_A Sigma_R P_A
   ```

   where `P_A = I - w_A w_A^T`.

This does not generate data. It uses the real intervention labels and the real
paired reference domains. The expected paper use is a diagnostic plot/table:
interventions with larger `lambda_Z / lambda_U` should be the cases where CAP's
reproducible validation signal is best aligned with downstream anomaly power.

## Encoder-Based Fixed Pairing

The encoder-based pairing should be a separate pretraining step. It must not be
the anomaly detector being tuned, and it must not use intervention labels for
model selection.

The goal is to learn a representation

```text
z_pair = g_psi(x)
```

from real `uniform_reference` data, then freeze `g_psi` and use nearest-neighbor
matching in `z_pair` space to construct fixed `normal/reference_normal` pairs.

Recommended Causal Chamber design:

1. Train the pair encoder only on the training split of `uniform_reference`.
2. Input: the same readout features used by the anomaly detector.
3. Auxiliary target: the experimental settings used by metadata-nearest pairing.
4. Loss: readout reconstruction plus supervised setting prediction, for example

   ```text
   L_pair = L_reco(readouts) + alpha * L_settings(settings)
   ```

5. Freeze the encoder.
6. Embed the held-out validation/test reference pools.
7. Construct one-to-one nearest-neighbor matches between `normal` and
   `reference_normal` in encoder space.
8. Store the resulting fixed pair indices, pair distances, encoder checkpoint
   path, and encoder training seed in the dataset contract.

This setup is useful because metadata-nearest is an interpretable Causal
Chamber baseline, while encoder-nearest is closer to the physics setting where
we may need learned representations to form comparable validation pairs. The
paper claim should not be that encoder pairing is an oracle. The claim should be
that a frozen encoder trained on normal data can recover pairings that are
competitive with known experimental-setting nearest neighbors and clearly
better than random pairing.

The branch `research/cchamber-pairs` in `bb511/ADatL1` uses an external
`capmetric.callback.ApproximationCapacityCallback` with named `data_pairs`.
That is useful inspiration for making pair definitions explicit in config.
For this repository, the missing piece is sample-level fixed-pair support:
the datamodule should construct pair indices and CAP should either consume
`pair_id` directly or receive pre-aligned datasets whose order is guaranteed by
the contract.

## Pairing Implementation Plan

1. Rename the current `nearest` strategy to `metadata_nearest`, keeping
   `nearest` as a backwards-compatible alias.
2. Keep `random` as a first-class strategy in the Causal Chamber config and in
   `scripts/generation.py` for CAP pairing ablations.
3. Add a shared pair-construction utility, for example
   `src/data/components/pairing.py`, with:
   - standardized nearest-neighbor matching;
   - random matching;
   - one-to-one greedy or Hungarian matching;
   - a `PairingResult` object containing indices, distances, strategy, seed, and
     diagnostics.
4. Add `pairing_type: fixed` or `pairing_type: pair_id` to both training-time
   and evaluation CAP callbacks. The callback should collect `pair_id` from
   batches, align scores by common pair IDs, and fail loudly if IDs are missing
   or duplicated.
5. Add a pair-encoder training path:
   - model/config: `configs/algorithm/cchamber_pair_encoder.yaml`;
   - script or Hydra experiment: train on `uniform_reference` train rows only;
   - outputs: checkpoint plus normalization/settings metadata.
6. Add `encoder_nearest` to the datamodule:
   - load the frozen pair encoder checkpoint;
   - compute embeddings for held-out reference pools;
   - build fixed nearest-neighbor pairs in embedding space;
   - cache pair indices under `data/causal_chamber/pairs/...` to avoid
     recomputation.
7. Extend the Causal Chamber contract with:
   - `pairing.strategy`;
   - `pairing.source`, e.g. metadata columns or encoder checkpoint;
   - `pairing.distance_mean`, `distance_median`, and optionally quantiles;
   - `pairing.seed`;
   - `pairing.n_pairs`.
8. Extend `scripts/generation.py` so CAP experiments can be generated for:

   ```text
   cchamber_<model>_cap_metadata_nearest
   cchamber_<model>_cap_encoder_nearest
   cchamber_<model>_cap_random
   ```

   Drift and Wasserstein can remain single-pairing baselines because they do not
   use sample-level pairing.
9. Add tests:
   - random pairing is reproducible under fixed seed and differs from nearest;
   - fixed CAP gives the same result when batches are shuffled but `pair_id` is
     preserved;
   - encoder-nearest can be exercised with a tiny dummy encoder;
   - the contract records the selected pairing strategy and diagnostics.

The final paper table for the pairing ablation should report, for each model:

| Pairing | Validation CAP | RankCorr | Pair distance | Mean intervention efficiency |
| --- | ---: | ---: | ---: | ---: |
| metadata-nearest | | | | |
| encoder-nearest | | | | |
| random | | | | |

The expected result is not that metadata-nearest and encoder-nearest are
identical. The useful result is that both produce substantially more reliable
CAP-based model selection than random pairing.

## Smoke Runs

Use the demo configs to check that the datamodule and model configs work:

```bash
uv run python tests/train.py experiment=demo/cchamber_ae
uv run python tests/train.py experiment=demo/cchamber_vae
uv run python tests/train.py experiment=demo/cchamber_svdd
uv run python tests/train.py experiment=demo/cchamber_realnvp
```

For quick local debugging without evaluation:

```bash
uv run python src/train.py \
    experiment=cchamber/ae_paired_agnostic \
    logger=none \
    trainer=cpu \
    trainer.max_epochs=1 \
    +trainer.limit_train_batches=2 \
    +trainer.limit_val_batches=1 \
    '~evaluation' \
    '~evaluation/callbacks'
```

## Hyperparameter Sweeps

Generate sweep scripts for all Causal Chamber experiments:

```bash
uv run python scripts/generation.py generate \
    --dataset cchamber \
    --stage sweep \
    --n-trials 600 \
    --seeds 123 \
    --trainer gpu \
    --devices "[0]"
```

This writes one folder per experiment under `scripts/generated`, each containing
`sweep.sh`, `manifest.json`, and `manifest.md`.

Run one sweep:

```bash
bash scripts/generated/cchamber_ae_cap/sweep.sh
```

Run all generated Causal Chamber sweeps by executing the corresponding
`sweep.sh` files:

```text
cchamber_ae_cap
cchamber_ae_drift
cchamber_ae_wasserstein
cchamber_vae_cap
cchamber_vae_drift
cchamber_vae_wasserstein
cchamber_svdd_cap
cchamber_svdd_drift
cchamber_svdd_wasserstein
cchamber_realnvp_cap
cchamber_realnvp_drift
cchamber_realnvp_wasserstein
```

For cluster execution, use a launcher:

```bash
uv run python scripts/generation.py generate \
    --dataset cchamber \
    --stage sweep \
    --n-trials 600 \
    --seeds 123 \
    --trainer gpu \
    --devices "[0]" \
    --launcher submitit_slurm_clariden \
    --cpus-per-task 4 \
    --gpus-per-node 1 \
    --timeout-min 240
```

The generator fixes the Causal Chamber data settings:

```text
data=causal_chamber
data.batch_size=512
data.max_val_batches=-1
data.pairing_strategy=nearest
algorithm.target_rate=0.01
algorithm.base_rate=null
```

## Selecting Trials

The sweep uses multi-objective Optuna:

- the first objective is the strategy metric;
- the second objective is the model-native validation objective, such as MSE,
  KL, SVDD distance, or RealNVP log-probability objective.

Use only validation quantities from `normal/reference_normal` for model
selection. Do not use intervention efficiencies to choose hyperparameters.

For each `(model, strategy)`, choose a trial from the Pareto front according to
the paper's selection rule. A practical rule is:

1. keep trials with acceptable model-native validation objective;
2. among them, choose the best strategy metric;
3. if several trials are statistically indistinguishable, choose the simpler or
   more stable model.

Record the selected Hydra overrides in a JSON file:

```json
[
  {
    "run_name": "cchamber_ae_cap_seed123",
    "seed": 123,
    "overrides": [
      "algorithm.optimizer.lr=0.0003",
      "algorithm.encoder.nodes=[32,16]",
      "algorithm.delta=5.0"
    ]
  }
]
```

The exact override keys must match the tuned parameters in the corresponding
`configs/hparams_search/*_optuna.yaml` file and the selected Optuna trial.

## Retraining

Generate retraining scripts from the selected overrides:

```bash
uv run python scripts/generation.py generate \
    --name cchamber_ae_cap \
    --stage retrain \
    --selected-overrides scripts/selected/cchamber_ae_cap.json \
    --trainer gpu \
    --devices "[0]"
```

Then run:

```bash
bash scripts/generated/cchamber_ae_cap/retrain.sh
```

For seed statistics, put one JSON object per seed in the selected-overrides file.
For example, after selecting one hyperparameter configuration, repeat the same
`overrides` with seeds such as `123`, `456`, `789`, `101112`, and `131415`.
Those seeds are reporting/statistical factors, not new hyperparameter choices.

## Evaluation

After retraining, evaluate the selected checkpoint on the intervention datasets.
Create a checkpoint manifest:

```json
[
  {
    "run_name": "cchamber_ae_cap_seed123_eval",
    "seed": 123,
    "ckpt_path": "checkpoints/cchamber_ae_cap_retrain/cchamber_ae_cap_seed123/summary/cap_ema_normal_vs_reference_normal/max/cap_ema_normal_vs_reference_normal.ckpt"
  }
]
```

The checkpoint path depends on the strategy:

- CAP: `summary/cap_ema_normal_vs_reference_normal/max/...`
- Drift: `summary/operational_drift_ema/min/...`
- Wasserstein: `summary/w1dist_ema_normal_vs_reference_normal/min/...`

Generate and run evaluation scripts:

```bash
uv run python scripts/generation.py generate \
    --name cchamber_ae_cap \
    --stage evaluate \
    --ckpt-manifest scripts/checkpoints/cchamber_ae_cap.json \
    --trainer gpu \
    --devices "[0]"

bash scripts/generated/cchamber_ae_cap/evaluate.sh
```

The evaluation callback computes anomaly efficiency on
`${data.signal_experiments}`. These are the held-out real intervention datasets.

## Reporting Statistics

For paper numbers, aggregate over:

- model family: AE, VAE, SVDD, RealNVP;
- model-selection strategy: CAP, drift, Wasserstein;
- seed;
- intervention dataset;
- intervention family and strength parsed from the dataset name.

Recommended reporting:

- mean and standard error over seeds;
- median and interquartile range over intervention datasets;
- per-family tables to avoid hiding failure modes;
- paired comparisons between model-selection strategies using the same model,
  seed, and intervention dataset.

The important statistical discipline is that hyperparameter selection must use
only agnostic validation metrics. Intervention labels enter only after the model
and checkpoint have been selected.

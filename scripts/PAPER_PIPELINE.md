# Paper experiment pipeline

`generation.py` owns Hydra commands. `paper_pipeline.py` owns label-free trial
selection, checkpoint manifests, result collection, and statistical summaries. The
interfaces are CSV/JSON so a local synthetic smoke test and a cluster run use the same
workflow.

## 1. Select from one replayed candidate pool

Create a long-form `candidate_metrics.csv` with one row for every candidate and
validation strategy:

```text
dataset,model,seed,candidate_id,strategy,value,params_json
cchamber,ae,123,0001,cap_random,0.71,"{""algorithm.optimizer.lr"":0.001}"
cchamber,ae,123,0001,drift,0.08,"{""algorithm.optimizer.lr"":0.001}"
```

Allowed strategies are `cap`, `cap_metadata_nearest`, `cap_encoder_nearest`,
`cap_random`, `drift`, and `wasserstein`. The tool rejects downstream metrics such as
AUPRC and requires identical candidate IDs and parameters for every strategy within a
dataset/model/seed group. Do not combine independent adaptive Optuna studies and call
them a shared pool: replay the same hyperparameter candidates under every selection
strategy first.

```bash
uv run python scripts/paper_pipeline.py select \
  --candidate-metrics results/candidate_metrics.csv \
  --output-dir results/selection
```

This writes:

- `selected_trials.csv`: auditable winners and validation values.
- `retrain_manifest.json`: Hydra overrides consumed by `generation.py`.
- `candidate_pool_audit.csv`: pool sizes and a parameter hash.
- `selection_provenance.json`: input checksum and selection rule.

## 2. Retrain selected candidates

Generate one retraining command per selected seed. A shared manifest is filtered by
`spec_name`, preventing parameters from being applied to the wrong model.

```bash
uv run python scripts/generation.py generate \
  --dataset cchamber \
  --stage retrain \
  --selected-overrides results/selection/retrain_manifest.json \
  --output-dir scripts/generated \
  --trainer gpu \
  --devices '[0]'
```

Run the generated `retrain.sh` files through the desired cluster launcher.

## 3. Resolve and evaluate selected checkpoints

The checkpoint resolver selects the strategy-specific validation checkpoint, not
`last.ckpt`.

```bash
uv run python scripts/paper_pipeline.py checkpoints \
  --selected-trials results/selection/selected_trials.csv \
  --checkpoints-dir checkpoints \
  --output results/selection/checkpoint_manifest.json

uv run python scripts/generation.py generate \
  --dataset cchamber \
  --stage evaluate \
  --ckpt-manifest results/selection/checkpoint_manifest.json \
  --output-dir scripts/generated \
  --trainer gpu \
  --devices '[0]'
```

Evaluation commands point `paths.checkpoints_dir`, `experiment_name`, and `run_name`
at the retrained run because that is the interface used by `src/train.py`. They disable
`last` and unrelated checkpoint criteria. Encoder-nearest sweeps use the validation
pair table. Final evaluation retains that table for evaluator validation and
supplies a separate held-out test table for evaluator testing. Both tables are
strictly checked against their ordered source tensors.

## 4. Collect callback values

The AUPRC and efficiency evaluation callbacks write `values.csv` files with:

```text
checkpoint,intervention,metric,value
```

Create a collection manifest with one selected callback file per seed and strategy:

```text
path,dataset,model,strategy,seed,pairing
/cluster/run/.../auprc/values.csv,cchamber,ae,cap_random,123,random
```

Relative paths are resolved from the collection manifest directory.

```bash
uv run python scripts/paper_pipeline.py collect \
  --manifest results/values_manifest.csv \
  --output results/evaluation_long.csv
```

The collector writes the canonical long-form CSV and a checksum provenance file.

## 5. Aggregate and report

```bash
uv run python scripts/paper_pipeline.py aggregate \
  --results results/evaluation_long.csv \
  --output-dir results/paper \
  --main-metric auprc
```

The aggregator requires paired seeds and identical intervention coverage. It first
averages interventions within each seed, then computes deterministic bootstrap
intervals across seeds. Outputs include:

- seed-level, overall, and per-intervention CSV tables;
- paired strategy-difference CSVs;
- a compact model/strategy comparison plot;
- an intervention heatmap;
- `report.md` with coverage, primary results, largest paired differences, and figure
  links;
- an input checksum and aggregation provenance record.

## 6. Deployment gate

Before submitting the generated matrix, run:

```bash
make preflight-local
export CCHAMBER_VALID_PAIR_TABLE=/shared/pairs/valid_ae_pairs.pt
export CCHAMBER_TEST_PAIR_TABLE=/shared/pairs/test_ae_pairs.pt
make preflight-cloud
```

The cloud profile rejects a dirty checkout, stale lockfile, missing runtime paths,
missing L1 parquet directories, invalid or mismatched pair tables, Hydra
composition errors, and shell syntax errors. It validates all 76 core
specifications with the same hyperparameter-search and launcher overrides used by
generated sweep commands.

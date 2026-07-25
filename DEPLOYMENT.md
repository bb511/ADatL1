# Cloud Deployment Gate

Cloud readiness is a sequence of evidence, not a promise based on local unit tests.
Use the following gates on the exact commit and environment that will run the
experiments.

## 1. Reproduce the environment

```bash
uv sync --locked --group dev
bash scripts/setup.sh
```

Edit the generated `.env` so `PROJECT_ROOT`, `DATA_DIR`, `RAW_DATA_DIR`, `LOG_DIR`,
`OUTPUT_DIR`, and `CHECKPOINT_DIR` point to durable cluster storage. Set
`WANDB_MODE=offline` or `online` explicitly.

## 2. Run local structural gates

```bash
make smoke
uv run pytest -k "not slow"
make preflight-local
```

`make smoke` covers analytical artifacts, four anomaly models, checkpoint
selection/evaluation/reporting, the controlled fixed-pair-table handoff, and the
same handoff on the real public Causal Chamber CSVs.
`make preflight-local` composes the 76 core paper sweeps with their real Optuna
overrides and parses all generated shell scripts.

## 3. Produce real Causal Chamber pair tables

Train the frozen pairing AE and build separate validation and test tables using
the commands in `cchamber.md`. Store them on durable shared storage, then export:

```bash
export CCHAMBER_VALID_PAIR_TABLE=/shared/pairs/valid_ae_pairs.pt
export CCHAMBER_TEST_PAIR_TABLE=/shared/pairs/test_ae_pairs.pt
```

Pair-table loading checks the schema, split, dataset names, source sizes, source
content fingerprints, encoder checkpoint fingerprint, uniqueness, bounds, and
finite distances.

## 4. Run the strict cloud preflight

From a clean checkout:

```bash
make preflight-cloud
```

The cloud profile requires:

- Python 3.10 and a lockfile consistent with `pyproject.toml`;
- a clean Git worktree;
- writable configured artifact directories;
- an explicit WandB mode;
- all expected physics raw-data directories with parquet files;
- valid Causal Chamber validation and test pair tables from one encoder;
- successful composition and shell parsing of every core generated sweep with
  the configured launcher.

The machine-readable report is written to `results/preflight.json`.

## 5. Run one target-environment canary

Generate one low-cost experiment with the target launcher, submit it, and verify
the GPU, CUDA build, scheduler account/partition, shared filesystem, checkpoint,
logger, and evaluator outputs. Local tests cannot prove those external services.
Only after that canary succeeds should the full sweep matrix be submitted.

## 6. Evidence boundary

A green smoke/preflight means the implementation and deployment inputs are
structurally ready. It does not establish the scientific result. Full-data sweeps,
shared candidate replay, at least three paired retraining seeds, every required
intervention/domain, final evaluation, and paired aggregation remain mandatory.

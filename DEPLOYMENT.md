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

On Clariden, after an IOPS scratch cleanup, recreate the native ARM64 environment
with:

```bash
bash scripts/clariden/bootstrap.sh
```

The bootstrap refuses non-ARM hosts and non-symlink `.venv` replacements. It
installs the locked Python 3.10 development environment and uv cache under the
project's IOPS scratch root.

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

If the external L1 parquet data is not yet available, run the explicitly
data-free deployment gate instead:

```bash
make preflight-cloud-synthetic
```

This preserves every environment, Git, configuration, generated-script, and
launcher check, but replaces the raw-data and frozen-table requirements with an
instantiated 57-feature `SyntheticL1ADDataModule` contract check. It verifies
normal, reference-normal, anomaly, and paper-compatible loader aliases. This mode
is suitable for infrastructure canaries only; it is not evidence that the
real-data gate or physics study is complete. `make preflight-cloud` continues to
default to strict real-data mode.

## 5. Run target-environment canaries

Submit bounded debug-partition jobs and verify the GPU, CUDA build, scheduler
account/partition, shared filesystem, checkpoint, logger, pairing, CAP, and
evaluator outputs. Keep each job's outputs and checkpoints in a job-specific
scratch directory. Local tests cannot prove those external services. Only after
the canaries succeed should the full sweep matrix be submitted.

On Clariden, the two reviewed canary jobs are:

```bash
mkdir -p /iopsstor/scratch/cscs/vjimenez/adatl1/logs/slurm
sbatch scripts/clariden/canary_synthetic.sbatch
sbatch scripts/clariden/canary_cchamber.sbatch
```

The first job covers the analytical smoke profile, a 57-feature L1-shaped VAE,
the model/checkpoint/CAP/evaluator workflow, and controlled frozen-table pairing.
The second independently covers the public Causal Chamber download, pairing AE,
separate 1,000-pair validation/test tables, CAP consumption, and the
`uniform_red_mid` intervention. Both use account `a0166`, the `debug` partition,
one GPU, a 90-minute limit, offline logging, and job-specific scratch artifacts.

## 6. Evidence boundary

A green smoke/preflight means the implementation and selected deployment inputs
are structurally ready. A green synthetic preflight says nothing about external
L1 data availability. Neither mode establishes the scientific result. Full-data
sweeps, shared candidate replay, at least three paired retraining seeds, every
required intervention/domain, final evaluation, and paired aggregation remain
mandatory.

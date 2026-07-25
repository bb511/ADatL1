[![Email Badge](https://img.shields.io/badge/blah-podagiu%40ethz.ch-blue?style=flat-square&logo=minutemailer&logoColor=white&label=%20&labelColor=grey)](mailto:podagiu@ethz.ch)
[![Python: version](https://img.shields.io/badge/python-3.10-blue?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black?style=flat-square&logo=black)](https://github.com/psf/black)

# Anomaly Detection @ Trigger

## Setup

This repository uses [uv](https://docs.astral.sh/uv/) for package management.

Create the local environment and install the development dependencies:

```
uv sync --group dev
```

To include the optional quantisation stack:

```
uv sync --group dev --extra quant
```

The repository reads local paths from `.env`. A default local file has been
generated for this checkout; regenerate it with:

```
bash scripts/setup.sh
```

## Data

The LHC L1 AD data runs on data produced by [this code](https://github.com/bb511/adl1t_datamaker).
For the LHC L1 AD dataset, you must download the data [here](https://cernbox.cern.ch/s/dRnVAa3ZDHWl2bs), unzip it, and then point `RAW_DATA_DIR` in `.env` or `paths.raw_data_dir` in Hydra at that directory.

CIFAR-10 is the lightweight demo dataset and is downloaded automatically under
`data/cifar10` by the CIFAR datamodule.

Causal Chamber `lt_interventions_standard_v1` is the controlled low-dimensional
tabular benchmark used for intervention/anomaly experiments. It is downloaded
automatically under `data/causal_chamber`; the default config trains on
`uniform_reference`, keeps a disjoint `reference_normal` split for agnostic CAP/W1
comparison, and exposes the `uniform_*` interventions as anomalies.

## Usage

Run the short smoke training:

```
uv run python tests/train.py experiment=demo/cifar10_ae
uv run python tests/train.py experiment=demo/l1_vae
uv run python tests/train.py experiment=demo/l1_vicreg
uv run python tests/train.py experiment=demo/l1_wnae
uv run python tests/train.py experiment=demo/l1_rvae
uv run python tests/train.py experiment=demo/cchamber_ae
uv run python tests/train.py experiment=demo/cchamber_vae
uv run python tests/train.py experiment=demo/cchamber_svdd
uv run python tests/train.py experiment=demo/cchamber_realnvp
```

Run a configured experiment:

```
uv run python src/train.py experiment=cifar10/ae
uv run python src/train.py experiment=cchamber/ae_agnostic
uv run python src/train.py experiment=physics/ae paths.raw_data_dir=/path/to/adl1t_data/parquet_files
```

Run the publication-scale controlled Gaussian-subspace analytical study:

```
make analytical
```

Run its fast end-to-end pipeline check:

```
make analytical-smoke
```

Run the complete data-free smoke suite:

```
make smoke
```

This runs the analytical artifact smoke, the four-model/three-seed
checkpoint-and-reporting smoke, the controlled frozen-encoder pairing smoke, and
a real-public-CSV Causal Chamber pairing producer/consumer smoke. Causal Chamber
is downloaded automatically if needed. The results are ignored artifacts under
`results/`. These are structural tests, not paper results.

Run the two model-oriented smokes separately with:

```bash
make model-smoke
make pairing-smoke
make cchamber-pairing-smoke
```

To run only one synthetic anomaly model and seed:

```
uv run python scripts/synthetic/smoke.py --models ae --seeds 123
```

Generate reproducible paper launch scripts:

```
uv run python scripts/generation.py list --dataset physics --model ae --strategy cap
uv run python scripts/generation.py generate --name physics_ae_cap --stage sweep
```

The generator writes shell scripts plus `manifest.json` and `manifest.md` under
`scripts/generated/<experiment>/`. The manifests record the Optuna-tuned
parameters from `configs/hparams_search`, fixed Hydra overrides, validation
strategy overrides, Optuna sweeper overrides, and reporting factors such as seeds
or benchmark domains.

Before generating expensive jobs, validate all 76 core sweep compositions and
generated shell scripts:

```bash
make preflight-local
```

The strict cloud gate additionally requires a clean commit, real physics parquet
directories, and versioned validation/test Causal Chamber pair tables:

```bash
export CCHAMBER_VALID_PAIR_TABLE=/shared/pairs/valid_ae_pairs.pt
export CCHAMBER_TEST_PAIR_TABLE=/shared/pairs/test_ae_pairs.pt
make preflight-cloud
```

See `DEPLOYMENT.md` for the exact handoff and for checks that still must happen on
the target cluster.

After sweeps, `scripts/paper_pipeline.py` validates label-free selection,
resolves retrained checkpoints, collects evaluator `values.csv` files, and
generates seed-aware tables, plots, provenance, and a Markdown report. See
`EXPERIMENT_STATUS.md` for the current evidence and exact remaining cluster work.

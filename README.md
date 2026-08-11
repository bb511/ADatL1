[![Email Badge](https://img.shields.io/badge/blah-podagiu%40ethz.ch-blue?style=flat-square&logo=minutemailer&logoColor=white&label=%20&labelColor=grey)](mailto:podagiu@ethz.ch)
[![Python: version](https://img.shields.io/badge/python-3.10-blue?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black?style=flat-square&logo=black)](https://github.com/psf/black)

# Anomaly Detection @ Trigger

## Setup

This repository uses [poetry](https://python-poetry.org/) for package management.
We recommend setting this up using poetry.
However, if you do not want to use poetry, skip to [here](#setup-without-poetry).

Install the dependencies using poetry by running the following command in the repository root:
```
poetry install --no-root
```

To install the dependencies required by the quantisation packages:
```
poetry install --extras quant --no-root
```

## Setup without Poetry

To install the dependencies using `pip`, use
```
pip install -r requirements.txt
```

## Project layout

Run this once. It creates `data/ logs/ outputs/ checkpoints/` and writes the `.env` that
`configs/paths/default.yaml` reads through `${oc.env:PROJECT_ROOT}`. Nothing composes
without it. Re-running never overwrites an existing `.env`.
```
bash scripts/setup.sh
```
To keep those directories on another filesystem, edit `RES_DIR` in `.env` and then run
`bash scripts/symbolink.sh`, which replaces them with symlinks. That step **deletes** any
real directory in the way, so run it before the first training, not after; it asks for
confirmation unless given `--force`.

## Data

The LHC L1 AD data is produced by [this code](https://github.com/bb511/adl1t_datamaker).
See more details about it there.
We recommend using the HuggingFace variant [`podagiu/anomaly_detection_cmsl1t`](https://huggingface.co/datasets/podagiu/anomaly_detection_cmsl1t) of the dataloader of the LHC L1 AD data.

## Licence

The code is MIT (see `LICENSE`). The dataset is released separately under CC0 1.0.

## Usage

Training and evaluation both run from `src/train.py`, selected by an experiment config:
```
python src/train.py \
    experiment=physics/ae \
    paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
    trainer=gpu trainer.devices=[0]
```
Add `data=basis_hf` to read the published record instead, in which case no data path is needed.
Domains are `physics/`, `cifar10/` and `robustad/`; `*_agnostic` variants validate with the
label-free objectives instead of signal efficiency. Hyperparameter searches use
`--multirun hparams_search=physics/ae_optuna`.

The exact commands behind every result in the paper are in [`scripts/`](scripts/README.md) —
one catalogue of commented, copy-pasteable `src/train.py` invocations per model and domain,
plus the tooling that generated them, submitted them to slurm, and harvested the results.
The experiment configs already carry the hyperparameter values reported in the paper.

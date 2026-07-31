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

Run this once, from the repository root. It creates `data/ logs/ outputs/ results/ checkpoints/`
and writes the `.env` that `configs/paths/default.yaml` reads through `${oc.env:PROJECT_ROOT}`.
Nothing composes without it.
```
bash scripts/setup.sh
```
Edit `RES_DIR` in `.env` if those directories should live elsewhere, then run
`bash scripts/symbolink.sh` to symlink them.

## Data

The LHC L1 AD data runs on data produced by [this code](https://github.com/bb511/adl1t_datamaker).
For the LHC L1 AD dataset, you must download the data [here](https://cernbox.cern.ch/s/dRnVAa3ZDHWl2bs), unzip it, and then point to it by configuring `paths.raw_data_dir` in the running scripts.

The data is one directory per dataset, each holding one subdirectory per trigger object
(`muons/`, `jets/`, `egammas/`, `taus/`, `ET/`, `FET/`, `HT/`, `event_info/`, `seeds/`, ...) of
parquet shards. Values are in L1 integer hardware units; `configs/data/l1_scales/default.yaml`
records the conversion factors to physical units.

## Licence

The code is MIT (see `LICENSE`). The dataset is released separately under CC-BY-4.0.

## Usage

Training and evaluation both run from `src/train.py`, selected by an experiment config:
```
python src/train.py \
    experiment=physics/ae \
    paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
    trainer=gpu trainer.devices=[0]
```
Domains are `physics/`, `cifar10/` and `robustad/`; `*_agnostic` variants validate with the
label-free objectives instead of signal efficiency. Hyperparameter searches use
`--multirun hparams_search=physics/ae_optuna`.

See the `scripts` directory for the exact commands used to run the experiments described in the
paper. The experiments are already configured with the hyperparameter values described for the
models shown in the paper.

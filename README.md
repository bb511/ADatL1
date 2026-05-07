[![Python: version](https://img.shields.io/badge/python-3.10-blue?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black?style=flat-square&logo=black)](https://github.com/psf/black)

# Anomaly Detection @ Trigger

## Setup

This repository uses [poetry](https://python-poetry.org/) for package management.
We recommend setting this up using poetry.
However, if you do not want to use poetry, skip to [here](#setup-without-poetry).

Install the dependencies using poetry by running the following command in the parent directory of the repository:
```
poetry install --no-root
```

To install the dependencies required by the quantisation packages:
```
poetry install --with quant --no-root
```

## Setup without Poetry

To install the dependencies using `pip`, use
```
pip install -r requirements.txt
```

## Data

The LHC L1 AD data runs on data produced by [this code](https://github.com/cdfpzmvpvg/info_ad_data).
For the LHC L1 AD dataset, you must download the data [here](https://cernbox.cern.ch/s/dRnVAa3ZDHWl2bs), unzip it, and then point to it by configuring `paths.raw_data_dir` in the running scripts.

## Usage

See the `scripts` directory for scripts used to run the experiments described in the paper.
The experiments are already configured with the hyperparameter values described for the mdoels shown in the paper.

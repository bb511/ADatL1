[![Email Badge](https://img.shields.io/badge/blah-podagiu%40ethz.ch-blue?style=flat-square&logo=minutemailer&logoColor=white&label=%20&labelColor=grey)](mailto:podagiu@ethz.ch)
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

To install the dependencies required by the approximation capacity package:
```
poetry install --with cap
```

To install the dependencies required by the quantisation packages:
```
poetry install --with quant
```

## Setup without Poetry

To install the dependencies using `pip`, use
```
pip install -r requirements.txt
```

## Data

This repository runs on data produced by [this code](https://github.com/bb511/adl1t_datamaker).
The mentioned repository requires access to L1TNtuple data.

## Usage

See the `scripts` directory for scripts used to run different parts of this repo.
For example, to train a simple AE


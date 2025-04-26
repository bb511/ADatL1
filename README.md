# AD@L1

![https://github.com/ashleve/lightning-hydra-template/blob/main/configs/debug/profiler.yaml]

![https://github.com/CausalLearningAI/CausalMNIST]

## Setup

- First, install the required dependencies in a virtual environment.

```{bash}
setup.sh
```

- Folders `data`, `logs`, `outputs` and `results` will be created. The repository structure should be as follows: 

```
pa-covariate-shift/
├── configs/                # Containing YAML configuration files
├── data/                   # Containing datasets and model checkpoints
├── logs/                   # Containing the logs of each execution
├── outputs/                # Containing the command output of each execution
├── results/                # Containing figures and tables resulting from the experiments
├── scripts/                # Containing pre-built scripts
├── src/                    # Containing the source code
|   ├── callbacks/          # Containing Callback implementations
|   ├── data/               # Containing LightningDataModule implementations
|   ├── models/             # Containing LightningModule implementations
|   ├── plot/               # Containing python scripts to generate plots
|   └── utils/              # Containing useful methods
├── tests/                  # Containing pre-built sanity checks
├── requirements.txt        # Containing python dependencies
└── README.md               # This file
```

- Create a `.env` file to define the environment variables.

```
poetry export --without-hashes --format=requirements.txt > requirements.txt
./scripts/requirements.txt
```

```
PROJECT_ROOT="."
RES_DIR = "." # set to the desired location
DATA_DIR = "${RES_DIR}/data"
LOG_DIR = "${RES_DIR}/logs"
OUTPUT_DIR = "${RES_DIR}/outputs"
```

- Create the three data folders according to the `RES_DIR` variable you have set up, or run the `./scripts/create_soft_links.sh` to directly create soft links to the specified directory paths (useful if you want to store the data in remote
locations).

- To flawlessly rely on `wandb` logging, you should set up automatic account login with your credentials (account and API key).

```
pip install wandb
wandb login
```

## Train

```{bash}
python src/train.py experiment=example
```

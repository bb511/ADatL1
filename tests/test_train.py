import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train

# These run the real training loop, so they need the L1AD parquet data on disk.
pytestmark = pytest.mark.skipif(
    "ADL1T_RAW_DATA_DIR" not in os.environ,
    reason="Set ADL1T_RAW_DATA_DIR to the parquet data to run the training tests.",
)


def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
    train(cfg_train)

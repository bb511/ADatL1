from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train


def test_train_skips_evaluator_when_unconfigured(cfg_train: DictConfig) -> None:
    """Training should complete without an evaluator when evaluation config is null."""
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.evaluation = None
        cfg_train.test = False
        cfg_train.callbacks = None

    HydraConfig().set_config(cfg_train)
    _, objects = train(cfg_train)

    assert objects["evaluator"] is None

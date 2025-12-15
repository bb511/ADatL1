# Main training script.
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import copy
import gc

import hydra
import pytorch_lightning as pl

import torch
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from omegaconf import OmegaConf, DictConfig
from colorama import Fore, Back, Style

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Add resolvers to evaluate operations in the .yaml configuration files
from src.utils.omegaconf import register_resolvers

register_resolvers()

from src.utils import RankedLogger
from src.utils import extras
from src.utils import get_metric_value
from src.utils import instantiate_callbacks
from src.utils import instantiate_eval_callbacks
from src.utils import instantiate_loggers
from src.utils import log_hyperparameters
from src.utils import task_wrapper
from src.evaluation.evaluator import Evaluator

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating algorithm <{cfg.algorithm._target_}>")
    algorithm: LightningModule = hydra.utils.instantiate(cfg.algorithm)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "algorithm": algorithm,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=algorithm, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    # Evaluate the training.
    if cfg.get("test"):
        log.info(Back.MAGENTA + 8*'-' + "STARTING TESTING" + 8*'-')
        test(cfg, datamodule, algorithm, logger)

    metric_dict = {**train_metrics}
    return metric_dict, object_dict

def test(cfg: DictConfig, datamodule, algorithm, logger):
    """Evaluate the training."""
    print(cfg.get('evaluator'))
    if cfg.get('evaluator') is None:
        log.info(Back.YELLOW + "No evaluator config found... Skipping testing")
        return

    # Merge the trainer configuration with the evaluator. This is done since the
    # Evaluator object is basically a wrapper around a trainer with extra steps.
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
    evaluator_config = OmegaConf.to_container(cfg.evaluator, resolve=True)
    merged_dict = {**trainer_config, **evaluator_config}
    evaluator_cfg = OmegaConf.create(merged_dict)

    log.info("Instantiating evaluator callbacks...")
    callbacks = instantiate_eval_callbacks(cfg.get('evaluator_callbacks'), datamodule)
    log.info(f"Instantiating evaluator <{evaluator_cfg._target_}>")
    evaluator = hydra.utils.instantiate(
        evaluator_cfg, callbacks=callbacks, logger=logger
    )

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    run_ckpts = Path(cfg.paths.checkpoints_dir) / cfg.experiment_name / cfg.run_name

    evaluator.evaluate_run(run_ckpts, algorithm, test_loader)

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()

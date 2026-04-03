# Main training script.
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import gc

import os

os.environ["KERAS_BACKEND"] = "torch"

import hydra
import pytorch_lightning as pl

import torch
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from omegaconf import OmegaConf, DictConfig
from colorama import Fore, Back, Style
from math import inf
from hydra.core.hydra_config import HydraConfig

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

log = RankedLogger(__name__, rank_zero_only=True)

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*LeafSpec.*TreeSpec.*is_leaf.*",
    category=FutureWarning,
)


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

    # Get validation report, and also set hp optimisation values.
    log.info(Fore.CYAN + "Instantiating evaluator...")
    evaluator = _get_evaluator(cfg, datamodule, logger)
    run_ckpts = Path(cfg.paths.checkpoints_dir) / cfg.experiment_name / cfg.run_name

    log.info(Back.MAGENTA + 8 * "-" + "STARTING RUN VALIDATION" + 8 * "-")
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    evaluator.evaluate_run(
        run_ckpts, algorithm, val_loader, "val", set_optimized_metric=True
    )
    object_dict.update({"evaluator": evaluator})

    # Evaluate once more on a held out test set for final performance.
    if cfg.get("test"):
        log.info(Back.MAGENTA + 8 * "-" + "STARTING RUN TESTING" + 8 * "-")
        datamodule.setup("test")
        test_loader = datamodule.test_dataloader()
        evaluator.evaluate_run(run_ckpts, algorithm, test_loader, "test")
        object_dict.update({"evaluator": evaluator})

    metric_dict = {**train_metrics}
    return metric_dict, object_dict


def _worst_for(direction: str) -> float:
    return inf if direction == "minimize" else -inf


def _get_evaluator(cfg: DictConfig, datamodule, logger):
    """Configure the evaluator object and return it."""
    if cfg.get("evaluator") is None:
        log.info(Back.YELLOW + "No evaluator config found... Skipping testing")
        return

    # Merge the trainer configuration with the evaluator. This is done since the
    # Evaluator object is basically a wrapper around a trainer with extra steps.
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
    evaluator_config = OmegaConf.to_container(cfg.evaluator, resolve=True)

    merged_dict = {**trainer_config, **evaluator_config}
    evaluator_cfg = OmegaConf.create(merged_dict)

    log.info("Instantiating evaluator callbacks...")
    callbacks = instantiate_eval_callbacks(cfg.get("evaluator_callbacks"), datamodule)
    log.info(f"Instantiating evaluator <{evaluator_cfg._target_}>")
    evaluator = hydra.utils.instantiate(
        evaluator_cfg,
        callbacks=callbacks,
        logger=logger,
        optimized_metric_config=cfg.get("optimized_metric_config"),
    )

    return evaluator


def _get_directions(cfg):
    # 1) Prefer your own config (always available)
    if "optimized_metric_config" in cfg:
        # multi-objective if sec_metric exists
        main_dir = cfg.optimized_metric_config.main_metric.direction
        if "sec_metric" in cfg.optimized_metric_config:
            sec_dir = cfg.optimized_metric_config.sec_metric.direction
            return [main_dir, sec_dir]
        return [main_dir]

    # 2) Fallback: hydra optuna sweeper (only in sweeps)
    try:
        hydra_cfg = HydraConfig.get()
        dirs = getattr(hydra_cfg.sweeper, "direction", None)
        if dirs is None:
            return None
        return list(dirs) if isinstance(dirs, (list, tuple)) else [dirs]
    except Exception:
        return None


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
    metric_dict, object_dict = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(
    # metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )
    evaluator = object_dict.get("evaluator", None)
    metric_value = evaluator.optimized_metric if evaluator else None

    # Clean up.
    del object_dict
    del metric_dict
    gc.collect()

    def _worst_for(direction: str) -> float:
        return float("inf") if direction == "minimize" else -float("inf")

    if metric_value is None or (
        isinstance(metric_value, (list, tuple)) and any(v is None for v in metric_value)
    ):
        dirs = _get_directions(cfg) or ["minimize"]
        worst = tuple(_worst_for(d) for d in dirs)
        return worst[0] if len(worst) == 1 else worst

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()

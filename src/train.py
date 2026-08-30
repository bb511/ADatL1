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

from src.evaluation.leakage_probe.persistence import (
    evaluate_and_record_loss_total_leakage_probes,
    log_leakage_probe_outcome_metadata,
)
from src.evaluation.leakage_probe.provenance import (
    make_leakage_probe_run_metadata,
)

from src.evaluation.leakage_probe.serialization import (
    log_four_probe_metrics,
    log_shuffled_target_metrics,
)

from src.utils import RankedLogger
from src.utils import extras
from src.utils import get_metric_value
from src.utils import instantiate_callbacks
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
        resume_ckpt_path = cfg.get("ckpt_path")

        trainer.fit(
            model=algorithm,
            datamodule=datamodule,
            ckpt_path=resume_ckpt_path,
            weights_only=False if resume_ckpt_path is not None else None,
        )

    train_metrics = trainer.callback_metrics

    post_training_metrics: Dict[str, float] = {}

    if cfg.get("train"):
        log.info("Releasing fit dataloaders before run validation...")
        _release_fit_dataloaders(trainer, datamodule)

    # Get validation report, and also set hp optimisation values.
    log.info(Fore.CYAN + "Instantiating evaluator...")
    evaluator = _get_evaluator(cfg, datamodule, logger)
    run_ckpts = Path(cfg.paths.checkpoints_dir) / cfg.experiment_name / cfg.run_name

    log.info(Back.MAGENTA + 8 * "-" + "STARTING RUN VALIDATION" + 8 * "-")
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    try:
        evaluator.evaluate_run(
            run_ckpts, algorithm, val_loader, "val", set_optimized_metric=True
        )
    finally:
        # The physics datamodule keeps every split in RAM. Release validation before
        # setup("test") loads another full copy of the model/control tensors.
        evaluator.release_dataloaders()
        del val_loader
        datamodule.teardown("validate")
        gc.collect()

    evaluation_cfg = cfg.get("evaluation")
    leakage_probe_cfg = (
        evaluation_cfg.get("leakage_probes")
        if evaluation_cfg is not None
        else None
    )
    if leakage_probe_cfg and leakage_probe_cfg.get("enabled", False):
        leakage_probe_evaluation_mode = str(
            leakage_probe_cfg.get("mode", "validation")
        )
        leakage_probe_run_metadata = (
            make_leakage_probe_run_metadata(
                autoencoder_seed=cfg.get("seed"),
                algorithm_config=cfg.algorithm,
            )
        )
        log.info(
            Back.MAGENTA
            + 8 * "-"
            + "STARTING "
            + leakage_probe_evaluation_mode.upper()
            + " LEAKAGE PROBES"
            + 8 * "-"
        )
        leakage_probe_outcome = (
            evaluate_and_record_loss_total_leakage_probes(
                algorithm,
                datamodule,
                run_ckpts,
                device=algorithm.device,
                run_shuffled_target_controls=bool(
                    leakage_probe_cfg.get(
                        "run_shuffled_target_controls",
                        False,
                    )
                ),
                evaluation_mode=leakage_probe_evaluation_mode,
                run_metadata=leakage_probe_run_metadata,
            )
        )

        leakage_probe_metadata = (
            log_leakage_probe_outcome_metadata(
                leakage_probe_outcome,
                logger,
            )
        )

        object_dict.update(
            {
                "leakage_probe_outcome": (
                    leakage_probe_outcome
                ),
                "leakage_probe_path": (
                    leakage_probe_outcome.output_path
                ),
                "leakage_probe_metadata": (
                    leakage_probe_metadata
                ),
            }
        )

        if leakage_probe_outcome.probe_valid:
            leakage_probe_result = (
                leakage_probe_outcome.result
            )

            if leakage_probe_result is None:
                raise RuntimeError(
                    "A valid leakage-probe outcome has "
                    "no four-probe result."
                )

            primary_probe_metrics = (
                log_four_probe_metrics(
                    leakage_probe_result,
                    logger,
                    step=trainer.global_step,
                )
            )

            shuffled_probe_metrics = (
                log_shuffled_target_metrics(
                    leakage_probe_result,
                    logger,
                    step=trainer.global_step,
                )
            )

            leakage_probe_metrics = {
                **primary_probe_metrics,
                **shuffled_probe_metrics,
            }

            post_training_metrics.update(
                leakage_probe_metrics
            )

            object_dict.update(
                {
                    "leakage_probe_result": (
                        leakage_probe_result
                    ),
                    "leakage_probe_metrics": (
                        leakage_probe_metrics
                    ),
                }
            )

            log.info(
                "Stored valid leakage probes at "
                f"{leakage_probe_outcome.output_path}."
            )
        else:
            diagnostic_probe_metrics: Dict[
                str,
                float,
            ] = {}

            diagnostic_result = (
                leakage_probe_outcome
                .diagnostic_result
            )

            if diagnostic_result is not None:
                diagnostic_probe_metrics = (
                    log_shuffled_target_metrics(
                        diagnostic_result,
                        logger,
                        step=trainer.global_step,
                    )
                )

                post_training_metrics.update(
                    diagnostic_probe_metrics
                )

            object_dict.update(
                {
                    "leakage_probe_result": None,
                    "leakage_probe_metrics": (
                        diagnostic_probe_metrics
                    ),
                    "leakage_probe_diagnostic_result": (
                        diagnostic_result
                    ),
                }
            )

            log.error(
                "Leakage-probe evaluation is invalid: "
                f"{leakage_probe_outcome.rejection_reason}: "
                f"{leakage_probe_outcome.rejection_message}. "
                "The invalid result was stored at "
                f"{leakage_probe_outcome.output_path}."
            )

    object_dict.update({"evaluator": evaluator})

    # Evaluate once more on a held out test set for final performance.
    if cfg.get("test"):
        log.info(Back.MAGENTA + 8 * "-" + "STARTING RUN TESTING" + 8 * "-")
        datamodule.setup("test")
        test_loader = datamodule.test_dataloader()
        try:
            evaluator.evaluate_run(run_ckpts, algorithm, test_loader, "test")
        finally:
            evaluator.release_dataloaders()
            del test_loader
            datamodule.teardown("test")
            gc.collect()
        object_dict.update({"evaluator": evaluator})

    metric_dict = {
        **train_metrics,
        **post_training_metrics,
    }
    return metric_dict, object_dict


def _release_fit_dataloaders(
    trainer: Trainer, datamodule: LightningDataModule
) -> None:
    """Release Lightning's train/validation loaders before standalone evaluation.

    Lightning 2.6 keeps the processed ``CombinedLoader`` and original dataloader
    source on both the fit loop and its nested validation loop after ``fit`` returns.
    The physics datasets own multi-gigabyte in-memory tensors, so those references
    must be cleared before ``datamodule.setup("validate")`` loads validation again.
    """
    fit_loop = trainer.fit_loop
    fit_validation_loop = getattr(getattr(fit_loop, "epoch_loop", None), "val_loop", None)

    for loop in (fit_loop, fit_validation_loop):
        if loop is None:
            continue

        if hasattr(loop, "_combined_loader"):
            loop._combined_loader = None

        data_source = getattr(loop, "_data_source", None)
        if data_source is not None:
            data_source.instance = None

    # Lightning calls teardown at the end of fit, but running it again after its
    # loader references are gone lets project datamodules release any remaining
    # split tensors. The repository datamodule teardown methods are idempotent.
    datamodule.teardown("fit")
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _worst_for(direction: str) -> float:
    return inf if direction == "minimize" else -inf


def _get_evaluator(cfg: DictConfig, datamodule, logger):
    """Configure the evaluator object and return it."""
    if cfg.get("evaluation") is None:
        log.info(Back.YELLOW + "No evaluation config found... Skipping testing")
        return

    eval_config = cfg.get("evaluation")
    # Merge the trainer configuration with the evaluation. This is done since the
    # Evaluator object is basically a wrapper around a trainer with extra steps.
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
    evaluator_config = OmegaConf.to_container(eval_config.evaluator, resolve=True)

    merged_dict = {**trainer_config, **evaluator_config}
    evaluator_cfg = OmegaConf.create(merged_dict)

    log.info("Instantiating evaluator callbacks...")
    callbacks = instantiate_callbacks(eval_config.get("callbacks"))
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

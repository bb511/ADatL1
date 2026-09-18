"""Stage 1 of 4: train the autoencoder.

This script fits the model and runs the ordinary evaluation callbacks composed
into ``evaluation.callbacks`` -- for ``experiment=physics/ae`` that is the plain
AE's usual plots (reconstruction, anomaly score, Wasserstein, threshold drift).
Its scientific output is the checkpoint:

    <checkpoints_dir>/<experiment_name>/<run_name>/loss_total.ckpt

The analysis that used to follow training in this same process now lives in its
own entrypoints, so that every autoencoder can be trained before anything is
analysed, and so each step can be scheduled with the resources it needs:

    stage 2  src/run_probes.py        the four leakage probes  (slow: ~27 min)
    stage 3  src/run_eval_metrics.py  the remaining Pareto metrics (memory-hungry)
    stage 4  scripts/collect_pareto_study.py + scripts/select_pareto_front.py

Stages 2 and 3 are independent of each other and may run concurrently. Both must
compose the same config as the stage-1 run that produced the checkpoint.
"""
from typing import Any, Dict, List, Optional, Tuple
import gc

import os

os.environ["KERAS_BACKEND"] = "torch"

import hydra
import pytorch_lightning as pl

from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig, open_dict
from colorama import Fore, Back
from math import inf
from hydra.core.hydra_config import HydraConfig

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Add resolvers to evaluate operations in the .yaml configuration files
from src.utils.omegaconf import register_resolvers

register_resolvers()

from src.utils.pareto_manifest import write_resolved_pareto_manifest
from src.utils.run_manifest import write_run_manifest, write_stage_status
from src.utils.stage import (
    get_evaluator,
    release_fit_dataloaders,
    run_checkpoint_dir,
)

from src.utils import RankedLogger
from src.utils.instrumentation import log_phase
from src.utils import extras
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
    pareto_manifest_path = write_resolved_pareto_manifest(cfg)
    if pareto_manifest_path is not None:
        # f-string, not %-args: RankedLogger.log binds the first positional
        # argument after the message to `rank`, so %-style lazy formatting is
        # silently broken throughout this codebase (see src/utils/pylogger.py).
        log.info(f"Saved resolved Pareto study manifest to {pareto_manifest_path}")

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

    train_enabled = bool(cfg.get("train"))

    resume_ckpt_path = cfg.get("ckpt_path")

    if train_enabled:
        log.info(
            f"Starting training (max_epochs={getattr(trainer, 'max_epochs', '?')}, "
            f"resume={resume_ckpt_path or 'none'})"
        )
        with log_phase("fit"):
            trainer.fit(
                model=algorithm,
                datamodule=datamodule,
                ckpt_path=resume_ckpt_path,
                weights_only=False if resume_ckpt_path is not None else None,
            )
    else:
        # train=false means evaluate checkpoints that already exist on disk. It
        # must NOT call trainer.fit: with ckpt_path unset that silently trains a
        # fresh model from scratch, which is what this branch used to do and is
        # the opposite of what the flag says. The evaluator loads each
        # checkpoint's weights itself (Evaluator.evaluate_ckpt does an explicit
        # load_state_dict), so the model passed through here only has to be
        # correctly shaped, not trained.
        log.info(
            "Skipping training (train=false): evaluating checkpoints already on disk."
        )

    _prepare_data_for_checkpoint_only_evaluation(
        datamodule,
        train_enabled=train_enabled,
    )

    train_metrics = trainer.callback_metrics

    if train_enabled:
        log.info("Releasing fit dataloaders before run validation...")
        with log_phase("release fit dataloaders", collect=True):
            release_fit_dataloaders(trainer, datamodule)

    run_ckpts = run_checkpoint_dir(cfg)

    # Get validation report, and also set hp optimisation values.
    log.info(Fore.CYAN + "Instantiating evaluator...")
    evaluator = get_evaluator(cfg, logger)

    log.info(Back.MAGENTA + 8 * "-" + "STARTING RUN VALIDATION" + 8 * "-")
    with log_phase("load validation split"):
        datamodule.setup("validate")
        val_loader = datamodule.val_dataloader()
        loader_names = (
            sorted(val_loader.keys())
            if hasattr(val_loader, "keys")
            else type(val_loader).__name__
        )
        log.info(f"Validation loaders ({len(loader_names)}): {loader_names}")
    try:
        with log_phase("run validation"):
            evaluator.evaluate_run(
                run_ckpts, algorithm, val_loader, "val", set_optimized_metric=True
            )
    finally:
        # The physics datamodule keeps every split in RAM. Release validation before
        # setup("test") loads another full copy of the model/control tensors.
        with log_phase("release validation split", collect=True):
            evaluator.release_dataloaders()
            del val_loader
            datamodule.teardown("validate")
            gc.collect()

    # The leakage probes that used to run here are now stage 2:
    #     python3 src/run_probes.py <same overrides as this run>
    # They are the single most expensive step (~27 min per run, independent of
    # epoch count) and they need nothing from this process except the
    # loss_total.ckpt written above, so keeping them here forced every training
    # job to carry their runtime and their sklearn memory profile.

    object_dict.update({"evaluator": evaluator})

    # The per-run manifest is written here, at the end, for two reasons. The
    # training callback ClearRunCheckpointDir wipes the run directory when a fit
    # starts, so anything written earlier would not survive it; and a manifest
    # present is then a truthful claim that stage 1 got this far, which is what
    # stage 4 relies on when it reads an experiment directory instead of a
    # pre-declared plan.
    manifest_path = write_run_manifest(
        cfg,
        run_ckpts,
        mlflow_run_id=_mlflow_run_id(logger),
    )
    write_stage_status(
        run_ckpts,
        stage_name="train",
        ok=True,
        detail=f"max_epochs={getattr(trainer, 'max_epochs', '?')}",
        artifacts=[str(run_ckpts / "loss_total.ckpt"), str(manifest_path)],
    )
    object_dict.update({"run_manifest_path": manifest_path})

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

    return dict(train_metrics), object_dict


def _mlflow_run_id(loggers) -> Optional[str]:
    """Return the MLflow run id of this run, if an MLflow logger is attached.

    Recorded in the manifest so the later stages can log into the SAME MLflow
    run rather than creating a second one with the same name.
    """
    for candidate in loggers or []:
        run_id = getattr(candidate, "run_id", None)
        if run_id:
            return str(run_id)
    return None


def _prepare_data_for_checkpoint_only_evaluation(
    datamodule: LightningDataModule,
    *,
    train_enabled: bool,
) -> None:
    """Initialize datamodule cache state when ``trainer.fit`` was skipped."""
    if train_enabled:
        return

    # trainer.fit() normally invokes prepare_data(). Checkpoint-only runs skip fit,
    # but the standalone evaluator and leakage probes still need the prepared cache
    # location recorded by the datamodule.
    log.info("Preparing data for checkpoint-only evaluation...")
    datamodule.prepare_data()


def _worst_for(direction: str) -> float:
    return inf if direction == "minimize" else -inf


def _get_directions(cfg):
    # 1) Our own config: present in any composable run, so 2) below is a dead fallback.
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
    # A numeric run_name (RUN_NAME=7) is resolved by OmegaConf as an int, and
    # MLflow's protobuf run_name field only accepts str, so create_run dies with
    # "TypeError: bad argument type for built-in operation" long before training
    # starts. A numeric run name is legitimate user input, so coerce it here --
    # once, before anything reads it -- rather than making the user rename runs.
    # Doing it on the config (not on the logger) keeps run_name consistent across
    # the checkpoint dirpaths, the binning plot dirs and the MLflow run.
    if "run_name" in cfg and not isinstance(cfg.run_name, str):
        coerced = str(cfg.run_name)
        with open_dict(cfg):
            cfg.run_name = coerced
        log.warning(
            f"run_name was not a string; coerced to {coerced!r} for MLflow compatibility."
        )

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, object_dict = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    evaluator = object_dict.get("evaluator", None)
    metric_value = evaluator.optimized_metric if evaluator else None

    # Clean up.
    del object_dict
    del metric_dict
    gc.collect()

    # A missing metric reports the worst value per direction, so optuna records the
    # trial instead of aborting; the order must match `hydra.sweeper.direction`.
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

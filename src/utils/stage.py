"""Shared construction for the four pipeline stages.

The pipeline is deliberately split into four executable stages so that every
autoencoder can be trained before anything is analysed, and so that each stage
can be scheduled with the resources it actually needs:

    stage 1  src/train.py             fit + the ordinary AE plots -> loss_total.ckpt
    stage 2  src/run_probes.py        four leakage probes on that checkpoint
    stage 3  scripts/collect_pareto_study.py + scripts/select_pareto_front.py

    (not a stage) src/run_eval_metrics.py  re-evaluation of a checkpoint without
                                           training, e.g. on the test split

Stage 2 and src/run_eval_metrics.py consume only
``<checkpoints_dir>/<experiment_name>/<run_name>/loss_total.ckpt`` and write
into disjoint subdirectories of that run folder, so they may run concurrently.

Every stage must compose the SAME Hydra config as the stage-1 run it analyses.
The checkpoint carries the weights but not the config, and both the evaluator and
the probe loader do a ``load_state_dict(..., strict=True)``: an architecture
override that differs from the trained run fails loudly rather than silently
measuring the wrong model.

Note on logging: ``RankedLogger.log`` binds the first positional argument after
the message to ``rank``, so %-style lazy formatting is silently broken (with
``rank_zero_only=False`` the record is dropped entirely). Always use f-strings
here. See src/utils/pylogger.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import gc

import hydra
import pytorch_lightning as pl
import torch

from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

from src.utils import RankedLogger
from src.utils import instantiate_callbacks
from src.utils import instantiate_loggers
from src.utils.mlflow_stage_run import find_stage1_run_id, tolerate_param_conflicts
from src.utils.run_manifest import verify_against_manifest

log = RankedLogger(__name__, rank_zero_only=True)

#: The single checkpoint every downstream stage consumes. Both the Evaluator
#: (via ``evaluation.evaluator.ckpts.loss_total``) and the leakage probes
#: (hardcoded in leakage_probe/persistence.py) agree on this name.
LOSS_TOTAL_CHECKPOINT = "loss_total.ckpt"


@dataclass
class StageContext:
    """Everything an analysis stage needs, built once from the config."""

    cfg: DictConfig
    datamodule: LightningDataModule
    algorithm: LightningModule
    logger: List[Logger]
    run_ckpts: Path
    #: The manifest stage 1 wrote for this run, when there is one.
    manifest: Optional[Dict[str, Any]] = None
    object_dict: Dict[str, Any] = field(default_factory=dict)
    #: The stage-1 MLflow run this stage logs into, or None when it opened its own.
    reused_mlflow_run_id: Optional[str] = None


def run_checkpoint_dir(cfg: DictConfig) -> Path:
    """Return the run's checkpoint folder.

    This mirrors the ``dirpath`` of the ModelCheckpoint callbacks
    (configs/callbacks/default.yaml) and the path train.py uses, so stage 1's
    output directory and the stages that read it can never drift apart.
    """
    return Path(cfg.paths.checkpoints_dir) / cfg.experiment_name / cfg.run_name


def reuse_stage1_mlflow_run(cfg: DictConfig, stage_name: str) -> Optional[str]:
    """Point the MLflow logger at the stage-1 run of this checkpoint.

    Without this every stage opens a new MLflow run with the same run name, so
    one checkpoint ends up with several runs and the training run (curves and
    artifacts) never receives the analysis metrics. With it, a checkpoint keeps
    one run, as before the pipeline was split. Falls back to a new run when no
    stage-1 run exists (e.g. a checkpoint copied in without its MLflow store).
    """
    logger_cfg = cfg.get("logger")
    if not logger_cfg or "mlflow" not in logger_cfg or logger_cfg.mlflow is None:
        return None
    mlflow_cfg = logger_cfg.mlflow
    if mlflow_cfg.get("run_id"):
        return str(mlflow_cfg.run_id)
    try:
        run_id = find_stage1_run_id(
            mlflow_cfg.get("tracking_uri"), str(cfg.experiment_name), str(cfg.run_name)
        )
    except Exception as err:  # never let bookkeeping kill an analysis job
        log.warning(f"[{stage_name}] Could not look up the stage-1 MLflow run: {err}")
        return None
    if run_id is None:
        log.warning(
            f"[{stage_name}] No stage-1 MLflow run for "
            f"{cfg.experiment_name}/{cfg.run_name}; this stage opens a new run."
        )
        return None
    with open_dict(mlflow_cfg):
        mlflow_cfg.run_id = run_id
    log.info(f"[{stage_name}] Logging into the stage-1 MLflow run {run_id}.")
    return run_id


def finish_stage_loggers(context: "StageContext", status: str = "success") -> None:
    """Close an MLflow run this stage opened itself.

    A reopened stage-1 run keeps the status stage 1 gave it. Only a run created
    here is terminated, so it no longer stays RUNNING forever.
    """
    if context.reused_mlflow_run_id:
        return
    for logger in context.logger or []:
        try:
            logger.finalize(status)
        except Exception as err:
            log.warning(f"Could not finalize logger {type(logger).__name__}: {err}")


def require_loss_total_checkpoint(run_ckpts: Path) -> Path:
    """Fail early and clearly when stage 1 has not produced its checkpoint.

    Without this the failure surfaces much later: the Evaluator quietly skips a
    missing root checkpoint, and the probe loader raises a ProbeExtractionError
    that is caught and written to disk as an *invalid result*, which then makes
    the whole configuration look scientifically rejected rather than simply not
    run yet.
    """
    checkpoint_path = run_ckpts / LOSS_TOTAL_CHECKPOINT
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing {LOSS_TOTAL_CHECKPOINT} in {run_ckpts}. "
            "Run stage 1 (src/train.py) for this experiment_name/run_name "
            "before running this stage."
        )
    return checkpoint_path


def build_stage_context(
    cfg: DictConfig,
    *,
    stage_name: str,
    require_checkpoint: bool = True,
    verify_manifest: bool = True,
    strict_manifest: bool = True,
) -> StageContext:
    """Instantiate the objects an analysis stage needs from an existing run.

    Deliberately does NOT build a Trainer or the training callbacks: stage 2 and
    run_eval_metrics never fit, and the training callbacks include ModelCheckpoint
    instances whose ``dirpath`` is the run folder. ClearRunCheckpointDir in
    particular would wipe the very checkpoint this stage is here to read.
    """
    run_ckpts = run_checkpoint_dir(cfg)
    log.info(f"[{stage_name}] Run checkpoint directory: {run_ckpts}")

    if require_checkpoint:
        checkpoint_path = require_loss_total_checkpoint(run_ckpts)
        log.info(f"[{stage_name}] Found checkpoint: {checkpoint_path}")

    # Check this config actually describes the checkpoint before spending minutes
    # loading data. strict=True by default: analysing a checkpoint with the wrong
    # gamma succeeds silently otherwise, because load_state_dict compares shapes
    # and nothing else. Pass manifest_strict=false on the command line for runs
    # trained before manifests existed.
    manifest = None
    if verify_manifest:
        manifest = verify_against_manifest(
            cfg, run_ckpts, stage_name=stage_name, strict=strict_manifest
        )

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating algorithm <{cfg.algorithm._target_}>")
    algorithm: LightningModule = hydra.utils.instantiate(cfg.algorithm)

    reused_run_id = reuse_stage1_mlflow_run(cfg, stage_name)
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    if reused_run_id:
        tolerate_param_conflicts(logger)

    # trainer.fit() normally calls prepare_data(). These stages skip fit, but the
    # evaluator and the probes still need the prepared cache location recorded by
    # the datamodule.
    log.info(f"[{stage_name}] Preparing data...")
    datamodule.prepare_data()

    object_dict: Dict[str, Any] = {
        "cfg": cfg,
        "datamodule": datamodule,
        "algorithm": algorithm,
        "logger": logger,
        "run_ckpts": run_ckpts,
        "stage": stage_name,
        "run_manifest": manifest,
    }

    return StageContext(
        cfg=cfg,
        datamodule=datamodule,
        algorithm=algorithm,
        logger=logger,
        run_ckpts=run_ckpts,
        manifest=manifest,
        object_dict=object_dict,
        reused_mlflow_run_id=reused_run_id,
    )


def get_evaluator(cfg: DictConfig, logger: Optional[List[Logger]]):
    """Configure the evaluator object and return it.

    The Evaluator is a wrapper around a Trainer, so the trainer config is merged
    underneath the evaluation-specific keys (accelerator, devices, precision come
    from the trainer; enable_checkpointing and ckpts from the evaluation block).
    """
    if cfg.get("evaluation") is None:
        log.warning("No evaluation config found; nothing to evaluate.")
        return None

    eval_config = cfg.get("evaluation")
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
    evaluator_config = OmegaConf.to_container(eval_config.evaluator, resolve=True)

    merged_dict = {**trainer_config, **evaluator_config}
    evaluator_cfg = OmegaConf.create(merged_dict)

    log.info("Instantiating evaluator callbacks...")
    callbacks = instantiate_callbacks(eval_config.get("callbacks"))
    log.info(f"Instantiating evaluator <{evaluator_cfg._target_}>")
    return hydra.utils.instantiate(
        evaluator_cfg,
        callbacks=callbacks,
        logger=logger,
        optimized_metric_config=cfg.get("optimized_metric_config"),
    )


def release_fit_dataloaders(
    trainer: Trainer, datamodule: LightningDataModule
) -> None:
    """Release Lightning's train/validation loaders before standalone evaluation.

    Lightning 2.6 keeps the processed ``CombinedLoader`` and original dataloader
    source on both the fit loop and its nested validation loop after ``fit``
    returns. The physics datasets own multi-gigabyte in-memory tensors, so those
    references must be cleared before ``datamodule.setup("validate")`` loads
    validation again.
    """
    fit_loop = trainer.fit_loop
    fit_validation_loop = getattr(
        getattr(fit_loop, "epoch_loop", None), "val_loop", None
    )

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
    release_accelerator_cache()


def release_accelerator_cache() -> None:
    """Drop Python and accelerator caches between phases."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

"""Stage 3 of 4: the remaining Pareto metrics on an already-trained checkpoint.

Reads ``<checkpoints_dir>/<experiment_name>/<run_name>/loss_total.ckpt``, which
stage 1 (src/train.py) wrote, replays the validation split through it, and lets
the evaluation callbacks write their summary artifacts:

    <run>/plots/val/loss_total/eff/eff_summary.json                 signal efficiency
    <run>/plots/val/loss_total/correlation_matrix/normal/
                                       mean_correlations.json       objective E
    <run>/plots/val/loss_total/latent_collapse/collapse_summary.json feasibility
    <run>/plots/val/loss_total/auroc/auroc_summary.json              diagnostic

Which of these actually appear is decided entirely by ``evaluation.callbacks``
in the composed experiment. ``experiment=physics/ae`` gives the ordinary AE
plots; ``experiment=physics/ae_metrics`` and ``experiment=physics/pareto_fet``
add the four artifacts above. This stage does not force any of them on, so that
one entrypoint serves both the plain-AE workflow and the Pareto study.

Compose the same config as the stage-1 run. Example:

    python3 src/run_eval_metrics.py \\
        experiment=physics/ae_metrics \\
        run_name=AE_LXPLUS_30ep \\
        paths.raw_data_dir=... \\
        trainer=cpu

This stage holds the validation split plus all 21 auxiliary signal datasets in
memory at once, so it is the memory-hungry analysis stage; the probes are the
slow one.
"""

from typing import Any, Dict, Tuple

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import hydra

from omegaconf import DictConfig
from colorama import Back

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.omegaconf import register_resolvers

register_resolvers()

from src.utils import RankedLogger
from src.utils import extras
from src.utils import task_wrapper
from src.utils.instrumentation import log_phase
from src.utils.run_manifest import write_stage_status
from src.utils.stage import (
    build_stage_context,
    get_evaluator,
    release_accelerator_cache,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def run_eval_metrics(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Replay the validation split through the stage-1 checkpoint.

    Returns (metric_dict, object_dict); task_wrapper unpacks this pair.
    """
    if cfg.get("evaluation") is None:
        raise ValueError(
            "No evaluation config. Compose the same experiment as the stage-1 "
            "run, for example experiment=physics/ae_metrics."
        )

    context = build_stage_context(
        cfg,
        stage_name="metrics",
        strict_manifest=bool(cfg.get("manifest_strict", True)),
    )

    log.info(Back.MAGENTA + 8 * "-" + "STAGE 3: EVALUATION METRICS" + 8 * "-")
    evaluator = get_evaluator(cfg, context.logger)
    if evaluator is None:
        raise ValueError("Could not instantiate the evaluator; nothing to do.")

    with log_phase("load validation split"):
        context.datamodule.setup("validate")
        val_loader = context.datamodule.val_dataloader()
        loader_names = (
            sorted(val_loader.keys())
            if hasattr(val_loader, "keys")
            else type(val_loader).__name__
        )
        log.info(f"Validation loaders ({len(loader_names)}): {loader_names}")

    try:
        with log_phase("run validation"):
            # set_optimized_metric=True keeps the Optuna/MLflow bookkeeping
            # identical to what the old single-process pipeline recorded.
            evaluator.evaluate_run(
                context.run_ckpts,
                context.algorithm,
                val_loader,
                "val",
                set_optimized_metric=True,
            )
    finally:
        # The physics datamodule keeps every split in RAM; release it even on
        # failure so a crash here does not also exhaust the node.
        with log_phase("release validation split", collect=True):
            evaluator.release_dataloaders()
            del val_loader
            context.datamodule.teardown("validate")
            release_accelerator_cache()

    context.object_dict.update({"evaluator": evaluator})

    metric_dict: Dict[str, Any] = {}
    optimized = getattr(evaluator, "optimized_metric", None)
    if optimized is not None:
        metric_dict["optimized_metric"] = optimized
        log.info(f"Optimized metric: {optimized}")

    write_stage_status(
        context.run_ckpts,
        stage_name="metrics",
        ok=True,
        detail=f"experiment={cfg.experiment_name}",
        artifacts=[str(context.run_ckpts / "plots" / "val" / "loss_total")],
    )

    log.info(f"Evaluation artifacts written under {context.run_ckpts / 'plots'}.")
    return metric_dict, context.object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Entry point for stage 3."""
    if "run_name" in cfg and not isinstance(cfg.run_name, str):
        # See the same coercion in src/train.py.
        from omegaconf import open_dict

        coerced = str(cfg.run_name)
        with open_dict(cfg):
            cfg.run_name = coerced
        log.warning(f"run_name was not a string; coerced to {coerced!r}.")

    extras(cfg)
    run_eval_metrics(cfg)


if __name__ == "__main__":
    main()

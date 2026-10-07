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

``eval_split=test`` replays the held-out TEST split instead (zero-bias test part
plus the test part of every auxiliary set) and writes the same artifacts under
``<run>/plots/test/``. That is what scripts/physics/runae_test.sh runs, and only
for configurations already selected on validation. On test:

* callbacks that run on one split only (``evaluation_split``, i.e. the latent
  collapse diagnostic) are pointed at test;
* no optimized metric is set: test must never feed back into model selection;
* efficiencies use the operating threshold stored in the checkpoint, which was
  fixed on validation data during training;
* the stage status is ``stage_status/metrics_test.yaml``, so the validation
  record of stage 3 is left alone.
"""

from typing import Any, Dict, List, Tuple

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import hydra

from omegaconf import DictConfig, open_dict
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
    finish_stage_loggers,
    get_evaluator,
    release_accelerator_cache,
)

log = RankedLogger(__name__, rank_zero_only=True)

#: ``eval_split`` -> (datamodule stage, dataloader method).
EVAL_SPLITS = {
    "val": ("validate", "val_dataloader"),
    "test": ("test", "test_dataloader"),
}


def resolve_eval_split(cfg: DictConfig) -> str:
    """The split this invocation replays: ``val`` (stage 3) or ``test``."""
    split = str(cfg.get("eval_split", "val"))
    if split not in EVAL_SPLITS:
        raise ValueError(
            f"eval_split must be one of {sorted(EVAL_SPLITS)}, got {split!r}."
        )
    return split


def stage_name_for(split: str) -> str:
    """``metrics`` for validation, ``metrics_<split>`` otherwise."""
    return "metrics" if split == "val" else f"metrics_{split}"


def follow_eval_split(cfg: DictConfig, split: str) -> List[str]:
    """Point every callback that runs on one split only at ``split``.

    Such callbacks carry an ``evaluation_split`` (latent_collapse: ``val``) and
    silently do nothing on any other split. Returns the names that were changed.
    """
    callbacks = (cfg.get("evaluation") or {}).get("callbacks") or {}
    changed = []
    with open_dict(cfg):
        for name, callback in callbacks.items():
            if callback is None or "evaluation_split" not in callback:
                continue
            if str(callback.evaluation_split) != split:
                callback.evaluation_split = split
                changed.append(str(name))
    return changed


@task_wrapper
def run_eval_metrics(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Replay the validation (or, with eval_split=test, the test) split through
    the stage-1 checkpoint.

    Returns (metric_dict, object_dict); task_wrapper unpacks this pair.
    """
    if cfg.get("evaluation") is None:
        raise ValueError(
            "No evaluation config. Compose the same experiment as the stage-1 "
            "run, for example experiment=physics/ae_metrics."
        )

    split = resolve_eval_split(cfg)
    stage_name = stage_name_for(split)
    datamodule_stage, loader_method = EVAL_SPLITS[split]
    for name in follow_eval_split(cfg, split):
        log.info(f"[{stage_name}] evaluation.callbacks.{name}.evaluation_split -> {split}")

    context = build_stage_context(
        cfg,
        stage_name=stage_name,
        strict_manifest=bool(cfg.get("manifest_strict", True)),
    )

    log.info(
        Back.MAGENTA + 8 * "-" + f"STAGE 3: EVALUATION METRICS ({split})" + 8 * "-"
    )
    evaluator = get_evaluator(cfg, context.logger)
    if evaluator is None:
        raise ValueError("Could not instantiate the evaluator; nothing to do.")

    with log_phase(f"load {split} split"):
        context.datamodule.setup(datamodule_stage)
        loader = getattr(context.datamodule, loader_method)()
        loader_names = (
            sorted(loader.keys())
            if hasattr(loader, "keys")
            else type(loader).__name__
        )
        log.info(f"{split} loaders ({len(loader_names)}): {loader_names}")

    try:
        with log_phase(f"run {split} evaluation"):
            # set_optimized_metric=True on validation keeps the Optuna/MLflow
            # bookkeeping identical to what the old single-process pipeline
            # recorded. Never on test: it must not feed back into selection.
            evaluator.evaluate_run(
                context.run_ckpts,
                context.algorithm,
                loader,
                split,
                set_optimized_metric=split == "val",
            )
    finally:
        # The physics datamodule keeps every split in RAM; release it even on
        # failure so a crash here does not also exhaust the node.
        with log_phase(f"release {split} split", collect=True):
            evaluator.release_dataloaders()
            del loader
            context.datamodule.teardown(datamodule_stage)
            release_accelerator_cache()

    context.object_dict.update({"evaluator": evaluator})

    metric_dict: Dict[str, Any] = {}
    optimized = getattr(evaluator, "optimized_metric", None)
    if optimized is not None and split == "val":
        metric_dict["optimized_metric"] = optimized
        log.info(f"Optimized metric: {optimized}")

    write_stage_status(
        context.run_ckpts,
        stage_name=stage_name,
        ok=True,
        detail=f"experiment={cfg.experiment_name} split={split}",
        artifacts=[str(context.run_ckpts / "plots" / split / "loss_total")],
    )

    log.info(f"Evaluation artifacts written under {context.run_ckpts / 'plots'}.")
    finish_stage_loggers(context)
    return metric_dict, context.object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Entry point for stage 3 (and, with eval_split=test, the test evaluation)."""
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

"""Stage 2 of 4: the four leakage probes on an already-trained checkpoint.

Reads ``<checkpoints_dir>/<experiment_name>/<run_name>/loss_total.ckpt``, which
stage 1 (src/train.py) wrote, and produces

    <run>/plots/<val|test>/loss_total/probes/leakage_probes.json
    <run>/plots/<val|test>/loss_total/probes/leakage_probes_summary.json
    <run>/plots/<val|test>/loss_total/probes/leakage_probes_loss_plots/*.png

This is the objective L of the Pareto study: the maximum clipped held-out R^2
over {MLP, linear} x {z_logits, reconstruction} when predicting the sensitive
variable. It is by far the most expensive analysis step (measured: ~27 min per
run, independent of epoch count), which is the main reason it is its own stage.

Compose the same config as the stage-1 run. Example:

    python3 src/run_probes.py \\
        experiment=physics/ae \\
        run_name=AE_LXPLUS_30ep \\
        paths.raw_data_dir=... \\
        trainer=cpu

Exit codes: 0 when the probes ran and the result is scientifically valid, 3 when
they ran but the protocol rejected the result (an invalid result IS written to
disk, so this is a meaningful distinction for a batch driver: the run is finished
but must not be reported).
"""

from typing import Any, Dict, Optional, Tuple

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import hydra

from omegaconf import DictConfig, OmegaConf
from colorama import Back

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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
from src.utils import task_wrapper
from src.utils.instrumentation import log_phase
from src.utils.run_manifest import write_stage_status
from src.utils.stage import build_stage_context, release_accelerator_cache

log = RankedLogger(__name__, rank_zero_only=True)

#: Returned by main() when the probes completed but the protocol rejected the
#: result. Distinct from a crash so a batch driver can tell "not reportable"
#: from "did not run".
PROBE_INVALID_EXIT_CODE = 3


@task_wrapper
def run_probes(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the four leakage probes against the stage-1 checkpoint.

    Returns (metric_dict, object_dict); task_wrapper unpacks this pair.
    """
    evaluation_cfg = cfg.get("evaluation")
    probe_cfg = (
        evaluation_cfg.get("leakage_probes") if evaluation_cfg is not None else None
    )
    if probe_cfg is None:
        raise ValueError(
            "No evaluation.leakage_probes config. Compose the same experiment "
            "as the stage-1 run, for example experiment=physics/ae."
        )
    if not probe_cfg.get("enabled", False):
        raise ValueError(
            "evaluation.leakage_probes.enabled is false. This stage exists only "
            "to run the probes; enable them or do not run this stage."
        )

    smoke_cfg = probe_cfg.get("smoke_test")
    smoke_enabled = bool(smoke_cfg is not None and smoke_cfg.get("enabled", False))
    sample_caps: Optional[Dict[str, int]] = None
    if smoke_enabled:
        sample_caps = OmegaConf.to_container(
            smoke_cfg.get("max_events_per_split"), resolve=True
        )

    evaluation_mode = str(probe_cfg.get("mode", "validation"))

    context = build_stage_context(
        cfg,
        stage_name="probes",
        strict_manifest=bool(cfg.get("manifest_strict", True)),
    )

    run_metadata = make_leakage_probe_run_metadata(
        autoencoder_seed=cfg.get("seed"),
        algorithm_config=cfg.algorithm,
    )

    log.info(
        Back.MAGENTA
        + 8 * "-"
        + "STAGE 2: "
        + ("SMOKE " if smoke_enabled else "")
        + evaluation_mode.upper()
        + " LEAKAGE PROBES"
        + 8 * "-"
    )

    # A plain `with` block, unlike train.py's manual __enter__/__exit__, so the
    # phase is closed and reported even when extraction raises.
    with log_phase("leakage probes"):
        outcome = evaluate_and_record_loss_total_leakage_probes(
            context.algorithm,
            context.datamodule,
            context.run_ckpts,
            # A freshly instantiated LightningModule that no Trainer has
            # touched reports device=cpu. That is correct here: probe
            # extraction and the sklearn fits run on CPU regardless of the
            # trainer config, exactly as they did inside train.py.
            device=context.algorithm.device,
            run_shuffled_target_controls=bool(
                probe_cfg.get("run_shuffled_target_controls", False)
            ),
            evaluation_mode=evaluation_mode,
            run_metadata=run_metadata,
            max_samples_by_split=sample_caps,
        )

    metadata = log_leakage_probe_outcome_metadata(outcome, context.logger)

    metrics: Dict[str, float] = {}
    context.object_dict.update(
        {
            "leakage_probe_outcome": outcome,
            "leakage_probe_path": outcome.output_path,
            "leakage_probe_metadata": metadata,
        }
    )

    if outcome.probe_valid:
        result = outcome.result
        if result is None:
            raise RuntimeError(
                "A valid leakage-probe outcome has no four-probe result."
            )

        if outcome.smoke_test:
            # Smoke scores stay in their explicitly non-reportable JSON. Do not
            # publish them under the scientific metric names that the Pareto
            # aggregation consumes.
            log.info(
                "Smoke-test probe metrics are not logged as scientific probe metrics."
            )
        else:
            # step=0: this stage has no training loop, so there is no global step
            # to attribute the metric to. The scientific record is the JSON.
            metrics.update(log_four_probe_metrics(result, context.logger, step=0))
            metrics.update(
                log_shuffled_target_metrics(result, context.logger, step=0)
            )

        context.object_dict.update(
            {"leakage_probe_result": result, "leakage_probe_metrics": metrics}
        )
        log.info(f"Stored valid leakage probes at {outcome.output_path}.")
    else:
        diagnostic = outcome.diagnostic_result
        if diagnostic is not None and not outcome.smoke_test:
            metrics.update(
                log_shuffled_target_metrics(diagnostic, context.logger, step=0)
            )

        context.object_dict.update(
            {
                "leakage_probe_result": None,
                "leakage_probe_metrics": metrics,
                "leakage_probe_diagnostic_result": diagnostic,
            }
        )
        log.error(
            "Leakage-probe evaluation is invalid: "
            f"{outcome.rejection_reason}: {outcome.rejection_message}. "
            f"The invalid result was stored at {outcome.output_path}."
        )

    write_stage_status(
        context.run_ckpts,
        stage_name="probes",
        ok=bool(outcome.probe_valid),
        detail=(
            f"mode={evaluation_mode}"
            if outcome.probe_valid
            else f"{outcome.rejection_reason}: {outcome.rejection_message}"
        ),
        artifacts=[str(outcome.output_path)] if outcome.output_path else [],
    )

    context.datamodule.teardown("validate")
    release_accelerator_cache()

    return metrics, context.object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Entry point for stage 2."""
    if "run_name" in cfg and not isinstance(cfg.run_name, str):
        # See the same coercion in src/train.py: MLflow's protobuf run_name only
        # accepts str, and a numeric RUN_NAME resolves to int.
        from omegaconf import open_dict

        coerced = str(cfg.run_name)
        with open_dict(cfg):
            cfg.run_name = coerced
        log.warning(f"run_name was not a string; coerced to {coerced!r}.")

    extras(cfg)
    _metric_dict, object_dict = run_probes(cfg)

    outcome = object_dict.get("leakage_probe_outcome") if object_dict else None
    if outcome is not None and not outcome.probe_valid:
        raise SystemExit(PROBE_INVALID_EXIT_CODE)


if __name__ == "__main__":
    main()

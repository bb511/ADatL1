"""Checkpoint orchestration and leakage-probe artifact persistence."""

from __future__ import annotations

import json
import logging
from time import perf_counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .constants import (
    LEAKAGE_PROBE_EVALUATION_MODES,
    LEAKAGE_PROBE_PROTOCOL_VERSION,
)
from .errors import (
    ProbeExtractionError,
    ProbeFitError,
    ProbePartitionError,
    ShuffledTargetGuardrailError,
)
from .evaluation import evaluate_four_leakage_probes
from .extraction import extract_probe_split
from .provenance import concatenate_probe_representation_sets
from .plotting import write_probe_loss_plots
from .serialization import (
    four_probe_result_payload,
    four_probe_summary_payload,
    leakage_probe_run_metadata_payload,
)

from .types import (
    FourProbeEvaluationResult,
    LeakageProbeRunMetadata,
    LeakageProbeRunOutcome,
)

log = logging.getLogger(__name__)

def leakage_probe_output_path(
    run_folder: str | Path,
    *,
    evaluation_mode: str = "validation",
    smoke_test: bool = False,
) -> Path:
    """Return the fixed leakage artifact path."""

    stage_folders = {
        "validation": "val",
        "final_test": "test",
    }
    try:
        stage_folder = stage_folders[evaluation_mode]
    except KeyError as error:
        raise ValueError(
            "Unknown leakage-probe evaluation mode "
            f"{evaluation_mode!r}; expected one of "
            f"{list(LEAKAGE_PROBE_EVALUATION_MODES)}."
        ) from error

    filename = (
        "leakage_probes_smoke.json"
        if smoke_test
        else "leakage_probes.json"
    )

    return (
        Path(run_folder)
        / "plots"
        / stage_folder
        / "loss_total"
        / "probes"
        / filename
    )


def leakage_probe_summary_output_path(
    run_folder: str | Path,
    *,
    evaluation_mode: str = "validation",
    smoke_test: bool = False,
) -> Path:
    """Return the compact summary path without mixing smoke and science."""

    detailed_path = leakage_probe_output_path(
        run_folder,
        evaluation_mode=evaluation_mode,
        smoke_test=smoke_test,
    )
    return detailed_path.with_name(
        "leakage_probes_smoke_summary.json"
        if smoke_test
        else "leakage_probes_summary.json"
    )


def _write_json_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_leakage_probe_summary(
    detailed_payload: Mapping[str, Any],
    detailed_path: Path,
    *,
    smoke_test: bool,
) -> Path:
    summary_path = detailed_path.with_name(
        "leakage_probes_smoke_summary.json"
        if smoke_test
        else "leakage_probes_summary.json"
    )
    summary_payload = four_probe_summary_payload(
        detailed_payload,
        source_artifact=detailed_path.name,
    )
    _write_json_artifact(summary_path, summary_payload)
    log.info("Saved compact probe summary JSON: %s", summary_path)
    return summary_path


def write_leakage_probe_results(
    result: FourProbeEvaluationResult,
    run_folder: str | Path,
) -> Path:
    """Write all four probe results below one checkpoint run."""

    smoke_test = (
        result.evaluation_context.development_data.max_samples
        is not None
        or result.evaluation_context.held_out_data.max_samples
        is not None
    )
    output_path = leakage_probe_output_path(
        run_folder,
        evaluation_mode=result.evaluation_context.mode,
        smoke_test=smoke_test,
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = four_probe_result_payload(result)
    write_probe_loss_plots(payload, output_path)

    _write_json_artifact(output_path, payload)
    _write_leakage_probe_summary(
        payload,
        output_path,
        smoke_test=smoke_test,
    )

    log.info("Saved four-probe JSON: %s", output_path)
    return output_path


def write_invalid_leakage_probe_result(
    run_folder: str | Path,
    error: (
        ProbeExtractionError
        | ProbePartitionError
        | ProbeFitError
    ),
    *,
    evaluation_mode: str = "validation",
    run_metadata: LeakageProbeRunMetadata | None = None,
    smoke_test: bool = False,
) -> Path:
    """Persist one expected protocol failure without a fake score."""

    output_path = leakage_probe_output_path(
        run_folder,
        evaluation_mode=evaluation_mode,
        smoke_test=smoke_test,
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if run_metadata is None:
        run_metadata = LeakageProbeRunMetadata(
            autoencoder_seed=None,
            configuration_id=None,
        )

    if isinstance(
        error,
        ShuffledTargetGuardrailError,
    ):
        payload = four_probe_result_payload(
            error.result
        )
        payload.update(
            {
                "probe_valid": False,
                "rejection_reason": error.reason,
                "rejection_message": str(error),
                "worst_probe": None,
                "leakage_worst": None,
            }
        )
        write_probe_loss_plots(payload, output_path)
    else:
        payload = {
            "leakage_probe_protocol_version": (
                LEAKAGE_PROBE_PROTOCOL_VERSION
            ),
            "probe_valid": False,
            "rejection_reason": error.reason,
            "rejection_message": str(error),
            "run": leakage_probe_run_metadata_payload(
                run_metadata
            ),
            "evaluation": {
                "mode": evaluation_mode,
                "purpose": (
                    "smoke_test" if smoke_test else "scientific"
                ),
                "reporting_eligible": not smoke_test,
                "development_data": None,
                "held_out_data": None,
            },
            "worst_probe": None,
            "leakage_worst": None,
            "probes": {},
        }

    _write_json_artifact(output_path, payload)
    _write_leakage_probe_summary(
        payload,
        output_path,
        smoke_test=smoke_test,
    )

    log.error("Saved invalid probe JSON (%s): %s", error.reason, output_path)
    return output_path


def _validated_smoke_sample_caps(
    sample_caps: Mapping[str, int] | None,
    *,
    evaluation_mode: str,
) -> dict[str, int]:
    """Validate the explicit non-reportable smoke-test sampling request."""

    if sample_caps is None:
        return {}
    if not isinstance(sample_caps, Mapping):
        raise ProbeExtractionError(
            "invalid_probe_smoke_sample_caps",
            "Smoke-test sample caps must be a mapping.",
        )
    if evaluation_mode != "validation":
        raise ProbeExtractionError(
            "smoke_test_final_test_forbidden",
            "Smoke-test event caps are allowed only in validation mode; "
            "the final test split must not be used for a smoke run.",
        )

    expected_splits = {"train", "valid"}
    if set(sample_caps) != expected_splits:
        raise ProbeExtractionError(
            "invalid_probe_smoke_sample_caps",
            "Validation smoke caps must define exactly train and valid; "
            f"received {sorted(sample_caps)}.",
        )

    validated: dict[str, int] = {}
    for split in ("train", "valid"):
        value = sample_caps[split]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 2
        ):
            raise ProbeExtractionError(
                "invalid_probe_smoke_sample_caps",
                f"Smoke cap for {split!r} must be an integer >= 2.",
            )
        validated[split] = int(value)

    return validated


def evaluate_and_write_loss_total_leakage_probes(
    model: Any,
    datamodule: Any,
    run_folder: str | Path,
    *,
    device: torch.device | str = "cpu",
    run_shuffled_target_controls: bool = True,
    evaluation_mode: str = "validation",
    run_metadata: LeakageProbeRunMetadata | None = None,
    max_samples_by_split: Mapping[str, int] | None = None,
) -> tuple[FourProbeEvaluationResult, Path]:
    """Evaluate the frozen loss-total checkpoint and persist four probes.

    The checkpoint is loaded explicitly and strictly. Raw datamodule splits
    are loaded and released one at a time; final-test mode then combines the
    extracted train and validation representations in CPU memory.
    """

    sample_caps = _validated_smoke_sample_caps(
        max_samples_by_split,
        evaluation_mode=evaluation_mode,
    )

    run_folder = Path(run_folder)
    checkpoint_path = run_folder / "loss_total.ckpt"

    started = perf_counter()
    log.info(
        "Leakage evaluation starting: mode=%s, sample caps=%s, shuffled controls=%s. "
        "Loading checkpoint %s.",
        evaluation_mode, sample_caps or "uncapped", run_shuffled_target_controls, checkpoint_path,
    )

    if not checkpoint_path.is_file():
        raise ProbeExtractionError(
            "loss_total_checkpoint_missing",
            f"Required checkpoint not found: {checkpoint_path}.",
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            weights_only=False,
            map_location="cpu",
        )
    except Exception as error:
        raise ProbeExtractionError(
            "loss_total_checkpoint_load_failed",
            f"Could not load {checkpoint_path}: {error}",
        ) from error

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ProbeExtractionError(
            "loss_total_state_dict_missing",
            f"Checkpoint {checkpoint_path} has no state_dict.",
        )

    state_dict = checkpoint["state_dict"]
    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except Exception as error:
        raise ProbeExtractionError(
            "loss_total_state_dict_load_failed",
            f"Could not restore {checkpoint_path} strictly: {error}",
        ) from error
    finally:
        del state_dict
        del checkpoint

    log.info("Frozen checkpoint restored; extracting development and held-out representations.")

    if run_metadata is None:
        run_metadata = LeakageProbeRunMetadata(
            autoencoder_seed=None,
            configuration_id=None,
        )

    if evaluation_mode == "validation":
        development_representations = extract_probe_split(
            model,
            datamodule,
            "train",
            device=device,
            **(
                {"max_samples": sample_caps["train"]}
                if sample_caps
                else {}
            ),
        )
        held_out_representations = extract_probe_split(
            model,
            datamodule,
            "valid",
            device=device,
            **(
                {"max_samples": sample_caps["valid"]}
                if sample_caps
                else {}
            ),
        )
    elif evaluation_mode == "final_test":
        train_representations = extract_probe_split(
            model,
            datamodule,
            "train",
            device=device,
        )
        validation_representations = extract_probe_split(
            model,
            datamodule,
            "valid",
            device=device,
        )
        development_representations = (
            concatenate_probe_representation_sets(
                (
                    train_representations,
                    validation_representations,
                ),
                split="train+valid",
            )
        )
        del train_representations
        del validation_representations
        held_out_representations = extract_probe_split(
            model,
            datamodule,
            "test",
            device=device,
        )
    else:
        raise ProbeExtractionError(
            "invalid_probe_evaluation_mode",
            "Unknown leakage-probe evaluation mode "
            f"{evaluation_mode!r}; expected one of "
            f"{list(LEAKAGE_PROBE_EVALUATION_MODES)}.",
        )

    result = evaluate_four_leakage_probes(
        development_representations,
        held_out_representations,
        run_shuffled_target_controls=run_shuffled_target_controls,
        evaluation_mode=evaluation_mode,
        run_metadata=run_metadata,
    )
    output_path = write_leakage_probe_results(
        result,
        run_folder,
    )

    log.info(
        "Leakage evaluation finished in %.1fs; output=%s%s.",
        perf_counter() - started, output_path,
        " (SMOKE ONLY; not reportable)" if sample_caps else "",
    )

    return result, output_path


def evaluate_and_record_loss_total_leakage_probes(
    model: Any,
    datamodule: Any,
    run_folder: str | Path,
    *,
    device: torch.device | str = "cpu",
    run_shuffled_target_controls: bool = True,
    evaluation_mode: str = "validation",
    run_metadata: LeakageProbeRunMetadata | None = None,
    max_samples_by_split: Mapping[str, int] | None = None,
) -> LeakageProbeRunOutcome:
    """Evaluate leakage and record expected protocol failures."""

    if run_metadata is None:
        run_metadata = LeakageProbeRunMetadata(
            autoencoder_seed=None,
            configuration_id=None,
        )

    try:
        result, output_path = (
            evaluate_and_write_loss_total_leakage_probes(
                model,
                datamodule,
                run_folder,
                device=device,
                run_shuffled_target_controls=run_shuffled_target_controls,
                evaluation_mode=evaluation_mode,
                run_metadata=run_metadata,
                max_samples_by_split=max_samples_by_split,
            )
        )
    except (
        ProbeExtractionError,
        ProbePartitionError,
        ProbeFitError,
    ) as error:
        output_path = write_invalid_leakage_probe_result(
            run_folder,
            error,
            evaluation_mode=evaluation_mode,
            run_metadata=run_metadata,
            smoke_test=(max_samples_by_split is not None),
        )

        diagnostic_result = (
            error.result
            if isinstance(
                error,
                ShuffledTargetGuardrailError,
            )
            else None
        )

        return LeakageProbeRunOutcome(
            probe_valid=False,
            result=None,
            output_path=output_path,
            rejection_reason=error.reason,
            rejection_message=str(error),
            evaluation_mode=evaluation_mode,
            run_metadata=run_metadata,
            diagnostic_result=diagnostic_result,
            smoke_test=(max_samples_by_split is not None),
        )

    return LeakageProbeRunOutcome(
        probe_valid=True,
        result=result,
        output_path=output_path,
        rejection_reason=None,
        rejection_message=None,
        evaluation_mode=evaluation_mode,
        run_metadata=run_metadata,
        smoke_test=(max_samples_by_split is not None),
    )


def log_leakage_probe_outcome_metadata(
    outcome: LeakageProbeRunOutcome,
    loggers: list[Any],
) -> dict[str, Any]:
    """Make probe validity queryable in the run table."""

    metadata: dict[str, Any] = {
        "leakage_probe_protocol_version": (
            LEAKAGE_PROBE_PROTOCOL_VERSION
        ),
        "probe_valid": outcome.probe_valid,
        "probe_rejection_reason": (
            outcome.rejection_reason
            if outcome.rejection_reason is not None
            else "none"
        ),
        "leakage_probe_evaluation_mode": (
            outcome.evaluation_mode
        ),
        "leakage_probe_smoke_test": outcome.smoke_test,
        "leakage_probe_reporting_eligible": (
            not outcome.smoke_test
        ),
        "autoencoder_seed": (
            outcome.run_metadata.autoencoder_seed
        ),
        "leakage_probe_configuration_id": (
            outcome.run_metadata.configuration_id
        ),
    }

    for output_logger in loggers:
        output_logger.log_hyperparams(metadata)

    return metadata

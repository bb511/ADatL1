"""Checkpoint orchestration and leakage-probe artifact persistence."""

from __future__ import annotations

import json
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
from .serialization import (
    four_probe_result_payload,
    leakage_probe_run_metadata_payload,
)
from .types import (
    FourProbeEvaluationResult,
    LeakageProbeRunMetadata,
    LeakageProbeRunOutcome,
)

def leakage_probe_output_path(
    run_folder: str | Path,
    *,
    evaluation_mode: str = "validation",
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

    return (
        Path(run_folder)
        / "plots"
        / stage_folder
        / "loss_total"
        / "probes"
        / "leakage_probes.json"
    )


def write_leakage_probe_results(
    result: FourProbeEvaluationResult,
    run_folder: str | Path,
) -> Path:
    """Write all four probe results below one checkpoint run."""

    output_path = leakage_probe_output_path(
        run_folder,
        evaluation_mode=result.evaluation_context.mode,
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = four_probe_result_payload(result)

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

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
) -> Path:
    """Persist one expected protocol failure without a fake score."""

    output_path = leakage_probe_output_path(
        run_folder,
        evaluation_mode=evaluation_mode,
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
                "development_data": None,
                "held_out_data": None,
            },
            "worst_probe": None,
            "leakage_worst": None,
            "probes": {},
        }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path


def evaluate_and_write_loss_total_leakage_probes(
    model: Any,
    datamodule: Any,
    run_folder: str | Path,
    *,
    device: torch.device | str = "cpu",
    run_shuffled_target_controls: bool = True,
    evaluation_mode: str = "validation",
    run_metadata: LeakageProbeRunMetadata | None = None,
) -> tuple[FourProbeEvaluationResult, Path]:
    """Evaluate the frozen loss-total checkpoint and persist four probes.

    The checkpoint is loaded explicitly and strictly. Raw datamodule splits
    are loaded and released one at a time; final-test mode then combines the
    extracted train and validation representations in CPU memory.
    """

    run_folder = Path(run_folder)
    checkpoint_path = run_folder / "loss_total.ckpt"

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
        )
        held_out_representations = extract_probe_split(
            model,
            datamodule,
            "valid",
            device=device,
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
        )

    return LeakageProbeRunOutcome(
        probe_valid=True,
        result=result,
        output_path=output_path,
        rejection_reason=None,
        rejection_message=None,
        evaluation_mode=evaluation_mode,
        run_metadata=run_metadata,
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

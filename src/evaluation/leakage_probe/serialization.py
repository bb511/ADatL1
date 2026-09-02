"""Serialize and log leakage-probe results."""

from collections.abc import Mapping
from typing import Any

import numpy as np

from .constants import (
    LEAKAGE_PROBE_PROTOCOL_VERSION,
    PROBE_INITIALIZATION_SEEDS,
)
from .errors import ProbeFitError
from .types import (
    FourProbeEvaluationResult,
    LeakageProbeRunMetadata,
    MLPProbeCandidateFailure,
    MLPProbeCandidateResult,
    MLPProbeOuterResult,
    MLPProbeSeedSelection,
    NamedLinearProbeResult,
    NamedMLPProbeResult,
    ProbeEvaluationContext,
    ProbeSplitProvenance,
    ShuffledTargetMLPResult,
)


SUMMARY_PROBE_NAMES = (
    "mlp/z_logits",
    "mlp/reconstruction",
    "linear/z_logits",
    "linear/reconstruction",
)


def _finite_or_none(value) -> float | None:
    """Keep optional diagnostics JSON-safe, including unavailable history points."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _mlp_training_history_payload(
    fit: MLPProbeCandidateResult | MLPProbeOuterResult,
) -> dict[str, Any]:
    return {
        "epochs": list(range(1, len(fit.loss_curve) + 1)),
        "loss": [_finite_or_none(value) for value in fit.loss_curve],
        "loss_definition": "sklearn training objective on standardized targets, including L2 penalty",
        "loss_units": "dimensionless",
        "early_stopping_validation_r2": [
            _finite_or_none(value) for value in fit.early_stopping_validation_scores
        ],
        "validation_scope": "internal early-stopping subset of the fit pool; not the held-out split",
        "note": (
            "Loss is recorded during optimization. Early stopping restores the best internal "
            "validation weights, so final_loss is not necessarily the restored model's loss."
        ),
    }


def leakage_probe_run_metadata_payload(
    metadata: LeakageProbeRunMetadata,
) -> dict[str, Any]:
    """Serialize the AE seed and seed-independent configuration identity."""

    return {
        "autoencoder_seed": metadata.autoencoder_seed,
        "configuration_id": metadata.configuration_id,
    }


def _probe_split_provenance_payload(
    provenance: ProbeSplitProvenance,
) -> dict[str, Any]:
    """Serialize one development or held-out data identity."""

    return {
        "split": provenance.split,
        "source_splits": list(provenance.source_splits),
        "n_events": int(provenance.n_events),
        "sample_seed": int(provenance.sample_seed),
        "max_samples": provenance.max_samples,
        "event_manifest_hash": provenance.event_manifest_hash,
        "data_cache_id": provenance.data_cache_id,
        "data_cache_path": provenance.data_cache_path,
    }


def probe_evaluation_context_payload(
    context: ProbeEvaluationContext,
) -> dict[str, Any]:
    """Serialize the probe split design and provenance."""

    smoke_test = (
        context.development_data.max_samples is not None
        or context.held_out_data.max_samples is not None
    )

    return {
        "mode": context.mode,
        "purpose": "smoke_test" if smoke_test else "scientific",
        "reporting_eligible": not smoke_test,
        "development_data": _probe_split_provenance_payload(
            context.development_data
        ),
        "held_out_data": _probe_split_provenance_payload(
            context.held_out_data
        ),
    }

def _successful_mlp_candidate_payload(
    candidate: MLPProbeCandidateResult,
    *,
    selected_seed: int,
) -> dict[str, Any]:
    """Serialize one successful inner-validation candidate."""

    return {
        "seed": int(candidate.seed),
        "status": "successful",
        "selected": candidate.seed == selected_seed,
        "inner_r2_raw": float(candidate.inner_r2_raw),
        "inner_mae_gev": float(candidate.inner_mae_gev),
        "convergence_warnings": list(
            candidate.convergence_warnings
        ),
        "n_iter": int(candidate.n_iter),
        "final_loss": float(candidate.final_loss),
        "training_history": _mlp_training_history_payload(candidate),
    }

def _failed_mlp_candidate_payload(
    failure: MLPProbeCandidateFailure,
) -> dict[str, Any]:
    """Serialize one failed MLP initialization."""

    return {
        "seed": int(failure.seed),
        "status": "failed",
        "selected": False,
        "reason": failure.reason,
        "message": failure.message,
    }

def _mlp_seed_selection_payload(
    selection: MLPProbeSeedSelection,
) -> dict[str, Any]:
    """Serialize every frozen MLP initialization exactly once."""

    successful_by_seed: dict[
        int,
        MLPProbeCandidateResult,
    ] = {}
    failed_by_seed: dict[
        int,
        MLPProbeCandidateFailure,
    ] = {}

    for candidate in selection.successful_candidates:
        seed = int(candidate.seed)

        if (
            seed not in PROBE_INITIALIZATION_SEEDS
            or seed in successful_by_seed
            or seed in failed_by_seed
        ):
            raise ProbeFitError(
                "invalid_mlp_candidate_diagnostics",
                "Successful MLP candidate diagnostics contain "
                f"an unknown or duplicate seed: {seed}.",
            )

        successful_by_seed[seed] = candidate

    for failure in selection.failed_candidates:
        seed = int(failure.seed)

        if (
            seed not in PROBE_INITIALIZATION_SEEDS
            or seed in successful_by_seed
            or seed in failed_by_seed
        ):
            raise ProbeFitError(
                "invalid_mlp_candidate_diagnostics",
                "Failed MLP candidate diagnostics contain "
                f"an unknown or duplicate seed: {seed}.",
            )

        failed_by_seed[seed] = failure

    recorded_seeds = (
        set(successful_by_seed)
        | set(failed_by_seed)
    )
    expected_seeds = set(
        PROBE_INITIALIZATION_SEEDS
    )

    if recorded_seeds != expected_seeds:
        missing_seeds = sorted(
            expected_seeds - recorded_seeds
        )
        extra_seeds = sorted(
            recorded_seeds - expected_seeds
        )

        raise ProbeFitError(
            "invalid_mlp_candidate_diagnostics",
            "MLP candidate diagnostics must record every "
            "frozen initialization exactly once. "
            f"Missing={missing_seeds}, extra={extra_seeds}.",
        )

    if (
        selection.selected_seed
        not in successful_by_seed
        or selection.selected_candidate.seed
        != selection.selected_seed
    ):
        raise ProbeFitError(
            "invalid_mlp_candidate_diagnostics",
            "The selected MLP seed must identify the recorded "
            "successful selected candidate.",
        )

    candidates: list[dict[str, Any]] = []

    # Iterate over the frozen protocol order, not dictionary or
    # completion order.
    for seed in PROBE_INITIALIZATION_SEEDS:
        if seed in successful_by_seed:
            candidates.append(
                _successful_mlp_candidate_payload(
                    successful_by_seed[seed],
                    selected_seed=selection.selected_seed,
                )
            )
        else:
            candidates.append(
                _failed_mlp_candidate_payload(
                    failed_by_seed[seed]
                )
            )

    return {
        "selected_seed": int(
            selection.selected_seed
        ),
        "candidates": candidates,
    }

def _probe_result_payload(
    probe: NamedMLPProbeResult | NamedLinearProbeResult,
) -> dict[str, Any]:
    """Convert one fitted probe result to JSON-safe values."""

    outer = probe.outer_result

    payload: dict[str, Any] = {
        "representation_name": probe.representation_name,
        "metric_name": probe.metric_name,
        "feature_dimension": int(probe.feature_dimension),
        "r2_raw": float(outer.outer_r2_raw),
        "r2_clipped": float(outer.outer_r2_clipped),
        "mae_gev": float(outer.outer_mae_gev),
        "n_development": int(outer.n_train),
        "n_held_out": int(outer.n_validation),
    }

    if isinstance(probe, NamedMLPProbeResult):
        payload.update(
            {
                "selected_seed": int(outer.selected_seed),
                "convergence_warnings": list(
                    outer.convergence_warnings
                ),
                "n_iter": int(outer.n_iter),
                "final_loss": float(outer.final_loss),
                "training_history": _mlp_training_history_payload(outer),
                "seed_selection": (
                    _mlp_seed_selection_payload(
                        probe.seed_selection
                    )
                ),
            }
        )
    elif isinstance(probe, NamedLinearProbeResult):
        payload["loss_summary"] = {
            "method": "direct_least_squares",
            "epochs": None,
            "development_mse_gev2": _finite_or_none(outer.train_mse_gev2),
            "held_out_mse_gev2": _finite_or_none(outer.outer_mse_gev2),
            "note": "LinearRegression is a direct solve; no epoch loss history exists.",
        }

    return payload

def four_probe_result_payload(
    result: FourProbeEvaluationResult,
) -> dict[str, Any]:
    """Build the complete four-probe JSON payload."""

    return {
        "leakage_probe_protocol_version": (
            LEAKAGE_PROBE_PROTOCOL_VERSION
        ),
        "probe_valid": True,
        "rejection_reason": None,
        "rejection_message": None,
        "run": leakage_probe_run_metadata_payload(
            result.run_metadata
        ),
        "evaluation": probe_evaluation_context_payload(
            result.evaluation_context
        ),
        "worst_probe": result.worst_probe,
        "leakage_worst": float(result.leakage_worst),
        "probes": {
            "mlp/z_logits": _probe_result_payload(
                result.mlp_latent_logits
            ),
            "mlp/reconstruction": _probe_result_payload(
                result.mlp_reconstructed_data
            ),
            "linear/z_logits": _probe_result_payload(
                result.linear_latent_logits
            ),
            "linear/reconstruction": _probe_result_payload(
                result.linear_reconstructed_data
            ),
        },
        "diagnostics": {
            "shuffled_targets": (
                _shuffled_target_controls_payload(
                    result.shuffled_target_controls
                )
            ),
        },
    }


def four_probe_summary_payload(
    detailed_payload: Mapping[str, Any],
    *,
    source_artifact: str,
) -> dict[str, Any]:
    """Build a compact, provenance-aware view of one detailed probe result."""

    detailed_probes = detailed_payload.get("probes", {})
    if not isinstance(detailed_probes, Mapping):
        detailed_probes = {}

    probes: dict[str, dict[str, float | None]] = {}
    for name in SUMMARY_PROBE_NAMES:
        detailed_probe = detailed_probes.get(name)
        clipped_r2 = (
            detailed_probe.get("r2_clipped")
            if isinstance(detailed_probe, Mapping)
            else None
        )
        probes[name] = {
            "r2_clipped": _finite_or_none(clipped_r2),
        }

    evaluation = detailed_payload.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        evaluation = {}
    development = evaluation.get("development_data")
    held_out = evaluation.get("held_out_data")

    def split_manifest(split: Any) -> str | None:
        return (
            split.get("event_manifest_hash")
            if isinstance(split, Mapping)
            else None
        )

    run = detailed_payload.get("run", {})
    return {
        "leakage_probe_summary_schema_version": 1,
        "source_artifact": source_artifact,
        "leakage_probe_protocol_version": detailed_payload.get(
            "leakage_probe_protocol_version"
        ),
        "probe_valid": bool(detailed_payload.get("probe_valid", False)),
        "rejection_reason": detailed_payload.get("rejection_reason"),
        "run": dict(run) if isinstance(run, Mapping) else {},
        "evaluation": {
            "mode": evaluation.get("mode"),
            "purpose": evaluation.get("purpose"),
            "reporting_eligible": evaluation.get("reporting_eligible"),
            "development_event_manifest_hash": split_manifest(development),
            "held_out_event_manifest_hash": split_manifest(held_out),
        },
        "worst_probe": detailed_payload.get("worst_probe"),
        "leakage_worst": _finite_or_none(
            detailed_payload.get("leakage_worst")
        ),
        "probes": probes,
    }


def four_probe_metric_values(
    result: FourProbeEvaluationResult,
) -> dict[str, float]:
    """Return the fixed primary leakage metrics for run logging."""

    probe_results = (
        (
            "probe/mlp/z_logits",
            result.mlp_latent_logits,
        ),
        (
            "probe/mlp/reconstruction",
            result.mlp_reconstructed_data,
        ),
        (
            "probe/linear/z_logits",
            result.linear_latent_logits,
        ),
        (
            "probe/linear/reconstruction",
            result.linear_reconstructed_data,
        ),
    )

    metrics: dict[str, float] = {}

    for metric_prefix, probe_result in probe_results:
        outer = probe_result.outer_result

        metrics[f"{metric_prefix}/r2_raw"] = float(
            outer.outer_r2_raw
        )
        metrics[f"{metric_prefix}/r2_clipped"] = float(
            outer.outer_r2_clipped
        )
        metrics[f"{metric_prefix}/mae_gev"] = float(
            outer.outer_mae_gev
        )

    metrics["probe/leakage_worst"] = float(
        result.leakage_worst
    )

    non_finite_metrics = [
        metric_name
        for metric_name, metric_value in metrics.items()
        if not np.isfinite(metric_value)
    ]

    if non_finite_metrics:
        raise ProbeFitError(
            "non_finite_primary_probe_metric",
            "Cannot log non-finite primary probe metrics: "
            f"{non_finite_metrics}.",
        )

    return metrics


def log_four_probe_metrics(
    result: FourProbeEvaluationResult,
    loggers: list[Any],
    *,
    step: int,
) -> dict[str, float]:
    """Log all four primary probe results through Lightning loggers."""

    metrics = four_probe_metric_values(result)

    for output_logger in loggers:
        output_logger.log_metrics(
            metrics,
            step=step,
        )

    return metrics


def _shuffled_target_controls_payload(
    controls: ShuffledTargetMLPResult | None,
) -> dict[str, Any]:
    """Serialize both shuffled-training-target controls."""

    if controls is None:
        return {"enabled": False}

    return {
        "enabled": True,
        "shuffle_seed": int(
            controls.shuffle_seed
        ),
        "permutation_manifest_hash": (
            controls.permutation_manifest_hash
        ),
        "z_logits": _probe_result_payload(
            controls.latent_logits
        ),
        "reconstruction": _probe_result_payload(
            controls.reconstructed_data
        ),
    }


def shuffled_target_metric_values(
    result: FourProbeEvaluationResult,
) -> dict[str, float]:
    """Return the fixed shuffled-control MLflow metrics."""

    controls = result.shuffled_target_controls
    if controls is None:
        return {}

    metrics = {
        "probe/shuffled/z_logits/r2_raw": float(
            controls.latent_logits
            .outer_result.outer_r2_raw
        ),
        "probe/shuffled/reconstruction/r2_raw": float(
            controls.reconstructed_data
            .outer_result.outer_r2_raw
        ),
    }

    non_finite_metrics = [
        metric_name
        for metric_name, metric_value in metrics.items()
        if not np.isfinite(metric_value)
    ]

    if non_finite_metrics:
        raise ProbeFitError(
            "non_finite_shuffled_target_metric",
            "Cannot log non-finite shuffled-target metrics: "
            f"{non_finite_metrics}.",
        )

    return metrics


def log_shuffled_target_metrics(
    result: FourProbeEvaluationResult,
    loggers: list[Any],
    *,
    step: int,
) -> dict[str, float]:
    """Log shuffled-target diagnostics separately."""

    metrics = shuffled_target_metric_values(
        result
    )

    if metrics:
        for output_logger in loggers:
            output_logger.log_metrics(
                metrics,
                step=step,
            )

    return metrics

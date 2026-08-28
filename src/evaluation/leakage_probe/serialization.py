"""Serialize and log leakage-probe results."""

from typing import Any

import numpy as np

from .constants import (
    LEAKAGE_PROBE_PROTOCOL_VERSION,
    PROBE_INITIALIZATION_SEEDS,
)
from .errors import ProbeFitError
from .types import (
    FourProbeEvaluationResult,
    LeakageProbeRunOutcome,
    MLPProbeCandidateFailure,
    MLPProbeCandidateResult,
    MLPProbeSeedSelection,
    NamedDummyBaselineResult,
    NamedLinearProbeResult,
    NamedMLPProbeResult,
)

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
        "n_train": int(outer.n_train),
        "n_validation": int(outer.n_validation),
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
                "seed_selection": (
                    _mlp_seed_selection_payload(
                        probe.seed_selection
                    )
                ),
            }
        )

    return payload

def _dummy_baseline_payload(
    baseline: NamedDummyBaselineResult,
) -> dict[str, Any]:
    """Serialize one target-mean diagnostic."""

    outer = baseline.outer_result

    return {
        "representation_name": (
            baseline.representation_name
        ),
        "metric_name": baseline.metric_name,
        "feature_dimension": int(
            baseline.feature_dimension
        ),
        "strategy": outer.estimator.strategy,
        "train_mean_gev": float(
            outer.train_mean_gev
        ),
        "r2_raw": float(outer.outer_r2_raw),
        "r2_clipped": float(
            outer.outer_r2_clipped
        ),
        "mae_gev": float(outer.outer_mae_gev),
        "n_train": int(outer.n_train),
        "n_validation": int(
            outer.n_validation
        ),
    }

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
            "dummy_baselines": {
                "z_logits": _dummy_baseline_payload(
                    result.dummy_baselines.latent_logits
                ),
                "reconstruction": (
                    _dummy_baseline_payload(
                        result.dummy_baselines
                        .reconstructed_data
                    )
                ),
            },
        },
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


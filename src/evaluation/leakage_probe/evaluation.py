"""Aggregate four primary probes and secondary diagnostics."""

from .errors import ProbeFitError
from .linear import evaluate_primary_linear_probes
from .mlp import evaluate_primary_mlp_probes
from .types import FourProbeEvaluationResult, ProbeRepresentationSet
from .diagnostics import (
    enforce_shuffled_target_guardrail,
    evaluate_latent_sample_mlp_diagnostic,
    evaluate_shuffled_target_mlp_controls,
)

def evaluate_four_leakage_probes(
    train_representations: ProbeRepresentationSet,
    validation_representations: ProbeRepresentationSet,
) -> FourProbeEvaluationResult:
    """Evaluate two MLP and two linear probes."""

    mlp_result = evaluate_primary_mlp_probes(
        train_representations,
        validation_representations,
    )

    latent_sample_diagnostic = (
        evaluate_latent_sample_mlp_diagnostic(
            train_representations,
            validation_representations,
            mlp_result.inner_partition,
        )
    )

    linear_result = evaluate_primary_linear_probes(
        train_representations,
        validation_representations,
    )

    shuffled_target_controls = (
        evaluate_shuffled_target_mlp_controls(
            train_representations,
            validation_representations,
        )
    )

    mlp_latent = mlp_result.latent_logits
    mlp_reconstruction = mlp_result.reconstructed_data
    linear_latent = linear_result.latent_logits
    linear_reconstruction = (
        linear_result.reconstructed_data
    )

    estimators = (
        mlp_latent.outer_result.estimator,
        mlp_reconstruction.outer_result.estimator,
        linear_latent.outer_result.estimator,
        linear_reconstruction.outer_result.estimator,
    )

    feature_scalers = (
        mlp_latent.outer_result.feature_scaler,
        mlp_reconstruction.outer_result.feature_scaler,
        linear_latent.outer_result.feature_scaler,
        linear_reconstruction.outer_result.feature_scaler,
    )

    if len({id(item) for item in estimators}) != 4:
        raise ProbeFitError(
            "four_probe_estimator_state_shared",
            "The four probes do not have four independent "
            "estimators.",
        )

    if len({id(item) for item in feature_scalers}) != 4:
        raise ProbeFitError(
            "four_probe_scaler_state_shared",
            "The four probes do not have four independent "
            "feature scalers.",
        )

    probe_scores = {
        "mlp/z_logits": (
            mlp_latent.outer_result.outer_r2_clipped
        ),
        "mlp/reconstruction": (
            mlp_reconstruction.outer_result.outer_r2_clipped
        ),
        "linear/z_logits": (
            linear_latent.outer_result.outer_r2_clipped
        ),
        "linear/reconstruction": (
            linear_reconstruction.outer_result.outer_r2_clipped
        ),
    }

    # Dictionary insertion order defines deterministic tie-breaking.
    worst_probe = max(
        probe_scores,
        key=probe_scores.__getitem__,
    )
    leakage_worst = probe_scores[worst_probe]

    result = FourProbeEvaluationResult(
        mlp_latent_logits=mlp_latent,
        mlp_reconstructed_data=mlp_reconstruction,
        linear_latent_logits=linear_latent,
        linear_reconstructed_data=linear_reconstruction,
        shuffled_target_controls=shuffled_target_controls,
        latent_sample_diagnostic=latent_sample_diagnostic,
        inner_partition=mlp_result.inner_partition,
        worst_probe=worst_probe,
        leakage_worst=float(leakage_worst),
    )
    enforce_shuffled_target_guardrail(result)

    return result

"""Aggregate four primary probes and secondary diagnostics."""

import logging

from .errors import ProbeFitError
from .linear import evaluate_primary_linear_probes
from .mlp import evaluate_primary_mlp_probes
from .provenance import make_probe_evaluation_context
from .types import (
    FourProbeEvaluationResult,
    LeakageProbeRunMetadata,
    ProbeRepresentationSet,
)
from .diagnostics import (
    enforce_shuffled_target_guardrail,
    evaluate_shuffled_target_mlp_controls,
)

log = logging.getLogger(__name__)

def evaluate_four_leakage_probes(
    development_representations: ProbeRepresentationSet,
    held_out_representations: ProbeRepresentationSet,
    *,
    run_shuffled_target_controls: bool = True,
    evaluation_mode: str = "validation",
    run_metadata: LeakageProbeRunMetadata | None = None,
) -> FourProbeEvaluationResult:
    """Evaluate four probes and optional shuffled-target controls."""

    evaluation_context = make_probe_evaluation_context(
        development_representations,
        held_out_representations,
        mode=evaluation_mode,
    )
    if run_metadata is None:
        run_metadata = LeakageProbeRunMetadata(
            autoencoder_seed=None,
            configuration_id=None,
        )

    log.info("Four-probe evaluation: starting the two primary MLP probes (1–2/4).")
    mlp_result = evaluate_primary_mlp_probes(
        development_representations,
        held_out_representations,
    )

    log.info("Four-probe evaluation: starting the two primary linear probes (3–4/4).")
    linear_result = evaluate_primary_linear_probes(
        development_representations,
        held_out_representations,
    )

    shuffled_target_controls = None
    if run_shuffled_target_controls:
        log.info("Starting optional shuffled-target MLP controls (not primary probes).")
        shuffled_target_controls = (
            evaluate_shuffled_target_mlp_controls(
                development_representations,
                held_out_representations,
            )
        )
    else:
        log.info("Shuffled-target controls disabled.")

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
        inner_partition=mlp_result.inner_partition,
        worst_probe=worst_probe,
        leakage_worst=float(leakage_worst),
        evaluation_context=evaluation_context,
        run_metadata=run_metadata,
    )
    if shuffled_target_controls is not None:
        enforce_shuffled_target_guardrail(result)

    log.info("Four-probe maximum: %s, clipped R2=%.6f.", worst_probe, leakage_worst)
    return result

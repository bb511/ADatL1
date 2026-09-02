import json
from unittest.mock import Mock

import pytest

from src.evaluation.leakage_probe import (
    FourProbeEvaluationResult,
    LinearProbeOuterResult,
    MLPProbeCandidateFailure,
    MLPProbeCandidateResult,
    MLPProbeOuterResult,
    MLPProbeSeedSelection,
    NamedLinearProbeResult,
    NamedMLPProbeResult,
    PROBE_INITIALIZATION_SEEDS,
    ProbeFitError,
    four_probe_result_payload,
    ShuffledTargetMLPResult,
    four_probe_result_payload,
)
from tests.helpers.leakage_probe import (
    make_probe_evaluation_context,
    make_probe_run_metadata,
)


def make_candidate(
    seed: int,
    inner_r2_raw: float,
    inner_mae_gev: float,
    *,
    convergence_warnings: tuple[str, ...] = (),
    n_iter: int = 25,
    final_loss: float = 0.05,
) -> MLPProbeCandidateResult:
    return MLPProbeCandidateResult(
        seed=seed,
        inner_r2_raw=inner_r2_raw,
        inner_mae_gev=inner_mae_gev,
        convergence_warnings=convergence_warnings,
        n_iter=n_iter,
        final_loss=final_loss,
        feature_scaler=Mock(),
        target_scaler=Mock(),
        estimator=Mock(),
    )


def make_mlp_outer_result(
    selected_seed: int,
) -> MLPProbeOuterResult:
    return MLPProbeOuterResult(
        selected_seed=selected_seed,
        outer_r2_raw=0.4,
        outer_r2_clipped=0.4,
        outer_mae_gev=7.5,
        convergence_warnings=(
            "Outer refit reached max_iter.",
        ),
        n_iter=500,
        final_loss=0.025,
        n_train=100,
        n_validation=40,
        feature_scaler=Mock(),
        target_scaler=Mock(),
        estimator=Mock(),
    )


def make_linear_probe(
    representation_name: str,
    metric_name: str,
) -> NamedLinearProbeResult:
    return NamedLinearProbeResult(
        representation_name=representation_name,
        metric_name=metric_name,
        feature_dimension=4,
        outer_result=LinearProbeOuterResult(
            outer_r2_raw=0.2,
            outer_r2_clipped=0.2,
            outer_mae_gev=9.0,
            n_train=100,
            n_validation=40,
            feature_scaler=Mock(),
            estimator=Mock(),
        ),
    )

def make_result(
    selection: MLPProbeSeedSelection,
) -> FourProbeEvaluationResult:
    latent_mlp = NamedMLPProbeResult(
        representation_name="latent_logits",
        metric_name="z_logits",
        feature_dimension=4,
        seed_selection=selection,
        outer_result=make_mlp_outer_result(
            selection.selected_seed
        ),
    )

    reconstruction_mlp = NamedMLPProbeResult(
        representation_name="reconstructed_data",
        metric_name="reconstruction",
        feature_dimension=8,
        seed_selection=selection,
        outer_result=make_mlp_outer_result(
            selection.selected_seed
        ),
    )

    return FourProbeEvaluationResult(
        mlp_latent_logits=latent_mlp,
        mlp_reconstructed_data=reconstruction_mlp,
        linear_latent_logits=make_linear_probe(
            "latent_logits",
            "z_logits",
        ),
        linear_reconstructed_data=make_linear_probe(
            "reconstructed_data",
            "reconstruction",
        ),
        shuffled_target_controls=ShuffledTargetMLPResult(
            latent_logits=latent_mlp,
            reconstructed_data=reconstruction_mlp,
            inner_partition=Mock(),
            shuffle_seed=12345,
            permutation_manifest_hash=(
                "test-shuffle-manifest"
            ),
        ),
        inner_partition=Mock(),
        worst_probe="mlp/z_logits",
        leakage_worst=0.4,
        evaluation_context=make_probe_evaluation_context(),
        run_metadata=make_probe_run_metadata(),
    )


def test_payload_contains_every_mlp_initialization() -> None:
    candidate_10 = make_candidate(
        10,
        inner_r2_raw=0.12,
        inner_mae_gev=11.0,
        convergence_warnings=(
            "Seed 10 reached max_iter.",
        ),
        n_iter=500,
        final_loss=0.08,
    )
    candidate_123 = make_candidate(
        123,
        inner_r2_raw=0.36,
        inner_mae_gev=8.2,
        n_iter=140,
        final_loss=0.03,
    )
    failure_500 = MLPProbeCandidateFailure(
        seed=500,
        reason="mlp_fit_failed",
        message="Synthetic seed 500 failure.",
    )

    selection = MLPProbeSeedSelection(
        selected_seed=123,
        selected_candidate=candidate_123,
        successful_candidates=(
            candidate_10,
            candidate_123,
        ),
        failed_candidates=(failure_500,),
    )

    payload = four_probe_result_payload(
        make_result(selection)
    )

    latent_payload = payload["probes"]["mlp/z_logits"]
    diagnostics = latent_payload["seed_selection"]

    # The existing fields retain their meaning; numeric histories are additive.
    without_histories = {
        **diagnostics,
        "candidates": [
            {key: value for key, value in candidate.items() if key != "training_history"}
            for candidate in diagnostics["candidates"]
        ],
    }
    assert without_histories == {
        "selected_seed": 123,
        "candidates": [
            {
                "seed": 10,
                "status": "successful",
                "selected": False,
                "inner_r2_raw": 0.12,
                "inner_mae_gev": 11.0,
                "convergence_warnings": [
                    "Seed 10 reached max_iter.",
                ],
                "n_iter": 500,
                "final_loss": 0.08,
            },
            {
                "seed": 123,
                "status": "successful",
                "selected": True,
                "inner_r2_raw": 0.36,
                "inner_mae_gev": 8.2,
                "convergence_warnings": [],
                "n_iter": 140,
                "final_loss": 0.03,
            },
            {
                "seed": 500,
                "status": "failed",
                "selected": False,
                "reason": "mlp_fit_failed",
                "message": "Synthetic seed 500 failure.",
            },
        ],
    }

    # The final full-training refit diagnostics remain available.
    assert latent_payload["selected_seed"] == 123
    assert latent_payload["n_iter"] == 500
    assert latent_payload["final_loss"] == pytest.approx(
        0.025
    )
    assert latent_payload["convergence_warnings"] == [
        "Outer refit reached max_iter.",
    ]

    # Both primary MLPs receive initialization diagnostics.
    assert (
        payload["probes"]["mlp/reconstruction"][
            "seed_selection"
        ]
        == diagnostics
    )

    # Linear probes have no initialization search.
    assert (
        "seed_selection"
        not in payload["probes"]["linear/z_logits"]
    )
    assert (
        "seed_selection"
        not in payload["probes"][
            "linear/reconstruction"
        ]
    )

    # Ensure no sklearn objects accidentally enter the JSON.
    json.dumps(
        payload,
        allow_nan=False,
    )


def test_candidate_diagnostics_follow_frozen_seed_order() -> None:
    candidates = {
        seed: make_candidate(
            seed,
            inner_r2_raw=float(index) / 10.0,
            inner_mae_gev=10.0 - index,
        )
        for index, seed in enumerate(
            reversed(PROBE_INITIALIZATION_SEEDS)
        )
    }

    # Deliberately provide candidates in reverse order.
    successful_candidates = tuple(
        candidates[seed]
        for seed in reversed(PROBE_INITIALIZATION_SEEDS)
    )

    selected = candidates[123]

    selection = MLPProbeSeedSelection(
        selected_seed=123,
        selected_candidate=selected,
        successful_candidates=successful_candidates,
        failed_candidates=(),
    )

    payload = four_probe_result_payload(
        make_result(selection)
    )

    serialized_seeds = [
        candidate["seed"]
        for candidate in payload["probes"][
            "mlp/z_logits"
        ]["seed_selection"]["candidates"]
    ]

    assert serialized_seeds == list(
        PROBE_INITIALIZATION_SEEDS
    )


def test_missing_candidate_diagnostic_invalidates_payload() -> None:
    candidate_10 = make_candidate(
        10,
        inner_r2_raw=0.1,
        inner_mae_gev=10.0,
    )
    candidate_123 = make_candidate(
        123,
        inner_r2_raw=0.2,
        inner_mae_gev=9.0,
    )

    # Seed 500 is neither successful nor recorded as failed.
    selection = MLPProbeSeedSelection(
        selected_seed=123,
        selected_candidate=candidate_123,
        successful_candidates=(
            candidate_10,
            candidate_123,
        ),
        failed_candidates=(),
    )

    with pytest.raises(ProbeFitError) as error:
        four_probe_result_payload(
            make_result(selection)
        )

    assert (
        error.value.reason
        == "invalid_mlp_candidate_diagnostics"
    )


def test_duplicate_candidate_diagnostic_invalidates_payload() -> None:
    candidate_10 = make_candidate(
        10,
        inner_r2_raw=0.1,
        inner_mae_gev=10.0,
    )
    candidate_123 = make_candidate(
        123,
        inner_r2_raw=0.2,
        inner_mae_gev=9.0,
    )
    candidate_500 = make_candidate(
        500,
        inner_r2_raw=0.3,
        inner_mae_gev=8.0,
    )

    selection = MLPProbeSeedSelection(
        selected_seed=500,
        selected_candidate=candidate_500,
        successful_candidates=(
            candidate_10,
            candidate_123,
            candidate_500,
            candidate_500,
        ),
        failed_candidates=(),
    )

    with pytest.raises(ProbeFitError) as error:
        four_probe_result_payload(
            make_result(selection)
        )

    assert (
        error.value.reason
        == "invalid_mlp_candidate_diagnostics"
    )

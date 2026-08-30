from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import src.evaluation.leakage_probe.evaluation as evaluation_module
from src.evaluation.leakage_probe import (
    FourProbeEvaluationResult,
    ProbeFitError,
    ShuffledTargetGuardrailError,
    ShuffledTargetMLPResult,
    evaluate_four_leakage_probes,
    four_probe_result_payload,
    log_shuffled_target_metrics,
    shuffled_target_metric_values,
)


def make_probe(
    representation_name: str,
    r2_raw: float,
    r2_clipped: float | None = None,
) -> SimpleNamespace:
    metric_names = {
        "latent_logits": "z_logits",
        "reconstructed_data": "reconstruction",
    }
    metric_name = metric_names[representation_name]

    if r2_clipped is None:
        r2_clipped = max(0.0, r2_raw)

    return SimpleNamespace(
        representation_name=representation_name,
        metric_name=metric_name,
        feature_dimension=4,
        outer_result=SimpleNamespace(
            outer_r2_raw=r2_raw,
            outer_r2_clipped=r2_clipped,
            outer_mae_gev=8.0,
            n_train=100,
            n_validation=40,
            estimator=object(),
            feature_scaler=object(),
            target_scaler=object(),
        ),
    )


def make_shuffled_controls(
    latent_r2: float = -0.04,
    reconstruction_r2: float = -0.06,
) -> ShuffledTargetMLPResult:
    return ShuffledTargetMLPResult(
        latent_logits=make_probe(
            "latent_logits",
            latent_r2,
        ),
        reconstructed_data=make_probe(
            "reconstructed_data",
            reconstruction_r2,
        ),
        inner_partition=Mock(),
        shuffle_seed=12345,
        permutation_manifest_hash="shuffle-manifest",
    )


def make_complete_result() -> FourProbeEvaluationResult:
    return FourProbeEvaluationResult(
        mlp_latent_logits=make_probe(
            "latent_logits",
            0.1,
        ),
        mlp_reconstructed_data=make_probe(
            "reconstructed_data",
            0.2,
        ),
        linear_latent_logits=make_probe(
            "latent_logits",
            0.3,
        ),
        linear_reconstructed_data=make_probe(
            "reconstructed_data",
            0.4,
        ),
        shuffled_target_controls=(
            make_shuffled_controls()
        ),
        inner_partition=Mock(),
        worst_probe="linear/reconstruction",
        leakage_worst=0.4,
    )


def test_four_probe_evaluation_rejects_failed_shuffled_guardrail(
    monkeypatch,
) -> None:
    mlp_result = SimpleNamespace(
        latent_logits=make_probe(
            "latent_logits",
            0.1,
        ),
        reconstructed_data=make_probe(
            "reconstructed_data",
            0.2,
        ),
        inner_partition=Mock(),
    )

    linear_result = SimpleNamespace(
        latent_logits=make_probe(
            "latent_logits",
            0.3,
        ),
        reconstructed_data=make_probe(
            "reconstructed_data",
            0.4,
        ),
    )

    # Both controls deliberately exceed the frozen 0.02
    # guardrail threshold.
    shuffled_controls = make_shuffled_controls(
        latent_r2=0.98,
        reconstruction_r2=0.99,
    )

    monkeypatch.setattr(
        evaluation_module,
        "evaluate_primary_mlp_probes",
        Mock(return_value=mlp_result),
    )
    monkeypatch.setattr(
        evaluation_module,
        "evaluate_primary_linear_probes",
        Mock(return_value=linear_result),
    )
    shuffled_evaluator = Mock(
        return_value=shuffled_controls
    )
    monkeypatch.setattr(
        evaluation_module,
        "evaluate_shuffled_target_mlp_controls",
        shuffled_evaluator,
    )
    train = Mock()
    validation = Mock()

    with pytest.raises(
        ShuffledTargetGuardrailError
    ) as error:
        evaluate_four_leakage_probes(
            train,
            validation,
        )

    shuffled_evaluator.assert_called_once_with(
        train,
        validation,
    )

    assert error.value.failed_controls == (
        "z_logits",
        "reconstruction",
    )

    diagnostic_result = error.value.result

    assert (
        diagnostic_result.shuffled_target_controls
        is shuffled_controls
    )

    # The internal primary maximum remains available only for
    # diagnostics; the caller must mark the measurement invalid.
    assert (
        diagnostic_result.worst_probe
        == "linear/reconstruction"
    )
    assert (
        diagnostic_result.leakage_worst
        == pytest.approx(0.4)
    )


def test_shuffled_controls_are_serialized_as_diagnostics() -> None:
    payload = four_probe_result_payload(
        make_complete_result()
    )

    assert set(payload["probes"]) == {
        "mlp/z_logits",
        "mlp/reconstruction",
        "linear/z_logits",
        "linear/reconstruction",
    }

    diagnostics = payload["diagnostics"]

    assert set(diagnostics) == {"shuffled_targets"}

    shuffled = diagnostics["shuffled_targets"]

    assert shuffled["shuffle_seed"] == 12345
    assert (
        shuffled["permutation_manifest_hash"]
        == "shuffle-manifest"
    )

    assert shuffled["z_logits"]["r2_raw"] == pytest.approx(
        -0.04
    )
    assert shuffled["z_logits"][
        "r2_clipped"
    ] == pytest.approx(0.0)

    assert shuffled["reconstruction"][
        "r2_raw"
    ] == pytest.approx(-0.06)
    assert shuffled["reconstruction"][
        "r2_clipped"
    ] == pytest.approx(0.0)

    # Controls remain outside the primary probe dictionary.
    assert all(
        "shuffled" not in probe_name
        for probe_name in payload["probes"]
    )


def test_shuffled_target_metric_names_are_exact() -> None:
    metrics = shuffled_target_metric_values(
        make_complete_result()
    )

    assert metrics == pytest.approx(
        {
            "probe/shuffled/z_logits/r2_raw": -0.04,
            "probe/shuffled/reconstruction/r2_raw": -0.06,
        }
    )


def test_shuffled_metrics_are_logged_separately() -> None:
    logger = Mock()
    result = make_complete_result()

    metrics = log_shuffled_target_metrics(
        result,
        [logger],
        step=31,
    )

    logger.log_metrics.assert_called_once_with(
        metrics,
        step=31,
    )

    assert set(metrics) == {
        "probe/shuffled/z_logits/r2_raw",
        "probe/shuffled/reconstruction/r2_raw",
    }

    assert "probe/leakage_worst" not in metrics


def test_non_finite_shuffled_metric_is_rejected() -> None:
    result = make_complete_result()
    (
        result.shuffled_target_controls
        .latent_logits
        .outer_result
        .outer_r2_raw
    ) = np.nan

    with pytest.raises(ProbeFitError) as error:
        shuffled_target_metric_values(result)

    assert (
        error.value.reason
        == "non_finite_shuffled_target_metric"
    )

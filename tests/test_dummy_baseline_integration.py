from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sklearn.dummy import DummyRegressor

import src.evaluation.leakage_probe.evaluation as leakage_probe
from src.evaluation.leakage_probe import (
    DummyBaselineOuterResult,
    FourProbeEvaluationResult,
    NamedDummyBaselineResult,
    PrimaryDummyBaselineResult,
    evaluate_four_leakage_probes,
    four_probe_metric_values,
    four_probe_result_payload,
    ShuffledTargetMLPResult,
)


def make_primary_probe(
    representation_name: str,
    clipped_r2: float,
) -> SimpleNamespace:
    metric_name = (
        "z_logits"
        if representation_name == "latent_logits"
        else "reconstruction"
    )

    return SimpleNamespace(
        representation_name=representation_name,
        metric_name=metric_name,
        feature_dimension=4,
        outer_result=SimpleNamespace(
            outer_r2_raw=clipped_r2,
            outer_r2_clipped=clipped_r2,
            outer_mae_gev=8.0,
            n_train=100,
            n_validation=40,
            estimator=object(),
            feature_scaler=object(),
        ),
    )


def make_named_dummy_baseline(
    representation_name: str,
    *,
    r2_raw: float,
    r2_clipped: float,
    mae_gev: float,
    train_mean_gev: float,
) -> NamedDummyBaselineResult:
    metric_name = (
        "z_logits"
        if representation_name == "latent_logits"
        else "reconstruction"
    )

    return NamedDummyBaselineResult(
        representation_name=representation_name,
        metric_name=metric_name,
        feature_dimension=4,
        outer_result=DummyBaselineOuterResult(
            outer_r2_raw=r2_raw,
            outer_r2_clipped=r2_clipped,
            outer_mae_gev=mae_gev,
            train_mean_gev=train_mean_gev,
            n_train=100,
            n_validation=40,
            estimator=DummyRegressor(
                strategy="mean"
            ),
        ),
    )


def make_dummy_baselines() -> PrimaryDummyBaselineResult:
    return PrimaryDummyBaselineResult(
        latent_logits=make_named_dummy_baseline(
            "latent_logits",
            r2_raw=-0.03,
            r2_clipped=0.0,
            mae_gev=12.0,
            train_mean_gev=95.0,
        ),
        reconstructed_data=make_named_dummy_baseline(
            "reconstructed_data",
            r2_raw=-0.03,
            r2_clipped=0.0,
            mae_gev=12.0,
            train_mean_gev=95.0,
        ),
    )

def make_shuffled_controls() -> ShuffledTargetMLPResult:
    return ShuffledTargetMLPResult(
        latent_logits=make_primary_probe(
            "latent_logits",
            0.0,
        ),
        reconstructed_data=make_primary_probe(
            "reconstructed_data",
            0.0,
        ),
        inner_partition=Mock(),
        shuffle_seed=12345,
        permutation_manifest_hash=(
            "test-shuffle-manifest"
        ),
    )

def test_four_probe_evaluation_attaches_dummy_diagnostics(
    monkeypatch,
) -> None:
    mlp_result = SimpleNamespace(
        latent_logits=make_primary_probe(
            "latent_logits",
            0.1,
        ),
        reconstructed_data=make_primary_probe(
            "reconstructed_data",
            0.2,
        ),
        inner_partition=object(),
    )

    linear_result = SimpleNamespace(
        latent_logits=make_primary_probe(
            "latent_logits",
            0.3,
        ),
        reconstructed_data=make_primary_probe(
            "reconstructed_data",
            0.4,
        ),
    )

    # Give the diagnostic an intentionally large score. It must
    # still be excluded from the primary maximum.
    dummy_baselines = PrimaryDummyBaselineResult(
        latent_logits=make_named_dummy_baseline(
            "latent_logits",
            r2_raw=0.98,
            r2_clipped=0.98,
            mae_gev=1.0,
            train_mean_gev=95.0,
        ),
        reconstructed_data=make_named_dummy_baseline(
            "reconstructed_data",
            r2_raw=0.99,
            r2_clipped=0.99,
            mae_gev=0.5,
            train_mean_gev=95.0,
        ),
    )

    mlp_evaluator = Mock(return_value=mlp_result)
    linear_evaluator = Mock(return_value=linear_result)
    dummy_evaluator = Mock(return_value=dummy_baselines)

    shuffled_controls = make_shuffled_controls()

    shuffled_evaluator = Mock(
        return_value=shuffled_controls
    )

    monkeypatch.setattr(
        leakage_probe,
        "evaluate_primary_mlp_probes",
        mlp_evaluator,
    )
    monkeypatch.setattr(
        leakage_probe,
        "evaluate_primary_linear_probes",
        linear_evaluator,
    )
    monkeypatch.setattr(
        leakage_probe,
        "evaluate_primary_dummy_baselines",
        dummy_evaluator,
    )
    monkeypatch.setattr(
        leakage_probe,
        "evaluate_shuffled_target_mlp_controls",
        shuffled_evaluator,
    )

    train_representations = Mock()
    validation_representations = Mock()

    result = evaluate_four_leakage_probes(
        train_representations,
        validation_representations,
    )

    dummy_evaluator.assert_called_once_with(
        train_representations,
        validation_representations,
    )

    assert result.dummy_baselines is dummy_baselines

    # Only the four primary scores determine the maximum.
    assert result.worst_probe == "linear/reconstruction"
    assert result.leakage_worst == pytest.approx(0.4)
    assert result.leakage_worst != pytest.approx(0.99)


def test_dummy_baselines_are_serialized_under_diagnostics() -> None:
    result = FourProbeEvaluationResult(
        mlp_latent_logits=make_primary_probe(
            "latent_logits",
            0.1,
        ),
        mlp_reconstructed_data=make_primary_probe(
            "reconstructed_data",
            0.2,
        ),
        linear_latent_logits=make_primary_probe(
            "latent_logits",
            0.3,
        ),
        linear_reconstructed_data=make_primary_probe(
            "reconstructed_data",
            0.4,
        ),
        dummy_baselines=make_dummy_baselines(),
        inner_partition=Mock(),
        worst_probe="linear/reconstruction",
        leakage_worst=0.4,
        shuffled_target_controls=make_shuffled_controls(),
    )

    payload = four_probe_result_payload(result)

    assert set(payload["probes"]) == {
        "mlp/z_logits",
        "mlp/reconstruction",
        "linear/z_logits",
        "linear/reconstruction",
    }

    assert set(payload["diagnostics"]) == {
        "dummy_baselines",
        "shuffled_targets",
    }

    assert payload["diagnostics"]["dummy_baselines"] == {
        "z_logits": {
            "representation_name": "latent_logits",
            "metric_name": "z_logits",
            "feature_dimension": 4,
            "strategy": "mean",
            "train_mean_gev": 95.0,
            "r2_raw": -0.03,
            "r2_clipped": 0.0,
            "mae_gev": 12.0,
            "n_train": 100,
            "n_validation": 40,
        },
        "reconstruction": {
            "representation_name": "reconstructed_data",
            "metric_name": "reconstruction",
            "feature_dimension": 4,
            "strategy": "mean",
            "train_mean_gev": 95.0,
            "r2_raw": -0.03,
            "r2_clipped": 0.0,
            "mae_gev": 12.0,
            "n_train": 100,
            "n_validation": 40,
        },
    }


def test_dummy_baselines_are_excluded_from_logged_metrics() -> None:
    result = FourProbeEvaluationResult(
        mlp_latent_logits=make_primary_probe(
            "latent_logits",
            0.1,
        ),
        mlp_reconstructed_data=make_primary_probe(
            "reconstructed_data",
            0.2,
        ),
        linear_latent_logits=make_primary_probe(
            "latent_logits",
            0.3,
        ),
        linear_reconstructed_data=make_primary_probe(
            "reconstructed_data",
            0.4,
        ),
        dummy_baselines=make_dummy_baselines(),
        inner_partition=Mock(),
        worst_probe="linear/reconstruction",
        leakage_worst=0.4,
        shuffled_target_controls=make_shuffled_controls(),
    )

    metrics = four_probe_metric_values(result)

    assert metrics["probe/leakage_worst"] == pytest.approx(
        0.4
    )
    assert all(
        "dummy" not in metric_name
        for metric_name in metrics
    )
    assert len(metrics) == 13

def make_shuffled_controls() -> ShuffledTargetMLPResult:
    return ShuffledTargetMLPResult(
        latent_logits=make_primary_probe(
            "latent_logits",
            0.0,
        ),
        reconstructed_data=make_primary_probe(
            "reconstructed_data",
            0.0,
        ),
        inner_partition=Mock(),
        shuffle_seed=12345,
        permutation_manifest_hash=(
            "test-shuffle-manifest"
        ),
    )
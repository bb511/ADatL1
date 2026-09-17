from unittest.mock import Mock

import numpy as np
import pytest

import src.evaluation.leakage_probe.mlp as leakage_probe
from src.evaluation.leakage_probe import (
    MLPProbeOuterResult,
    NamedMLPProbeResult,
    ProbeFitError,
    ProbeRepresentationSet,
    evaluate_mlp_probe_representation,
    evaluate_primary_mlp_probes,
)


def make_representation_set(
    split: str,
    n_events: int,
) -> ProbeRepresentationSet:
    target = np.linspace(
        80.0,
        180.0,
        num=n_events,
        dtype=np.float64,
    )

    latent_logits = np.column_stack(
        [
            target,
            target**2,
        ]
    )

    reconstructed_data = np.column_stack(
        [
            target,
            target**2,
            np.sin(target),
        ]
    )

    latent_sample = (
        latent_logits > latent_logits.mean(axis=0)
    ).astype(np.float64)

    return ProbeRepresentationSet(
        split=split,
        latent_logits=latent_logits,
        latent_sample=latent_sample,
        reconstructed_data=reconstructed_data,
        sensitive_target=target,
        n_events=n_events,
        sample_seed=12345,
        max_samples=None,
        manifest_hash=f"{split}-manifest",
        data_cache_id="test-cache",
        data_cache_path="/test/cache",
        source_splits=(split,),
    )


def make_outer_result(
    seed: int,
    clipped_r2: float,
) -> MLPProbeOuterResult:
    return MLPProbeOuterResult(
        seed=seed,
        outer_r2_raw=clipped_r2,
        outer_r2_clipped=clipped_r2,
        outer_mae_gev=10.0,
        convergence_warnings=(),
        n_iter=10,
        final_loss=0.1,
        n_train=20,
        n_validation=10,
        feature_scaler=Mock(),
        target_scaler=Mock(),
        estimator=Mock(),
    )


def test_named_probe_uses_requested_representation(
    monkeypatch,
) -> None:
    train = make_representation_set("train", 20)
    validation = make_representation_set("valid", 10)

    outer = make_outer_result(
        seed=123,
        clipped_r2=0.4,
    )

    fit_calls = []

    def fake_fit(
        train_features,
        train_target,
        validation_features,
        validation_target,
    ):
        fit_calls.append(
            (
                train_features,
                train_target,
                validation_features,
                validation_target,
            )
        )
        return outer

    monkeypatch.setattr(
        leakage_probe,
        "fit_mlp_probe",
        fake_fit,
    )

    result = evaluate_mlp_probe_representation(
        train,
        validation,
        representation_name="latent_logits",
    )

    assert result.representation_name == "latent_logits"
    assert result.metric_name == "z_logits"
    assert result.feature_dimension == 2
    assert result.outer_result is outer

    assert fit_calls == [
        (
            train.latent_logits,
            train.sensitive_target,
            validation.latent_logits,
            validation.sensitive_target,
        )
    ]


def test_primary_evaluator_evaluates_both_representations(
    monkeypatch,
) -> None:
    train = make_representation_set("train", 20)
    validation = make_representation_set("valid", 10)

    received_representations = []

    def fake_evaluate(
        train_representations,
        validation_representations,
        *,
        representation_name,
    ):
        received_representations.append(representation_name)

        if representation_name == "latent_logits":
            return NamedMLPProbeResult(
                representation_name="latent_logits",
                metric_name="z_logits",
                feature_dimension=2,
                outer_result=make_outer_result(123, 0.2),
            )

        return NamedMLPProbeResult(
            representation_name="reconstructed_data",
            metric_name="reconstruction",
            feature_dimension=3,
            outer_result=make_outer_result(123, 0.7),
        )

    monkeypatch.setattr(
        leakage_probe,
        "evaluate_mlp_probe_representation",
        fake_evaluate,
    )

    result = evaluate_primary_mlp_probes(
        train,
        validation,
    )

    assert received_representations == [
        "latent_logits",
        "reconstructed_data",
    ]
    assert result.latent_logits.outer_result.seed == 123
    assert result.reconstructed_data.outer_result.seed == 123


def test_worst_leakage_is_maximum_not_average(
    monkeypatch,
) -> None:
    train = make_representation_set("train", 20)
    validation = make_representation_set("valid", 10)

    def fake_evaluate(
        train_representations,
        validation_representations,
        *,
        representation_name,
    ):
        if representation_name == "latent_logits":
            clipped = 0.1
            seed = 10
            metric_name = "z_logits"
            feature_dimension = 2
        else:
            clipped = 0.9
            seed = 123
            metric_name = "reconstruction"
            feature_dimension = 3

        return NamedMLPProbeResult(
            representation_name=representation_name,
            metric_name=metric_name,
            feature_dimension=feature_dimension,
            outer_result=make_outer_result(
                seed,
                clipped,
            ),
        )

    monkeypatch.setattr(
        leakage_probe,
        "evaluate_mlp_probe_representation",
        fake_evaluate,
    )

    result = evaluate_primary_mlp_probes(
        train,
        validation,
    )

    assert result.leakage_worst == pytest.approx(0.9)
    assert result.leakage_worst != pytest.approx(0.5)


def test_primary_probes_do_not_share_fitted_state(
    monkeypatch,
) -> None:
    train = make_representation_set("train", 20)
    validation = make_representation_set("valid", 10)

    def fake_evaluate(
        train_representations,
        validation_representations,
        *,
        representation_name,
    ):
        seed = (
            10
            if representation_name == "latent_logits"
            else 123
        )

        metric_name = (
            "z_logits"
            if representation_name == "latent_logits"
            else "reconstruction"
        )

        feature_dimension = (
            2
            if representation_name == "latent_logits"
            else 3
        )

        return NamedMLPProbeResult(
            representation_name=representation_name,
            metric_name=metric_name,
            feature_dimension=feature_dimension,
            outer_result=make_outer_result(
                seed,
                0.3,
            ),
        )

    monkeypatch.setattr(
        leakage_probe,
        "evaluate_mlp_probe_representation",
        fake_evaluate,
    )

    result = evaluate_primary_mlp_probes(
        train,
        validation,
    )

    latent = result.latent_logits.outer_result
    reconstruction = (
        result.reconstructed_data.outer_result
    )

    assert latent.estimator is not reconstruction.estimator
    assert (
        latent.feature_scaler
        is not reconstruction.feature_scaler
    )
    assert (
        latent.target_scaler
        is not reconstruction.target_scaler
    )


def test_shared_primary_probe_state_is_rejected(
    monkeypatch,
) -> None:
    train = make_representation_set("train", 20)
    validation = make_representation_set("valid", 10)

    shared_estimator = Mock()

    def fake_evaluate(
        train_representations,
        validation_representations,
        *,
        representation_name,
    ):
        outer = make_outer_result(10, 0.3)
        object.__setattr__(
            outer,
            "estimator",
            shared_estimator,
        )

        metric_name = (
            "z_logits"
            if representation_name == "latent_logits"
            else "reconstruction"
        )

        return NamedMLPProbeResult(
            representation_name=representation_name,
            metric_name=metric_name,
            feature_dimension=2,
            outer_result=outer,
        )

    monkeypatch.setattr(
        leakage_probe,
        "evaluate_mlp_probe_representation",
        fake_evaluate,
    )

    with pytest.raises(ProbeFitError) as error:
        evaluate_primary_mlp_probes(
            train,
            validation,
        )

    assert error.value.reason == "primary_probe_state_shared"


@pytest.mark.parametrize(
    ("train_split", "validation_split", "expected_reason"),
    [
        (
            "valid",
            "valid",
            "invalid_probe_training_split",
        ),
        (
            "train",
            "train",
            "invalid_probe_outer_split",
        ),
        (
            "train",
            "test",
            "invalid_probe_outer_split",
        ),
    ],
)
def test_primary_evaluator_enforces_outer_split_protocol(
    train_split: str,
    validation_split: str,
    expected_reason: str,
) -> None:
    train = make_representation_set(
        train_split,
        20,
    )
    validation = make_representation_set(
        validation_split,
        10,
    )

    with pytest.raises(ProbeFitError) as error:
        evaluate_primary_mlp_probes(
            train,
            validation,
        )

    assert error.value.reason == expected_reason


def test_unknown_representation_is_rejected() -> None:
    train = make_representation_set("train", 20)
    validation = make_representation_set("valid", 10)

    with pytest.raises(ProbeFitError) as error:
        evaluate_mlp_probe_representation(
            train,
            validation,
            representation_name="control_x",
        )

    assert (
        error.value.reason
        == "unknown_probe_representation"
    )

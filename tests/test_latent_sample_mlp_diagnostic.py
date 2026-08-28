from unittest.mock import Mock

import numpy as np
import pytest

import src.evaluation.leakage_probe.diagnostics as diagnostics_module
import src.evaluation.leakage_probe.mlp as mlp_module
from src.evaluation.leakage_probe import (
    MLPProbeCandidateResult,
    MLPProbeOuterResult,
    MLPProbeSeedSelection,
    NamedMLPProbeResult,
    ProbeFitError,
    ProbeRepresentationSet,
    evaluate_latent_sample_mlp_diagnostic,
    evaluate_mlp_probe_representation,
    make_probe_inner_partition,
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

    latent_sample = (
        latent_logits
        > latent_logits.mean(axis=0)
    ).astype(np.float64)

    reconstructed_data = np.column_stack(
        [
            target,
            target**2,
            np.sin(target),
        ]
    )

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
    )


def make_selection(
    seed: int = 123,
) -> MLPProbeSeedSelection:
    candidate = MLPProbeCandidateResult(
        seed=seed,
        inner_r2_raw=0.3,
        inner_mae_gev=8.0,
        convergence_warnings=(),
        n_iter=20,
        final_loss=0.1,
        feature_scaler=Mock(),
        target_scaler=Mock(),
        estimator=Mock(),
    )

    return MLPProbeSeedSelection(
        selected_seed=seed,
        selected_candidate=candidate,
        successful_candidates=(candidate,),
        failed_candidates=(),
    )


def make_outer_result(
    seed: int = 123,
) -> MLPProbeOuterResult:
    return MLPProbeOuterResult(
        selected_seed=seed,
        outer_r2_raw=0.45,
        outer_r2_clipped=0.45,
        outer_mae_gev=7.0,
        convergence_warnings=(),
        n_iter=30,
        final_loss=0.08,
        n_train=20,
        n_validation=10,
        feature_scaler=Mock(),
        target_scaler=Mock(),
        estimator=Mock(),
    )


def make_named_latent_sample_probe(
) -> NamedMLPProbeResult:
    selection = make_selection()
    outer_result = make_outer_result()

    return NamedMLPProbeResult(
        representation_name="latent_sample",
        metric_name="z_sample",
        feature_dimension=2,
        seed_selection=selection,
        outer_result=outer_result,
    )


def test_generic_mlp_evaluator_supports_latent_sample(
    monkeypatch,
) -> None:
    train = make_representation_set(
        "train",
        20,
    )
    validation = make_representation_set(
        "valid",
        10,
    )
    partition = make_probe_inner_partition(
        train.n_events
    )

    selection = make_selection()
    outer_result = make_outer_result()

    select_seed = Mock(
        return_value=selection
    )
    refit = Mock(
        return_value=outer_result
    )

    monkeypatch.setattr(
        mlp_module,
        "select_mlp_probe_seed",
        select_seed,
    )
    monkeypatch.setattr(
        mlp_module,
        "refit_selected_mlp_probe",
        refit,
    )

    result = evaluate_mlp_probe_representation(
        train,
        validation,
        partition,
        representation_name="latent_sample",
    )

    assert result.representation_name == "latent_sample"
    assert result.metric_name == "z_sample"
    assert result.feature_dimension == 2
    assert result.seed_selection is selection
    assert result.outer_result is outer_result

    select_call = select_seed.call_args
    assert select_call is not None
    assert select_call.args[0] is train.latent_sample
    assert (
        select_call.args[1]
        is train.sensitive_target
    )
    assert select_call.args[2] is partition

    refit_call = refit.call_args
    assert refit_call is not None
    assert refit_call.args[0] is train.latent_sample
    assert (
        refit_call.args[1]
        is train.sensitive_target
    )
    assert (
        refit_call.args[2]
        is validation.latent_sample
    )
    assert (
        refit_call.args[3]
        is validation.sensitive_target
    )
    assert refit_call.args[4] is selection


def test_latent_sample_diagnostic_delegates_to_frozen_mlp(
    monkeypatch,
) -> None:
    train = make_representation_set(
        "train",
        20,
    )
    validation = make_representation_set(
        "valid",
        10,
    )
    partition = make_probe_inner_partition(
        train.n_events
    )

    expected_result = (
        make_named_latent_sample_probe()
    )
    generic_evaluator = Mock(
        return_value=expected_result
    )

    monkeypatch.setattr(
        diagnostics_module,
        "evaluate_mlp_probe_representation",
        generic_evaluator,
    )

    result = (
        evaluate_latent_sample_mlp_diagnostic(
            train,
            validation,
            partition,
        )
    )

    assert result is expected_result

    call = generic_evaluator.call_args
    assert call is not None
    assert call.args[0] is train
    assert call.args[1] is validation
    assert call.args[2] is partition
    assert call.kwargs == {
        "representation_name": "latent_sample",
    }


@pytest.mark.parametrize(
    (
        "train_split",
        "validation_split",
        "expected_reason",
    ),
    [
        (
            "valid",
            "valid",
            "invalid_latent_sample_diagnostic_training_split",
        ),
        (
            "train",
            "test",
            "invalid_latent_sample_diagnostic_outer_split",
        ),
    ],
)
def test_latent_sample_diagnostic_rejects_wrong_splits(
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
    partition = make_probe_inner_partition(
        train.n_events
    )

    with pytest.raises(
        ProbeFitError
    ) as error:
        evaluate_latent_sample_mlp_diagnostic(
            train,
            validation,
            partition,
        )

    assert error.value.reason == expected_reason
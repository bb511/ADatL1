from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.evaluation.leakage_probe.evaluation as evaluation_module
from src.evaluation.leakage_probe import (
    FourProbeEvaluationResult,
    evaluate_four_leakage_probes,
)


def make_probe(
    representation_name: str,
    clipped_r2: float,
) -> SimpleNamespace:
    metric_names = {
        "latent_logits": "z_logits",
        "reconstructed_data": "reconstruction",
        "latent_sample": "z_sample",
    }

    return SimpleNamespace(
        representation_name=representation_name,
        metric_name=metric_names[
            representation_name
        ],
        feature_dimension=4,
        outer_result=SimpleNamespace(
            outer_r2_raw=clipped_r2,
            outer_r2_clipped=clipped_r2,
            outer_mae_gev=8.0,
            n_train=100,
            n_validation=40,
            estimator=object(),
            feature_scaler=object(),
            target_scaler=object(),
        ),
    )


def test_four_probe_evaluation_attaches_latent_sample_diagnostic(
    monkeypatch,
) -> None:
    inner_partition = object()

    mlp_result = SimpleNamespace(
        latent_logits=make_probe(
            "latent_logits",
            0.1,
        ),
        reconstructed_data=make_probe(
            "reconstructed_data",
            0.2,
        ),
        inner_partition=inner_partition,
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

    dummy_baselines = SimpleNamespace(
        latent_logits=object(),
        reconstructed_data=object(),
    )

    shuffled_target_controls = SimpleNamespace(
        latent_logits=make_probe(
            "latent_logits",
            0.0,
        ),
        reconstructed_data=make_probe(
            "reconstructed_data",
            0.0,
        ),
    )

    # Deliberately larger than every primary score.
    # It must remain outside leakage_worst.
    latent_sample_diagnostic = make_probe(
        "latent_sample",
        0.99,
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
    monkeypatch.setattr(
        evaluation_module,
        "evaluate_primary_dummy_baselines",
        Mock(return_value=dummy_baselines),
    )
    monkeypatch.setattr(
        evaluation_module,
        "evaluate_shuffled_target_mlp_controls",
        Mock(
            return_value=shuffled_target_controls
        ),
    )

    diagnostic_evaluator = Mock(
        return_value=latent_sample_diagnostic
    )
    monkeypatch.setattr(
        evaluation_module,
        "evaluate_latent_sample_mlp_diagnostic",
        diagnostic_evaluator,
    )

    train = Mock()
    validation = Mock()

    result = evaluate_four_leakage_probes(
        train,
        validation,
    )

    assert isinstance(
        result,
        FourProbeEvaluationResult,
    )

    diagnostic_call = (
        diagnostic_evaluator.call_args
    )
    assert diagnostic_call is not None
    assert diagnostic_call.args[0] is train
    assert diagnostic_call.args[1] is validation

    # The exact primary MLP partition must be reused.
    assert (
        diagnostic_call.args[2]
        is inner_partition
    )

    assert (
        result.latent_sample_diagnostic
        is latent_sample_diagnostic
    )
    assert (
        result.inner_partition
        is inner_partition
    )

    # Exactly the four primary probes determine leakage_worst.
    assert result.worst_probe == "linear/reconstruction"
    assert result.leakage_worst == pytest.approx(0.4)
    assert result.leakage_worst != pytest.approx(0.99)

    primary_probes = (
        result.mlp_latent_logits,
        result.mlp_reconstructed_data,
        result.linear_latent_logits,
        result.linear_reconstructed_data,
    )

    assert len(primary_probes) == 4
    assert (
        result.latent_sample_diagnostic
        not in primary_probes
    )
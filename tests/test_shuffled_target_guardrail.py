import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.evaluation.leakage_probe.persistence as persistence_module
from src.evaluation.leakage_probe import (
    FourProbeEvaluationResult,
    LEAKAGE_PROBE_PROTOCOL_VERSION,
    LeakageProbeRunOutcome,
    SHUFFLED_TARGET_R2_CLIPPED_MAX,
    ShuffledTargetGuardrailError,
    ShuffledTargetMLPResult,
    enforce_shuffled_target_guardrail,
    shuffled_target_guardrail_failures,
)


def make_probe(
    representation_name: str,
    r2_clipped: float,
) -> SimpleNamespace:
    metric_names = {
        "latent_logits": "z_logits",
        "reconstructed_data": "reconstruction",
        "latent_sample": "z_sample",
    }
    metric_name = metric_names[representation_name]

    return SimpleNamespace(
        representation_name=representation_name,
        metric_name=metric_name,
        feature_dimension=4,
        outer_result=SimpleNamespace(
            outer_r2_raw=r2_clipped,
            outer_r2_clipped=r2_clipped,
            outer_mae_gev=8.0,
            n_train=100,
            n_validation=40,
            estimator=object(),
            feature_scaler=object(),
            target_scaler=object(),
        ),
    )


def make_result(
    *,
    shuffled_latent: float,
    shuffled_reconstruction: float,
) -> FourProbeEvaluationResult:
    shuffled_controls = ShuffledTargetMLPResult(
        latent_logits=make_probe(
            "latent_logits",
            shuffled_latent,
        ),
        reconstructed_data=make_probe(
            "reconstructed_data",
            shuffled_reconstruction,
        ),
        inner_partition=Mock(),
        shuffle_seed=12345,
        permutation_manifest_hash="shuffle-manifest",
    )

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
        shuffled_target_controls=shuffled_controls,
        latent_sample_diagnostic=make_probe(
            "latent_sample",
            0.5,
        ),
        inner_partition=Mock(),
        worst_probe="linear/reconstruction",
        leakage_worst=0.4,
    )


@pytest.mark.parametrize(
    (
        "shuffled_latent",
        "shuffled_reconstruction",
    ),
    [
        (0.0, 0.0),
        (0.02, 0.0),
        (0.0, 0.02),
        (0.02, 0.02),
    ],
)
def test_guardrail_accepts_scores_at_or_below_threshold(
    shuffled_latent: float,
    shuffled_reconstruction: float,
) -> None:
    result = make_result(
        shuffled_latent=shuffled_latent,
        shuffled_reconstruction=(
            shuffled_reconstruction
        ),
    )

    assert (
        SHUFFLED_TARGET_R2_CLIPPED_MAX
        == pytest.approx(0.02)
    )
    assert shuffled_target_guardrail_failures(
        result
    ) == ()

    enforce_shuffled_target_guardrail(result)


@pytest.mark.parametrize(
    (
        "shuffled_latent",
        "shuffled_reconstruction",
        "expected_failures",
    ),
    [
        (
            0.020001,
            0.0,
            ("z_logits",),
        ),
        (
            0.0,
            0.020001,
            ("reconstruction",),
        ),
        (
            0.03,
            0.04,
            (
                "z_logits",
                "reconstruction",
            ),
        ),
    ],
)
def test_guardrail_rejects_scores_above_threshold(
    shuffled_latent: float,
    shuffled_reconstruction: float,
    expected_failures: tuple[str, ...],
) -> None:
    result = make_result(
        shuffled_latent=shuffled_latent,
        shuffled_reconstruction=(
            shuffled_reconstruction
        ),
    )

    assert shuffled_target_guardrail_failures(
        result
    ) == expected_failures

    with pytest.raises(
        ShuffledTargetGuardrailError
    ) as error:
        enforce_shuffled_target_guardrail(
            result
        )

    assert (
        error.value.reason
        == "shuffled_target_guardrail_failed"
    )
    assert (
        error.value.failed_controls
        == expected_failures
    )
    assert error.value.result is result


def test_guardrail_failure_is_recorded_without_leakage_score(
    monkeypatch,
    tmp_path,
) -> None:
    result = make_result(
        shuffled_latent=0.03,
        shuffled_reconstruction=0.01,
    )

    with pytest.raises(
        ShuffledTargetGuardrailError
    ) as captured_error:
        enforce_shuffled_target_guardrail(
            result
        )

    strict_evaluator = Mock(
        side_effect=captured_error.value
    )
    monkeypatch.setattr(
        persistence_module,
        "evaluate_and_write_loss_total_leakage_probes",
        strict_evaluator,
    )

    outcome = (
        persistence_module
        .evaluate_and_record_loss_total_leakage_probes(
            Mock(),
            Mock(),
            tmp_path,
            device="cpu",
        )
    )

    assert isinstance(
        outcome,
        LeakageProbeRunOutcome,
    )
    assert outcome.probe_valid is False
    assert outcome.result is None
    assert outcome.diagnostic_result is result
    assert (
        outcome.rejection_reason
        == "shuffled_target_guardrail_failed"
    )

    payload = json.loads(
        outcome.output_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["leakage_probe_protocol_version"] == (
        LEAKAGE_PROBE_PROTOCOL_VERSION
    )
    assert payload["probe_valid"] is False
    assert (
        payload["rejection_reason"]
        == "shuffled_target_guardrail_failed"
    )

    # An invalid score must be absent/null, never zero.
    assert payload["worst_probe"] is None
    assert payload["leakage_worst"] is None

    # The completed probes remain available for audit.
    assert set(payload["probes"]) == {
        "mlp/z_logits",
        "mlp/reconstruction",
        "linear/z_logits",
        "linear/reconstruction",
    }

    shuffled = payload["diagnostics"][
        "shuffled_targets"
    ]

    assert shuffled["z_logits"][
        "r2_clipped"
    ] == pytest.approx(0.03)

    assert shuffled["reconstruction"][
        "r2_clipped"
    ] == pytest.approx(0.01)

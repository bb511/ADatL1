from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.evaluation.leakage_probe import (
    ProbeFitError,
    four_probe_metric_values,
    log_four_probe_metrics,
)


def make_probe(
    r2_raw: float,
    r2_clipped: float,
    mae_gev: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        outer_result=SimpleNamespace(
            outer_r2_raw=r2_raw,
            outer_r2_clipped=r2_clipped,
            outer_mae_gev=mae_gev,
        )
    )


def make_four_probe_result() -> SimpleNamespace:
    return SimpleNamespace(
        mlp_latent_logits=make_probe(
            r2_raw=0.31,
            r2_clipped=0.31,
            mae_gev=8.1,
        ),
        mlp_reconstructed_data=make_probe(
            r2_raw=-0.04,
            r2_clipped=0.0,
            mae_gev=10.2,
        ),
        linear_latent_logits=make_probe(
            r2_raw=0.18,
            r2_clipped=0.18,
            mae_gev=9.3,
        ),
        linear_reconstructed_data=make_probe(
            r2_raw=0.42,
            r2_clipped=0.42,
            mae_gev=7.4,
        ),
        worst_probe="linear/reconstruction",
        leakage_worst=0.42,
    )


def expected_metrics() -> dict[str, float]:
    return {
        "probe/mlp/z_logits/r2_raw": 0.31,
        "probe/mlp/z_logits/r2_clipped": 0.31,
        "probe/mlp/z_logits/mae_gev": 8.1,
        "probe/mlp/reconstruction/r2_raw": -0.04,
        "probe/mlp/reconstruction/r2_clipped": 0.0,
        "probe/mlp/reconstruction/mae_gev": 10.2,
        "probe/linear/z_logits/r2_raw": 0.18,
        "probe/linear/z_logits/r2_clipped": 0.18,
        "probe/linear/z_logits/mae_gev": 9.3,
        "probe/linear/reconstruction/r2_raw": 0.42,
        "probe/linear/reconstruction/r2_clipped": 0.42,
        "probe/linear/reconstruction/mae_gev": 7.4,
        "probe/leakage_worst": 0.42,
    }


def test_four_probe_metric_values_returns_exact_contract_names() -> None:
    metrics = four_probe_metric_values(
        make_four_probe_result()
    )

    assert set(metrics) == set(expected_metrics())
    assert metrics == pytest.approx(expected_metrics())
    assert all(isinstance(value, float) for value in metrics.values())


def test_log_four_probe_metrics_logs_to_every_configured_logger() -> None:
    first_logger = Mock()
    second_logger = Mock()

    metrics = log_four_probe_metrics(
        make_four_probe_result(),
        [first_logger, second_logger],
        step=27,
    )

    assert metrics == pytest.approx(expected_metrics())

    first_logger.log_metrics.assert_called_once_with(
        metrics,
        step=27,
    )
    second_logger.log_metrics.assert_called_once_with(
        metrics,
        step=27,
    )


def test_log_four_probe_metrics_supports_no_logger() -> None:
    metrics = log_four_probe_metrics(
        make_four_probe_result(),
        [],
        step=0,
    )

    assert metrics == pytest.approx(expected_metrics())


def test_four_probe_metric_values_rejects_non_finite_values() -> None:
    result = make_four_probe_result()
    result.mlp_latent_logits.outer_result.outer_r2_raw = (
        np.nan
    )

    with pytest.raises(ProbeFitError) as error:
        four_probe_metric_values(result)

    assert error.value.reason == "non_finite_primary_probe_metric"
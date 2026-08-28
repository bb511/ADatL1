import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import src.evaluation.leakage_probe as leakage_probe
from src.evaluation.leakage_probe import (
    FourProbeEvaluationResult,
    LEAKAGE_PROBE_PROTOCOL_VERSION,
    LeakageProbeRunOutcome,
    ProbeExtractionError,
    ProbeFitError,
    ProbePartitionError,
    evaluate_and_record_loss_total_leakage_probes,
    log_leakage_probe_outcome_metadata,
)


def expected_output_path(run_folder: Path) -> Path:
    return (
        run_folder
        / "plots"
        / "val"
        / "loss_total"
        / "probes"
        / "leakage_probes.json"
    )


@pytest.mark.parametrize(
    ("error_type", "reason"),
    [
        (
            ProbeExtractionError,
            "loss_total_checkpoint_missing",
        ),
        (
            ProbePartitionError,
            "inner_partition_too_small",
        ),
        (
            ProbeFitError,
            "all_mlp_candidates_failed",
        ),
    ],
)
def test_expected_probe_failure_is_persisted_as_invalid(
    monkeypatch,
    tmp_path,
    error_type,
    reason: str,
) -> None:
    failure = error_type(
        reason,
        "Synthetic expected probe failure.",
    )

    strict_evaluator = Mock(side_effect=failure)
    monkeypatch.setattr(
        leakage_probe,
        "evaluate_and_write_loss_total_leakage_probes",
        strict_evaluator,
    )

    outcome = evaluate_and_record_loss_total_leakage_probes(
        Mock(),
        Mock(),
        tmp_path,
        device="cpu",
    )

    output_path = expected_output_path(tmp_path)

    assert isinstance(outcome, LeakageProbeRunOutcome)
    assert outcome.probe_valid is False
    assert outcome.result is None
    assert outcome.output_path == output_path
    assert outcome.rejection_reason == reason
    assert (
        outcome.rejection_message
        == "Synthetic expected probe failure."
    )

    assert output_path.is_file()

    payload = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert payload == {
        "leakage_probe_protocol_version": (
            LEAKAGE_PROBE_PROTOCOL_VERSION
        ),
        "probe_valid": False,
        "rejection_reason": reason,
        "rejection_message": (
            "Synthetic expected probe failure."
        ),
        "worst_probe": None,
        "leakage_worst": None,
        "probes": {},
    }

    # Invalid leakage must be absent/null, never represented as zero.
    assert payload["leakage_worst"] is None


def test_successful_probe_evaluation_returns_valid_outcome(
    monkeypatch,
    tmp_path,
) -> None:
    result = Mock(spec=FourProbeEvaluationResult)
    output_path = expected_output_path(tmp_path)

    strict_evaluator = Mock(
        return_value=(result, output_path)
    )
    invalid_writer = Mock()

    monkeypatch.setattr(
        leakage_probe,
        "evaluate_and_write_loss_total_leakage_probes",
        strict_evaluator,
    )
    monkeypatch.setattr(
        leakage_probe,
        "write_invalid_leakage_probe_result",
        invalid_writer,
    )

    outcome = evaluate_and_record_loss_total_leakage_probes(
        Mock(),
        Mock(),
        tmp_path,
        device="cpu",
    )

    assert outcome == LeakageProbeRunOutcome(
        probe_valid=True,
        result=result,
        output_path=output_path,
        rejection_reason=None,
        rejection_message=None,
    )

    invalid_writer.assert_not_called()


def test_unexpected_programming_error_is_not_hidden(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        leakage_probe,
        "evaluate_and_write_loss_total_leakage_probes",
        Mock(
            side_effect=TypeError(
                "Unexpected implementation bug."
            )
        ),
    )

    with pytest.raises(
        TypeError,
        match="Unexpected implementation bug",
    ):
        evaluate_and_record_loss_total_leakage_probes(
            Mock(),
            Mock(),
            tmp_path,
        )

    assert not expected_output_path(tmp_path).exists()


def test_invalid_outcome_metadata_is_logged_without_leakage(
    tmp_path,
) -> None:
    logger = Mock()

    outcome = LeakageProbeRunOutcome(
        probe_valid=False,
        result=None,
        output_path=expected_output_path(tmp_path),
        rejection_reason="constant_target",
        rejection_message="Target is constant.",
    )

    metadata = log_leakage_probe_outcome_metadata(
        outcome,
        [logger],
    )

    assert metadata == {
        "leakage_probe_protocol_version": (
            LEAKAGE_PROBE_PROTOCOL_VERSION
        ),
        "probe_valid": False,
        "probe_rejection_reason": "constant_target",
    }

    logger.log_hyperparams.assert_called_once_with(metadata)
    logger.log_metrics.assert_not_called()


def test_valid_outcome_metadata_is_logged(
    tmp_path,
) -> None:
    logger = Mock()
    result = Mock(spec=FourProbeEvaluationResult)

    outcome = LeakageProbeRunOutcome(
        probe_valid=True,
        result=result,
        output_path=expected_output_path(tmp_path),
        rejection_reason=None,
        rejection_message=None,
    )

    metadata = log_leakage_probe_outcome_metadata(
        outcome,
        [logger],
    )

    assert metadata == {
        "leakage_probe_protocol_version": (
            LEAKAGE_PROBE_PROTOCOL_VERSION
        ),
        "probe_valid": True,
        "probe_rejection_reason": "none",
    }

    logger.log_hyperparams.assert_called_once_with(metadata)
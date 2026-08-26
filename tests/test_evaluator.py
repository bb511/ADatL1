from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.evaluation.evaluator import Evaluator
from src.evaluation.callbacks.reco import ReconstructionPlots


def test_evaluate_root_checkpoint_uses_named_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "loss_total.ckpt"
    checkpoint.touch()

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.evaluator = SimpleNamespace(
        strat_name=None,
        metric_name="old_metric",
        criterion_name="old_criterion",
        ckpt_path_name=None,
    )
    evaluator.optimized_metric_config = None
    evaluator.evaluate_ckpt = Mock()
    evaluator._get_optimized_metric = Mock()
    evaluator._make_criterion_summary_plots = Mock()

    model = object()
    test_loaders = {"normal": object()}
    evaluator.evaluate_root_checkpoint(
        tmp_path,
        model,
        test_loaders,
        checkpoint_name="loss_total",
    )

    evaluator.evaluate_ckpt.assert_called_once_with(
        checkpoint,
        model,
        test_loaders,
    )
    evaluator._get_optimized_metric.assert_called_once_with(None)
    evaluator._make_criterion_summary_plots.assert_called_once_with(tmp_path)
    assert evaluator.evaluator.ckpt_path_name == "loss_total.ckpt"
    assert evaluator.evaluator.strat_name is None
    assert evaluator.evaluator.metric_name is None
    assert evaluator.evaluator.criterion_name is None


def test_evaluate_root_checkpoint_requires_existing_checkpoint(
    tmp_path: Path,
) -> None:
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.evaluator = SimpleNamespace(
        strat_name=None,
        metric_name=None,
        criterion_name=None,
        ckpt_path_name=None,
    )

    with pytest.raises(FileNotFoundError, match="loss_total[.]ckpt"):
        evaluator.evaluate_root_checkpoint(
            tmp_path,
            model=object(),
            test_loaders={},
            checkpoint_name="loss_total",
        )


def test_reconstruction_callback_supports_named_root_checkpoint() -> None:
    callback = ReconstructionPlots(
        warmup_batches=0.2,
        output_name="reconstructed_data",
        ckpts={"loss_total": True},
    )
    loss_total = SimpleNamespace(
        strat_name="loss_total",
        metric_name=None,
        criterion_name=None,
    )
    last = SimpleNamespace(
        strat_name="last",
        metric_name=None,
        criterion_name=None,
    )

    assert callback._should_run_for_current_ckpt(loss_total)
    assert not callback._should_run_for_current_ckpt(last)

import csv
from unittest.mock import Mock

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from src.evaluation.callbacks.utils.misc import write_metric_values
from src.train import (
    _allowed_evaluation_callback_configs,
    _setup_algorithm_for_evaluation,
    train,
)


def test_train_skips_evaluator_when_unconfigured(cfg_train: DictConfig) -> None:
    """Training should complete without an evaluator when evaluation config is null."""
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.evaluation = None
        cfg_train.test = False
        cfg_train.callbacks = None

    HydraConfig().set_config(cfg_train)
    _, objects = train(cfg_train)

    assert objects["evaluator"] is None


def test_evaluation_initializes_data_dependent_pairing_state() -> None:
    """Checkpoint-only JetCLR evaluation must reconstruct its augmenters."""
    algorithm = Mock()
    datamodule = Mock()

    _setup_algorithm_for_evaluation(algorithm, datamodule)

    algorithm.setup_pairing.assert_called_once_with(datamodule, setup_lorentz=True)


def test_evaluation_setup_is_optional_for_other_algorithms() -> None:
    """Algorithms without data-dependent setup remain valid for evaluation."""
    algorithm = object()

    _setup_algorithm_for_evaluation(algorithm, Mock())


def test_evaluation_callback_allowlist_selects_only_exact_named_configs() -> None:
    """An explicit allowlist retains only its exact callback config keys."""
    callbacks = OmegaConf.create(
        {
            "eff_native": {"_target_": "package.Native"},
            "eff_oas": {"_target_": "package.OAS"},
            "cap": {"_target_": "package.CAP"},
        }
    )

    selected = _allowed_evaluation_callback_configs(
        callbacks, OmegaConf.create(["eff_native", "eff_oas"])
    )

    assert list(selected) == ["eff_native", "eff_oas"]
    assert selected.eff_native._target_ == "package.Native"
    assert selected.eff_oas._target_ == "package.OAS"


def test_evaluation_callback_allowlist_fails_closed_on_unknown_name() -> None:
    """A misspelled allowlist entry must not silently reduce evaluation coverage."""
    callbacks = OmegaConf.create({"eff_native": {"_target_": "package.Native"}})

    with pytest.raises(ValueError, match="Unknown or non-instantiable"):
        _allowed_evaluation_callback_configs(callbacks, OmegaConf.create(["missing"]))


def test_skip_validation_evaluation_still_runs_test(monkeypatch, cfg_train: DictConfig) -> None:
    """The opt-in validation skip leaves downstream test evaluation enabled."""
    datamodule = Mock()
    datamodule.test_dataloader.return_value = {"normal": object()}
    algorithm = Mock()
    trainer = Mock(callback_metrics={})
    evaluator = Mock()
    instantiated = iter([datamodule, algorithm, trainer])

    with open_dict(cfg_train):
        cfg_train.train = False
        cfg_train.test = True
        cfg_train.callbacks = None
        cfg_train.evaluation = OmegaConf.create({"evaluator": {"_target_": "unused"}})
        cfg_train.skip_validation_evaluation = True

    monkeypatch.setattr(
        "src.train.hydra.utils.instantiate", lambda *args, **kwargs: next(instantiated)
    )
    monkeypatch.setattr("src.train._get_evaluator", lambda *args, **kwargs: evaluator)

    _, objects = train(cfg_train)

    datamodule.setup.assert_called_once_with("test")
    datamodule.val_dataloader.assert_not_called()
    evaluator.evaluate_run.assert_called_once()
    assert evaluator.evaluate_run.call_args.args[-1] == "test"
    assert objects["evaluator"] is evaluator


def test_metric_values_use_the_pipeline_raw_csv_contract(tmp_path) -> None:
    """Evaluation metric CSV output retains the pipeline's raw schema."""
    path = tmp_path / "values.csv"
    write_metric_values(
        path,
        [
            {
                "checkpoint": "last",
                "intervention": "synthetic_signal",
                "metric": "auprc",
                "value": 0.75,
            }
        ],
    )

    with path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [
            {
                "checkpoint": "last",
                "intervention": "synthetic_signal",
                "metric": "auprc",
                "value": "0.75",
            }
        ]

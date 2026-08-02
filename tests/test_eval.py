import csv
from unittest.mock import Mock

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.evaluation.callbacks.utils.misc import write_metric_values
from src.train import _setup_algorithm_for_evaluation, train


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
    algorithm = object()

    _setup_algorithm_for_evaluation(algorithm, Mock())


def test_metric_values_use_the_pipeline_raw_csv_contract(tmp_path) -> None:
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

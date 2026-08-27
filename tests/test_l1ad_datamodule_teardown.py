from types import SimpleNamespace

from src.data.L1AD_datamodule import L1ADDataModule


def _datamodule_state() -> SimpleNamespace:
    return SimpleNamespace(
        _main={"train": object(), "valid": object(), "test": object()},
        _aux={"valid": {"signal": object()}, "test": {"signal": object()}},
        _train_loader=object(),
        _val_loaders=object(),
        _test_loaders=object(),
    )


def test_validate_teardown_releases_only_validation_tensors() -> None:
    datamodule = _datamodule_state()

    L1ADDataModule.teardown(datamodule, "validate")

    assert set(datamodule._main) == {"train", "test"}
    assert datamodule._aux["valid"] == {}
    assert datamodule._aux["test"]
    assert datamodule._val_loaders is None
    assert datamodule._train_loader is not None
    assert datamodule._test_loaders is not None


def test_fit_teardown_still_releases_train_and_validation_tensors() -> None:
    datamodule = _datamodule_state()

    L1ADDataModule.teardown(datamodule, "fit")

    assert set(datamodule._main) == {"test"}
    assert datamodule._aux["valid"] == {}
    assert datamodule._aux["test"]
    assert datamodule._train_loader is None
    assert datamodule._val_loaders is None

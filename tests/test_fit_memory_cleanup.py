from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.train import _release_fit_dataloaders


def test_release_fit_dataloaders_clears_lightning_references_and_device_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_source = SimpleNamespace(instance=object())
    validation_source = SimpleNamespace(instance=object())
    validation_loop = SimpleNamespace(
        _combined_loader=object(),
        _data_source=validation_source,
    )
    fit_loop = SimpleNamespace(
        _combined_loader=object(),
        _data_source=train_source,
        epoch_loop=SimpleNamespace(val_loop=validation_loop),
    )
    trainer = SimpleNamespace(fit_loop=fit_loop)
    datamodule = Mock()
    collect = Mock()
    cuda_empty_cache = Mock()
    mps_empty_cache = Mock()

    monkeypatch.setattr("src.train.gc.collect", collect)
    monkeypatch.setattr("src.train.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("src.train.torch.cuda.empty_cache", cuda_empty_cache)
    monkeypatch.setattr("src.train.torch.backends.mps.is_available", lambda: True)
    monkeypatch.setattr("src.train.torch.mps.empty_cache", mps_empty_cache)

    _release_fit_dataloaders(trainer, datamodule)

    assert fit_loop._combined_loader is None
    assert train_source.instance is None
    assert validation_loop._combined_loader is None
    assert validation_source.instance is None
    datamodule.teardown.assert_called_once_with("fit")
    collect.assert_called_once_with()
    cuda_empty_cache.assert_called_once_with()
    mps_empty_cache.assert_called_once_with()

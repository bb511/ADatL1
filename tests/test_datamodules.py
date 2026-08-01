from types import SimpleNamespace
from unittest.mock import Mock

import torch

from src.data.CIFAR10_datamodule import CIFAR10DataModule
from src.data.L1AD_datamodule import L1ADDataModule, SplitTensors
from src.data.synthetic import SyntheticL1ADDataModule


def test_cifar10_datamodule_smoke() -> None:
    """Verify the lightweight demo datamodule can prepare train/validation loaders."""
    dm = CIFAR10DataModule(
        data_dir="data/cifar10",
        normal_classes=[0],
        signal_classes=[1],
        batch_size=32,
        max_val_batches=1,
        val_fraction=0.2,
        reference_fraction=0.5,
        seed=123,
        num_workers=0,
    )

    dm.prepare_data()
    dm.setup("fit")

    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (32, 3, 32, 32)
    assert y.shape == (32,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64

    val_loaders = dm.val_dataloader()
    assert list(val_loaders) == ["normal", "1", "reference_normal"]


def test_l1_validation_setup_resolves_cache_without_preparing_data(tmp_path, monkeypatch) -> None:
    """Evaluation-only runs must load an existing deterministic mlready cache."""
    mlready = SimpleNamespace(
        cache_root_dir=str(tmp_path),
        name="aad_default",
        prepare=Mock(side_effect=AssertionError("preprocessing must not run")),
    )
    expected = tmp_path / "mlready" / "aad_default" / "standard"
    expected.mkdir(parents=True)
    (expected / "FET_norm_params.pkl").touch()
    normalizer = Mock(name="normalizer")
    normalizer.name = "standard"
    normalizer.norm_params = {}

    def import_params(_path, object_name):
        normalizer.norm_params[object_name] = {"Et": {"shift": 0.0, "scale": 1.0}}

    normalizer.import_norm_params.side_effect = import_params
    loader = SimpleNamespace(object_feature_map={"FET": {"Et": [0]}})
    datamodule = L1ADDataModule(
        zerobias={},
        signal={},
        background={},
        data_extractor=Mock(),
        data_processor=Mock(),
        data_normalizer=normalizer,
        data_mlready=mlready,
        data_awkward2torch=loader,
        train_features={},
        l1_scales={},
        batch_size=8,
    )
    split = SplitTensors(
        x=torch.zeros(2, 3),
        mask=torch.ones(2, 3, dtype=torch.bool),
        l1bit=torch.ones(2, dtype=torch.bool),
        y=torch.zeros(2),
    )
    load_main = Mock(return_value=split)
    load_aux = Mock(return_value={"signal": split})
    monkeypatch.setattr(datamodule, "_load_main_split", load_main)
    monkeypatch.setattr(datamodule, "_load_aux_split", load_aux)

    datamodule.setup("validate")

    assert datamodule.main_cache_folder == expected
    load_main.assert_called_once_with(expected, "valid", label=0)
    load_aux.assert_called_once_with(expected, "valid")
    normalizer.import_norm_params.assert_called_once_with(expected / "FET_norm_params.pkl", "FET")
    mlready.prepare.assert_not_called()


def test_gaussian_subspace_synthetic_l1_datamodule() -> None:
    """Verify the controlled Gaussian-subspace generator exposes L1 loaders."""
    dm = SyntheticL1ADDataModule(
        n_features=4,
        n_train=64,
        n_val=512,
        n_test=512,
        batch_size=128,
        seed=123,
        generator="gaussian_subspace",
        reference_shift=1.5,
        reference_shift_dim=1,
        anomaly_shift=4.0,
        anomaly_dim=0,
    )

    dm.setup("test")
    loaders = dm.test_dataloader()
    assert list(loaders) == ["normal", "reference_normal", "synthetic_signal"]

    x_normal, mask, l1bit, y_normal = next(iter(loaders["normal"]))
    x_reference, _, _, y_reference = next(iter(loaders["reference_normal"]))
    x_signal, _, _, y_signal = next(iter(loaders["synthetic_signal"]))

    assert x_normal.shape == (128, 4)
    assert mask.dtype == torch.bool
    assert l1bit.dtype == torch.bool
    assert torch.all(y_normal == 0)
    assert torch.all(y_reference == -1)
    assert torch.all(y_signal == 1)
    assert x_reference[:, 1].mean() > x_normal[:, 1].mean() + 0.75
    assert x_signal[:, 0].mean() > x_normal[:, 0].mean() + 2.5


def test_gaussian_subspace_supports_score_independent_paired_views() -> None:
    dm = SyntheticL1ADDataModule(
        n_features=3,
        n_train=64,
        n_val=4096,
        n_test=4096,
        batch_size=4096,
        seed=123,
        generator="gaussian_subspace",
        paired_reliability=0.8,
    )

    dm.setup("validate")
    loaders = dm.val_dataloader()
    x_normal = next(iter(loaders["normal"]))[0]
    x_reference = next(iter(loaders["reference_normal"]))[0]

    correlations = [
        torch.corrcoef(torch.stack([x_normal[:, idx], x_reference[:, idx]]))[0, 1]
        for idx in range(x_normal.shape[1])
    ]
    torch.testing.assert_close(
        torch.tensor(correlations).mean(),
        torch.tensor(0.8),
        atol=0.04,
        rtol=0.0,
    )

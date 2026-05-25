import torch

from src.data.CIFAR10_datamodule import CIFAR10DataModule
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

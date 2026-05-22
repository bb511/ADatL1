import torch

from src.data.CIFAR10_datamodule import CIFAR10DataModule


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

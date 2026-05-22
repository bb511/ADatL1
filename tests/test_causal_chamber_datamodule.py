import csv
from pathlib import Path

import torch

from src.data.CausalChamber_datamodule import (
    META_COLUMNS,
    READOUT_FEATURES,
    CausalChamberDataModule,
)


def _write_experiment(path: Path, n_rows: int, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(META_COLUMNS) + list(READOUT_FEATURES)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx in range(n_rows):
            meta = [float(idx), "standard", float(idx), offset, 1.0]
            features = [offset + idx + j / 10.0 for j in range(len(READOUT_FEATURES))]
            writer.writerow(meta + features)


def test_causal_chamber_datamodule_splits_and_labels(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "lt_interventions_standard_v1"
    _write_experiment(dataset_dir / "uniform_reference.csv", n_rows=40, offset=0.0)
    _write_experiment(dataset_dir / "uniform_red_mid.csv", n_rows=10, offset=100.0)
    _write_experiment(dataset_dir / "uniform_green_mid.csv", n_rows=10, offset=200.0)

    dm = CausalChamberDataModule(
        data_dir=str(tmp_path),
        signal_experiments=["uniform_red_mid", "uniform_green_mid"],
        batch_size=4,
        max_val_batches=1,
        train_fraction=0.5,
        val_fraction=0.25,
        reference_fraction=0.5,
        signal_val_fraction=0.6,
        seed=123,
    )

    dm.prepare_data()
    dm.setup("fit")

    x, mask, l1bit, y = next(iter(dm.train_dataloader()))
    assert x.shape == (4, len(READOUT_FEATURES))
    assert mask.shape == x.shape
    assert mask.dtype == torch.bool
    assert not l1bit.any()
    assert torch.all(y == 0)
    assert dm.feature_names == list(READOUT_FEATURES)

    val_loaders = dm.val_dataloader()
    assert list(val_loaders) == [
        "normal",
        "reference_normal",
        "uniform_red_mid",
        "uniform_green_mid",
    ]

    _, _, _, y_ref = next(iter(val_loaders["reference_normal"]))
    _, _, _, y_red = next(iter(val_loaders["uniform_red_mid"]))
    assert torch.all(y_ref < 0)
    assert torch.all(y_red > 0)

    dm.setup("test")
    test_loaders = dm.test_dataloader()
    assert list(test_loaders) == [
        "normal",
        "reference_normal",
        "uniform_red_mid",
        "uniform_green_mid",
    ]

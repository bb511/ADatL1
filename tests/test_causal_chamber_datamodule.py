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
    control_features = ["red", "green"]
    header = list(META_COLUMNS) + control_features + list(READOUT_FEATURES)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx in range(n_rows):
            meta = [float(idx), "standard", float(idx), offset, 1.0]
            controls = [offset + idx, offset + n_rows - idx]
            features = [offset + idx + j / 10.0 for j in range(len(READOUT_FEATURES))]
            writer.writerow(meta + controls + features)


def test_train_seed_is_independent_from_split_seed(tmp_path: Path) -> None:
    common = {"data_dir": str(tmp_path), "seed": 314159}
    first = CausalChamberDataModule(**common, train_seed=1001)
    repeat = CausalChamberDataModule(**common, train_seed=1001)
    second = CausalChamberDataModule(**common, train_seed=1002)
    fallback = CausalChamberDataModule(**common)

    first_order = torch.randperm(64, generator=first.shuffler)
    assert torch.equal(first_order, torch.randperm(64, generator=repeat.shuffler))
    assert not torch.equal(first_order, torch.randperm(64, generator=second.shuffler))
    assert fallback.train_seed == 314159
    assert first.hparams.seed == second.hparams.seed == 314159


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

    batch = next(iter(dm.train_dataloader()))
    x = batch["x"]
    mask = batch["mask"]
    l1bit = batch["l1bit"]
    y = batch["y"]
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

    normal = next(iter(val_loaders["normal"]))
    reference = next(iter(val_loaders["reference_normal"]))
    red = next(iter(val_loaders["uniform_red_mid"]))
    y_ref = reference["y"]
    y_red = red["y"]
    assert torch.all(y_ref < 0)
    assert torch.all(y_red > 0)
    assert torch.equal(normal["pair_id"], reference["pair_id"])
    assert dm.contract["pairing"]["cap_pairing_type"] == "none"
    assert dm.contract["model_features"] == list(READOUT_FEATURES)
    assert dm.contract["pairing_features"] == ["red", "green"]
    assert "red" in dm.contract["excluded_columns"]
    assert not torch.equal(normal["sample_id"], reference["sample_id"])

    dm.setup("test")
    test_loaders = dm.test_dataloader()
    assert list(test_loaders) == [
        "normal",
        "reference_normal",
        "uniform_red_mid",
        "uniform_green_mid",
    ]

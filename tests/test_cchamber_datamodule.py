import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.algorithms import ADLightningModule
from src.algorithms.utils.object_feature_map_loader import maybe_get_object_feature_map
from src.data.CausalChamber_datamodule import CausalChamberDataModule

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DATASET = "lt_interventions_standard_v1"


def _compose(data: str) -> DictConfig:
    overrides = [f"data={data}", "paths.raw_data_dir=/x"]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        return compose("train", overrides=overrides).data


@pytest.fixture(scope="module")
def lt_readouts() -> list[str]:
    return list(_compose("cchamber").readouts)


def _datamodule(
    data: SimpleNamespace, readouts: list[str], **overrides
) -> CausalChamberDataModule:
    kwargs = dict(
        data_dir=str(data.data_dir),
        dataset_name=DATASET,
        url="https://example.invalid/archive.zip",
        md5="0" * 32,
        readouts=readouts,
        reference={"file": "uniform_reference"},
        signals={name: {"file": name} for name in data.signals},
        batch_size=8,
    )
    return CausalChamberDataModule(**(kwargs | overrides))


@pytest.fixture
def datamodule(
    cchamber_data: SimpleNamespace, lt_readouts: list[str]
) -> CausalChamberDataModule:
    dm = _datamodule(cchamber_data, lt_readouts)
    dm.setup("fit")
    return dm


def _batches(loader) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*loader)
    return torch.cat(xs), torch.cat(ys)


def _raw(dm: CausalChamberDataModule, name: str) -> torch.Tensor:
    with (dm.dataset_dir / f"{name}.csv").open() as f:
        rows = list(csv.DictReader(f))
    readouts = dm.hparams.readouts
    return torch.tensor([[float(row[k]) for k in readouts] for row in rows]).double()


def _raw_indices(dm, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    restored = x.double() * dm.scale.double() + dm.median.double()
    distance, index = torch.cdist(restored, raw).min(dim=1)
    assert distance.max() < 1e-3
    return index


def test_splits_partition_the_rows(datamodule, cchamber_data):
    raw = _raw(datamodule, "uniform_reference")
    parts = [datamodule.data_train.tensors[0]]
    for loaders in (datamodule.val_dataloader(), datamodule.test_dataloader()):
        parts += [_batches(loaders[k])[0] for k in ("normal", "reference_normal")]
    assert [len(x) for x in parts] == [60, 10, 10, 10, 10]
    index = torch.cat([_raw_indices(datamodule, x, raw) for x in parts])
    assert torch.equal(index.sort().values, torch.arange(100))

    for name in cchamber_data.signals:
        x_valid = _batches(datamodule.val_dataloader()[name])[0]
        x_test = _batches(datamodule.test_dataloader()[name])[0]
        assert (len(x_valid), len(x_test)) == (30, 20)
        if name != cchamber_data.clipped:
            x = torch.cat([x_valid, x_test])
            index = _raw_indices(datamodule, x, _raw(datamodule, name))
            assert torch.equal(index.sort().values, torch.arange(50))


def test_loader_order_and_labels(datamodule, cchamber_data):
    expected = {"normal": [0], "reference_normal": [0]}
    expected |= {name: [i] for i, name in enumerate(cchamber_data.signals, start=1)}
    assert not datamodule.data_train.tensors[1].any()
    for loaders in (datamodule.val_dataloader(), datamodule.test_dataloader()):
        labels = {name: _batches(dl)[1] for name, dl in loaders.items()}
        assert list(labels) == list(expected)
        assert {name: y.unique().tolist() for name, y in labels.items()} == expected


def test_normalization_is_fitted_on_the_training_rows_and_clipped(
    datamodule, cchamber_data, lt_readouts
):
    raw = _raw(datamodule, "uniform_reference")
    x_train = datamodule.data_train.tensors[0]
    raw_train = raw[_raw_indices(datamodule, x_train, raw)]
    quantiles = torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)
    q = torch.quantile(raw_train, quantiles, dim=0)

    median, scale = datamodule.median.double(), datamodule.scale.double()
    torch.testing.assert_close(median, q[1], rtol=0, atol=1e-4)
    torch.testing.assert_close(scale, q[2] - q[0], rtol=0, atol=1e-4)
    assert (median - torch.quantile(raw, 0.5, dim=0)).abs().max() > 1e-2

    clipped = _batches(datamodule.val_dataloader()[cchamber_data.clipped])[0]
    assert torch.all(clipped[:, lt_readouts.index("ir_1")] == 10.0)
    for loaders in (datamodule.val_dataloader(), datamodule.test_dataloader()):
        for loader in loaders.values():
            assert _batches(loader)[0].abs().max() <= 10.0


def test_readouts_and_object_feature_map(datamodule, lt_readouts):
    names = "current angle_1 angle_2 ir_1 vis_1 ir_2 vis_2 ir_3 vis_3 v_board v_reg"
    assert lt_readouts == names.split()

    fit = SimpleNamespace(trainer=SimpleNamespace(datamodule=datamodule))
    loaders = datamodule.test_dataloader()
    evaluation = SimpleNamespace(
        trainer=SimpleNamespace(datamodule=None, test_dataloaders=loaders)
    )
    feature_map = maybe_get_object_feature_map(fit)
    expected = {"chamber": {name: [i] for i, name in enumerate(lt_readouts)}}
    assert feature_map == expected
    assert maybe_get_object_feature_map(evaluation) == feature_map

    module = SimpleNamespace(object_feature_map=feature_map)
    x = datamodule.data_train.tensors[0][:5]
    x_by_type, _ = ADLightningModule._split_by_type_from_flat(module, x, None)
    assert list(x_by_type) == ["chamber"]
    assert torch.equal(x_by_type["chamber"], x.unsqueeze(1))


def test_train_seed_changes_the_batch_order_only(cchamber_data, lt_readouts):
    dm_a, dm_b = (_datamodule(cchamber_data, lt_readouts, train_seed=s) for s in (1, 2))
    dm_a.setup("fit")
    dm_b.setup("fit")
    xa, _ = _batches(dm_a.train_dataloader())
    xb, _ = _batches(dm_b.train_dataloader())
    assert torch.equal(dm_a.data_train.tensors[0], dm_b.data_train.tensors[0])
    assert not torch.equal(xa, xb)
    assert torch.equal(xa.sort(dim=0).values, xb.sort(dim=0).values)


WT_DATASET = "wt_walks_v1"


def _regime(block: int) -> dict:
    return {"file": "regime_jumps_single", "where": {"flag": block}}


def _wt_datamodule(data: SimpleNamespace, **overrides) -> CausalChamberDataModule:
    kwargs = dict(
        data_dir=str(data.data_dir),
        dataset_name=WT_DATASET,
        url="https://example.invalid/archive.zip",
        md5="0" * 32,
        readouts=data.readouts,
        reference=_regime(0),
        signals={name: _regime(b) for name, b in data.signal_blocks.items()},
        nulls={name: _regime(b) for name, b in data.null_blocks.items()},
        batch_size=8,
    )
    return CausalChamberDataModule(**(kwargs | overrides))


@pytest.fixture
def wt_datamodule(cchamber_wt_data: SimpleNamespace) -> CausalChamberDataModule:
    dm = _wt_datamodule(cchamber_wt_data)
    dm.setup("fit")
    return dm


def test_blocks_are_partitioned_and_labeled(wt_datamodule, cchamber_wt_data):
    dm, data = wt_datamodule, cchamber_wt_data
    expected = {"normal": [0], "reference_normal": [0]}
    expected |= {name: [i] for i, name in enumerate(data.signal_blocks, start=1)}
    expected |= {name: [0] for name in data.null_blocks}
    parts = {0: [dm.data_train.tensors[0]]}
    for loaders in (dm.val_dataloader(), dm.test_dataloader()):
        labels = {name: _batches(dl)[1] for name, dl in loaders.items()}
        assert list(labels) == list(expected)
        assert {name: y.unique().tolist() for name, y in labels.items()} == expected
        parts[0] += [_batches(loaders[k])[0] for k in ("normal", "reference_normal")]
        for name, block in (data.signal_blocks | data.null_blocks).items():
            parts.setdefault(block, []).append(_batches(loaders[name])[0])

    assert sorted(parts) == sorted(data.raw)
    for block, xs in parts.items():
        raw = torch.from_numpy(data.raw[block])
        index = torch.cat([_raw_indices(dm, x, raw) for x in xs])
        assert torch.equal(index.sort().values, torch.arange(len(raw)))


def test_registered_set_counts():
    # Pinned, since a config edit that dropped a set would change the benchmark
    # silently.
    lt, wt = _compose("cchamber"), _compose("cchamber_wt")
    assert len(lt.signals) == 58 and not lt.get("nulls")
    assert len(wt.signals) == 26 and len(wt.nulls) == 5
    sets = [wt.reference, *wt.signals.values(), *wt.nulls.values()]
    assert sorted(s.where.flag for s in sets) == list(range(32))

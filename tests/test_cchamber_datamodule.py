import csv
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.algorithms import ADLightningModule
from src.algorithms.utils.object_feature_map_loader import maybe_get_object_feature_map
from src.data.CausalChamber_datamodule import CausalChamberDataModule
from src.data.components.chamber_columns import read_columns

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
    names = "current angle_1 angle_2 ir_1 vis_1 ir_2 vis_2 ir_3 vis_3 v_reg"
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


def _pool(blocks: list[int]) -> dict:
    return {"members": [_regime(b) for b in blocks]}


def test_a_pool_concatenates_its_members_split_as_plain_sets(cchamber_wt_data):
    members = {"signal_pool": [3, 4], "null_pool": [5, 2]}
    signals = {f"regime_{b}": _regime(b) for b in range(1, 6)}
    plain = _wt_datamodule(cchamber_wt_data, signals=signals, nulls=None)
    pooled = _wt_datamodule(
        cchamber_wt_data,
        signals={"regime_1": _regime(1), "signal_pool": _pool(members["signal_pool"])},
        nulls={"null_pool": _pool(members["null_pool"])},
    )
    plain.setup("fit")
    pooled.setup("fit")

    order = ["normal", "reference_normal", "regime_1", "signal_pool", "null_pool"]
    labels = {"regime_1": 1, "signal_pool": 2, "null_pool": 0}
    for split in ("val_dataloader", "test_dataloader"):
        expected, loaders = getattr(plain, split)(), getattr(pooled, split)()
        assert list(loaders) == order
        for name in ("normal", "reference_normal", "regime_1"):
            assert torch.equal(_batches(loaders[name])[0], _batches(expected[name])[0])
        for name, blocks in members.items():
            x = torch.cat([_batches(expected[f"regime_{b}"])[0] for b in blocks])
            assert torch.equal(_batches(loaders[name])[0], x)
        for name, label in labels.items():
            assert _batches(loaders[name])[1].unique().tolist() == [label]


def _assert_reads_the_served_rows(
    dm: CausalChamberDataModule, names: list[str]
) -> None:
    clip = dm.hparams.clip_value
    loaders = {"valid": dm.val_dataloader(), "test": dm.test_dataloader()}
    for split in loaders:
        for name in names:
            raw = read_columns(dm, name, split, list(dm.hparams.readouts))
            assert raw.dtype == torch.float64
            x = ((raw.float() - dm.median) / dm.scale).clamp(-clip, clip)
            assert torch.equal(x, _batches(loaders[split][name])[0])


def test_read_columns_follows_the_loader_order(cchamber_data, lt_readouts):
    pool = {"members": [{"file": name} for name in cchamber_data.signals[1:]]}
    dm = _datamodule(cchamber_data, lt_readouts, nulls={"pool": pool})
    dm.setup("fit")
    _assert_reads_the_served_rows(
        dm, ["normal", "reference_normal", "uniform_red_mid", "pool"]
    )

    with (dm.dataset_dir / "uniform_red_mid.csv").open() as f:
        rows = list(csv.DictReader(f))
    red = {tuple(float(r[k]) for k in lt_readouts): float(r["red"]) for r in rows}
    x = read_columns(dm, "uniform_red_mid", "test", [*lt_readouts, "red"])
    assert x[:, -1].tolist() == [red[tuple(row)] for row in x[:, :-1].tolist()]


def test_read_columns_filters_as_the_datamodule_filters(cchamber_wt_data):
    dm = _wt_datamodule(cchamber_wt_data, nulls={"null_pool": _pool([5, 2])})
    dm.setup("fit")
    _assert_reads_the_served_rows(
        dm, ["normal", "reference_normal", "regime_03_b", "null_pool"]
    )


def _to_tensor_before(table, readouts: list[str], relative_to) -> torch.Tensor:
    # _to_tensor before relative_to existed.
    columns = [table.column(name).to_numpy() for name in readouts]
    return torch.as_tensor(np.stack(columns, axis=1), dtype=torch.float32)


def _tensors(dm: CausalChamberDataModule) -> list[torch.Tensor]:
    loaders = [*dm.val_dataloader().values(), *dm.test_dataloader().values()]
    return [dm.data_train.tensors[0], *(_batches(dl)[0] for dl in loaders)]


def test_relative_to_off_leaves_every_tensor_unchanged(cchamber_wt_data, monkeypatch):
    current = [_wt_datamodule(cchamber_wt_data, relative_to=r) for r in (None, {})]
    for dm in current:
        dm.setup("fit")
    monkeypatch.setattr(
        "src.data.CausalChamber_datamodule._to_tensor", _to_tensor_before
    )
    before = _wt_datamodule(cchamber_wt_data)
    before.setup("fit")

    for dm in current:
        assert torch.equal(dm.median, before.median)
        assert torch.equal(dm.scale, before.scale)
        for x, x_before in zip(_tensors(dm), _tensors(before), strict=True):
            assert torch.equal(x, x_before)


def test_relative_to_subtracts_the_column_in_float64(cchamber_wt_data):
    ambient = "pressure_ambient"
    readouts = [name for name in cchamber_wt_data.readouts if name != ambient]
    relative_to = {"pressure_upwind": ambient, "pressure_intake": ambient}
    dm = _wt_datamodule(cchamber_wt_data, readouts=readouts, relative_to=relative_to)
    plain = _wt_datamodule(cchamber_wt_data, readouts=readouts)
    assert ambient in dm._read_table("regime_jumps_single").column_names
    assert ambient not in plain._read_table("regime_jumps_single").column_names

    x, x_plain = dm._read_set(_regime(0)), plain._read_set(_regime(0))
    with (dm.dataset_dir / "regime_jumps_single.csv").open() as f:
        rows = [row for row in csv.DictReader(f) if float(row["flag"]) == 0]
    for i, name in enumerate(readouts):
        if name not in relative_to:
            assert torch.equal(x[:, i], x_plain[:, i])
            continue
        values = np.array([[float(row[name]), float(row[ambient])] for row in rows])
        expected = torch.from_numpy(values[:, 0] - values[:, 1]).float()
        assert torch.equal(x[:, i], expected)
        # Subtracting after the cast to float32 gives other values.
        values = torch.from_numpy(values).float()
        assert not torch.equal(x[:, i], values[:, 0] - values[:, 1])

    dm.setup("fit")
    assert dm.data_train.tensors[0].shape[1] == len(readouts)
    assert list(dm.loader.object_feature_map["chamber"]) == readouts
    with pytest.raises(ValueError, match="non-readout"):
        _wt_datamodule(
            cchamber_wt_data, readouts=readouts, relative_to={ambient: "mic"}
        )


# The registered lists, in the order of configs/data/cchamber{,_wt}.yaml.
LT_SIGNALS = [
    f"uniform_{target}_{grade}"
    for target in ("blue", "green", "pol_1", "pol_2", "red")
    for grade in ("mid", "strong")
]
LT_POOLS = ["sensor_weak", "sensor_mid", "sensor_strong"]
LT_LEDS = [f"uniform_l_{led}_mid" for led in (11, 12, 21, 22, 31, 32)]
WT_BAROMETERS = ["pressure_upwind", "pressure_downwind", "pressure_intake"]
CHAMBER_EXPERIMENTS = [
    f"{tunnel}/{model}"
    for tunnel in ("cchamber", "cchamber_wt")
    for model in ("ae", "realnvp", "svdd", "vae")
]


def _plain_sets(sets: dict) -> list[dict]:
    return [m for spec in sets.values() for m in spec.get("members") or [spec]]


def test_registered_set_counts():
    # Pinned, since a config edit that dropped a set would change the benchmark
    # silently.
    lt, wt = (OmegaConf.to_container(_compose(d)) for d in ("cchamber", "cchamber_wt"))
    lt_nulls = {name: len(_plain_sets({name: s})) for name, s in lt["nulls"].items()}
    assert len(lt["readouts"]) == 10 and list(lt["signals"]) == LT_SIGNALS
    assert lt_nulls == {name: 9 for name in LT_POOLS} | {name: 1 for name in LT_LEDS}
    files = [s["file"] for s in _plain_sets(lt["signals"] | lt["nulls"])]
    assert len(set(files)) == 10 + 27 + 6

    assert len(wt["readouts"]) == 10 and "pressure_ambient" not in wt["readouts"]
    assert wt["relative_to"] == {name: "pressure_ambient" for name in WT_BAROMETERS}
    wt_pools = {name: len(pool["members"]) for name, pool in wt["nulls"].items()}
    assert len(wt["signals"]) == 7 and wt_pools == {"sensor_osr": 10, "walk_null": 5}
    sets = [wt["reference"], *_plain_sets(wt["signals"] | wt["nulls"])]
    left_out = {3, 6, 11, 12, 14, 21, 22, 25, 31}
    flags = [flag for flag in range(32) if flag not in left_out]
    assert sorted(s["where"]["flag"] for s in sets) == flags


def test_registered_lt_sets_are_labeled_in_order(cchamber_data):
    lt = OmegaConf.to_container(_compose("cchamber"))
    dataset_dir = cchamber_data.data_dir / DATASET
    for spec in _plain_sets(lt["signals"] | lt["nulls"]):
        path = dataset_dir / f"{spec['file']}.csv"
        if not path.exists():
            shutil.copy(dataset_dir / "uniform_red_mid.csv", path)
    dm = _datamodule(
        cchamber_data, lt["readouts"], signals=lt["signals"], nulls=lt["nulls"]
    )
    dm.setup("fit")

    assert dm.data_train.tensors[0].shape[1] == 10
    expected = {"normal": [0], "reference_normal": [0]}
    expected |= {name: [i] for i, name in enumerate(LT_SIGNALS, start=1)}
    expected |= {name: [0] for name in LT_POOLS + LT_LEDS}
    for loaders in (dm.val_dataloader(), dm.test_dataloader()):
        labels = {name: _batches(dl)[1] for name, dl in loaders.items()}
        assert list(labels) == list(expected)
        assert {name: y.unique().tolist() for name, y in labels.items()} == expected


def _experiment(name: str) -> DictConfig:
    overrides = [f"experiment={name}", "paths.raw_data_dir=/x", "run_name=dbg"]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        return compose("train", overrides=overrides)


@pytest.mark.parametrize("experiment", CHAMBER_EXPERIMENTS)
def test_the_models_take_one_input_per_readout(experiment):
    cfg = _experiment(experiment)
    algorithm = cfg.algorithm
    dim = algorithm.flow.input_dim if "flow" in algorithm else algorithm.encoder.in_dim
    assert dim == len(cfg.data.readouts) == 10


@pytest.mark.parametrize("tunnel", ["cchamber", "cchamber_wt"])
def test_the_vae_decodes_the_posterior_mean_in_eval_mode(tunnel):
    assert _experiment(f"{tunnel}/vae").algorithm.eval_mean is True

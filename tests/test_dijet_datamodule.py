from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.Dijet_datamodule import DijetDataModule

COLUMNS = (
    "pT_j1 eta_j1 phi_j1 pT_j2 eta_j2 phi_j2 m_jj tau21_j1 tau21_j2 tau32_j1 "
    "tau32_j2 met phi_met min_dPhi ht"
).split()
INPUTS = ["ht", "met", "m_jj", "tau21_j1", "tau21_j2", "tau32_j1", "tau32_j2"]
LOGGED = ["ht", "met", "m_jj"]
# File: (rows, rows with a -nan ratio, rows with met = 0).
FILES = {
    "qcd_data_samples/qcd_0.txt": (60, 2, 1),
    "qcd_data_samples/qcd_1.txt": (60, 1, 0),
    "qcd_test_samples/qcd_0.txt": (50, 1, 0),
    "qcd_mc_samples/qcd_0.txt": (50, 0, 1),
    "sig_samples/sig_a.txt": (25, 1, 0),
    "sig_samples/sig_b.txt": (25, 0, 0),
}


def _write(path: Path, rows: int, nans: int, zeros: int, seed: int) -> None:
    values = np.random.default_rng(seed).uniform(1.0, 2.0, size=(rows, 15))
    values[:zeros, COLUMNS.index("met")] = 0.0
    text = [[f"{v:.6f}" for v in row] for row in values]
    for row in text[zeros : zeros + nans]:
        row[COLUMNS.index("tau21_j1")] = "-nan"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Every line ends with a space, as in the record.
    lines = [" ".join(COLUMNS) + " "] + [" ".join(row) + " " for row in text]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data")
    for seed, (name, counts) in enumerate(FILES.items()):
        _write(root / "dijet" / name, *counts, seed=seed)
    return root


def _raw(data_dir: Path, name: str) -> torch.Tensor:
    table = np.loadtxt(data_dir / "dijet" / name, skiprows=1, dtype=str)
    x = table[:, [COLUMNS.index(c) for c in INPUTS]].astype(np.float64)
    with np.errstate(divide="ignore"):
        x[:, :3] = np.log(x[:, :3])
    return torch.as_tensor(x[np.isfinite(x).all(axis=1)])


def _datamodule(data_dir: Path, **overrides) -> DijetDataModule:
    kwargs = dict(
        data_dir=str(data_dir),
        dataset_name="dijet",
        url="https://example.invalid/{name}",
        archives={},
        inputs=INPUTS,
        log_inputs=LOGGED,
        reference={"files": list(FILES)[:2], "n_train": 60, "n_valid": 20},
        test_reference={"files": [list(FILES)[2]], "n_test": 20},
        signals={
            "sig_a": {"files": ["sig_samples/sig_a.txt"]},
            "sig_b": {"files": ["sig_samples/sig_b.txt"]},
        },
        nulls={
            "simulation": {
                "files": [list(FILES)[3]],
                "n_valid": 15,
                "n_test": 15,
            }
        },
        batch_size=8,
    )
    dm = DijetDataModule(**(kwargs | overrides))
    dm.setup("fit")
    return dm


@pytest.fixture(scope="module")
def datamodule(data_dir) -> DijetDataModule:
    return _datamodule(data_dir)


def _batches(loader) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*loader)
    return torch.cat(xs), torch.cat(ys)


def _sets(dm: DijetDataModule, split: str) -> dict[str, torch.Tensor]:
    loaders = dm.val_dataloader() if split == "valid" else dm.test_dataloader()
    return {name: _batches(loader)[0] for name, loader in loaders.items()}


def _raw_indices(dm, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    restored = x.double() * dm.spread.double() + dm.median.double()
    distance, index = torch.cdist(restored, raw).min(dim=1)
    assert distance.max() < 1e-4
    return index


def test_sizes_and_counts(datamodule):
    valid, test = _sets(datamodule, "valid"), _sets(datamodule, "test")
    sizes = {"normal": 20, "reference_normal": 20, "sig_a": 14, "sig_b": 15}
    assert {k: len(x) for k, x in valid.items()} == sizes | {"simulation": 15}
    sizes |= {"sig_a": 10, "sig_b": 10, "simulation": 15}
    assert {k: len(x) for k, x in test.items()} == sizes
    assert len(datamodule.data_train) == 60
    assert datamodule.counts["reference"] == {"read": 120, "dropped": 4, "unused": 16}
    assert datamodule.counts["test_reference"] == {
        "read": 50,
        "dropped": 1,
        "unused": 9,
    }
    assert datamodule.counts["simulation"] == {"read": 50, "dropped": 1, "unused": 19}
    assert datamodule.counts["sig_a"] == {"read": 25, "dropped": 1, "unused": 0}


def test_sets_are_disjoint_draws_of_their_sources(datamodule, data_dir):
    valid, test = _sets(datamodule, "valid"), _sets(datamodule, "test")
    names = list(FILES)
    reference = torch.cat([_raw(data_dir, n) for n in names[:2]])
    parts = [datamodule.data_train.tensors[0], valid["normal"]]
    parts.append(valid["reference_normal"])
    index = torch.cat([_raw_indices(datamodule, x, reference) for x in parts])
    assert index.unique().numel() == 100
    sources = {
        names[2]: [test["normal"], test["reference_normal"]],
        names[3]: [valid["simulation"], test["simulation"]],
        names[4]: [valid["sig_a"], test["sig_a"]],
    }
    for name, parts in sources.items():
        raw = _raw(data_dir, name)
        x = torch.cat(parts)
        assert _raw_indices(datamodule, x, raw).unique().numel() == len(x)


def test_log_and_non_finite_drop(datamodule, data_dir):
    rows = {n: _raw(data_dir, n) for n in FILES}
    every = torch.cat(
        [datamodule.data_train.tensors[0], *_sets(datamodule, "valid").values()]
    )
    every = every.double() * datamodule.spread.double() + datamodule.median.double()
    assert torch.isfinite(every).all()
    # Inputs drawn in [1, 2): the logged columns lie in [0, log 2).
    assert every[:, :3].min() >= 0 and every[:, :3].max() < np.log(2.0) + 1e-6
    assert every[:, 3:].min() >= 1 - 1e-6
    assert sum(len(x) for x in rows.values()) == 270 - 7


def test_normalization_from_training_rows(datamodule, data_dir):
    reference = torch.cat([_raw(data_dir, n) for n in list(FILES)[:2]])
    x_train = datamodule.data_train.tensors[0]
    raw_train = reference[_raw_indices(datamodule, x_train, reference)]
    q = torch.quantile(raw_train, torch.tensor([0.05, 0.5, 0.95]).double(), dim=0)
    median, spread = datamodule.median.double(), datamodule.spread.double()
    torch.testing.assert_close(median, q[1], rtol=0, atol=1e-5)
    torch.testing.assert_close(spread, q[2] - q[0], rtol=0, atol=1e-5)
    assert (median - torch.quantile(reference, 0.5, dim=0)).abs().max() > 1e-4


def test_loader_order_and_labels(datamodule):
    expected = {"normal": [0], "reference_normal": [0], "sig_a": [1], "sig_b": [2]}
    expected["simulation"] = [0]
    assert not datamodule.data_train.tensors[1].any()
    for loaders in (datamodule.val_dataloader(), datamodule.test_dataloader()):
        labels = {name: _batches(dl)[1] for name, dl in loaders.items()}
        assert list(labels) == list(expected)
        assert {name: y.unique().tolist() for name, y in labels.items()} == expected


def test_splits_fixed_by_seed(datamodule, data_dir):
    again = _datamodule(data_dir, train_seed=7)
    other = _datamodule(data_dir, seed=1)
    for split in ("valid", "test"):
        for name, x in _sets(datamodule, split).items():
            assert torch.equal(x, _sets(again, split)[name])
    assert torch.equal(datamodule.data_train.tensors[0], again.data_train.tensors[0])
    assert not torch.equal(
        datamodule.data_train.tensors[0], other.data_train.tensors[0]
    )


def test_batch_order_follows_train_seed(data_dir):
    first = [_datamodule(data_dir, train_seed=s) for s in (1, 1, 2)]
    batches = [next(iter(dm.train_dataloader()))[0] for dm in first]
    assert torch.equal(batches[0], batches[1])
    assert not torch.equal(batches[0], batches[2])

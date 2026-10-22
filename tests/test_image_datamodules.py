from pathlib import Path

import pytest
import torch

from src.data.CIFAR10_datamodule import CIFAR10DataModule, CIFARSplit
from src.data.RobustAD_datamodule import RobustADDataModule

PCB_SPLITS = ["train", *RobustADDataModule.SHIFT_MAP["pcb"]]
SHIFTED_SIZES = {"normal": (6, 5), "anomaly": (3, 3)}


def _rows(x: torch.Tensor) -> set[bytes]:
    return {row.numpy().tobytes() for row in x}


def _raw(data_dir: Path, split: str) -> dict:
    return torch.load(data_dir / "PCB" / "tensor_cache" / f"pcb_{split}_4x4_raw.pt")


@pytest.fixture
def robustad_dir(tmp_path: Path) -> Path:
    # The raw caches stand in for the image folders, which setup() reads only when no
    # raw cache exists. Each target domain holds 11 normals and 7 anomalies, the last
    # two of them one image twice, so 6 distinct anomalies.
    cache = tmp_path / "PCB" / "tensor_cache"
    cache.mkdir(parents=True)
    gen = torch.Generator().manual_seed(0)
    for split in PCB_SPLITS:
        n_normal, n_anomaly = (40, 15) if split == "train" else (11, 7)
        y = torch.cat([torch.zeros(n_normal), torch.ones(n_anomaly)]).long()
        x = torch.rand(y.numel(), 3, 4, 4, generator=gen)
        x[-1] = x[-2]
        torch.save({"x": x, "y": y}, cache / f"pcb_{split}_4x4_raw.pt")
        # Files under the names of earlier builds, which must never be read.
        stale = {"x": torch.zeros_like(x), "y": y}
        torch.save(stale, cache / f"pcb_{split}_4x4_norm.pt")
    stale_stats = {"mean": torch.full((3,), 9.0), "std": torch.ones(3)}
    torch.save(stale_stats, tmp_path / "PCB" / "pcb_normal_stats.pt")
    return tmp_path


def _robustad(data_dir: Path, **overrides) -> RobustADDataModule:
    kwargs = dict(data_dir=str(data_dir), subset="pcb", image_size=[4, 4], batch_size=4)
    dm = RobustADDataModule(**(kwargs | overrides))
    dm.setup()
    return dm


def test_robustad_shifted_halves_are_disjoint(robustad_dir):
    dm = _robustad(robustad_dir)
    valid, test = dm._aux["valid"], dm._aux["test"]
    assert list(valid) == list(test)
    for name in valid:
        assert not _rows(valid[name].x) & _rows(test[name].x)
    for i, split in enumerate(PCB_SPLITS[1:]):
        raw = _raw(robustad_dir, split)
        x = (raw["x"] - dm.mean[:, None, None]) / dm.std[:, None, None]
        for label, kind in enumerate(SHIFTED_SIZES):
            v, t = valid[f"shifted_{kind}_{i}"].x, test[f"shifted_{kind}_{i}"].x
            assert (len(v), len(t)) == SHIFTED_SIZES[kind]
            assert _rows(v) | _rows(t) == _rows(x[raw["y"] == label])


def _assert_stats_of_training_part(dm, normal: torch.Tensor) -> None:
    # normal is (n, 3, h, w) in [0, 1]; its rows that land in the training part, once
    # scaled as the datamodule scales them, must be the only ones the stats cover.
    scaled = (normal - dm.mean[:, None, None]) / dm.std[:, None, None]
    train = _rows(dm._main["train"].x)
    in_train = torch.tensor([row.numpy().tobytes() in train for row in scaled])
    assert int(in_train.sum()) == len(dm._main["train"].x) < len(normal)
    assert torch.allclose(dm.mean, normal[in_train].mean(dim=(0, 2, 3)))
    assert torch.allclose(dm.std, normal[in_train].std(dim=(0, 2, 3)))
    assert not torch.allclose(dm.mean, normal.mean(dim=(0, 2, 3)))


def test_robustad_stats_come_from_the_training_part(robustad_dir):
    dm = _robustad(robustad_dir)
    raw = _raw(robustad_dir, "train")
    _assert_stats_of_training_part(dm, raw["x"][raw["y"] == 0])
    assert (robustad_dir / "PCB" / "pcb_4x4_seed314159_normal_stats.pt").exists()


def test_robustad_split_ignores_train_seed(robustad_dir):
    dm_a, dm_b = (_robustad(robustad_dir, train_seed=s) for s in (12345, 54321))
    for key in ("train", "valid", "test"):
        assert torch.equal(dm_a._main[key].x, dm_b._main[key].x)
    for part in ("valid", "test"):
        for name, split in dm_a._aux[part].items():
            assert torch.equal(split.x, dm_b._aux[part][name].x)
    xa, xb = (torch.cat([x for x, _ in dm.train_dataloader()]) for dm in (dm_a, dm_b))
    assert not torch.equal(xa, xb)
    assert _rows(xa) == _rows(xb)


def _cifar_split(train: bool) -> CIFARSplit:
    n_per_class = 50 if train else 20
    targets = [c for c in range(3) for _ in range(n_per_class)]
    gen = torch.Generator().manual_seed(int(train))
    data = torch.randint(0, 256, (len(targets), 4, 4, 3), generator=gen)
    return CIFARSplit(data.to(torch.uint8).numpy(), targets)


@pytest.fixture
def cifar_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(
        CIFAR10DataModule, "_load_split", lambda self, train: _cifar_split(train)
    )
    stale_stats = {"mean": torch.full((3,), 9.0), "std": torch.ones(3)}
    torch.save(stale_stats, tmp_path / "cifar10_normal_stats.pt")
    return tmp_path


def _cifar(data_dir: Path, **overrides) -> CIFAR10DataModule:
    kwargs = dict(
        data_dir=str(data_dir),
        normal_classes=[0],
        signal_classes=[1, 2],
        batch_size=4,
        val_fraction=0.2,
    )
    dm = CIFAR10DataModule(**(kwargs | overrides))
    dm.setup()
    return dm


def test_cifar_split_ignores_train_seed(cifar_dir):
    dm_a, dm_b = (_cifar(cifar_dir, train_seed=s) for s in (12345, 54321))
    for key in ("train", "valid", "test"):
        assert torch.equal(dm_a._main[key].x, dm_b._main[key].x)
    for part in ("valid", "test"):
        for name, split in dm_a._aux[part].items():
            assert torch.equal(split.x, dm_b._aux[part][name].x)
    xa, xb = (torch.cat([x for x, _ in dm.train_dataloader()]) for dm in (dm_a, dm_b))
    assert not torch.equal(xa, xb)
    assert _rows(xa) == _rows(xb)


def test_cifar_stats_come_from_the_training_part(cifar_dir):
    dm = _cifar(cifar_dir)
    images = torch.as_tensor(_cifar_split(train=True).data[:50], dtype=torch.float32)
    _assert_stats_of_training_part(dm, images.permute(0, 3, 1, 2) / 255.0)
    assert (cifar_dir / "cifar10_normal0_seed314159_stats.pt").exists()

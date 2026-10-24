from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.CIFAR10_datamodule import CIFAR10DataModule, CIFARSplit, _member_rng
from src.data.components.corruptions import corrupt


def _rows(x: torch.Tensor) -> set[bytes]:
    return {row.numpy().tobytes() for row in x}


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


MEMBERS = [
    "gaussian_noise",
    "shot_noise",
    "gaussian_blur",
    "brightness",
    "contrast",
    "jpeg_compression",
]
# The second pool lists its members in reverse, so that block order follows the config.
SHIFTS = {
    "sensor_weak": {"severity": 1, "members": MEMBERS},
    "sensor_strong": {"severity": 5, "members": MEMBERS[::-1]},
}
PARTS = {"valid": True, "test": False}


def _normalized(dm, images: np.ndarray) -> torch.Tensor:
    x = torch.as_tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return (x - dm.mean[None, :, None, None]) / dm.std[None, :, None, None]


def _sources(dm, x: torch.Tensor, kind: str | None = None, severity: int = 1):
    # (split, index) of the class-0 image of either split whose image, corrupted by the
    # deterministic kind if one is given, each row of x is.
    index = {}
    for train in PARTS.values():
        split = _cifar_split(train)
        images = split.data[np.asarray(split.targets) == 0]
        if kind is not None:
            images = corrupt(images, kind, severity, np.random.default_rng(0))
        rows = _normalized(dm, images)
        index |= {row.numpy().tobytes(): (train, i) for i, row in enumerate(rows)}
    return [index[row.numpy().tobytes()] for row in x]


def _raw_normal(dm, part: str) -> np.ndarray:
    split = _cifar_split(PARTS[part])
    images = split.data[np.asarray(split.targets) == 0]
    return images[[i for _, i in _sources(dm, dm._main[part].x)]]


def _loader_sets(loaders: dict) -> dict[str, torch.Tensor]:
    return {name: torch.cat([x for x, _ in loader]) for name, loader in loaders.items()}


def _eval_sets(dm) -> list[dict[str, torch.Tensor]]:
    return [_loader_sets(dm.val_dataloader()), _loader_sets(dm.test_dataloader())]


def test_cifar_pools_come_after_every_set_with_label_zero(cifar_dir):
    dm = _cifar(cifar_dir, shifts=SHIFTS)
    for part, loaders in zip(PARTS, (dm.val_dataloader(), dm.test_dataloader())):
        names = ["normal", "1", "2", "reference_normal", *SHIFTS]
        assert list(loaders) == names
        n_normal = len(dm._main[part].x)
        for pool in SHIFTS:
            assert len(dm._aux[part][pool].x) == len(MEMBERS) * n_normal
            assert set(torch.cat([y for _, y in loaders[pool]]).tolist()) == {0}


def test_cifar_pool_blocks_corrupt_normal_row_by_row_in_member_order(cifar_dir):
    dm = _cifar(cifar_dir, shifts=SHIFTS)
    for part in PARTS:
        raw = _raw_normal(dm, part)
        for pool, spec in SHIFTS.items():
            blocks = dm._aux[part][pool].x.split(len(raw))
            for member, block in zip(spec["members"], blocks, strict=True):
                rng = _member_rng(dm.hparams.seed, pool, member, part)
                expected = corrupt(raw, member, spec["severity"], rng)
                assert torch.equal(block, _normalized(dm, expected))


def test_cifar_pools_corrupt_their_own_split_normal_only(cifar_dir):
    dm = _cifar(cifar_dir, shifts=SHIFTS)
    for part in PARTS:
        normal = set(_sources(dm, dm._main[part].x))
        reference = set(_sources(dm, dm._aux[part]["reference_normal"].x))
        assert normal and not normal & reference
        for pool, spec in SHIFTS.items():
            blocks = dm._aux[part][pool].x.split(len(normal))
            for member in ("gaussian_blur", "contrast", "jpeg_compression"):
                block = blocks[spec["members"].index(member)]
                assert set(_sources(dm, block, member, spec["severity"])) == normal


def test_cifar_warm_pools_read_the_cold_caches(cifar_dir, monkeypatch):
    cold = _cifar(cifar_dir, shifts=SHIFTS)
    caches = list(cifar_dir.glob("cifar10_normal0_seed314159_shift314159_*.pt"))
    assert len(caches) == len(PARTS) * sum(len(s["members"]) for s in SHIFTS.values())

    def no_corruption(*args):
        raise AssertionError("a warm build corrupts nothing")

    monkeypatch.setattr("src.data.CIFAR10_datamodule.corrupt", no_corruption)
    warm = _cifar(cifar_dir, shifts=SHIFTS)
    for part in PARTS:
        for pool in SHIFTS:
            assert torch.equal(cold._aux[part][pool].x, warm._aux[part][pool].x)


def test_cifar_pool_cache_under_another_key_is_rebuilt(cifar_dir):
    dm = _cifar(cifar_dir, shifts=SHIFTS)
    name = "cifar10_normal0_seed314159_shift314159_valid_sensor_weak_contrast_severity1"
    path = cifar_dir / f"{name}.pt"
    stale = torch.load(path)
    stale["images"] = torch.zeros_like(stale["images"])
    stale["key"]["source"] = "the normal images of another val_fraction"
    torch.save(stale, path)
    rebuilt = _cifar(cifar_dir, shifts=SHIFTS)
    pool = rebuilt._aux["valid"]["sensor_weak"].x
    assert torch.equal(pool, dm._aux["valid"]["sensor_weak"].x)
    assert torch.load(path)["key"]["source"] != stale["key"]["source"]


def test_cifar_pools_leave_every_other_set_and_the_stats_unchanged(cifar_dir):
    builds = {
        name: _cifar(cifar_dir / name, shifts=shifts)
        for name, shifts in (("null", None), ("empty", {}), ("pools", SHIFTS))
    }
    stats = {
        name: torch.load(cifar_dir / name / "cifar10_normal0_seed314159_stats.pt")
        for name in builds
    }
    # One pass each: a second pass would draw the next batch order.
    trains = {
        name: torch.cat([x for x, _ in dm.train_dataloader()])
        for name, dm in builds.items()
    }
    plain_sets = _eval_sets(builds["null"])
    for name, dm in builds.items():
        for key in ("mean", "std"):
            assert torch.equal(stats["null"][key], stats[name][key])
        extra = list(SHIFTS) if name == "pools" else []
        for plain, part in zip(plain_sets, _eval_sets(dm)):
            assert list(part) == [*plain, *extra]
            assert all(torch.equal(x, part[set_]) for set_, x in plain.items())
        assert torch.equal(trains[name], trains["null"])
    assert not list((cifar_dir / "null").glob("*_shift*"))


MEMBER_SETS = [f"{pool}_{m}" for pool, spec in SHIFTS.items() for m in spec["members"]]


def test_cifar_member_sets_are_the_pool_blocks_with_label_minus_one(cifar_dir):
    dm = _cifar(cifar_dir, shifts=SHIFTS, member_sets=True)
    for part, loaders in zip(PARTS, (dm.val_dataloader(), dm.test_dataloader())):
        names = ["normal", "1", "2", "reference_normal", *SHIFTS, *MEMBER_SETS]
        assert list(loaders) == names
        n_normal = len(dm._main[part].x)
        for pool, spec in SHIFTS.items():
            blocks = dm._aux[part][pool].x.split(n_normal)
            for member, block in zip(spec["members"], blocks, strict=True):
                member_set = dm._aux[part][f"{pool}_{member}"]
                assert torch.equal(member_set.x, block)
                assert member_set.y.tolist() == [-1] * n_normal


def test_cifar_member_sets_off_leave_every_loader_unchanged(cifar_dir):
    builds = {
        "default": _cifar(cifar_dir, shifts=SHIFTS),
        "off": _cifar(cifar_dir, shifts=SHIFTS, member_sets=False),
        "on": _cifar(cifar_dir, shifts=SHIFTS, member_sets=True),
    }
    plain_sets = _eval_sets(builds["default"])
    for name, dm in builds.items():
        extra = MEMBER_SETS if name == "on" else []
        for plain, part in zip(plain_sets, _eval_sets(dm)):
            assert list(part) == [*plain, *extra]
            assert all(torch.equal(x, part[set_]) for set_, x in plain.items())

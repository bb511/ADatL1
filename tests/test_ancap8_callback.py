import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks import ancap8 as a8
from src.callbacks.ancap8 import ANCAP8Callback
from src.data.utils import unpack_batch
from src.evaluation.callbacks.ancap8 import ANCAP8

SIZES = {"normal": 120, "gun": 150}
# Seeds and shifts of the toy sets, whose eighths sum exactly in float32 in any order.
SEEDS = {"normal": 0, "gun": 1, "kind_1": 2, "kind_2": 3}
SHIFTS = {"normal": 0.0, "gun": 0.25, "kind_1": 2.0, "kind_2": 3.0}
DECLARED = {"n1": 120, "n2": 150}


class SumScorer(pl.LightningModule):
    # A weight that a learning rate of 0 keeps at 1: every pass scores the row sums.
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def model_step(self, batch):
        x = unpack_batch(batch).x
        score = x.reshape(len(x), -1).sum(dim=1) * self.weight
        return {"loss": (score**2).mean(), "ascore/full": score.detach()}

    def training_step(self, batch, batch_idx):
        return self.model_step(batch)["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model_step(batch)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model_step(batch)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


def _rows(name: str, n: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(SEEDS[name])
    return torch.randint(-40, 40, (n, 4), generator=g) / 8 + SHIFTS[name]


def _loader(x: torch.Tensor) -> DataLoader:
    return DataLoader(TensorDataset(x, torch.zeros(len(x), dtype=torch.long)), 32)


class SumData(pl.LightningDataModule):
    def train_dataloader(self):
        return _loader(_rows("normal", 64))

    def val_dataloader(self):
        return {name: _loader(_rows(name, n)) for name, n in SIZES.items()}


def _md5(array: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(array).tobytes()).hexdigest()


def _spec(path: Path) -> dict:
    # The entry of extra_sets.points that declares the file as it is.
    with np.load(path) as npz:
        mask = _md5(npz["mask"]) if "mask" in npz.files else None
        return {"path": str(path), "md5_x": _md5(npz["x"]), "md5_mask": mask}


@pytest.fixture
def masked_points(tmp_path) -> tuple[dict, torch.Tensor]:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((25, 4)).astype(np.float32) + 1.0
    mask = rng.random((25, 4)) < 0.7
    np.savez(tmp_path / "masked.npz", x=x, mask=mask)
    # One batch of the file's shape, as the callback scores it.
    scores = torch.where(torch.from_numpy(mask), torch.from_numpy(x), 0).sum(dim=1)
    return _spec(tmp_path / "masked.npz"), scores


def _same(got: float, want: float) -> bool:
    return (math.isnan(got) and math.isnan(want)) or abs(got - want) <= 1e-12


def test_training_callback_logs_every_key(points_files, masked_points, monkeypatch):
    calls, candidate = [], a8.candidate

    def counted(*args, **kwargs):
        calls.append(1)
        return candidate(*args, **kwargs)

    monkeypatch.setattr(a8, "candidate", counted)
    specs = {name: _spec(points_files[name]) for name in ("near", "far")}
    specs["masked"] = masked_points[0]
    callback = ANCAP8Callback(
        "ascore/full", "normal", {"gun": "gun"}, specs, kinds=True, sizes=DECLARED
    )
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        num_sanity_val_steps=-1,
        callbacks=[callback],
    )
    trainer.fit(SumScorer(), datamodule=SumData())
    # Two epochs and one cell; the sanity pass computes nothing.
    assert len(calls) == 2
    sums = {name: _rows(name, n).sum(dim=1) for name, n in SIZES.items()}
    points = [
        torch.from_numpy(np.load(points_files[k])["x"]).sum(dim=1)
        for k in ("near", "far")
    ]
    points.append(masked_points[1])
    views = {"gun": sums["gun"]}
    expected = a8.ancap8(sums["normal"], views, points, True, DECLARED)
    assert len(expected) == len(a8.keys(3))
    for key, want in expected.items():
        got = trainer.callback_metrics[f"val/ancap8/{key}"]
        assert got.dtype == torch.float64 and _same(got.item(), want), key
    assert expected["usable_gun"] == 1 and expected["n_k3_gun"] == 25


def test_training_callback_refuses_unchecked_points(tmp_path, masked_points):
    np.savez(tmp_path / "double.npz", x=np.zeros((3, 4)))
    # The md5s declared are the file's own: the dtype alone is refused.
    double = {"double": _spec(tmp_path / "double.npz")}
    altered = {"masked": masked_points[0] | {"md5_x": _md5(np.zeros(1))}}
    for specs in (double, altered):
        callback = ANCAP8Callback("ascore/full", "normal", {}, specs)
        with pytest.raises(ValueError, match="declared"):
            callback.on_fit_start(None, None)


def _evaluate(callback, loaders: dict) -> dict:
    trainer = pl.Trainer(
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[callback],
    )
    trainer.metric_records = {}
    module = SumScorer()
    module._ckpt_path = "epoch_0003.pt"
    trainer.test(module, dataloaders=loaders, verbose=False)
    return trainer.metric_records


def test_evaluation_mirror_records_every_key():
    sets = {"normal": 120, "gun": 150, "kind_1": 30, "kind_2": 40}
    rows = {name: _rows(name, n) for name, n in sets.items()}
    callback = ANCAP8(
        "ascore/full",
        "normal",
        {"gun": "gun"},
        ["kind_1", "kind_2"],
        kinds=True,
        sizes=DECLARED | {"n_p": 70},
        name="v8",
    )
    records = _evaluate(callback, {name: _loader(x) for name, x in rows.items()})
    sums = {name: x.sum(dim=1) for name, x in rows.items()}
    points = [sums["kind_1"], sums["kind_2"]]
    expected = a8.ancap8(
        sums["normal"], {"gun": sums["gun"]}, points, True, DECLARED | {"n_p": 70}
    )
    assert set(records) == {f"v8/{key}" for key in expected}
    for key, want in expected.items():
        assert _same(records[f"v8/{key}"], want), key
    assert expected["n_k1_gun"] == 30 and expected["L_k2_gun"] > 0


def test_evaluation_mirror_reads_a_misshapen_set_as_absent():
    sets = {"normal": 120, "gun": 150, "kind_1": 30}
    loaders = {name: _loader(_rows(name, n)) for name, n in sets.items()}
    views, sizes = {"gun": "gun"}, {"n2": 99}
    callback = ANCAP8("ascore/full", "normal", views, ["kind_1"], sizes=sizes)
    records = _evaluate(callback, loaders)
    assert records["ancap8/usable_gun"] == 0 and math.isnan(records["ancap8/n2_gun"])
    assert records["ancap8/n1_gun"] == 120 and math.isnan(records["ancap8/I_gun"])


def test_evaluation_mirror_requires_its_loaders():
    loaders = {"normal": _loader(_rows("normal", 120))}
    callback = ANCAP8("ascore/full", "normal", {"gun": "gun"}, ["kind_1"])
    with pytest.raises(ValueError, match="requires test dataloaders"):
        _evaluate(callback, loaders)

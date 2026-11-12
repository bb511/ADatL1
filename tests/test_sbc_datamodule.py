import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.data.SBC_datamodule import SBCDataModule

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
COLUMNS = (
    "Id,Age,Sex,Diagnosis,Center,Set,Sender,Episode,Time,TargetIcu,SecToIcu,CRP,HGB,"
    "MCV,PCT,PLT,RBC,WBC"
).split(",")
# Cases per (Center, Set, Diagnosis); the ids restart per center, so that a case is
# told apart by its center too.
CASES = {
    ("Leipzig", "Training", "Control"): 100,
    ("Leipzig", "Training", "Sepsis"): 10,
    ("Leipzig", "Training", "SIRS"): 6,
    ("Leipzig", "Validation", "Control"): 16,
    ("Leipzig", "Validation", "Sepsis"): 7,
    ("Leipzig", "Validation", "SIRS"): 5,
    ("Greifswald", "Validation", "Control"): 16,
    ("Greifswald", "Validation", "Sepsis"): 8,
    ("Greifswald", "Validation", "SIRS"): 5,
}
SIZES = dict(train_size=50, eval_size=5, null_size=5)
# MCV carries the case serial plus this offset and RBC the row's position in the file,
# so that every served row names its case and its row.
MCV_OFFSET = 50
# Incomplete rows: a missing count, a platelet count of zero, a negative white count.
DEFECTS = ({"HGB": ""}, {"PLT": "0"}, {"WBC": "-2"}, {"MCV": "", "PLT": ""})


def _counts(rng: np.random.Generator, serial: int, defect: dict | None) -> dict:
    counts = {
        "HGB": f"{rng.uniform(5, 11):.1f}",
        "MCV": str(MCV_OFFSET + serial),
        "PLT": str(int(rng.integers(20, 600))),
        "WBC": f"{rng.uniform(1, 30):.2f}",
    }
    return counts | (defect or {})


def _case_rows(rng, key, case_id, serial) -> list[dict]:
    center, set_, diagnosis = key
    rows = []
    for _ in range(int(rng.integers(1, 4))):
        defect = DEFECTS[rng.integers(4)] if rng.uniform() < 0.25 else None
        row = dict(Id=case_id, Age=60, Sex="W", Diagnosis=diagnosis, Center=center)
        row |= dict(Set=set_, Sender="GEN", Episode=1, Time=int(rng.integers(0, 3)))
        row |= dict(TargetIcu="", SecToIcu="", CRP="1.0", PCT="")
        rows.append(row | _counts(rng, serial, defect))
    return rows


def _synthetic_rows(rng: np.random.Generator) -> list[dict]:
    rows, serial = [], 0
    next_id = {"Leipzig": 1, "Greifswald": 1}
    for key, n in CASES.items():
        for _ in range(n):
            rows += _case_rows(rng, key, next_id[key[0]], serial)
            next_id[key[0]] += 1
            serial += 1
    rows = [rows[i] for i in rng.permutation(len(rows))]
    for position, row in enumerate(rows, start=1):
        row["RBC"] = str(position)
    return rows


def _complete(row: dict) -> bool:
    counts = [row[k] for k in ("HGB", "MCV", "PLT", "RBC", "WBC")]
    return "" not in counts and float(row["PLT"]) > 0 and float(row["WBC"]) > 0


def _first_complete(rows: list[dict]) -> dict[int, int]:
    first = {}
    for position, row in sorted(
        enumerate(rows, start=1), key=lambda p: (int(p[1]["Time"]), p[0])
    ):
        serial = int(row["MCV"] or -1) - MCV_OFFSET
        if _complete(row) and serial not in first:
            first[serial] = position
    return first


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def sbc_data(tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("sbc")
    rows = _synthetic_rows(np.random.default_rng(0))
    _write(root / "extdata" / "sbcdata.csv", rows)
    return SimpleNamespace(data_dir=root, rows=rows)


@pytest.fixture(scope="module")
def sbc_config() -> dict:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        data = compose("train", overrides=["data=sbc", "paths.raw_data_dir=/x"]).data
    keys = ("url", "md5", "inputs", "log_inputs", "reference", "signals", "nulls")
    return {k: data[k] for k in keys}


def _datamodule(sbc_data, sbc_config, **overrides) -> SBCDataModule:
    kwargs = sbc_config | SIZES | dict(data_dir=str(sbc_data.data_dir), batch_size=8)
    # No clipping, so that every served row maps back to its raw counts.
    dm = SBCDataModule(**(kwargs | dict(clip_value=1e9) | overrides))
    dm.setup("fit")
    return dm


@pytest.fixture(scope="module")
def datamodule(sbc_data, sbc_config) -> SBCDataModule:
    return _datamodule(sbc_data, sbc_config)


def _restored(dm: SBCDataModule, x: torch.Tensor) -> torch.Tensor:
    return x.double() * dm.spread.double() + dm.median.double()


def _serials(dm, x: torch.Tensor) -> set[int]:
    return set((_restored(dm, x)[:, 1].round().long() - MCV_OFFSET).tolist())


def _served(dm: SBCDataModule) -> dict[str, torch.Tensor]:
    sets = {"train": dm.data_train.tensors[0]}
    for split in ("valid", "test"):
        sets[f"{split}/normal"] = dm._main[split].x
        sets |= {f"{split}/{name}": part.x for name, part in dm._aux[split].items()}
    return sets


def test_split_sizes_and_labels(datamodule):
    assert datamodule.data_train.tensors[0].shape == (50, 5)
    assert datamodule.n_unused == datamodule.case_counts["reference"][1] - 70
    signals = list(datamodule.hparams.signals)
    nulls = ["leipzig_validation", "greifswald"]
    assert list(datamodule.val_dataloader()) == [
        "normal",
        "reference_normal",
        *signals,
        *nulls,
    ]
    for split in ("valid", "test"):
        aux = datamodule._aux[split]
        assert datamodule._main[split].x.size(0) == 5
        assert aux["reference_normal"].x.size(0) == 5
        for name in nulls:
            assert aux[name].x.size(0) == 5 and set(aux[name].y.tolist()) == {0}
        for label, name in enumerate(signals, start=1):
            n = datamodule.case_counts[name][1]
            n_valid = round(0.6 * n)
            assert aux[name].x.size(0) == (n_valid if split == "valid" else n - n_valid)
            assert set(aux[name].y.tolist()) <= {label}


def test_sets_are_disjoint_by_case(datamodule):
    served = {k: _serials(datamodule, x) for k, x in _served(datamodule).items()}
    names = list(served)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not served[a] & served[b], (a, b)


def test_each_case_is_its_first_complete_count(datamodule, sbc_data):
    first = _first_complete(sbc_data.rows)
    served = torch.cat(list(_served(datamodule).values()))
    restored = _restored(datamodule, served)
    serials = restored[:, 1].round().long() - MCV_OFFSET
    positions = restored[:, 3].round().long()
    assert {s: p for s, p in zip(serials.tolist(), positions.tolist())} == {
        s: first[s] for s in serials.tolist()
    }
    kept = sum(k for _, k in datamodule.case_counts.values())
    assert kept == len(first)
    assert sum(n for n, _ in datamodule.case_counts.values()) == sum(CASES.values())
    assert len(first) < sum(CASES.values())


def test_log_and_normalisation_from_the_training_rows(datamodule, sbc_data):
    by_position = {i: row for i, row in enumerate(sbc_data.rows, start=1)}
    x_train = datamodule.data_train.tensors[0]
    positions = _restored(datamodule, x_train)[:, 3].round().long().tolist()
    inputs = ("HGB", "MCV", "PLT", "RBC", "WBC")
    raw = np.array([[float(by_position[p][k]) for k in inputs] for p in positions])
    raw[:, [2, 4]] = np.log(raw[:, [2, 4]])
    raw = torch.as_tensor(raw, dtype=torch.float32)
    restored = _restored(datamodule, x_train)
    torch.testing.assert_close(restored, raw.double(), rtol=1e-5, atol=1e-4)
    lower, median, upper = torch.quantile(raw, torch.tensor([0.05, 0.5, 0.95]), dim=0)
    torch.testing.assert_close(datamodule.median, median, rtol=0, atol=0)
    spread = (upper - lower).clamp_min(1e-6)
    torch.testing.assert_close(datamodule.spread, spread, rtol=0, atol=0)


def test_splits_depend_on_the_seed_alone(sbc_data, sbc_config, datamodule):
    again = _datamodule(sbc_data, sbc_config, train_seed=7)
    other = _datamodule(sbc_data, sbc_config, seed=271828)
    for name, x in _served(datamodule).items():
        torch.testing.assert_close(_served(again)[name], x, rtol=0, atol=0)
    assert not torch.equal(
        other.data_train.tensors[0], datamodule.data_train.tensors[0]
    )


def test_batch_order_follows_the_train_seed(sbc_data, sbc_config):
    a, b, c = (_datamodule(sbc_data, sbc_config, train_seed=s) for s in (1, 1, 2))
    first = [next(iter(dm.train_dataloader()))[0] for dm in (a, b, c)]
    assert torch.equal(first[0], first[1])
    assert not torch.equal(first[0], first[2])


def test_a_case_with_two_diagnoses_raises(sbc_data, sbc_config, tmp_path):
    rows = [dict(row) for row in sbc_data.rows]
    twin = next(r for r in rows if r["Diagnosis"] == "Control")
    rows.append(twin | {"Diagnosis": "Sepsis", "RBC": str(len(rows) + 1)})
    _write(tmp_path / "extdata" / "sbcdata.csv", rows)
    with pytest.raises(ValueError, match="more than one value"):
        _datamodule(SimpleNamespace(data_dir=tmp_path), sbc_config)

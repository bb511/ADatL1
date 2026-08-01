from __future__ import annotations

import torch

from src.data.L1AD_datamodule import L1ADDataModule, SplitTensors


def _split(n_events: int, label: int) -> SplitTensors:
    x = torch.arange(n_events * 3, dtype=torch.float32).reshape(n_events, 1, 3)
    mask = torch.ones_like(x, dtype=torch.bool)
    l1bit = torch.zeros(n_events, 1, dtype=torch.bool)
    y = torch.full((n_events,), label, dtype=torch.int64)
    return SplitTensors(x=x, mask=mask, l1bit=l1bit, y=y)


def _datamodule(*, normal_cap: int | None, seed: int = 42) -> L1ADDataModule:
    dm = L1ADDataModule(
        zerobias={},
        signal={},
        background={},
        data_extractor=object(),
        data_processor=object(),
        data_normalizer=object(),
        data_mlready=object(),
        data_awkward2torch=object(),
        train_features={},
        l1_scales={},
        batch_size=4,
        max_val_batches=1,
        max_normal_eval_batches=normal_cap,
        seed=seed,
    )
    dm.batch_size_per_device = 4
    dm._main["valid"] = _split(11, label=0)
    dm._main["test"] = _split(11, label=0)
    dm._aux["valid"] = {"signal": _split(11, label=1)}
    dm._aux["test"] = {"signal": _split(11, label=1)}
    return dm


def _rows(loader) -> torch.Tensor:
    return torch.cat([batch[0] for batch in loader], dim=0)


def test_normal_eval_cap_is_independent_and_deterministic() -> None:
    dm = _datamodule(normal_cap=2)

    valid = dm.val_dataloader()
    test = dm.test_dataloader()

    assert len(valid["normal"]) == 2
    assert len(valid["signal"]) == 1
    assert len(test["normal"]) == 2
    assert len(test["signal"]) == 1
    torch.testing.assert_close(_rows(valid["normal"]), _rows(valid["normal"]))

    # The cap always chooses the same prefix, independent of the training RNG seed.
    other_candidate = _datamodule(normal_cap=2, seed=999)
    torch.testing.assert_close(
        _rows(valid["normal"]), _rows(other_candidate.val_dataloader()["normal"])
    )


def test_none_normal_eval_cap_preserves_full_split() -> None:
    dm = _datamodule(normal_cap=None)

    assert len(dm.val_dataloader()["normal"]) == 3
    assert _rows(dm.val_dataloader()["normal"]).shape[0] == 11


def test_minus_one_normal_eval_cap_aliases_unbounded() -> None:
    dm = _datamodule(normal_cap=-1)

    assert len(dm.test_dataloader()["normal"]) == 3

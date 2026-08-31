import json
import pickle

import numpy as np
import pytest
import torch

from src.data.components.awkward2torch import L1DataAwkward2Torch


def make_tensor_cache(tmp_path, n_events: int = 20):
    split_folder = tmp_path / "cache" / "train"
    split_folder.mkdir(parents=True)

    data = torch.arange(
        n_events * 6,
        dtype=torch.float32,
    ).reshape(n_events, 2, 3)
    mask = (data % 2) == 0
    l1bit = (torch.arange(n_events) % 3) == 0

    torch.save(data, split_folder / "torch_cache.pt")
    torch.save(mask, split_folder / "torch_mask.pt")
    torch.save(l1bit, split_folder / "torch_l1bit.pt")
    with (split_folder / "cached_objs.pkl").open("wb") as handle:
        pickle.dump(set(), handle)
    (split_folder.parent / "object_feature_map.json").write_text(
        json.dumps({"jets": {"Et": [0, 1]}}),
        encoding="utf-8",
    )

    return split_folder, data, mask, l1bit


def test_memory_mapped_smoke_cap_is_deterministic_and_ordered(
    tmp_path,
) -> None:
    split_folder, data, mask, l1bit = make_tensor_cache(tmp_path)
    loader = L1DataAwkward2Torch(workers=1, nconst={})

    first = loader.load_folder(
        split_folder,
        max_events=7,
        sample_seed=12345,
    )
    second = loader.load_folder(
        split_folder,
        max_events=7,
        sample_seed=12345,
    )

    expected_indices = np.sort(
        np.random.default_rng(12345).choice(
            data.shape[0],
            size=7,
            replace=False,
        )
    )
    index = torch.from_numpy(expected_indices)
    expected = (
        data.index_select(0, index),
        mask.index_select(0, index),
        l1bit.index_select(0, index),
    )

    for first_tensor, second_tensor, expected_tensor in zip(
        first,
        second,
        expected,
        strict=True,
    ):
        torch.testing.assert_close(first_tensor, expected_tensor)
        torch.testing.assert_close(second_tensor, expected_tensor)
        assert first_tensor.shape[0] == 7


@pytest.mark.parametrize("invalid_cap", [0, -1, 1.5, True])
def test_memory_mapped_smoke_cap_rejects_invalid_values(
    tmp_path,
    invalid_cap,
) -> None:
    split_folder, *_ = make_tensor_cache(tmp_path)
    loader = L1DataAwkward2Torch(workers=1, nconst={})

    with pytest.raises(ValueError, match="max_events"):
        loader.load_folder(
            split_folder,
            max_events=invalid_cap,
        )

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from src.data.L1AD_datamodule import L1ADDataModule, SplitTensors


def make_split(num_events: int = 5) -> SplitTensors:
    x = torch.arange(
        num_events * 2,
        dtype=torch.float32,
    ).reshape(num_events, 2)

    mask = torch.ones_like(x, dtype=torch.bool)
    l1bit = torch.zeros(num_events, dtype=torch.bool)
    labels = torch.zeros(num_events, dtype=torch.float32)

    control_x = torch.arange(
        num_events * 3,
        dtype=torch.float32,
    ).reshape(num_events, 3)

    control_mask = torch.ones_like(
        control_x,
        dtype=torch.bool,
    )

    return SplitTensors(
        x=x,
        mask=mask,
        l1bit=l1bit,
        y=labels,
        control_x=control_x,
        control_mask=control_mask,
    )


def make_datamodule(tmp_path: Path) -> L1ADDataModule:
    datamodule = L1ADDataModule(
        zerobias={},
        signal={},
        background={},
        data_extractor=Mock(),
        data_processor=Mock(),
        data_normalizer=Mock(),
        data_mlready=Mock(),
        data_awkward2torch=Mock(),
        train_features={},
        l1_scales={},
        batch_size=2,
        max_val_batches=1,
        seed=42,
    )

    datamodule.main_cache_folder = tmp_path
    return datamodule


def collect_loader(loader) -> tuple[torch.Tensor, ...]:
    batches = list(loader)

    assert batches
    assert all(len(batch) == 6 for batch in batches)

    return tuple(
        torch.cat(
            [batch[field_index] for batch in batches],
            dim=0,
        )
        for field_index in range(6)
    )


@pytest.mark.parametrize("split_name", ["train", "valid", "test"])
def test_setup_probe_split_loads_only_requested_main_split(
    tmp_path: Path,
    split_name: str,
) -> None:
    datamodule = make_datamodule(tmp_path)
    split = make_split()

    datamodule._load_main_split = Mock(return_value=split)
    datamodule._load_aux_split = Mock(
        side_effect=AssertionError(
            "Probe setup must not load auxiliary datasets."
        )
    )

    datamodule.setup_probe_split(split_name)

    datamodule._load_main_split.assert_called_once_with(
        tmp_path,
        split_name,
        label=0,
    )
    datamodule._load_aux_split.assert_not_called()

    assert datamodule._probe_split is split
    assert datamodule._probe_split_name == split_name


def test_setup_probe_split_rejects_unknown_split(
    tmp_path: Path,
) -> None:
    datamodule = make_datamodule(tmp_path)

    with pytest.raises(
        ValueError,
        match="Unknown probe split",
    ):
        datamodule.setup_probe_split("validation")


def test_setup_probe_split_requires_prepared_cache(
    tmp_path: Path,
) -> None:
    datamodule = make_datamodule(tmp_path)
    datamodule.main_cache_folder = None

    with pytest.raises(
        RuntimeError,
        match="main_cache_folder is not set",
    ):
        datamodule.setup_probe_split("train")


def test_setup_probe_split_requires_previous_split_release(
    tmp_path: Path,
) -> None:
    datamodule = make_datamodule(tmp_path)
    datamodule._load_main_split = Mock(return_value=make_split())

    datamodule.setup_probe_split("train")

    with pytest.raises(
        RuntimeError,
        match="already loaded",
    ):
        datamodule.setup_probe_split("valid")

    assert datamodule._load_main_split.call_count == 1
    assert datamodule._probe_split_name == "train"


def test_failed_probe_split_load_does_not_set_resident_state(
    tmp_path: Path,
) -> None:
    datamodule = make_datamodule(tmp_path)
    datamodule._load_main_split = Mock(
        side_effect=OSError("failed to load split")
    )

    with pytest.raises(
        OSError,
        match="failed to load split",
    ):
        datamodule.setup_probe_split("train")

    assert datamodule._probe_split is None
    assert datamodule._probe_split_name is None


def test_probe_dataloader_requires_loaded_split(
    tmp_path: Path,
) -> None:
    datamodule = make_datamodule(tmp_path)

    with pytest.raises(
        RuntimeError,
        match="No probe split is loaded",
    ):
        datamodule.probe_dataloader()


def test_probe_dataloader_is_unshuffled_and_repeatable(
    tmp_path: Path,
) -> None:
    datamodule = make_datamodule(tmp_path)
    split = make_split(num_events=5)
    datamodule._load_main_split = Mock(return_value=split)

    datamodule.setup_probe_split("train")
    loader = datamodule.probe_dataloader()

    assert loader.dataset.shuffler is None

    first_pass = collect_loader(loader)
    second_pass = collect_loader(loader)

    assert len(first_pass) == len(second_pass) == 6

    for first_tensor, second_tensor in zip(
        first_pass,
        second_pass,
        strict=True,
    ):
        torch.testing.assert_close(
            first_tensor,
            second_tensor,
        )

    torch.testing.assert_close(first_pass[0], split.x)
    torch.testing.assert_close(first_pass[1], split.mask)
    torch.testing.assert_close(first_pass[2], split.l1bit)
    torch.testing.assert_close(first_pass[3], split.y)
    torch.testing.assert_close(first_pass[4], split.control_x)
    torch.testing.assert_close(first_pass[5], split.control_mask)


def test_probe_dataloader_ignores_max_val_batches(
    tmp_path: Path,
) -> None:
    datamodule = make_datamodule(tmp_path)
    split = make_split(num_events=5)
    datamodule._load_main_split = Mock(return_value=split)

    # max_val_batches=1 was configured in make_datamodule().
    datamodule.setup_probe_split("valid")
    loader = datamodule.probe_dataloader()

    batches = list(loader)

    # Five events at batch size two require three batches. The probe
    # loader must use all three rather than max_val_batches=1.
    assert len(batches) == 3

    collected_x = torch.cat(
        [batch[0] for batch in batches],
        dim=0,
    )
    torch.testing.assert_close(collected_x, split.x)


def test_probe_dataloader_keeps_main_and_control_feature_maps(
    tmp_path: Path,
) -> None:
    datamodule = make_datamodule(tmp_path)
    datamodule.object_feature_map = {
        "jets": {
            "Et": [0, 1],
        }
    }
    datamodule.control_object_feature_map = {
        "jets": {
            "Et": [0, 1],
        },
        "FET": {
            "Et": [2],
        },
    }

    datamodule._load_main_split = Mock(
        return_value=make_split()
    )

    datamodule.setup_probe_split("train")
    loader = datamodule.probe_dataloader()

    assert (
        loader.dataset.object_feature_map
        == datamodule.object_feature_map
    )
    assert (
        loader.dataset.control_object_feature_map
        == datamodule.control_object_feature_map
    )


def test_release_probe_split_clears_resident_state(
    tmp_path: Path,
) -> None:
    datamodule = make_datamodule(tmp_path)
    datamodule._load_main_split = Mock(return_value=make_split())

    datamodule.setup_probe_split("train")
    assert datamodule._probe_split is not None

    datamodule.release_probe_split()

    assert datamodule._probe_split is None
    assert datamodule._probe_split_name is None

    # Release is intentionally idempotent.
    datamodule.release_probe_split()


def test_different_probe_splits_can_be_loaded_sequentially(
    tmp_path: Path,
) -> None:
    datamodule = make_datamodule(tmp_path)

    train_split = make_split(num_events=5)
    valid_split = make_split(num_events=3)

    datamodule._load_main_split = Mock(
        side_effect=[train_split, valid_split]
    )

    datamodule.setup_probe_split("train")
    train_loader = datamodule.probe_dataloader()
    train_values = collect_loader(train_loader)
    datamodule.release_probe_split()

    datamodule.setup_probe_split("valid")
    valid_loader = datamodule.probe_dataloader()
    valid_values = collect_loader(valid_loader)
    datamodule.release_probe_split()

    assert train_values[0].shape[0] == 5
    assert valid_values[0].shape[0] == 3

    assert datamodule._probe_split is None
    assert datamodule._probe_split_name is None

    assert datamodule._load_main_split.call_args_list == [
        ((tmp_path, "train"), {"label": 0}),
        ((tmp_path, "valid"), {"label": 0}),
    ]
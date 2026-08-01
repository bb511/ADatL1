from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.callbacks.cap import CAPCallback
from src.evaluation.callbacks.cap import CAP as EvaluationCAP
from src.utils.pairing.table import (
    atomic_torch_save,
    load_pair_table,
    sha256_file,
    sha256_tensor,
    validate_pair_table,
)
from src.utils.pairing.utils import (
    PairingResult,
    closure_metrics,
    mutual_nearest_pairs,
    one_to_one_nearest_pairs,
    pair_table_dict,
)


def _table(
    tmp_path: Path,
    *,
    split: str = "validate",
    idx_1: torch.Tensor | None = None,
    idx_2: torch.Tensor | None = None,
    n_1: int = 3,
    n_2: int = 3,
) -> dict:
    checkpoint = tmp_path / "encoder.ckpt"
    checkpoint.write_bytes(b"pairing-checkpoint")
    idx_1 = torch.tensor([0, 1]) if idx_1 is None else idx_1
    idx_2 = torch.tensor([1, 2]) if idx_2 is None else idx_2
    pairs = PairingResult(
        idx_1=idx_1,
        idx_2=idx_2,
        distance=torch.linspace(0.1, 0.2, idx_1.numel()),
        rank_1_to_2=torch.ones(idx_1.numel(), dtype=torch.long),
        rank_2_to_1=torch.ones(idx_1.numel(), dtype=torch.long),
    )
    source_1 = torch.arange(n_1 * 2, dtype=torch.float32).reshape(n_1, 2)
    source_2 = torch.arange(n_2 * 2, dtype=torch.float32).reshape(n_2, 2) + 1
    return pair_table_dict(
        pairs,
        dataset_1="normal",
        dataset_2="reference_normal",
        split=split,
        encoder_ckpt=str(checkpoint),
        metadata={
            "n_dataset_1": n_1,
            "n_dataset_2": n_2,
            "n_pairs": idx_1.numel(),
            "encoder_checkpoint_sha256": sha256_file(checkpoint),
            "source_1_sha256": sha256_tensor(source_1),
            "source_2_sha256": sha256_tensor(source_2),
            "data_seed": 123,
        },
    )


def test_pairing_algorithms_validate_inputs_and_return_unique_pairs() -> None:
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
    z2 = z1 + 0.01

    mutual = mutual_nearest_pairs(z1, z2, k=2)
    greedy = one_to_one_nearest_pairs(z1, z2, k=None, normalize=True)

    assert mutual.idx_1.numel() > 0
    assert torch.unique(mutual.idx_1).numel() == mutual.idx_1.numel()
    assert torch.unique(mutual.idx_2).numel() == mutual.idx_2.numel()
    assert greedy.idx_1.numel() == 3
    assert torch.unique(greedy.idx_2).numel() == 3

    with pytest.raises(ValueError, match="positive"):
        mutual_nearest_pairs(z1, z2, k=0)
    with pytest.raises(ValueError, match="finite"):
        mutual_nearest_pairs(z1, z2.fill_(float("nan")), k=1)
    with pytest.raises(ValueError, match="same number"):
        closure_metrics(z1, z2[:2])


def test_closure_metrics_are_exact_across_query_chunks() -> None:
    generator = torch.Generator().manual_seed(123)
    z1 = torch.randn(23, 7, generator=generator)
    z2 = z1 + 0.2 * torch.randn(23, 7, generator=generator)

    chunked = closure_metrics(z1, z2, ks=(1, 5, 10), chunk_size=4)
    dense = closure_metrics(z1, z2, ks=(1, 5, 10), chunk_size=23)

    assert chunked == pytest.approx(dense)
    with pytest.raises(ValueError, match="chunk_size"):
        closure_metrics(z1, z2, chunk_size=0)


def test_versioned_pair_table_round_trip_and_overwrite_protection(tmp_path: Path) -> None:
    path = tmp_path / "pairs.pt"
    table = _table(tmp_path)

    atomic_torch_save(table, path)
    loaded = load_pair_table(
        path,
        expected_dataset_1="normal",
        expected_dataset_2="reference_normal",
        expected_split="validate",
        n_dataset_1=3,
        n_dataset_2=3,
    )

    torch.testing.assert_close(loaded["idx_1"], table["idx_1"])
    with pytest.raises(FileExistsError, match="overwrite"):
        atomic_torch_save(table, path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda table: table.pop("schema_version"), "schema_version"),
        (lambda table: table.update(split="test"), "split"),
        (lambda table: table.update(idx_1=torch.tensor([-1, 1])), "non-negative"),
        (lambda table: table.update(idx_2=torch.tensor([1, 1])), "unique"),
        (
            lambda table: table["metadata"].update(n_dataset_1=1),
            "source sizes",
        ),
    ],
)
def test_pair_table_rejects_stale_or_invalid_artifacts(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    table = _table(tmp_path)
    mutation(table)

    with pytest.raises(ValueError, match=message):
        validate_pair_table(table, expected_split="validate")


def test_training_cap_rejects_pair_table_for_different_source_size(tmp_path: Path) -> None:
    path = tmp_path / "pairs.pt"
    atomic_torch_save(_table(tmp_path), path)
    callback = CAPCallback(
        output_name="ascore/full",
        dataset_1="normal",
        dataset_2="reference_normal",
        pairing_type="precomputed",
        pairing_index_path=str(path),
        cap_metric_config={},
    )

    with pytest.raises(ValueError, match="CAP collected 4"):
        callback._pair_indices(torch.ones(4), torch.ones(3))


def test_training_cap_seeded_random_pairing_is_stable() -> None:
    callback = CAPCallback(
        output_name="ascore/full",
        dataset_1="normal",
        dataset_2="reference_normal",
        pairing_type="random",
        pairing_seed=271828,
        metric_name="cap_random",
        cap_metric_config={},
    )

    first = callback._pair_indices(torch.ones(10), torch.ones(10))
    second = callback._pair_indices(torch.ones(10), torch.ones(10))

    torch.testing.assert_close(first[0], torch.arange(10))
    torch.testing.assert_close(first[1], second[1])
    assert torch.unique(first[1]).numel() == 10


def test_evaluation_cap_uses_distinct_validation_and_test_tables(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.pt"
    test_path = tmp_path / "test.pt"
    atomic_torch_save(_table(tmp_path, split="validate"), valid_path)
    atomic_torch_save(
        _table(
            tmp_path,
            split="test",
            idx_1=torch.tensor([1, 2]),
            idx_2=torch.tensor([0, 1]),
        ),
        test_path,
    )
    callback = EvaluationCAP(
        output_name="ascore/full",
        dataset_1="normal",
        dataset_2="reference_normal",
        pairing_type="precomputed",
        pairing_index_path=str(valid_path),
        pairing_test_index_path=str(test_path),
        cap_metric_config={},
    )

    callback.pair_table_split = "validate"
    valid_indices = callback._pair_indices(torch.ones(3), torch.ones(3))
    callback.pair_table_split = "test"
    test_indices = callback._pair_indices(torch.ones(3), torch.ones(3))

    torch.testing.assert_close(valid_indices[0].cpu(), torch.tensor([0, 1]))
    torch.testing.assert_close(test_indices[0].cpu(), torch.tensor([1, 2]))


def test_evaluation_cap_seeded_random_pairing_is_stable() -> None:
    callback = EvaluationCAP(
        output_name="ascore/full",
        dataset_1="normal",
        dataset_2="reference_normal",
        pairing_type="random",
        pairing_seed=271828,
        cap_metric_config={},
    )

    first = callback._pair_indices(torch.ones(10), torch.ones(10))
    second = callback._pair_indices(torch.ones(10), torch.ones(10))

    torch.testing.assert_close(first[0], torch.arange(10))
    torch.testing.assert_close(first[1], second[1])

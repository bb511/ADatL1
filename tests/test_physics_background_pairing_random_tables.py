"""Contracts for frozen physics random-bijection CAP control tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.utils.pairing import random_tables
from src.utils.pairing.random_tables import (
    TEST_SEEDS,
    VALIDATION_SEEDS,
    build_random_pair_table,
    freeze_random_pairing_tables,
    seed_contract,
    validate_frozen_random_pairing_tables,
)
from src.utils.pairing.table import (
    atomic_torch_save,
    load_pair_table,
    sha256_file,
    validate_pair_table,
)


def _sources(n_events: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two distinct ordered synthetic source tensors."""
    source_1 = torch.arange(n_events * 3, dtype=torch.float32).reshape(n_events, 3)
    source_2 = source_1 + 1000
    return source_1, source_2


def test_preregistered_seed_contract_is_explicit_disjoint_and_hashed() -> None:
    """The five validation and five test seeds are immutable explicit constants."""
    contract = seed_contract()
    assert contract["validate"] == list(VALIDATION_SEEDS)
    assert contract["test"] == list(TEST_SEEDS)
    assert len(set(VALIDATION_SEEDS)) == len(set(TEST_SEEDS)) == 5
    assert set(VALIDATION_SEEDS).isdisjoint(TEST_SEEDS)
    assert contract == seed_contract()
    assert len(contract["sha256"]) == 64


def test_random_table_is_reproducible_total_bijection_and_seed_specific() -> None:
    """A fixed seed reproduces one full permutation and another seed changes it."""
    source_1, source_2 = _sources()
    first = build_random_pair_table(source_1, source_2, split="validate", seed=VALIDATION_SEEDS[0])
    repeat = build_random_pair_table(
        source_1, source_2, split="validate", seed=VALIDATION_SEEDS[0]
    )
    other = build_random_pair_table(source_1, source_2, split="validate", seed=VALIDATION_SEEDS[1])
    assert torch.equal(first["idx_1"], torch.arange(len(source_1)))
    assert torch.equal(torch.sort(first["idx_2"]).values, torch.arange(len(source_1)))
    assert torch.equal(first["map_0_to_1"], first["idx_2"])
    assert torch.equal(first["idx_2"], repeat["idx_2"])
    assert not torch.equal(first["idx_2"], other["idx_2"])
    assert first["metadata"]["permutation_sha256"] == repeat["metadata"]["permutation_sha256"]


def test_random_table_rejects_unregistered_seed_and_wrong_split() -> None:
    """Callers cannot silently introduce post-outcome seeds or split reuse."""
    source_1, source_2 = _sources()
    with pytest.raises(ValueError, match="not preregistered"):
        build_random_pair_table(source_1, source_2, split="validate", seed=7)
    with pytest.raises(ValueError, match="not preregistered"):
        build_random_pair_table(source_1, source_2, split="test", seed=VALIDATION_SEEDS[0])


def test_random_table_load_binds_split_source_names_and_hashes(tmp_path: Path) -> None:
    """The ordinary pair-table loader enforces exact ordered source identity."""
    source_1, source_2 = _sources()
    table = build_random_pair_table(source_1, source_2, split="validate", seed=VALIDATION_SEEDS[0])
    path = atomic_torch_save(table, tmp_path / "random.pt")
    with pytest.raises(ValueError, match="source_1_sha256 does not match"):
        load_pair_table(
            path,
            expected_dataset_1=random_tables.DATASET_1,
            expected_dataset_2=random_tables.DATASET_2,
            expected_split="validate",
            n_dataset_1=len(source_1),
            n_dataset_2=len(source_2),
            source_1_sha256="0" * 64,
            source_2_sha256=table["metadata"]["source_2_sha256"],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("permutation_hash", "permutation SHA-256 differs"),
        ("partial", "metadata reports"),
        ("generator", "generator must be a non-empty string"),
    ],
)
def test_random_specific_table_validation_fails_closed(mutation: str, message: str) -> None:
    """Random provenance and total permutation semantics are actively validated."""
    source_1, source_2 = _sources()
    table = build_random_pair_table(source_1, source_2, split="validate", seed=VALIDATION_SEEDS[0])
    if mutation == "permutation_hash":
        table["metadata"]["permutation_sha256"] = "0" * 64
    elif mutation == "partial":
        table["metadata"]["n_pairs"] -= 1
    else:
        table["metadata"]["generator"] = ""
    with pytest.raises(ValueError, match=message):
        validate_pair_table(table)


def _campaign(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal source-authenticated campaign and structured artifact marker."""
    root = tmp_path / "campaign"
    structured = tmp_path / "structured_pairing"
    structured.mkdir(parents=True)
    marker = structured / "unchanged.txt"
    marker.write_text("structured", encoding="utf-8")
    structured_table = structured / "structured.pt"
    structured_table.write_bytes(b"frozen structured table")
    preflight = {
        "schema_version": 1,
        "cache_root": str((tmp_path / "cache").resolve()),
        "artifact_dir": str(structured.resolve()),
        "dataset_1": random_tables.DATASET_1,
        "dataset_2": random_tables.DATASET_2,
        "events_per_source": 8,
        "tables": [
            {
                "authenticated": True,
                "path": str(structured_table.resolve()),
                "sha256": sha256_file(structured_table),
            }
        ],
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    (tmp_path / "cache").mkdir()
    root.mkdir()
    (root / "design.json").write_text(
        json.dumps(
            {
                "code_commit": "a" * 40,
                "preflight": str(preflight_path.resolve()),
                "preflight_sha256": sha256_file(preflight_path),
            }
        ),
        encoding="utf-8",
    )
    return root, marker


def test_freeze_is_atomic_reauthenticatable_and_preserves_structured_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ten tables freeze separately, validate, and cannot be overwritten."""
    root, structured_marker = _campaign(tmp_path)
    source_1, source_2 = _sources(8)

    def fake_load_split(cache_root, split, *, dataset, limit, source_metadata_dir):
        """Return exact synthetic source prefixes for the requested source name."""
        del cache_root, split, limit, source_metadata_dir
        source = source_1 if dataset == random_tables.DATASET_1 else source_2
        return source, torch.ones_like(source, dtype=torch.bool), tmp_path / dataset

    monkeypatch.setattr(random_tables, "EVENTS", 8)
    monkeypatch.setattr(random_tables, "load_split", fake_load_split)
    manifest = freeze_random_pairing_tables(root)
    assert len(manifest["tables"]) == 10
    assert structured_marker.read_text(encoding="utf-8") == "structured"
    assert validate_frozen_random_pairing_tables(root) == manifest
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        freeze_random_pairing_tables(root)


def test_freeze_cleans_partial_directory_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source failure leaves no apparently frozen random-control directory."""
    root, _ = _campaign(tmp_path)
    monkeypatch.setattr(random_tables, "EVENTS", 8)

    def fail_load_split(*args, **kwargs):
        """Simulate an input failure before any table can be authenticated."""
        del args, kwargs
        raise RuntimeError("source failed")

    monkeypatch.setattr(random_tables, "load_split", fail_load_split)
    with pytest.raises(RuntimeError, match="source failed"):
        freeze_random_pairing_tables(root)
    assert not (root / "random_pairing").exists()
    assert not list(root.glob(".random_pairing.*"))


def test_freeze_rejects_changed_structured_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Random controls cannot freeze from a campaign whose structured input changed."""
    root, marker = _campaign(tmp_path)
    monkeypatch.setattr(random_tables, "EVENTS", 8)
    structured_table = marker.parent / "structured.pt"
    structured_table.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="structured pair table changed"):
        freeze_random_pairing_tables(root)
    assert not (root / "random_pairing").exists()

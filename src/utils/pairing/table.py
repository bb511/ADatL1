from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

PAIR_TABLE_SCHEMA_VERSION = 1
_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"File does not exist: {resolved}")

    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    """Hash tensor values together with their dtype and shape."""
    if not torch.is_tensor(value):
        raise TypeError("sha256_tensor expects a torch tensor.")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    raw = memoryview(tensor.numpy()).cast("B")
    for offset in range(0, len(raw), 1024 * 1024):
        digest.update(raw[offset : offset + 1024 * 1024])
    return digest.hexdigest()


def load_pair_table(
    path: str | Path,
    *,
    expected_dataset_1: str | None = None,
    expected_dataset_2: str | None = None,
    expected_split: str | None = None,
    n_dataset_1: int | None = None,
    n_dataset_2: int | None = None,
    source_1_sha256: str | None = None,
    source_2_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and strictly validate a versioned pair-table artifact."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Pair table does not exist: {resolved}")
    table = torch.load(resolved, map_location="cpu", weights_only=False)
    validate_pair_table(
        table,
        expected_dataset_1=expected_dataset_1,
        expected_dataset_2=expected_dataset_2,
        expected_split=expected_split,
        n_dataset_1=n_dataset_1,
        n_dataset_2=n_dataset_2,
        source_1_sha256=source_1_sha256,
        source_2_sha256=source_2_sha256,
    )
    return table


def validate_pair_table(
    table: Mapping[str, Any],
    *,
    expected_dataset_1: str | None = None,
    expected_dataset_2: str | None = None,
    expected_split: str | None = None,
    n_dataset_1: int | None = None,
    n_dataset_2: int | None = None,
    source_1_sha256: str | None = None,
    source_2_sha256: str | None = None,
) -> None:
    """Validate identity, provenance, tensor shape, uniqueness, and index bounds."""
    if not isinstance(table, Mapping):
        raise TypeError("Pair table must be a mapping.")

    version = table.get("schema_version")
    if version != PAIR_TABLE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported or unversioned pair table. "
            f"Expected schema_version={PAIR_TABLE_SCHEMA_VERSION}, got {version!r}. "
            "Regenerate the table with the current build_pair_table command."
        )

    idx_1 = _index_tensor(table, "idx_1")
    idx_2 = _index_tensor(table, "idx_2")
    if idx_1.numel() != idx_2.numel():
        raise ValueError("Pair-table idx_1 and idx_2 must have equal length.")
    if idx_1.numel() == 0:
        raise ValueError("Pair table contains no pairs.")
    if torch.any(idx_1 < 0) or torch.any(idx_2 < 0):
        raise ValueError("Pair-table indices must be non-negative.")
    if torch.unique(idx_1).numel() != idx_1.numel():
        raise ValueError("Pair-table idx_1 values must be unique.")
    if torch.unique(idx_2).numel() != idx_2.numel():
        raise ValueError("Pair-table idx_2 values must be unique.")

    dataset_1 = _nonempty_string(table, "dataset_1")
    dataset_2 = _nonempty_string(table, "dataset_2")
    split = _nonempty_string(table, "split")
    _nonempty_string(table, "encoder_ckpt")

    if expected_dataset_1 is not None and dataset_1 != expected_dataset_1:
        raise ValueError(
            f"Pair table dataset_1 is {dataset_1!r}, expected {expected_dataset_1!r}."
        )
    if expected_dataset_2 is not None and dataset_2 != expected_dataset_2:
        raise ValueError(
            f"Pair table dataset_2 is {dataset_2!r}, expected {expected_dataset_2!r}."
        )
    if expected_split is not None and split != expected_split:
        raise ValueError(f"Pair table split is {split!r}, expected {expected_split!r}.")

    metadata = table.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Pair table metadata must be a mapping.")
    stored_n_1 = _positive_metadata_int(metadata, "n_dataset_1")
    stored_n_2 = _positive_metadata_int(metadata, "n_dataset_2")
    stored_pairs = _positive_metadata_int(metadata, "n_pairs")
    if stored_pairs != idx_1.numel():
        raise ValueError(
            f"Pair table metadata reports {stored_pairs} pairs, found {idx_1.numel()}."
        )
    if idx_1.max().item() >= stored_n_1 or idx_2.max().item() >= stored_n_2:
        raise ValueError("Pair-table indices exceed the source sizes recorded in metadata.")
    if n_dataset_1 is not None and stored_n_1 != int(n_dataset_1):
        raise ValueError(
            f"Pair table was built for {stored_n_1} {dataset_1} samples, "
            f"but CAP collected {n_dataset_1}."
        )
    if n_dataset_2 is not None and stored_n_2 != int(n_dataset_2):
        raise ValueError(
            f"Pair table was built for {stored_n_2} {dataset_2} samples, "
            f"but CAP collected {n_dataset_2}."
        )

    dense_map = table.get("map_0_to_1")
    if dense_map is not None:
        if (
            not torch.is_tensor(dense_map)
            or dense_map.ndim != 1
            or dense_map.dtype not in _INTEGER_DTYPES
        ):
            raise ValueError("Pair-table map_0_to_1 must be a one-dimensional integer tensor.")
        dense_map = dense_map.long().cpu()
        if dense_map.numel() != stored_n_1:
            raise ValueError("Pair-table map_0_to_1 must have one entry per dataset-1 row.")
        if idx_1.numel() != stored_n_1 or not torch.equal(
            idx_1, torch.arange(stored_n_1, dtype=torch.long)
        ):
            raise ValueError("Dense map_0_to_1 requires idx_1 == arange(n_dataset_1).")
        if not torch.equal(dense_map, idx_2):
            raise ValueError("Pair-table map_0_to_1 must be identical to idx_2.")

    for name in (
        "encoder_checkpoint_sha256",
        "source_1_sha256",
        "source_2_sha256",
    ):
        digest = metadata.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"Pair table metadata must contain a valid {name}.")
    for name, expected in (
        ("source_1_sha256", source_1_sha256),
        ("source_2_sha256", source_2_sha256),
    ):
        if expected is not None and metadata[name] != expected:
            raise ValueError(
                f"Pair table {name} does not match the samples collected by CAP. "
                "Regenerate the table with the exact data configuration used by this run."
            )

    for name in ("distance", "rank_1_to_2", "rank_2_to_1"):
        value = table.get(name)
        if not torch.is_tensor(value) or value.ndim != 1:
            raise ValueError(f"Pair-table {name} must be a one-dimensional tensor.")
        if value.numel() != idx_1.numel():
            raise ValueError(f"Pair-table {name} must have one value per pair.")
    distance = table["distance"]
    if not torch.is_floating_point(distance) or not torch.isfinite(distance).all():
        raise ValueError("Pair-table distances must be finite floating-point values.")
    if torch.any(distance < 0):
        raise ValueError("Pair-table distances must be non-negative.")
    for name in ("rank_1_to_2", "rank_2_to_1"):
        if table[name].dtype not in _INTEGER_DTYPES:
            raise ValueError(f"Pair-table {name} must contain integers.")
        if torch.any(table[name] < 0):
            raise ValueError(f"Pair-table {name} values must be non-negative.")


def atomic_torch_save(value: Any, path: str | Path, *, overwrite: bool = False) -> Path:
    """Atomically write a torch artifact and refuse accidental replacement by default."""
    target = _prepare_target(path, overwrite=overwrite)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_json_dump(
    value: Any,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write strict JSON and reject NaN/Infinity values."""
    target = _prepare_target(path, overwrite=overwrite)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _prepare_target(path: str | Path, *, overwrite: bool) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing artifact: {target}. Pass --overwrite explicitly."
        )
    return target


def _index_tensor(table: Mapping[str, Any], name: str) -> torch.Tensor:
    value = table.get(name)
    if not torch.is_tensor(value) or value.ndim != 1 or value.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"Pair-table {name} must be a one-dimensional integer tensor.")
    return value.long().cpu()


def _nonempty_string(table: Mapping[str, Any], name: str) -> str:
    value = table.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Pair-table {name} must be a non-empty string.")
    return value


def _positive_metadata_int(metadata: Mapping[str, Any], name: str) -> int:
    value = metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Pair-table metadata.{name} must be a positive integer.")
    return value

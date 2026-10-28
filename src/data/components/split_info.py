"""Record and check which data split a cache was built with.

The train/valid/test split of the zerobias data (and the valid/test split of every
auxiliary sample) is drawn once with ``data.data_mlready.seed`` and frozen under
``mlready/<name>/splits/``. The frozen files used to hold only the event indices,
so a run asking for another seed under the same cache name silently got the old
split. Two records close that gap:

* every frozen ``.npz`` stores the seed and fractions it was drawn with, and
  ``check_frozen`` refuses to reuse it for different settings;
* ``split_info.yaml`` next to the frozen splits is written by
  ``src/prepare_split.py`` once the whole cache (split, normalised parquet and
  torch caches) is complete. Jobs started with ``require_split_info=true`` refuse
  to run without it, so they never build or half-build a cache themselves.

Only numpy and PyYAML are needed here.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SPLIT_INFO_FILENAME = "split_info.yaml"

#: The settings every cache written before the seed was recorded was built with:
#: configs/data/data_mlready/default.yaml has not changed them since the initial
#: commit (413f96a7). A frozen file without a record is taken to be this split.
LEGACY_SETTINGS = {"seed": 42, "train": 0.6, "valid": 0.2, "test": 0.2, "split_aux": 0.6}

_FRACTIONS = ("train", "valid", "test", "split_aux")
_NPZ_PREFIX = "meta_"


class SplitMismatchError(ValueError):
    """A frozen split or prepared cache was built with other split settings."""


class SplitNotPreparedError(RuntimeError):
    """A job requires a prepared cache that does not exist (yet)."""


def split_settings(seed: Any, split: Mapping[str, Any], split_aux: Any) -> dict[str, Any]:
    """The values that decide a split, in the form stored and compared here."""

    return {
        "seed": int(seed),
        "train": float(split["train"]),
        "valid": float(split["valid"]),
        "test": float(split["test"]),
        "split_aux": float(split_aux),
    }


def same_settings(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return int(left["seed"]) == int(right["seed"]) and all(
        math.isclose(float(left[key]), float(right[key]), abs_tol=1e-12) for key in _FRACTIONS
    )


def describe(settings: Mapping[str, Any]) -> str:
    return (f"seed {int(settings['seed'])}, train/valid/test "
            f"{float(settings['train']):g}/{float(settings['valid']):g}/{float(settings['test']):g}, "
            f"aux valid {float(settings['split_aux']):g}")


def npz_fields(settings: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Extra arrays saved into a frozen split ``.npz``."""

    fields = {f"{_NPZ_PREFIX}seed": np.asarray(int(settings["seed"]), dtype=np.int64)}
    fields.update({f"{_NPZ_PREFIX}{key}": np.asarray(float(settings[key])) for key in _FRACTIONS})
    return fields


def stored_settings(frozen) -> dict[str, Any] | None:
    """Settings recorded in a loaded ``.npz`` (``np.load`` result), or None if absent."""

    if f"{_NPZ_PREFIX}seed" not in frozen.files:
        return None
    settings = {"seed": int(frozen[f"{_NPZ_PREFIX}seed"])}
    settings.update({key: float(frozen[f"{_NPZ_PREFIX}{key}"]) for key in _FRACTIONS})
    return settings


def check_frozen(frozen, requested: Mapping[str, Any], path: Path) -> None:
    """Refuse a frozen split drawn with other settings than ``requested``.

    A file without a record predates it and is the legacy seed-42 60/20/20 split.
    """

    stored = stored_settings(frozen)
    legacy = stored is None
    if legacy:
        stored = LEGACY_SETTINGS
    if same_settings(stored, requested):
        return
    origin = "it predates the split record, so it is the original" if legacy else "it was drawn with"
    raise SplitMismatchError(
        f"Frozen split {path}: {origin} {describe(stored)}, but this run asks for "
        f"{describe(requested)}. A different split needs its own cache: set "
        f"data.data_mlready.split_suffix (scripts/physics/prepare_split.sh does this)."
    )


def check_partition(parts: Sequence[np.ndarray], expected_total: int | None = None) -> list[int]:
    """Check that index arrays are disjoint and together cover 0..n-1; return their sizes."""

    sizes = [int(len(part)) for part in parts]
    joined = np.concatenate([np.asarray(part, dtype=np.int64) for part in parts])
    total = int(joined.size)
    if expected_total is not None and total != int(expected_total):
        raise SplitMismatchError(f"The split covers {total} events, expected {expected_total}.")
    if total and (joined.min() < 0 or not np.array_equal(np.sort(joined), np.arange(total))):
        unique = int(np.unique(joined).size)
        raise SplitMismatchError(
            f"The split is not a partition of 0..{total - 1}: {total - unique} duplicated "
            f"or out-of-range indices."
        )
    return sizes


def index_digest(indices: np.ndarray) -> str:
    """SHA-256 of the indices as int64, to recognise a split later."""

    return hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int64).tobytes()).hexdigest()


def overlap_fraction(part: np.ndarray, reference: np.ndarray) -> float:
    """Fraction of ``part`` that is also in ``reference``."""

    if len(part) == 0:
        return float("nan")
    return float(np.isin(part, reference, assume_unique=True).mean())


def split_info_path(splits_dir: Path) -> Path:
    return Path(splits_dir) / SPLIT_INFO_FILENAME


def write_split_info(splits_dir: Path, record: Mapping[str, Any]) -> Path:
    """Write ``split_info.yaml`` atomically (temporary file, then rename)."""

    import yaml

    path = split_info_path(splits_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(yaml.safe_dump(dict(record), sort_keys=False))
    temporary.replace(path)
    return path


def read_split_info(splits_dir: Path) -> dict[str, Any] | None:
    import yaml

    path = split_info_path(splits_dir)
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text()) or {}


def require_split_info(splits_dir: Path, requested: Mapping[str, Any], cache_name: str) -> dict[str, Any]:
    """The prepared-cache record, or an error telling how to build it."""

    record = read_split_info(splits_dir)
    if record is None:
        raise SplitNotPreparedError(
            f"The data cache '{cache_name}' has not been prepared for {describe(requested)}: "
            f"{split_info_path(splits_dir)} is missing. Build it once before training with "
            f"DATA_SPLIT_SEED={int(requested['seed'])} bash scripts/physics/prepare_split.sh "
            f"(on lxplus: condor_submit batch/prepare_split.sub)."
        )
    path = split_info_path(splits_dir)
    settings = record.get("settings")
    if not settings:
        raise SplitMismatchError(f"{path} records no split settings.")
    if not same_settings(settings, requested):
        raise SplitMismatchError(
            f"{path} records {describe(settings)}, but this run asks for {describe(requested)}."
        )
    if record.get("status") != "complete":
        raise SplitNotPreparedError(
            f"{path} says the preparation of '{cache_name}' is {record.get('status')!r}, "
            f"not 'complete'. Rerun scripts/physics/prepare_split.sh."
        )
    return record


def iter_frozen(splits_dir: Path) -> Iterable[Path]:
    """The frozen split files of a cache, zerobias first."""

    splits_dir = Path(splits_dir)
    main = splits_dir / "zerobias.npz"
    if main.is_file():
        yield main
    yield from sorted(splits_dir.glob("aux__*.npz"))

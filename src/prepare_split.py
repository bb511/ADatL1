"""Build and check the data cache of a new train/valid/test split, once, before a study.

Run it through the wrapper, which sets the split from DATA_SPLIT_SEED::

    DATA_SPLIT_SEED=666 bash scripts/physics/prepare_split.sh    # local
    condor_submit batch/prepare_split.sub                        # lxplus

For ``data.data_mlready.seed = DATA_SPLIT_SEED`` and the cache
``<processor>_default_split<seed>`` it

1. runs ``prepare_data()``: draws and freezes the split (the frozen ``.npz``
   records seed and fractions), fits the normaliser on the NEW training set and
   writes the normalised parquet of train/valid/test and of the valid/test halves
   of every auxiliary sample;
2. builds every torch cache (``torch_cache.pt`` and friends), one folder at a
   time, so training jobs only ever read;
3. checks that train/valid/test and every auxiliary valid/test pair are disjoint
   and cover all events, and measures how much of the new split overlaps the
   original seed-42 one (about 60 % of the new training events were training
   events before; an identical split is an error);
4. writes ``splits/split_info.yaml`` with ``status: complete``.

Jobs started with ``DATA_SPLIT_SEED`` (``data.data_mlready.require_split_info``)
refuse to run until step 4 is done. While this runs the record says
``in_progress``, so a job submitted too early fails at once instead of building
the same files concurrently.

Rerunning is safe: finished parquet and torch caches are reused and the checks
are repeated, so an interrupted preparation is completed by running it again.
The weight seed and the model are not involved; only ``cfg.data`` is used.
"""

from __future__ import annotations

import gc
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.omegaconf import register_resolvers  # noqa: E402

register_resolvers()

from src.data.components import split_info as split_record  # noqa: E402
from src.utils import RankedLogger  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)

MAIN_SPLITS = ("train", "valid", "test")
AUX_SPLITS = ("valid", "test")


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - no git in the container is fine
        return "unknown"


def _load_indices(path: Path, keys: tuple[str, ...]) -> list[np.ndarray]:
    with np.load(path) as frozen:
        return [np.asarray(frozen[f"i{key}"]) for key in keys]


def check_splits(splits_dir: Path, reference_dir: Path | None) -> dict[str, Any]:
    """Partition checks of every frozen split, and the overlap with the reference split."""

    splits_dir = Path(splits_dir)
    main_file = splits_dir / "zerobias.npz"
    if not main_file.is_file():
        raise split_record.SplitNotPreparedError(f"No frozen zerobias split at {main_file}.")
    main = _load_indices(main_file, MAIN_SPLITS)
    sizes = split_record.check_partition(main)
    record: dict[str, Any] = {
        "zerobias": {
            "events": int(sum(sizes)),
            **dict(zip(MAIN_SPLITS, sizes)),
            "sha256": {name: split_record.index_digest(part) for name, part in zip(MAIN_SPLITS, main)},
        },
        "aux": {},
    }
    for path in sorted(splits_dir.glob("aux__*.npz")):
        parts = _load_indices(path, AUX_SPLITS)
        record["aux"][path.stem[len("aux__"):]] = dict(
            zip(AUX_SPLITS, split_record.check_partition(parts))
        )

    reference = None if reference_dir is None else Path(reference_dir) / "zerobias.npz"
    if reference is not None and reference.is_file():
        ref_train, ref_valid, ref_test = _load_indices(reference, MAIN_SPLITS)
        train, valid, test = main
        if len(train) == len(ref_train) and np.array_equal(np.sort(train), np.sort(ref_train)):
            raise split_record.SplitMismatchError(
                f"The new training set is identical to the reference one ({reference})."
            )
        record["overlap_with_reference"] = {
            "reference": str(reference),
            "train_in_reference_train": split_record.overlap_fraction(train, ref_train),
            "valid_in_reference_train": split_record.overlap_fraction(valid, ref_train),
            "test_in_reference_train": split_record.overlap_fraction(test, ref_train),
            "test_in_reference_test": split_record.overlap_fraction(test, ref_test),
            "valid_in_reference_valid": split_record.overlap_fraction(valid, ref_valid),
        }
    else:
        record["overlap_with_reference"] = None
    return record


def build_torch_caches(datamodule) -> list[str]:
    """Load every folder once, so its torch caches exist; drop the tensors right away."""

    loader = datamodule.hparams.data_awkward2torch
    cache = Path(datamodule.main_cache_folder)
    folders = [cache / split for split in MAIN_SPLITS]
    aux_dir = cache / "aux"
    if aux_dir.is_dir():
        for dataset in sorted(p for p in aux_dir.iterdir() if p.is_dir() and not p.name.startswith("._")):
            folders += [dataset / split for split in AUX_SPLITS]
    for number, folder in enumerate(folders, start=1):
        log.info(f"[{number}/{len(folders)}] torch cache of {folder.relative_to(cache)}")
        tensors = loader.load_folder(folder)
        del tensors
        gc.collect()
    return [str(folder.relative_to(cache)) for folder in folders]


def _reference_splits_dir(mlready_dir: Path, name: str, suffix: str) -> Path | None:
    if not suffix or not name.endswith(suffix):
        return None
    return mlready_dir / name[: -len(suffix)] / "splits"


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    ready_cfg = cfg.data.data_mlready
    if not ready_cfg.get("split_suffix"):
        raise SystemExit(
            "prepare_split builds the cache of a NEW split and needs "
            "data.data_mlready.split_suffix (DATA_SPLIT_SEED in "
            "scripts/physics/prepare_split.sh sets it). The original seed-42 cache "
            "is never touched."
        )
    requested = split_record.split_settings(ready_cfg.seed, ready_cfg.split, ready_cfg.split_aux)
    # This job is the one that builds the cache, so it must not require it.
    with open_dict(cfg):
        cfg.data.data_mlready.require_split_info = False

    mlready_dir = Path(str(ready_cfg.cache_root_dir)) / "mlready"
    name = str(ready_cfg.name)
    splits_dir = mlready_dir / name / "splits"
    existing = split_record.read_split_info(splits_dir)
    if existing and existing.get("settings") and not split_record.same_settings(existing["settings"], requested):
        raise split_record.SplitMismatchError(
            f"{split_record.split_info_path(splits_dir)} records "
            f"{split_record.describe(existing['settings'])}, not {split_record.describe(requested)}."
        )
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prepared_by = {
        "host": socket.gethostname(),
        "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "started": started,
    }
    split_record.write_split_info(splits_dir, {
        "status": "in_progress", "settings": requested, "cache_name": name, "prepared_by": prepared_by,
    })
    log.info(f"Preparing cache '{name}' for {split_record.describe(requested)}")

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    mlready = datamodule.hparams.data_mlready
    if Path(mlready.splits_dir) != splits_dir:
        raise RuntimeError(f"Unexpected splits folder {mlready.splits_dir}, expected {splits_dir}.")

    torch_caches = build_torch_caches(datamodule)
    checks = check_splits(splits_dir, _reference_splits_dir(mlready_dir, name, str(ready_cfg.split_suffix)))

    prepared_by["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record = {
        "status": "complete",
        "settings": requested,
        "cache_name": name,
        "cache_folder": str(datamodule.main_cache_folder),
        "normalizer": str(datamodule.normalizer.name),
        **checks,
        "torch_caches": torch_caches,
        "prepared_by": prepared_by,
    }
    path = split_record.write_split_info(splits_dir, record)

    seed = requested["seed"]
    overlap = checks.get("overlap_with_reference") or {}
    log.info(f"Split seed {seed} prepared: {path}")
    log.info(
        f"zerobias train/valid/test = {checks['zerobias']['train']}/{checks['zerobias']['valid']}/"
        f"{checks['zerobias']['test']}, {len(checks['aux'])} auxiliary samples"
    )
    if overlap:
        log.info(
            f"new train in seed-42 train: {overlap['train_in_reference_train']:.3f}, "
            f"new test in seed-42 train: {overlap['test_in_reference_train']:.3f}"
        )
    print(
        "\nTrain a study on this split with, e.g.\n"
        f"  condor_submit EXPERIMENT_NAME=Pareto-Front-261015-split{seed} DATA_SPLIT_SEED={seed} "
        "batch/runae_pareto.sub\n"
        f"and pass the same EXPERIMENT_NAME and DATA_SPLIT_SEED={seed} to runprobes_pareto.sub.",
        flush=True,
    )


if __name__ == "__main__":
    main()

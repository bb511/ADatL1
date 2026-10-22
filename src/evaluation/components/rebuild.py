"""A dumped run rebuilt from its configuration, and what its evaluation takes from
the rebuilt datamodule and callbacks: the signal and null sets, the loader layouts
and the SDS pseudo-sets, cached across the runs of a worker."""

import hashlib
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule

from src.callbacks.score_dump import _layout
from src.callbacks.sds import sds_inputs
from src.evaluation.callbacks.efficiency import AnomalyEfficiencyCallback

ROOT = Path(__file__).resolve().parents[3]
DATA_CONFIGS = ROOT / "configs" / "data"


def rebuild(run_dir: Path, data_dir: str | None, sets: str | None = None) -> tuple:
    """The run's module and datamodule, instantiated from its ``config.yaml``.

    With ``sets``, the ``signals`` and ``nulls`` of ``configs/data/<sets>.yaml`` replace
    the run's; the rest of the run's data configuration is kept.
    """
    cfg = OmegaConf.load(run_dir / "config.yaml")
    if data_dir is not None:
        cfg.data.data_dir = str(data_dir)
    if sets is not None:
        lists = OmegaConf.load(DATA_CONFIGS / f"{sets}.yaml")
        cfg.data.signals, cfg.data.nulls = lists.signals, lists.get("nulls")
    return hydra.utils.instantiate(cfg.algorithm), hydra.utils.instantiate(cfg.data)


# One entry, kept across the runs a worker evaluates: the runs of a lattice share their
# training inputs, and building the pseudo-sets costs a nearest-neighbour search (on
# CIFAR-10, 75 s and 6 GB), so LatticeEvaluation builds them once in the parent and the
# pool's initialiser copies them into every worker. A module global, since it outlives
# each RunEvaluation of a worker and the initialiser, pickled by name, reaches no
# instance.
SDS_CACHE: dict = {}


def cached_sds(x_train: torch.Tensor, max_rows: int, row_seed: int) -> dict:
    digest = hashlib.sha1(x_train.contiguous().numpy().tobytes()).hexdigest()
    key = (digest, tuple(x_train.shape), max_rows, row_seed)
    if key not in SDS_CACHE:
        SDS_CACHE.clear()
        SDS_CACHE[key] = sds_inputs(x_train, max_rows, row_seed)
    return SDS_CACHE[key]


def signal_sets(layout: dict) -> list[str]:
    """The sets whose labels are all positive, the efficiency callback's signals."""
    return [name for name in layout["names"] if min(layout["labels"][name]) > 0]


def loader_layouts(datamodule: LightningDataModule) -> dict[str, dict]:
    """The validation and test layouts of a set-up datamodule, in meta.pt's form."""
    loaders = {"val": datamodule.val_dataloader(), "test": datamodule.test_dataloader()}
    return {split: _layout(split_loaders) for split, split_loaders in loaders.items()}


def null_sets(callbacks: list) -> list[str]:
    efficiency = [cb for cb in callbacks if isinstance(cb, AnomalyEfficiencyCallback)]
    return list(efficiency[0].null_ds) if efficiency else []

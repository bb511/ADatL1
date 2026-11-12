"""A dumped run rebuilt from its configuration, and what its evaluation takes from it.

From the rebuilt datamodule and callbacks: the signal and null sets, the loader
layouts, the row-capped loaders, and the SDS pseudo-sets and the points files, each
cached across the runs of a worker. Before any datamodule is built, the check of a hub
run's record cache (``check_record_cache``).
"""

import hashlib
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler, TensorDataset

from src.callbacks.components.points import load_points
from src.callbacks.components.utils import row_subset
from src.callbacks.score_dump import _layout
from src.callbacks.sds import SDS_PUSH_SCALE, sds_inputs
from src.data.utils import BatchView, unpack_batch
from src.evaluation.callbacks.efficiency import AnomalyEfficiencyCallback

ROOT = Path(__file__).resolve().parents[3]
DATA_CONFIGS = ROOT / "configs" / "data"
# The L1AD datamodules, whose sets the data pipeline fixes: they have no signals and
# nulls lists for ``sets`` to replace.
L1AD_DATAMODULES = (
    "src.data.L1AD_datamodule.L1ADDataModule",
    "src.data.L1AD_HF_datamodule.L1ADHFDataModule",
)
# The datamodules that download a record of the hub: LatticeEvaluation prepares it
# once in its own process and spawns its workers offline.
HUB_DATAMODULES = ("src.data.L1AD_HF_datamodule.L1ADHFDataModule",)


def _data_target(cfg) -> str | None:
    return OmegaConf.select(cfg, "data._target_", throw_on_resolution_failure=False)


def reads_hub(run_dir: Path) -> bool:
    """Whether the run's datamodule reads a record of the hub."""
    return _data_target(OmegaConf.load(run_dir / "config.yaml")) in HUB_DATAMODULES


def _replace_data_dir(cfg, data_dir: str) -> None:
    # The raw L1AD datamodule reads its paths from the paths config, and the hub one
    # keeps the record's derived caches under cache_dir.
    target = _data_target(cfg)
    if target == L1AD_DATAMODULES[0]:
        raise ValueError("data_dir cannot replace the paths of the raw L1AD data.")
    cfg.data["cache_dir" if target in HUB_DATAMODULES else "data_dir"] = str(data_dir)


# The record's simulated categories, cached per data set under its cache_dir as its
# loader lays them out (adl1t_datamaker 97b465d, publish/assets/loader):
# <data>/extracted/<stage>/<category>/<set>, <data>/processed/<stage>/<category>/<set>
# and <data>/mlready/<stage>/<normaliser>/aux/<set>, <data> being its paths.data_dir.
SIMULATED = ("background", "signal")


def _simulated_sets(cache: Path) -> list[Path]:
    staged = [f"*/{s}/*/{c}/*" for s in ("extracted", "processed") for c in SIMULATED]
    globs = [*staged, "*/mlready/*/*/aux/*"]
    return sorted(p for glob in globs for p in cache.glob(glob) if p.is_dir())


def _record_cache(run_dir: Path, cfg, data_dir: str | None) -> Path:
    cache = data_dir or OmegaConf.select(cfg, "data.cache_dir")
    if cache is None:
        raise ValueError(f"{run_dir.name} names no record cache to check.")
    return Path(cache)


def check_record_cache(
    run_dir: Path, data_dir: str | None, val_sets: list[str] | None
) -> None:
    """Refuse a hub run's record cache that holds a simulated set outside ``val_sets``.

    The record's reader loads every simulated set of its ml-ready cache, and its
    pipeline carries every set of an earlier stage on to the next, so a set outside
    ``val_sets`` at any stage would be read. Directory names alone are listed, before
    any datamodule is built; no file of the cache is opened. The cache is
    ``data_dir``, else the run's ``data.cache_dir``. A run of another datamodule, or
    ``val_sets`` None, passes unchecked.

    Raises:
        ValueError: Naming the cache and the sets it holds outside ``val_sets``, or
            where the run names no cache.
    """
    cfg = OmegaConf.load(run_dir / "config.yaml")
    if val_sets is None or _data_target(cfg) not in HUB_DATAMODULES:
        return
    cache = _record_cache(run_dir, cfg, data_dir)
    found = _simulated_sets(cache)
    outside = [str(p.relative_to(cache)) for p in found if p.name not in val_sets]
    if outside:
        raise ValueError(
            f"The record cache {cache} holds sets outside val_sets: {outside}."
        )


def _add_record_overrides(cfg, data_overrides: list[str]) -> None:
    if not data_overrides:
        return
    if _data_target(cfg) not in HUB_DATAMODULES:
        raise ValueError("data_overrides apply to the hub datamodule's record alone.")
    cfg.data.overrides = [*(cfg.data.get("overrides") or []), *data_overrides]


def _replace_sets(cfg, sets: str | None) -> None:
    if sets is None:
        return
    if _data_target(cfg) in L1AD_DATAMODULES:
        raise ValueError("sets cannot replace the sets of the L1AD data pipeline.")
    lists = OmegaConf.load(DATA_CONFIGS / f"{sets}.yaml")
    cfg.data.signals, cfg.data.nulls = lists.signals, lists.get("nulls")


def _empty_signals(cfg, empty_signals: bool) -> None:
    if not empty_signals:
        return
    if _data_target(cfg) in L1AD_DATAMODULES:
        raise ValueError(
            "empty_signals cannot apply to the L1AD data: data_overrides compose its "
            "record without signal sets."
        )
    if "signals" not in cfg.data:
        raise ValueError(f"empty_signals: {_data_target(cfg)} takes no signals.")
    cfg.data.signals = {}


def rebuild(
    run_dir: Path,
    data_dir: str | None,
    sets: str | None = None,
    data_overrides: list[str] | None = None,
    empty_signals: bool = False,
) -> tuple:
    """The run's module and datamodule, instantiated from its ``config.yaml``.

    With ``data_dir``, the data config's ``data_dir`` is replaced (``cache_dir`` for
    the hub datamodule of the L1AD data). With ``sets``, the ``signals`` and ``nulls``
    of ``configs/data/<sets>.yaml`` replace the run's; the rest of the run's data
    configuration is kept. Both raise ValueError where the L1AD data has no such field.
    With ``empty_signals``, ``signals`` is then emptied, so that the datamodule reads
    no signal set; it raises ValueError for the L1AD data and for a data config without
    ``signals``. ``data_overrides`` are appended to the hub datamodule's ``overrides``,
    those of the record's own configuration (``data.signal={}`` composes the record
    without its signal sets), and raise ValueError for any other datamodule.
    """
    cfg = OmegaConf.load(run_dir / "config.yaml")
    if data_dir is not None:
        _replace_data_dir(cfg, data_dir)
    _replace_sets(cfg, sets)
    _empty_signals(cfg, empty_signals)
    _add_record_overrides(cfg, list(data_overrides or []))
    return hydra.utils.instantiate(cfg.algorithm), hydra.utils.instantiate(cfg.data)


# The pseudo-sets kept across the runs a worker evaluates: the runs of a lattice share
# their training inputs and validation normal, and building the pseudo-sets costs a
# nearest-neighbour search (on CIFAR-10, 75 s and 6 GB), so LatticeEvaluation builds
# them once in the parent and the pool's initialiser copies them into every worker. A
# module global, since it outlives each RunEvaluation of a worker and the initialiser,
# pickled by name, reaches no instance. At most two entries: the pseudo-sets of the
# training rows and of validation normal.
SDS_CACHE: dict = {}
SDS_CACHE_SIZE = 2


def _digest(tensor: torch.Tensor | None) -> str:
    # Hashed through the buffer protocol: tobytes() would copy the training inputs,
    # 5.9 GB on the L1AD data.
    if tensor is None:
        return ""
    return hashlib.sha1(tensor.contiguous().numpy()).hexdigest()


def cached_sds(
    x: torch.Tensor,
    max_rows: int,
    row_seed: int,
    mask: torch.Tensor | None = None,
    push_scale: float = SDS_PUSH_SCALE,
) -> dict:
    """``sds_inputs`` of the rows x and their data mask, through ``SDS_CACHE``."""
    key = (_digest(x), _digest(mask), tuple(x.shape), max_rows, row_seed)
    if push_scale != SDS_PUSH_SCALE:
        key += (push_scale,)
    if key not in SDS_CACHE:
        if len(SDS_CACHE) >= SDS_CACHE_SIZE:
            SDS_CACHE.clear()
        SDS_CACHE[key] = sds_inputs(x, max_rows, row_seed, mask, push_scale)
    return SDS_CACHE[key]


# The points files of ``extra_sets.points``, keyed by path and declared md5s: loaded and
# checked once per lattice, in the process that maps the runs, whose pool initialiser
# copies them into every worker, as SDS_CACHE.
POINTS_CACHE: dict = {}


def cached_points(points: dict | None) -> dict[str, tuple]:
    """``load_points`` of every entry ``{name: {path, md5_x, md5_mask}}`` of
    ``extra_sets.points``, through ``POINTS_CACHE``."""
    loaded = {}
    for name, spec in (points or {}).items():
        key = (str(spec["path"]), spec["md5_x"], spec.get("md5_mask"))
        if key not in POINTS_CACHE:
            POINTS_CACHE[key] = load_points(*key)
        loaded[name] = POINTS_CACHE[key]
    return loaded


def loader_inputs(loader: DataLoader) -> BatchView:
    """The inputs, labels and data masks of every batch of a loader, concatenated."""
    views = [unpack_batch(batch) for batch in loader]
    masks = [view.mask for view in views]
    return BatchView(
        x=torch.cat([view.x for view in views]),
        y=torch.cat([view.y for view in views]),
        mask=None if masks[0] is None else torch.cat(masks),
    )


def capped_loader(
    loader: DataLoader, max_rows: int | None, seed: int
) -> tuple[DataLoader, torch.Tensor | None]:
    """A loader of the ``row_subset(n, max_rows, seed)`` rows of a loader's n, and the
    sorted row indices; the loader itself and None where the cap keeps every row.

    The rows keep their order and are batched with the size of the loader's first
    batch; the dataset carries the loader's ``object_feature_map``, if any.
    """
    batches = list(loader)
    tensors = [torch.cat(parts) for parts in zip(*batches)]
    rows = row_subset(len(tensors[0]), max_rows, seed)
    if rows is None:
        return loader, None
    dataset = TensorDataset(*(tensor[rows] for tensor in tensors))
    feature_map = getattr(loader.dataset, "object_feature_map", None)
    if feature_map is not None:
        dataset.object_feature_map = feature_map
    # Whole batches indexed at once: a per-row collate costs 16.5 us a row on L1AD.
    size = len(unpack_batch(batches[0]).y)
    sampler = BatchSampler(SequentialSampler(dataset), size, drop_last=False)
    return DataLoader(dataset, sampler=sampler, batch_size=None), rows


def capped_meta(meta: dict, loaders: dict, rows: dict) -> dict:
    """``meta`` with the batch sizes of every capped set those of its capped loader."""
    meta = {**meta}
    for split, capped in rows.items():
        sizes = {**meta[split]["batch_sizes"]}
        for name in capped:
            sizes[name] = [len(batch) for batch in loaders[split][name].sampler]
        meta[split] = {**meta[split], "batch_sizes": sizes}
    return meta


def capped_record(record: dict, rows: dict) -> dict:
    """An epoch's record with the stored scores of every capped set at its rows."""
    record = {**record}
    for split, capped in rows.items():
        stored = {**record[split]}
        for name in capped.keys() & stored.keys():
            stored[name] = {k: v[capped[name]] for k, v in stored[name].items()}
        record[split] = stored
    return record


def signal_sets(layout: dict) -> list[str]:
    """The sets whose labels are all positive, the efficiency callback's signals."""
    return [name for name in layout["names"] if min(layout["labels"][name]) > 0]


def loader_layouts(
    datamodule: LightningDataModule, splits: tuple[str, ...] = ("val", "test")
) -> dict[str, dict]:
    """The layouts of a set-up datamodule's loaders of ``splits``, in meta.pt's form."""
    loaders = {"val": datamodule.val_dataloader, "test": datamodule.test_dataloader}
    return {split: _layout(loaders[split]()) for split in splits}


def null_sets(callbacks: list) -> list[str]:
    efficiency = [cb for cb in callbacks if isinstance(cb, AnomalyEfficiencyCallback)]
    return list(efficiency[0].null_ds) if efficiency else []

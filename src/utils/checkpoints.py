from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Iterator
from pathlib import Path
from collections import defaultdict
import re

import rootutils

ROOTDIR = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

_FILENAME_RE = re.compile(
    r"^ds=(?P<dataset>.+?)__value=(?P<metric_value>.+?)__epoch=(?P<epoch>\d+)\.ckpt$"
)


def _parse_filename(filename: str) -> Dict[str, str]:
    m = _FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"Unexpected filename format: {filename}")
    return {key: m.group(key) for key in ["dataset", "metric_value", "epoch"]}


PREFIXES = ("single", "loo", "lko", "cap")
CRITERIA = ("min", "max", "last", "stable")


def is_valid_ckpt(p: Path) -> bool:
    return p.is_file() and p.suffix == ".ckpt" and not p.name.startswith("._")


def _find_ckpt_files(
    dirpath: Path,
    list_prefix: Optional[List[str]] = [],
    list_metric_name: Optional[List[str]] = [],
    list_criterion: Optional[List[str]] = [],
) -> Iterator[Dict[str, Union[str, Path]]]:
    """
    Yield checkpoint descriptors:
        {
            "prefix": <prefix>,
            "metric_name": <metric_name>,
            "criterion": <criterion>,
            "path": Path,
        }
    from:
        <prefix>/<metric_name>/<criterion>/filename.ckpt
    """
    list_prefix = list_prefix if len(list_prefix) > 0 else list(PREFIXES)
    list_criterion = list_criterion if len(list_criterion) > 0 else list(CRITERIA)

    for prefix in list_prefix:
        prefix_dir = dirpath / prefix
        if not prefix_dir.is_dir():
            continue

        for metric_dir in prefix_dir.iterdir():
            if not metric_dir.is_dir():
                continue

            metric_name = metric_dir.name
            if len(list_metric_name) > 0 and metric_name not in list_metric_name:
                continue

            for criterion in list_criterion:
                criterion_dir = metric_dir / criterion
                if not criterion_dir.is_dir():
                    continue

                for ckpt_path in criterion_dir.glob("*.ckpt"):
                    if not is_valid_ckpt(ckpt_path):
                        continue

                    yield {
                        "prefix": prefix,
                        "metric_name": metric_name,
                        "criterion": criterion,
                        "path": ckpt_path,
                    }


def _iterate_checkpoints(
    dirpath: Path,
    include_prefix: Optional[List[str]] = None,
    exclude_prefix: Optional[List[str]] = None,
    include_ds: Optional[List[str]] = None,
    exclude_ds: Optional[List[str]] = None,
) -> Iterator[Dict[str, Union[str, Path]]]:
    """
    Internal helper used by find_scan_checkpoints.
    If no filters are provided, behavior is identical to calling _iter_ckpt_files(dirpath).
    """
    exclude_prefix = exclude_prefix or []
    list_prefix = include_prefix or list(PREFIXES)
    list_prefix = [prefix for prefix in list_prefix if prefix not in exclude_prefix]

    include_ds_set = set(include_ds) if include_ds else None
    exclude_ds_set = set(exclude_ds) if exclude_ds else None
    for comb in _find_ckpt_files(
        dirpath=dirpath,
        list_prefix=list_prefix,
    ):
        path = comb["path"]
        ds = _parse_filename(path.name)["dataset"]

        if include_ds_set is not None and ds not in include_ds_set:
            continue
        if exclude_ds_set is not None and ds in exclude_ds_set:
            continue

        yield comb


def find_checkpoints(
    dirpath: str,
    experiment_name: str,
    run_name: str,
    by_combination: bool = True,
    include_prefix: Optional[List[str]] = None,
    exclude_prefix: Optional[List[str]] = None,
    include_ds: Optional[List[str]] = None,
    exclude_ds: Optional[List[str]] = None,
) -> Dict[Union[int, Tuple[str, str, str, str]], List[str]]:
    """
    :param dirpath: Base directory containing experiment subdirectories.
    :param experiment_name: Name given to the experiment.
    :param run_name: Name given to the run.
    :param filter_groups: Optional callable to filter <prefix> subdirectories.
    :param filter_checkpoint: Optional callable to filter checkpoint files (by absolute path string).
    :param by_combination: Instead of retrieving just the set of checkpoints, retrieve the set of hparams.

    # Filters:
    :param include_prefix: List prefixes to include. If None, all PREFIXES will be included.
    :param include_ds: List of dset_key to include. If None, all will be included.
    :param exclude_ds: List of dset_key to exclude. If None, all will be included.
    """

    dirpath = Path(dirpath).expanduser().resolve()
    run_dir = (dirpath / experiment_name / run_name).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if by_combination is False:
        by_epoch: Dict[int, List[str]] = {}
        for comb in _iterate_checkpoints(
            run_dir, include_prefix, exclude_prefix, include_ds, exclude_ds
        ):
            path = comb.get("path")
            epoch = _parse_filename(path.name).get("epoch")
            by_epoch.setdefault(int(epoch), []).append(str(path))

        for paths in by_epoch.values():
            paths.sort()
        return by_epoch

    # By (prefix, metric_name, criterion, dataset)
    combinations: Dict[Tuple[str, str, str, str], List[str]] = defaultdict(list)
    for comb in _iterate_checkpoints(
        run_dir, include_prefix, exclude_prefix, include_ds, exclude_ds
    ):
        prefix = comb.get("prefix")
        metric_name = comb.get("metric_name")
        criterion = comb.get("criterion")
        dataset = _parse_filename(comb.get("path").name).get("dataset")
        combinations[(prefix, metric_name, criterion, dataset)].append(
            str(comb.get("path"))
        )

    for k in combinations:
        combinations[k].sort()
    return dict(combinations)

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Union, Iterable, Iterator
from pathlib import Path
from collections import defaultdict
import re

import rootutils
ROOTDIR = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

_FILENAME_RE = re.compile(
    r'^ds=(?P<dataset>.+?)__value=(?P<metric_value>.+?)__epoch=(?P<epoch>\d+)\.ckpt$'
)

def _parse_filename(filename: str) -> Dict[str, str]:
    m = _FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"Unexpected filename format: {filename}")
    return {
        key: m.group(key)
        for key in ["dataset", "metric_value", "epoch"]
    }

PREFIXES = ("single", "loo", "lko")
CRITERIA = ("min", "max", "last", "stable")

def _iter_ckpt_files(
    dirpath: Path,
    list_prefix: Optional[List[str]] = [],
    list_metric_name: Optional[List[str]] = [],
    list_criterion: Optional[List[str]] = [],
) -> Iterator[Path]:
    """
    Yield *.ckpt files under:
        <prefix>/<metric_name>/<criterion>/filename.ckpt
    restricted to known criteria.
    """
    
    list_prefix = list_prefix if len(list_prefix) > 0 else list(PREFIXES)
    list_criterion = list_criterion if len(list_criterion) > 0 else list(CRITERIA)
    for prefix in list_prefix:
        prefix_dir = dirpath / prefix
        if not prefix_dir.is_dir():
            continue

        for metric_dir in prefix_dir.iterdir():
            metric_name = metric_dir.name
            if len(list_metric_name) > 0 and metric_name not in list_metric_name:
                continue
            
            for criterion in list_criterion:
                criterion_dir = metric_dir / criterion
                if not criterion_dir.is_dir():
                    continue

                for ckpt_path in criterion_dir.glob("*.ckpt"):
                    yield {
                        "prefix": prefix,
                        "metric_name": metric_name,
                        "criterion": criterion,
                        "path": ckpt_path,
                    }


def find_scan_checkpoints(
    dirpath: str,
    experiment_name: str,
    run_name: str,
    by_combination: bool = True
) -> Dict[Union[int, Tuple[str, str, str]], List[Union[str, Path]]]:
    """
    :param dirpath: Base directory containing experiment subdirectories.
    :param run_name: Name given to the experiment.
    :param run_name: Name given to the run.
    :param filter_groups: Optional callable to filter <prefix> subdirectories.
    :param filter_checkpoint: Optional callable to filter checkpoint files (by absolute path string).
    :param by_combination: Instead of retrieving just the set of checkpoints, retrieve the set of hparams.
    """

    dirpath = Path(dirpath).expanduser().resolve()
    run_dir = (dirpath / experiment_name / run_name).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if by_combination == False:
        by_epoch: Dict[int, List[str]] = {}
        for comb in _iter_ckpt_files(run_dir):
            path = comb.get("path")
            epoch = _parse_filename(path.name).get("epoch")
            by_epoch.setdefault(int(epoch), []).append(str(path))

        # For reproducibility:
        for paths in by_epoch.values():
            paths.sort()
        return by_epoch

    # Else: By (prefix, metric_name, criterion, dataset)
    combinations: Dict[Tuple[str, str, str, str], List[str]] = defaultdict(list)
    for comb in _iter_ckpt_files(run_dir):
        prefix = comb.get("prefix")
        metric_name = comb.get("metric_name")
        criterion = comb.get("criterion")
        dataset = _parse_filename(comb.get("path").name).get("dataset")
        combinations[(prefix, metric_name, criterion, dataset)].append(str(comb.get("path")))

    # For reproducibility:
    for k in combinations:
        combinations[k].sort()
    return dict(combinations)


if __name__ == "__main__":
    import os
    dirpath = os.path.join(ROOTDIR, "checkpoints")
    experiment_name = "axov4"
    run_name = "test"
    checkpoint_dict = find_scan_checkpoints(
        dirpath=dirpath,
        experiment_name=experiment_name,
        run_name=run_name,
        by_combination=True,
    )
    import ipdb; ipdb.set_trace()
from __future__ import annotations
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, Iterable
from pathlib import Path
from collections import defaultdict
from omegaconf import DictConfig
import re

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _build_scan_dirname(scan_dict: Mapping[str, Any]) -> str:
    parts = [f"{k}={scan_dict[k]}" for k in sorted(scan_dict.keys())]
    return "scan_" + "_".join(parts)


_FILENAME_RE = re.compile(
    r'^prefix=(?P<prefix>.+?)__ds=(?P<dset_key>.+?)__metric=(?P<metric_name>.+?)__value=(?P<metric_value>.+?)__epoch=(?P<epoch>\d+)__step=(?P<step>\d+)\.ckpt$'
)

def _extract_dset_prefix_metric(path: Path) -> Tuple[str, str, str]:
    """Return (dset_key, prefix, metric_name) parsed from the checkpoint filename."""
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return m.group('dset_key'), m.group('prefix'), m.group('metric_name')

def _extract_metric_value(path: Path) -> str:
    """Return metric_value parsed from the checkpoint filename."""
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return m.group('metric_value')

def _extract_epoch_step(path: Path) -> Optional[EpochStep]:
    """Return epoch and step values parsed from the checkpoint filename."""
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return int(m.group('epoch')), int(m.group('step'))


EpochStep = Tuple[int, int]
FilterGroups = Callable[[str], bool]
FilterCheckpoint = Callable[[str], bool] 

def _extract_dataset_keys(
    scan_dir: Union[str, Path],
    filter_groups: Optional[FilterGroups] = None
) -> List[Path]:

    group_dirs: List[Path] = []
    for entry in Path(scan_dir).iterdir():
        if entry.is_dir():
            if filter_groups(entry):
                group_dirs.append(entry)

    return group_dirs


def _extract_checkpoint_paths(
    dset_key: Union[str, Path],
    filter_checkpoint: Optional[FilterCheckpoint] = None
) -> Iterable[Tuple[Tuple[int, int], Path]]:

    # Only look at immediate .ckpt files inside each group (non-recursive).
    for ckpt in Path(dset_key).glob("*.ckpt"):
        if not filter_checkpoint(str(ckpt)):
            continue
        
        key = _extract_epoch_step(ckpt)
        if key is None:
            continue    
        
        epoch, step = key
        yield (epoch, step), ckpt


def find_scan_checkpoints(
    dirpath: str,
    scan: DictConfig,
    filter_groups: Optional[FilterGroups] = None,
    filter_checkpoint: Optional[FilterCheckpoint] = None,
    by_combination: bool = True
) -> Dict[EpochStep, List[str]]:
    """
    Discover checkpoints organized as:
        <dirpath>/
          scan_<k1=v1>_<k2=v2>_.../
            <groupA>/
              ... epoch=X-step=Y.ckpt
            <groupB>/
              ... epoch=X-step=Y.ckpt
            ...

    - The 'scan_*' directory name is deterministically derived from cfg.scan by sorting keys.
    - 'filter_groups' selects which group subdirectories to include.
    - 'filter_checkpoint' selects which .ckpt files inside a selected group to include.
    - The result is a dict keyed by (epoch, step) -> list of all paths (strings) that match.

    :param dirpath: Base directory containing scan subdirectories.
    :param scan: DictConfig with scan parameters to build the scan directory name.
    :param filter_groups: Optional callable to filter group subdirectories.
    :param filter_checkpoint: Optional callable to filter checkpoint files.
    :param by_combination: If False, keys the result by (epoch, step) tuples. Otherwise, by (dset_key, prefix, metric_name).
    """

    base_dir = Path(dirpath).expanduser().resolve()
    scan_dict = dict(scan)
    scan_dirname = _build_scan_dirname(scan_dict)
    scan_dir = base_dir / scan_dirname

    if not scan_dir.exists() or not scan_dir.is_dir():
        raise FileNotFoundError(f"Scan directory not found: {scan_dir}")
    
    # Load the filters:
    filter_groups = filter_groups or (lambda *_, **__: True)
    filter_checkpoint = filter_checkpoint or (lambda *_, **__: True)
    
    # Extract dataset keys:
    dset_keys = _extract_dataset_keys(
        scan_dir=scan_dir,
        filter_groups=filter_groups
    )

    if not by_combination:
        by_epoch_step: Dict[EpochStep, List[str]] = {}
        seen_paths: set[str] = set()  # prevent duplicates across groups
        for dset_key in dset_keys:
            for (epoch, step), ckpt in _extract_checkpoint_paths(
                    dset_key=dset_key,
                    filter_checkpoint=filter_checkpoint
                ):

                spath = str(ckpt)
                if spath in seen_paths:
                    continue
                seen_paths.add(spath)

                by_epoch_step.setdefault((epoch, step), []).append(spath)

        # For reproducibility:
        for paths in by_epoch_step.values():
            paths.sort()
        return by_epoch_step
    

    # Else: just provide the standard format (dset_key, prefix, metric_name)
    dset_keys = [dset_key.stem for dset_key in dset_keys if filter_groups(dset_key)]
    by_combo: Dict[Tuple[str, str, str], List[Path]] = defaultdict(list)
    for ckpt in scan_dir.glob("*/*.ckpt"):
        try:
            dset_key, prefix, metric_name = _extract_dset_prefix_metric(ckpt)
            if dset_key in dset_keys and filter_checkpoint(str(ckpt)):
                by_combo[(dset_key, prefix, metric_name)].append(ckpt)
        except ValueError:
            continue

    # For reproducibility:
    for k in by_combo:
        by_combo[k].sort()
    return dict(by_combo)

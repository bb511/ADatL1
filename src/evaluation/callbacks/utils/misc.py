# Misc methods for the evaluation callbacks.
import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path


def get_ckpt_ds_name(ckpt_name: str):
    """Gets the data set name from the checkpoint name.

    The format of the checkpoint name is specified in     callbacks/checkpointing/dataset_aware.py
    and this method expects exactly that format. If the format changes, this method will stop
    working.
    """
    ckpt_ds = ckpt_name.split("ds=")
    if len(ckpt_ds) > 1:
        ckpt_ds = ckpt_ds[1].split("__")[0]
    else:
        ckpt_ds = ckpt_ds[0]

    return ckpt_ds


def to_plain_dict(d: defaultdict | dict):
    if isinstance(d, defaultdict):
        return {k: to_plain_dict(v) for k, v in d.items()}
    return d


def write_metric_values(
    path: Path,
    rows: Sequence[Mapping[str, str | int | float]],
) -> None:
    """Write callback-level values using the paper pipeline's raw CSV contract."""
    if not rows:
        return

    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError("All metric-value rows must have the same columns.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

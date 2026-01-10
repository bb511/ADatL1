# Misc methods for the evaluation callbacks.

from pathlib import Path
from collections import defaultdict


def get_ckpt_ds_name(ckpt_name: str):
    """Gets the data set name from the checkpoint name.

    The format of the checkpoint name is specified in
        callbacks/checkpointing/dataset_aware.py
    and this method expects exactly that format. If the format changes, this method
    will stop working.
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

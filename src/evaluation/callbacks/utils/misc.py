# Misc methods for the evaluation callbacks.
from collections import defaultdict
from pathlib import Path


def get_ckpt_ds_name(ckpt_name: str):
    """Gets the data set name from the checkpoint name.

    Returns the part of the name between 'ds=' and the next '__', or the whole name when
    it carries no 'ds=', as no current checkpoint or dump epoch does.
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

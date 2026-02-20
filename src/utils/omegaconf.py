from omegaconf import OmegaConf
import torch
import re

import hashlib


def register_resolvers():
    """
    OmegaConf resolvers allow us to perform operations in the .yaml configuration files. Since these are
    needed for all my scripts, I will load all of them everywhere using this function.
    """
    OmegaConf.register_new_resolver(
        "eval", eval
    )  # general: parse a str expression and evaluate it
    OmegaConf.register_new_resolver("len", len)  # compute length
    OmegaConf.register_new_resolver("str", lambda x: str(x))  # convert to string
    OmegaConf.register_new_resolver(
        "ifelse",
        lambda if_not_x, then_x: if_not_x if str(if_not_x) != "None" else then_x,
    )  # if-else
    OmegaConf.register_new_resolver(
        "classname", lambda classpath: classpath.split(".")[-1].lower()
    )  # extract class name from DictConfig path
    OmegaConf.register_new_resolver("lower", lambda x: x.lower())
    OmegaConf.register_new_resolver("prod", lambda x, y: x * y)
    OmegaConf.register_new_resolver("div", lambda x, y: float(x) / float(y))
    OmegaConf.register_new_resolver("sqrt", lambda x: float(x) ** 0.5)
    OmegaConf.register_new_resolver("intdiv", lambda x, y: x // y)
    OmegaConf.register_new_resolver("substract", lambda x, y: x - y)
    OmegaConf.register_new_resolver("arange_list", lambda lst: list(range(len(lst))))
    OmegaConf.register_new_resolver(
        "arange", lambda start, stop, step: list(range(start, stop, step))
    )
    OmegaConf.register_new_resolver(
        "list_at_idx", lambda list, idx: list[idx]
    )  # get 'list' element at index 'idx'
    OmegaConf.register_new_resolver(
        "dict_at_key", lambda dict, key: dict[key]
    )  # get 'dict' element with key 'key'
    OmegaConf.register_new_resolver(
        "percent_integer", lambda percent, value: int(value * percent / 100.0)
    )  # compute integer percentage of a value
    OmegaConf.register_new_resolver(
        "num_training_steps",
        lambda n_epochs, len_data, effective_batch_size: n_epochs
        * (len_data // effective_batch_size),
    )  # compute number of training steps
    OmegaConf.register_new_resolver(
        "torch_full_float",
        lambda value, length: torch.full((length,), float(value)),
    )
    OmegaConf.register_new_resolver(
        "scan_dirname",
        lambda scan_dictionary: "scan_"
        + "_".join(
            f"{skey}={svalue}" for skey, svalue in sorted(scan_dictionary.items())
        ),
    )
    OmegaConf.register_new_resolver("short_hash", short_hash, replace=True)
    OmegaConf.register_new_resolver("reverse", lambda xs: list(reversed(xs)))


def short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]

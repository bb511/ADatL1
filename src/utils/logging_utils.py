from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def resolve_for_logging(cfg: Any) -> Any:
    """Return ``cfg`` as plain dicts/lists with every interpolation resolved.

    Loggers stringify whatever they are given, so an unresolved config shows up in
    MLflow as e.g. ``callbacks/losses/gamma = ${algorithm.mi_gamma}`` instead of 0.1.
    A value whose interpolation cannot be resolved (an unset ``${oc.env:...}``, a
    missing key) keeps its raw ``${...}`` string, so one bad key never drops the rest.
    """
    if not isinstance(cfg, (DictConfig, ListConfig)):
        return cfg
    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception as err:  # omegaconf raises several unrelated error types here
        log.warning(
            f"Could not resolve the whole config for logging ({type(err).__name__}: {err}); "
            "resolving it key by key and keeping the unresolvable values as written."
        )
        return _resolve_each(cfg)


def _resolve_each(node: Any) -> Any:
    raw = OmegaConf.to_container(node, resolve=False)
    if isinstance(node, DictConfig):
        keys, out = list(node.keys()), {}
    else:
        keys, out = range(len(node)), [None] * len(node)
    for key in keys:
        try:
            value = node[key]
        except Exception:
            value = raw[key]
        if isinstance(value, (DictConfig, ListConfig)):
            value = _resolve_each(value)
        out[key] = value
    return out


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"algorithm"`: The Lightning algorithm.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = resolve_for_logging(object_dict["cfg"])
    algorithm = object_dict["algorithm"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["algorithm"] = cfg["algorithm"]

    # save number of algorithm parameters
    hparams["algorithm/params/total"] = sum(p.numel() for p in algorithm.parameters())
    hparams["algorithm/params/trainable"] = sum(
        p.numel() for p in algorithm.parameters() if p.requires_grad
    )
    hparams["algorithm/params/non_trainable"] = sum(
        p.numel() for p in algorithm.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

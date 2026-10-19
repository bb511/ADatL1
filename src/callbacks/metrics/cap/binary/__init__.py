import functools
from typing import Callable, Literal, Optional, Tuple

import torch
from torch import Tensor

__all__ = ["get_pairing_fn", "get_normalizer_fn", "get_energy_fn"]


from src.callbacks.metrics.cap.binary.pairing import cdf


def get_pairing_fn(
    pairing_type: str,
):
    """Get the pairing function based on type."""

    if pairing_type == "none":  # get the first elements
        return lambda tensor1, tensor2: (
            torch.arange(min(len(tensor1), len(tensor2))),
            torch.arange(min(len(tensor1), len(tensor2))),
        )

    elif pairing_type == "cdf":
        return cdf

    else:
        raise ValueError(f"Unknown pairing type: {pairing_type}")


from src.callbacks.metrics.cap.binary.normalization import (
    log_sigmoid,
    minmax,
    rank,
    rank_mid,
    sigmoid,
    sigmoid_robust,
    softmax,
)


def get_normalizer_fn(
    normalization_type: str, normalization_params: Optional[dict] = None
):
    """Get the normalization function based on type."""

    # Copied: the CAP callbacks rebuild their metric from one shared config at every
    # epoch or checkpoint, so popping mode from the caller's dict would leave every
    # metric after the first in joint mode.
    normalization_params = dict(normalization_params or {})
    if normalization_type == "none":
        return lambda x, y: (x, y)

    # Split or joint normalization:
    mode = normalization_params.pop("mode", "joint")

    if normalization_type == "minmax":
        normalizer_fn = functools.partial(minmax, **normalization_params)

    elif normalization_type == "sigmoid":
        normalizer_fn = functools.partial(sigmoid, **normalization_params)

    elif normalization_type == "sigmoid_robust":
        normalizer_fn = functools.partial(sigmoid_robust, **normalization_params)

    elif normalization_type == "softmax":
        normalizer_fn = softmax

    elif normalization_type == "rank":
        normalizer_fn = rank

    elif normalization_type == "rank_mid":
        normalizer_fn = rank_mid

    elif normalization_type == "log_sigmoid":
        normalizer_fn = log_sigmoid

    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")

    def _normalize(
        tensor1: Tensor,
        tensor2: Tensor,
        normalizer_fn: Callable,
        mode: Literal["split", "joint"] = "joint",
    ) -> Tuple[Tensor, Tensor]:
        if mode == "joint":
            combined = torch.cat([tensor1, tensor2], dim=0)
            normalized_combined = normalizer_fn(combined)
            return (
                normalized_combined[: len(tensor1)],
                normalized_combined[len(tensor1) :],
            )

        return normalizer_fn(tensor1), normalizer_fn(tensor2)

    return functools.partial(_normalize, normalizer_fn=normalizer_fn, mode=mode)


from src.callbacks.metrics.cap.binary.energy import (
    adaptive,
    baseline,
    contrastive,
    exponential,
    focal,
    margin,
)


def get_energy_fn(energy_type: str, energy_params: Optional[dict] = None):
    """Get the energy function based on type."""
    energy_params = energy_params or {}

    if energy_type == "baseline":
        return baseline

    elif energy_type == "focal":
        return functools.partial(focal, **energy_params)

    elif energy_type == "exponential":
        return functools.partial(exponential, **energy_params)

    elif energy_type == "margin":
        return functools.partial(margin, **energy_params)

    elif energy_type == "contrastive":
        return functools.partial(contrastive, **energy_params)

    elif energy_type == "adaptive":
        return functools.partial(adaptive, **energy_params)

    else:
        raise ValueError(f"Unknown energy type: {energy_type}")

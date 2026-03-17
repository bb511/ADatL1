from typing import Optional, Callable, Tuple, Literal
import functools
import torch
from torch import Tensor


__all__ = ["get_pairing_fn", "get_normalizer_fn", "get_energy_fn", "get_regularizer_fn"]


from src.evaluation.callbacks.metrics.cap.binary.pairing import (
    random,
    label,
    absolute,
    cdf
)

def get_pairing_fn(
    pairing_type: str,
):
    """Get the regularization function based on type."""

    if pairing_type == "none": # get the first elements
        return lambda tensor1, tensor2: (
            torch.arange(min(len(tensor1), len(tensor2))),
            torch.arange(min(len(tensor1), len(tensor2)))
        )

    elif pairing_type == "random":
        return random

    elif pairing_type == "label":
        return label

    elif pairing_type == "absolute":
        return absolute

    elif pairing_type == "cdf":
        return cdf

    else:
        raise ValueError(f"Unknown pairing type: {pairing_type}")


from src.evaluation.callbacks.metrics.cap.binary.normalization import (
    minmax,
    sigmoid,
    log_sigmoid,
    softmax,
    rank,
    rank_mid
)

def get_normalizer_fn(
    normalization_type: str,
    normalization_params: Optional[dict] = None
):
    """Get the normalization function based on type."""

    normalization_params = normalization_params or {}
    if normalization_type == "none":
        return lambda x, y: (x, y)

    # Split or joint normalization:
    mode = normalization_params.pop('mode', 'joint')

    if normalization_type == "minmax":
        normalizer_fn = functools.partial(
            minmax,
            **normalization_params
        )

    elif normalization_type == "sigmoid":
        normalizer_fn = functools.partial(
            sigmoid,
            **normalization_params
        )

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
            mode: Literal["split", "joint"] = "joint"
        ) -> Tuple[Tensor, Tensor]:

        if mode == "joint":
            combined = torch.cat([tensor1, tensor2], dim=0)
            normalized_combined = normalizer_fn(combined)
            return normalized_combined[:len(tensor1)], normalized_combined[len(tensor1):]

        return normalizer_fn(tensor1), normalizer_fn(tensor2)

    return functools.partial(
        _normalize,
        normalizer_fn=normalizer_fn,
        mode=mode
    )


from src.evaluation.callbacks.metrics.cap.binary.energy import (
    baseline,
    exponential,
    focal,
    contrastive,
    margin,
    adaptive
)

def get_energy_fn(
    energy_type: str,
    energy_params: Optional[dict] = None
):
    """Get the regularization function based on type."""
    energy_params = energy_params or {}

    if energy_type == "baseline":
        return baseline

    elif energy_type == "focal":
        return functools.partial(
            focal,
            **energy_params
        )

    elif energy_type == "exponential":
        return functools.partial(
            exponential,
            **energy_params
        )

    elif energy_type == "margin":
        return functools.partial(
            margin,
            **energy_params
        )

    elif energy_type == "contrastive":
        return functools.partial(
            contrastive,
            **energy_params
        )

    elif energy_type == "adaptive":
        return functools.partial(
            adaptive,
            **energy_params
        )

    else:
        raise ValueError(f"Unknown energy type: {energy_type}")


from src.evaluation.callbacks.metrics.cap.binary.regularization import (
    threshold,
    smooth,
    percentile
)

def get_regularizer_fn(
    regularization_type: str,
    regularization_params: Optional[dict] = None
):
    """Get the regularization function based on type."""

    regularization_params = regularization_params or {}
    if regularization_type == "none":
        return lambda x: x

    elif regularization_type == "threshold_max":
        return functools.partial(
            threshold,
            **regularization_params,
            mode="max"
        )

    elif regularization_type == "threshold_mean":
        return functools.partial(
            threshold,
            **regularization_params,
            mode="mean"
        )

    elif regularization_type == "threshold_zero":
        return functools.partial(
            threshold,
            **regularization_params,
            mode="zero"
        )

    elif regularization_type == "smooth_sigmoid":
        return functools.partial(
            smooth,
            **regularization_params,
            mode="sigmoid"
        )

    elif regularization_type == "smooth_exponential":
        return functools.partial(
            smooth,
            **regularization_params,
            mode="exponential"
        )

    elif regularization_type == "smooth_linear":
        return functools.partial(
            smooth,
            **regularization_params,
            mode="linear"
        )

    elif regularization_type == "percentile":
        return functools.partial(
            percentile,
            **regularization_params,
        )

    else:
        raise ValueError(f"Unknown regularization type: {regularization_type}")

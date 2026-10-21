from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def minmax(
    tensor: Tensor, min_val: Optional[float] = None, max_val: Optional[float] = None
) -> Tensor:
    """
    Min-max normalization: scales tensor to [0, 1] range.

    Rationale: Preserves original distribution shape.

    Args:
        tensor: Input tensor to normalize
        min_val: Minimum value (computed if None)
        max_val: Maximum value (computed if None)

    Returns:
        Normalized tensor in [0, 1] range
    """
    if min_val is None:
        min_val = tensor.min().item()
    if max_val is None:
        max_val = tensor.max().item()

    # Avoid division by zero
    if abs(max_val - min_val) < 1e-8:
        return torch.full_like(tensor, 0.5)

    return (tensor - min_val) / (max_val - min_val)


def sigmoid(
    tensor: Tensor, center: Optional[float] = None, scale: Optional[float] = None
) -> Tensor:
    """
    Sigmoid normalization: maps to (0, 1) using sigmoid function.

    Rationale: Provides smooth mapping while roughly preserving distribution shape.

    Args:
        tensor: Input tensor to normalize
        center: Center point for sigmoid (uses median if None)
        scale: Scale parameter (uses std/4 if None for reasonable steepness)
    """
    if center is None:
        center = tensor.median().item()
    if scale is None:
        scale = tensor.std().item() / 4.0
        if scale < 1e-8:
            scale = 1.0

    return torch.sigmoid((tensor - center) / scale)


def sigmoid_robust(
    tensor: Tensor, center: Optional[float] = None, scale: Optional[float] = None
) -> Tensor:
    """
    Robust sigmoid normalization: sigmoid with a median-absolute-deviation scale.

    Rationale: Same mapping as sigmoid, but no single extreme score sets its steepness.

    Args:
        tensor: Input tensor to normalize
        center: Center point for sigmoid (uses median if None)
        scale: Scale parameter (uses 1.4826 * MAD / 4 if None)
    """
    median = tensor.median().item()
    if center is None:
        center = median
    if scale is None:
        # 1.4826 MAD estimates the std of Gaussian scores, so this equals sigmoid's
        # std / 4 for them, while one extreme score no longer sets the scale.
        scale = 1.4826 * (tensor - median).abs().median().item() / 4.0
        if scale < 1e-8:
            scale = 1.0

    return torch.sigmoid((tensor - center) / scale)


def softmax(
    tensor: Tensor,
) -> Tensor:
    """
    Softmax normalization: smooth distribution over (0, 1) summing to 1 over `dim`.

    Rationale: Standard normalization in classification settings for log-probabilistic outputs.

    Args:
        tensor: Input tensor to normalize
    """
    return F.softmax(tensor, dim=-1)


def _tied_ranks(tensor: Tensor) -> Tensor:
    """Return the zero-based rank of every element, averaged over tied values."""
    _, inverse, counts = torch.unique(tensor, return_inverse=True, return_counts=True)
    first = torch.cumsum(counts, dim=0) - counts
    return (first + (counts - 1) / 2).to(torch.float)[inverse]


def rank(
    tensor: Tensor,
) -> Tensor:
    """
    Rank normalization: maps tensor values to [0, 1] using empirical CDF.

    Rationale:
        Robust to heavy-tailed distributions and extreme outliers.
        Preserves ordering of anomaly scores while ensuring a uniform
        distribution over [0, 1]. Tied values share their mean rank.

    Args:
        tensor: Input tensor to normalize

    Returns:
        Normalized tensor in [0, 1] range based on rank.
    """
    n = tensor.numel()

    if n <= 1:
        return torch.full_like(tensor, 0.5)

    return _tied_ranks(tensor) / (n - 1)


def rank_mid(
    tensor: Tensor,
) -> Tensor:
    """Rank normalization using mid-ranks.

    Rationale:
        Same as rank normalization but avoids exact 0 and 1, which
        can improve numerical stability in exponential kernels.
        Tied values share their mean rank.

    Args:
        tensor: Input tensor to normalize

    Returns:
        Normalized tensor in (0, 1) based on empirical CDF.
    """
    n = tensor.numel()

    if n <= 1:
        return torch.full_like(tensor, 0.5)

    return (_tied_ranks(tensor) + 0.5) / n


def log_sigmoid(
    tensor: Tensor,
    center: Optional[float] = None,
    scale: Optional[float] = None,
    eps: Optional[float] = None,
) -> Tensor:
    """
    Log-sigmoid normalization: applies a log1p transform followed by sigmoid mapping.

    Rationale:
        Useful for positive, heavy-tailed anomaly scores such as reconstruction MSE.
        The log1p transform compresses extreme tail values while preserving ordering,
        and the sigmoid then maps the transformed scores smoothly into (0, 1).

    Args:
        tensor: Input tensor to normalize
        center: Center point for sigmoid after log1p transform
            (uses median of transformed tensor if None)
        scale: Scale parameter after log1p transform
            (uses std/4 of transformed tensor if None for reasonable steepness)
        eps: Inputs below -1 + eps are clamped there before log1p (uses the machine
            epsilon of the tensor's dtype if None, so that -1 + eps differs from -1
            in that dtype)

    Returns:
        Normalized tensor in (0, 1) range
    """
    if eps is None:
        eps = torch.finfo(tensor.dtype).eps
    transformed = torch.log1p(torch.clamp(tensor, min=-1.0 + eps))

    if center is None:
        center = transformed.median().item()
    if scale is None:
        scale = transformed.std().item() / 4.0
        if scale < 1e-8:
            scale = 1.0

    return torch.sigmoid((transformed - center) / scale)

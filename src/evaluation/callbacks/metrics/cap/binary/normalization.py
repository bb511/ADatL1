from typing import Tuple, Optional, Literal, Dict
import torch
from torch import Tensor
import torch.nn.functional as F


def minmax(
    tensor: Tensor,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
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
    tensor: Tensor,
    center: Optional[float] = None,
    scale: Optional[float] = None
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


def rank(
    tensor: Tensor,
) -> Tensor:
    """
    Rank normalization: maps tensor values to [0, 1] using empirical CDF.

    Rationale:
        Robust to heavy-tailed distributions and extreme outliers.
        Preserves ordering of anomaly scores while ensuring a uniform
        distribution over [0, 1].

    Args:
        tensor: Input tensor to normalize

    Returns:
        Normalized tensor in [0, 1] range based on rank.
    """
    n = tensor.numel()

    if n <= 1:
        return torch.full_like(tensor, 0.5)

    # Compute ranks
    sorted_indices = torch.argsort(tensor)
    ranks = torch.empty_like(sorted_indices, dtype=torch.float)
    ranks[sorted_indices] = torch.arange(n, device=tensor.device, dtype=torch.float)

    # Normalize ranks to [0, 1]
    return ranks / (n - 1)


def rank_mid(
    tensor: Tensor,
) -> Tensor:
    """
    Rank normalization using mid-ranks.

    Rationale:
        Same as rank normalization but avoids exact 0 and 1, which
        can improve numerical stability in exponential kernels.

    Args:
        tensor: Input tensor to normalize

    Returns:
        Normalized tensor in (0, 1) based on empirical CDF.
    """
    n = tensor.numel()

    if n <= 1:
        return torch.full_like(tensor, 0.5)

    sorted_indices = torch.argsort(tensor)
    ranks = torch.empty_like(sorted_indices, dtype=torch.float)
    ranks[sorted_indices] = torch.arange(n, device=tensor.device, dtype=torch.float)

    return (ranks + 0.5) / n

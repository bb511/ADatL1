from typing import Tuple

import torch
from torch import Tensor


def _get_small_large(tensor1: Tensor, tensor2: Tensor) -> Tuple[Tensor, Tensor]:
    """Helper function to return tensors in (small, large) order."""
    if len(tensor1) <= len(tensor2):
        return tensor1, tensor2, False
    else:
        return tensor2, tensor1, True


def cdf(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Empirical CDF pairing: matches samples based on their percentile ranks.

    Rationale: Preserves distributional relationships by pairing samples at similar
    percentile ranks in their respective distributions.
    """
    small_tensor, large_tensor, swapped = _get_small_large(tensor1, tensor2)

    # Get sorted indices (ranks) for both tensors
    small_sorted_indices = torch.argsort(small_tensor)
    large_sorted_indices = torch.argsort(large_tensor)

    # For each position in small tensor, find corresponding percentile in large tensor
    small_size = len(small_tensor)
    large_size = len(large_tensor)

    # Map each rank in small tensor to corresponding rank in large tensor
    large_ranks = torch.round(
        (torch.arange(small_size, dtype=torch.float) / (small_size - 1))
        * (large_size - 1)
    ).long()

    # Handle edge case for single element
    if small_size == 1:
        large_ranks = torch.tensor([large_size // 2])  # Use median

    # Get the actual indices
    small_indices = small_sorted_indices
    large_indices = large_sorted_indices[large_ranks]

    if not swapped:
        return small_indices, large_indices
    return large_indices, small_indices

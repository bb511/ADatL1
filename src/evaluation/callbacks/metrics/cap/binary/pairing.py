from typing import Tuple
import torch
from torch import Tensor


def _get_small_large(
    tensor1: Tensor,
    tensor2: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Helper function to return tensors in (small, large) order.
    """
    if len(tensor1) <= len(tensor2):
        return tensor1, tensor2, False
    else:
        return tensor2, tensor1, True


def random(
    tensor1: Tensor,
    tensor2: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Random pairing: randomly samples from larger tensor to match smaller tensor size.
    
    Rationale: Baseline solution, no pairing.
    """
    small_tensor, large_tensor, swapped = _get_small_large(tensor1, tensor2)
    
    small_indices = torch.arange(len(small_tensor))
    large_indices = torch.randperm(len(large_tensor))[:len(small_tensor)]
    
    # Map back to original tensor order
    if not swapped:
        return small_indices, large_indices
    return large_indices, small_indices


def label(
    tensor1: Tensor,
    tensor2: Tensor,
    labels1: Tensor,
    labels2: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Label-based pairing: matches samples with same labels, then randomly pairs within groups.
    
    Rationale: Ensures comparisons are made between samples of the same class/type,
    controlling for label-based differences in loss distributions.
    """
    small_tensor, large_tensor, swapped = _get_small_large(tensor1, tensor2)
    small_labels, large_labels = _get_small_large(labels1, labels2)
    
    small_indices_list = []
    large_indices_list = []
    
    unique_labels = torch.unique(small_labels)
    
    for label in unique_labels:
        small_mask = small_labels == label
        large_mask = large_labels == label
        
        small_label_indices = torch.where(small_mask)[0]
        large_label_indices = torch.where(large_mask)[0]
        
        n_small = len(small_label_indices)
        n_large = len(large_label_indices)
        
        if n_large == 0:
            continue
            
        if n_large >= n_small:
            selected_large = large_label_indices[torch.randperm(n_large)[:n_small]]
        else:
            selected_large = large_label_indices[torch.randint(0, n_large, (n_small,))]
        
        small_indices_list.append(small_label_indices)
        large_indices_list.append(selected_large)
    
    if len(small_indices_list) == 0:
        return torch.tensor([]), torch.tensor([])
    
    small_indices = torch.cat(small_indices_list)
    large_indices = torch.cat(large_indices_list)
    
    if not swapped:
        return small_indices, large_indices
    return large_indices, small_indices


def absolute(
    tensor1: Tensor,
    tensor2: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Absolute value pairing: matches samples based on absolute values.

    It uses searchsorted to find where each value from the small tensor would be 
    inserted into the sorted large tensor, which pairs values based on their 
    absolute magnitudes.
    
    Rationale: Pair based on the value of the tensors, to provide an upper bound on the metric.
    """
    small_tensor, large_tensor, swapped = _get_small_large(tensor1, tensor2)
    
    # Sort large tensor to enable searchsorted
    large_sorted, large_sort_indices = torch.sort(large_tensor)
    
    # For each value in small tensor, find its position in large tensor's distribution
    large_indices = torch.searchsorted(large_sorted, small_tensor)
    large_indices = torch.clamp(large_indices, 0, len(large_tensor) - 1)
    large_indices = large_sort_indices[large_indices]
    small_indices = torch.arange(len(small_tensor))
    
    if not swapped:
        return small_indices, large_indices
    return large_indices, small_indices


def cdf(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor
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
        (torch.arange(small_size, dtype=torch.float) / (small_size - 1)) * (large_size - 1)
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

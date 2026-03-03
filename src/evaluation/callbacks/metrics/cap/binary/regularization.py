import torch
from torch import Tensor
from typing import Literal, Optional


def threshold(
    prob: Tensor,
    maxcount: int,
    mode: Literal["max", "mean", "zero"] = "max",
    **kwargs
) -> Tensor:
    """
    Regularize probabilities to naturally enforce maxcount positive classifications.
    
    Sets the smallest (N - maxcount) probabilities to reduce their anomaly likelihood.
    
    Args:
        prob: Normalized probability values [batch_size] 
        maxcount: Maximum number of positive classifications (top-k to preserve)
        mode: How to set the smallest (N-m) values
            - "max": Set to max of smallest (N-m) values (threshold effect)
            - "mean": Set to mean of smallest (N-m) values (averaging)
            - "zero": Set to zero (suppression)
    
    Returns:
        Regularized probability values [batch_size]
    """
    n = len(prob)
    if maxcount >= n:
        return prob
    
    # Get indices of smallest (n - maxcount) values
    _, indices = torch.topk(prob, n - maxcount, largest=False)
    
    regularized_prob = prob.clone()
    smallest_values = prob[indices]
    
    if mode == "max":
        new_value = smallest_values.max()
    elif mode == "mean": 
        new_value = smallest_values.mean()
    elif mode == "zero":
        new_value = 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    regularized_prob[indices] = new_value
    return regularized_prob


def smooth(
    prob: Tensor,
    maxcount: int,
    temperature: Optional[float] = 1.0,
    mode: Literal["sigmoid", "exponential", "linear"] = "sigmoid",
    **kwargs
) -> Tensor:
    """
    Smooth regularization that gradually suppresses smallest probabilities.
    
    Args:
        prob: Normalized probability values [batch_size]
        maxcount: Maximum number of positive classifications  
        temperature: Controls smoothness (lower = sharper transition)
        mode: Type of smooth transformation
            - "sigmoid": Sigmoid-based smooth threshold. Values below threshold get suppressed, above threshold preserved.
            - "exponential": Exponential decay for values below threshold.
            - "linear": Linear interpolation to minimum
    """
    n = len(prob)
    if maxcount >= n:
        return prob
    
    # Find the threshold (value of the maxcount-th largest element)
    threshold_value = torch.topk(prob, maxcount, largest=True)[0][-1]
    
    if mode == "sigmoid":
        centered = (prob - threshold_value) / temperature
        weight = torch.sigmoid(centered)
        return prob * weight
    
    elif mode == "exponential":
        regularized_prob = prob.clone()
        below_threshold = prob < threshold_value
        
        if below_threshold.any():
            # Exponential decay based on distance from threshold
            distance = (threshold_value - prob[below_threshold]) / temperature
            regularized_prob[below_threshold] = prob[below_threshold] * torch.exp(-distance)
        
        return regularized_prob
    
    elif mode == "linear":
        # Linear interpolation: map smallest values linearly to [0, threshold]
        min_val = prob.min()
        
        # Create linear mapping for values below threshold
        below_threshold = prob < threshold_value
        if below_threshold.any():
            # Map [min_val, threshold] -> [0, threshold] linearly
            normalized = (prob[below_threshold] - min_val) / (threshold_value - min_val + 1e-8)
            regularized_prob = prob.clone()
            regularized_prob[below_threshold] = normalized * threshold_value
            return regularized_prob
        
        return prob
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def percentile(
    prob: Tensor,
    maxcount: int,
    suppression_percentile: Optional[float] = 0.1,
    enhancement_percentile: Optional[float] = 0.9,
    **kwargs
) -> Tensor:
    """
    Regularization based on percentile thresholds.
    
    Maps probability values to create clear separation between anomaly and normal regions
    using percentile-based transformations.
    
    Args:
        prob: Normalized probability values [batch_size]
        maxcount: Maximum number of positive classifications
        suppression_percentile: Target percentile for suppressed values
        enhancement_percentile: Target percentile for enhanced values
    """
    n = len(prob)
    if maxcount >= n:
        return prob
    
    # Find threshold value
    threshold_value = torch.topk(prob, maxcount, largest=True)[0][-1]
    
    regularized_prob = prob.clone()
    
    # Map values below threshold to low percentile region
    below_threshold = prob < threshold_value
    if below_threshold.any():
        below_values = prob[below_threshold]
        # Map to [0, suppression_percentile] range
        normalized = (below_values - below_values.min()) / (below_values.max() - below_values.min() + 1e-8)
        regularized_prob[below_threshold] = normalized * suppression_percentile
    
    # Map values above threshold to high percentile region
    above_threshold = prob >= threshold_value
    if above_threshold.any():
        above_values = prob[above_threshold]
        # Map to [enhancement_percentile, 1.0] range
        normalized = (above_values - above_values.min()) / (above_values.max() - above_values.min() + 1e-8)
        regularized_prob[above_threshold] = enhancement_percentile + normalized * (1.0 - enhancement_percentile)
    
    return regularized_prob

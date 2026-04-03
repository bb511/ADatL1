from typing import Optional
import torch
from torch import Tensor


def baseline(prob: Tensor, y: Tensor, **kwargs) -> Tensor:
    """
    Baseline energy penalizing misalignment between predictions and labels.

    Rationale: Proportional penalization of misclassifications.

    Args:
        prob: Normalized classification scores [batch_size]
        y: Binary labels [batch_size]
    """
    return y * (1 - prob) + (1 - y) * prob


def exponential(
    prob: Tensor, y: Tensor, gamma: Optional[float] = 2.0, **kwargs
) -> Tensor:
    """
    Exponential energy applying heavy penalties for misalignments.

    Rationale: Heavier penalization of misclassifications.

    Args:
        prob: Normalized classification scores [batch_size]
        y: Binary labels [batch_size]
        gamma: Exponential scaling factor
    """

    exp0 = torch.exp(gamma * prob) + prob
    exp1 = torch.exp(-gamma * prob) + (1 - prob)
    return y * exp1 + (1 - y) * exp0


def focal(prob: Tensor, y: Tensor, gamma: Optional[float] = 2.0, **kwargs) -> Tensor:
    """
    Focal energy emphasizing hard-to-classify examples.

    Rationale: Focus penalization on hard examples.

    Args:
        prob: Normalized classification scores [batch_size]
        y: Binary labels [batch_size]
        gamma: Focusing parameter (higher = more focus on hard examples)
    """
    eps = 1e-8  # Numerical stability

    focal0 = -torch.log(1 - prob + eps) * (prob**gamma)
    focal1 = -torch.log(prob + eps) * ((1 - prob) ** gamma)
    return y * focal1 + (1 - y) * focal0


def contrastive(
    prob: Tensor,
    y: Tensor,
    center: Optional[float] = 0.75,
    margin: Optional[float] = 0.1,
    **kwargs,
) -> Tensor:
    """
    Contrastive energy promoting class separation.

    Rationale: Favor class separation.

    Args:
        prob: Normalized classification scores [batch_size]
        y: Binary labels [batch_size]
        center: Target center for y = 1.
        margin: Minimum separation between classes
    """
    centers = [1 - center, center]  # symmetric

    # Attraction to own class center
    att0 = torch.abs(prob - centers[0])
    att1 = torch.abs(prob - centers[1])

    # Repulsion from opposite class (with margin)
    rep0 = torch.relu(prob - (centers[1] - margin))
    rep1 = torch.relu((centers[0] + margin) - prob)

    # Return sum of contributions
    return y * (att1 + rep1) + (1 - y) * (att0 + rep0)


def margin(
    prob: Tensor,
    y: Tensor,
    margin: Optional[float] = 0.1,
    threshold: Optional[float] = 0.5,
    **kwargs,
) -> Tensor:
    """
    Margin-based energy enforcing clear decision boundaries.

    Rationale: Favor confident classification.

    Args:
        prob: Normalized classification scores [batch_size]
        y: Binary labels [batch_size]
        margin: Safety margin width
        threshold: Decision boundary
    """
    # Violations of the threshold + margin:
    viol0 = torch.relu(prob - (threshold - margin))
    viol1 = torch.relu((threshold + margin) - prob)

    # Base energy:
    pen0 = prob
    pen1 = 1 - prob

    # Return sum of contributions
    return y * (pen1 + viol1) + (1 - y) * (pen0 + viol0)


def adaptive(
    prob: Tensor,
    y: Tensor,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    scale: Optional[float] = 0.5,
    **kwargs,
) -> Tensor:
    """
    Adaptive margin energy with data-driven thresholds.
    Requires external statistics to maintain consistency across batches.

    Rationale: Favor confident classification adaptively.

    Args:
        prob: Normalized classification scores [batch_size]
        y: Binary labels [batch_size]
        mean: Overall mean of the probabilities
        std: Overall standard deviation of the probabilities
        scale: Scaling factor for adaptive thresholds
    """
    mean = mean if mean is not None else prob.mean().item()
    std = std if std is not None else prob.std().item()

    # Violations of the adaptive thresholds: mean - scale * std
    viol0 = torch.relu(prob - (mean - scale * std))
    viol1 = torch.relu((mean + scale * std) - prob)

    # Base energy:
    pen0 = prob
    pen1 = 1 - prob

    # Return sum of contributions
    return y * (pen1 + viol1) + (1 - y) * (pen0 + viol0)

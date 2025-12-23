import torch
from typing import Literal, Optional

import ot  # type: ignore

from src.losses.components import L1ADLoss


class WassersteinLoss(L1ADLoss):
    """Compute the Wasserstein distance between two batches.

    :param scale: Scaling factor applied to the computed distance.
    :param reduction: How to reduce the per-sample distances over the batch.
    :param p: The order of the norm to compute pairwise distances in the
        underlying metric space.  ``p=2`` corresponds to the squared
        Euclidean distance.  See ``ot.dist`` for details.
    """

    name: str = "wasserstein"

    def __init__(
        self,
        scale: float = 1.0,
        reduction: Literal["none", "mean", "sum"] = "none",
        p: int = 2,
    ) -> None:
        super().__init__(scale=scale, reduction=reduction)
        self.p = p

    def forward(
        self,
        target: torch.Tensor,
        negative_samples: torch.Tensor,
        **kwargs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the Wasserstein distance between two sets of samples.

        The method expects two positional arguments with shapes
        ``(batch, dim)`` or higher.  Higher dimensional inputs are
        flattened to 2D before computing pairwise distances.

        :param target: Tensor of real samples.
        :param negative_samples: Tensor of negative samples drawn from the model.
        """
        if target.shape != negative_samples.shape:
            raise ValueError(
                f"target and negative_samples must have the same shape, got {target.shape} and {negative_samples.shape}"
            )

        # Flatten all but the batch dimension to compute pairwise distances
        pos = target.view(target.size(0), -1)
        neg = negative_samples.view(negative_samples.size(0), -1)

        # Compute pairwise p-norm cost matrix using POT's differentiable API
        # The cost matrix is of shape (batch, batch)
        cost_matrix = ot.dist(pos, neg, p=self.p)

        # Uniform weights for both distributions
        n = pos.size(0)
        weights = torch.ones(n, device=pos.device, dtype=pos.dtype) / n

        # Compute the 2-Wasserstein distance (i.e., squared EMD)
        # We leave numItermax large to ensure convergence
        emd = ot.emd2(weights, weights, cost_matrix, numItermax=1_000_000)

        # emd is a scalar tensor representing the transport cost
        loss = self.scale * emd
        return self.reduce(loss)


class NAE(L1ADLoss):
    """
    Compute the energy difference between positive and negative samples.

    :param scale: Scaling factor applied to the computed distance.
    """

    name: str = "nae"

    def __init__(self, scale: float = 1.0, **kwargs):
        super().__init__(scale=scale, reduction="none")

    def forward(
        self,
        energy_pos: torch.Tensor,
        energy_neg: torch.Tensor,
    ) -> torch.Tensor:
        return energy_pos.mean() - energy_neg.mean()

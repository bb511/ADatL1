import torch
import torch.nn.functional as F

from src.algorithms.losses.components import ADLoss


class MSEReconstructionLoss(ADLoss):
    """Reconstruction loss.

    For padded data, pass mask to only include real features into the reconstruction.

    :param scale: Float scaling factor for the loss.
    :param reduction: String of the reduction method to apply to the batch of samples
        given to this loss.
    """

    name: str = "mse_reco"

    def __init__(self, scale: float = 1.0, reduction: str = "none"):
        super().__init__(scale=scale, reduction=reduction)

    def forward(
        self, target: torch.Tensor, reco: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        mse = F.mse_loss(target, reco, reduction="none")
        mse_flat = mse.view(mse.shape[0], -1)

        if mask is None:
            mse_per_observation = mse_flat.mean(dim=1)
        else:
            mask_flat = mask.view(mask.shape[0], -1).to(dtype=mse.dtype)
            nreal_feats = mask_flat.sum(dim=1).clamp_min(1.0)
            mse_per_observation = (mse_flat * mask_flat).sum(dim=1) / nreal_feats

        return self.scale * self.reduce(mse_per_observation)


class HuberReconstructionLoss(ADLoss):
    """The SmoothL1 loss.

    For padded data, pass mask to only include real features into the reconstruction.

    :param scale: Float scaling factor for the loss.
    :param reduction: String of the reduction method to apply to the batch of samples
        given to this loss.
    """

    name: str = "huber_reco"

    def __init__(self, scale: float = 1.0, delta: float = 1.0, reduction: str = "none"):
        super().__init__(scale=scale, reduction=reduction)
        self.delta = delta

    def forward(
        self, target: torch.Tensor, reco: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        huber = F.smooth_l1_loss(target, reco, reduction="none", beta=self.delta)
        huber_flat = huber.view(huber.shape[0], -1)

        if mask is None:
            huber_per_observation = huber_flat.mean(dim=1)
        else:
            mask_flat = mask.view(mask.shape[0], -1).to(dtype=huber.dtype)
            nreal_feats = mask_flat.sum(dim=1).clamp_min(1.0)
            huber_per_observation = (huber_flat * mask_flat).sum(dim=1) / nreal_feats

        return self.scale * self.reduce(huber_per_observation)

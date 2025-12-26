from typing import Literal
import torch
import torch.nn.functional as F

from src.losses.components import L1ADLoss


class ReconstructionLoss(L1ADLoss):
    """Reconstruction loss for VAE.

    :param scale: Float scaling factor for the loss.
    :param reduction: String of the reduction method to apply to the batch of samples
        given to this loss.
    """

    name: str = "reco"

    def __init__(
        self,
        scale: float = 1.0,
        reduction: Literal["none", "mean", "sum"] = "none",
    ):
        super().__init__(scale=scale, reduction=reduction)

    def forward(self, target: torch.Tensor, reco: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(target, reco, reduction="none")
        mse_per_observation = mse.view(mse.shape[0], -1).mean(dim=1)

        return self.scale * self.reduce(mse_per_observation)


class CylPtPzReconstructionLoss(ReconstructionLoss):
    """Convert the pT, eta, phi input into pT, pz and compute loss there.

    This trick is done since the training did not converge otherwise. This is legacy
    from axov4. Try to avoid using this trick in the future.
    """
    name: str = "cylreco"

    def forward(self, target: torch.Tensor, reco: torch.Tensor) -> torch.Tensor:
        # Indexes of where the pTs of the objects are, given the axo v4 input data.
        pT_idxs = list(range(0, 55, 3))
        # Indexes of where the eta of the objects are, given the axo v4 input data.
        MET_eta_idx = 2
        eta_idxs = list(range(4, 56, 3))
        eta_idxs.append(MET_eta_idx)

        # Compute the projection from pT to pz. Use real eta for the reco.
        pT, eta = target[:, pT_idxs], target[:, eta_idxs]
        pz = pT * torch.sinh(eta)
        # The phi value of the MET object is in the 2nd position of the torch tensor
        # for any given sample.
        MET_phi = target[:, 2]
        MET_phi_pred = reco[:, 2]

        pT_pred = reco[:, pT_idxs]
        pz_pred = pT_pred * torch.sinh(eta)

        loss_per_observation = torch.mean(
            torch.abs(pT - pT_pred) + torch.abs(pz - pz_pred), dim=1
        )
        loss_per_observation += torch.abs(MET_phi - MET_phi_pred)

        return self.scale * self.reduce(loss_per_observation)

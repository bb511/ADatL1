from typing import Literal
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


class CylPtPzReconstructionLoss(MSEReconstructionLoss):
    """Convert the pT, eta, phi input into pT, pz and compute loss there.

    Although the initial space is not in cylindrical coordinates, the reco is done in
    cylindrical coordinates. This is because for axov4, the training would not converge
    otherwise. This trick should not be used in the future.
    """
    name: str = "cylreco"

    def __init__(self, scale: float = 1.0, reduction: str = "none"):
        super().__init__(scale=scale, reduction=reduction)
        self.pt_idxs = None
        self.eta_idxs = None
        self.met_phi_idxs = None

    def set_object_feature_map(self, object_feature_map: dict):
        pt_idxs = []
        eta_idxs = []
        met_phi_idxs = []
        for obj_name, feature_map in object_feature_map.items():
            if "Et" in feature_map:
                pt_idxs.extend(feature_map["Et"])

            if "eta" in feature_map:
                eta_idxs.extend(feature_map["eta"])

            if obj_name.lower() == "met" and "phi" in feature_map:
                met_phi_idxs.extend(feature_map["phi"])

        self.pt_idxs = pt_idxs
        self.eta_idxs = eta_idxs
        self.met_phi_idxs = met_phi_idxs

    def forward(
        self,
        target: torch.Tensor,
        reco: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if self.pt_idxs is None:
            raise RuntimeError("object_feature_map not set.")

        # build cylindrical representation
        pt = target[:, self.pt_idxs]
        eta = target[:, self.eta_idxs]
        pz = pt * torch.sinh(eta)

        pt_pred = reco[:, self.pt_idxs]
        pz_pred = pt_pred * torch.sinh(eta)

        met_phi = target[:, self.met_phi_idxs]
        met_phi_pred = reco[:, self.met_phi_idxs]

        # concatenate features used for loss
        target_cyl = torch.cat([pt, pz, met_phi], dim=1)
        reco_cyl = torch.cat([pt_pred, pz_pred, met_phi_pred], dim=1)

        # reuse parent implementation
        return super().forward(target_cyl, reco_cyl, mask=None)

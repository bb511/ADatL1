import torch
from src.models.losses.base import L1ADBaseLoss


class CylPtPzMAELoss(L1ADBaseLoss):
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.view(-1, self.NOF_CONSTITUENTS, 3)
        y_true = y_true.view(-1, self.NOF_CONSTITUENTS, 3)

        y_true = y_true * self.scales + self.biases
        y_pred = y_pred * self.scales + self.biases

        pt, eta = y_true[:, :, 0], y_true[:, :, 1]
        pz = pt * torch.sinh(eta)
        pt_pred, eta_pred = (
            y_pred[:, :, 0],
            y_true[:, :, 1],
        )  # Using true eta as per legacy implementation
        pz_pred = pt_pred * torch.sinh(eta)  # Legacy implementation

        return torch.mean(torch.abs(pt - pt_pred) + torch.abs(pz - pz_pred), dim=1)


class CylPtPzLoss(L1ADBaseLoss):
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.view(-1, self.NOF_CONSTITUENTS, 3)
        y_true = y_true.view(-1, self.NOF_CONSTITUENTS, 3)

        y_true = y_true * self.scales + self.biases
        y_pred = y_pred * self.scales + self.biases

        pt, eta = y_true[:, :, 0], y_true[:, :, 1]
        pz = pt * torch.sinh(eta)
        pt_pred, eta_pred = (
            y_pred[:, :, 0],
            y_true[:, :, 1],
        )  # Using true eta as per legacy implementation
        pz_pred = pt_pred * torch.sinh(eta)  # Legacy implementation

        return torch.mean((pt - pt_pred).pow(2) + (pz - pz_pred).pow(2), dim=1)
from __future__ import annotations

import torch
from torch import nn

class BernoulliMILoss(torch.nn.Module):

    def __init__(self, temperature: float = 6.0, use_quantized_sigmoid: bool = False,
        bits_bernoulli_sigmoid: int = 8, eps: float = 1e-12, input_is_logits: bool = True) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.use_quantized_sigmoid = bool(use_quantized_sigmoid)
        self.bits_bernoulli_sigmoid = int(bits_bernoulli_sigmoid)
        self.eps = float(eps)
        self.input_is_logits = bool(input_is_logits)

        if self.use_quantized_sigmoid:
            raise NotImplementedError(
                "Quantized sigmoid is not implemented yet in this PyTorch version."
            )

    def forward(self, latent: torch.Tensor, sensitive: torch.Tensor) -> torch.Tensor:
        if latent.ndim < 2:
            raise ValueError(f"Expected latent shape [batch, latent_dim, ...], got {tuple(latent.shape)}")

        latent = torch.flatten(latent, start_dim=1).float()
        sensitive = sensitive.reshape(-1).to(device=latent.device, dtype=torch.long)

        if sensitive.numel() != latent.shape[0]:
            raise ValueError(
                f"Sensitive variable length ({sensitive.numel()}) must match batch size ({latent.shape[0]})."
            )

        probs = self._bernoulli_probs(latent)

        h_marginal = self._h_bernoulli_from_probs(probs) # Σ_r H(L_r)
        h_conditional = probs.new_zeros(()) # Σ_s p(S=s) Σ_r H(L_r | S=s)

        batch_size = probs.shape[0]
        for value in torch.unique(sensitive):
            mask = sensitive == value
            probs_value = probs[mask]
            weight = probs_value.shape[0] / batch_size
            h_conditional = h_conditional + weight * self._h_bernoulli_from_probs(probs_value)

        mi = h_marginal - h_conditional
        mi = torch.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)

        # MI is theoretically non-negative. Small negative values can occur from
        # finite precision and should not create an incentive during minimisation.
        return mi.clamp_min(0.0)


 # ----------------------------------------
 # Helpers
    def _bernoulli_probs(self, latent: torch.Tensor) -> torch.Tensor:
        if self.input_is_logits:
            probs = torch.sigmoid(self.temperature * latent)
        else:
            probs = latent
        return probs.clamp(self.eps, 1.0 - self.eps)

    def _log2(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x) / torch.log(x.new_tensor(2.0))

    def _h_bernoulli_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        theta = probs.mean(dim=0).clamp(self.eps, 1.0 - self.eps) 

        entropy_per_unit = -(1.0-theta) * self._log2(1.0 - theta) - theta * self._log2(theta)
        return entropy_per_unit.sum()

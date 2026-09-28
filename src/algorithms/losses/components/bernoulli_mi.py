from __future__ import annotations

import math

import torch
from torch import nn


class BernoulliMILoss(nn.Module):
    """HepInfo-compatible Bernoulli mutual-information estimator.

    Computes:

        I(L; S) = H(L) - sum_s p(S=s) H(L | S=s)

    Each latent activation is interpreted as a Bernoulli logit and mapped to

        p = sigmoid(temperature * latent)

    The returned dtype intentionally follows hepinfo and is float32.
    """

    def __init__(
        self,
        temperature: float = 6.0,
        eps: float = 1e-20,
        use_float64: bool = True,
    ) -> None:
        super().__init__()

        self.temperature = float(temperature)
        self.eps = float(eps)
        self.use_float64 = bool(use_float64)

    def forward(self, latent: torch.Tensor, sensitive: torch.Tensor) -> torch.Tensor:
        if latent.ndim < 2:
            raise ValueError(
                f"Expected latent shape [batch, latent_dim, ...], got {tuple(latent.shape)}."
            )

        work_dtype = self._work_dtype(latent)

        latent = torch.flatten(latent, start_dim=1).to(dtype=work_dtype)
        sensitive = self._prepare_sensitive(
            sensitive=sensitive,
            batch_size=latent.shape[0],
            device=latent.device,
        )

        h_marginal = self._h_bernoulli(latent)

        batch_size = latent.shape[0]
        h_conditional = latent.new_zeros(())

        for value in torch.unique(sensitive, sorted=True):
            mask = sensitive == value
            latent_value = latent[mask]

            h_value = self._h_bernoulli(latent_value)
            weight = latent.new_tensor(latent_value.shape[0] / batch_size)

            h_conditional = h_conditional + weight * h_value

        mi = h_marginal - h_conditional

        # hepinfo masks NaNs. Do not silently squash +/-inf to zero.
        mi = torch.where(torch.isnan(mi), mi.new_zeros(()), mi)

        # hepinfo returns tf.float32.
        return mi.to(dtype=torch.float32)

    # ------------------------------------------------------------------
    # Diagnostics (never part of the training objective)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def permutation_null(
        self,
        latent: torch.Tensor,
        sensitive: torch.Tensor,
        num_permutations: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """MI against randomly permuted sensitive labels, one value per permutation.

        Shuffling S inside the batch destroys any real dependence between L and S
        while keeping the batch size, the bin occupancies and the latent
        distribution unchanged. The result is therefore the plug-in estimator's
        finite-sample bias ("noise floor") for this very batch: the value
        ``forward`` would return if the latent carried no information about S.

        All permutations are evaluated in one vectorised scatter on the same
        Bernoulli probabilities, so the cost is independent of the number of
        sensitive bins and there are no per-bin host syncs.

        Permutation indices are drawn on the CPU from ``generator`` (if given)
        so the global RNG streams used by training are left untouched.

        :return: float32 tensor of shape ``[num_permutations]``.
        """
        if num_permutations < 1:
            raise ValueError(
                f"num_permutations must be >= 1, got {num_permutations}."
            )

        probs, labels, num_groups = self._diagnostic_inputs(latent, sensitive)
        batch_size, latent_dim = probs.shape

        perms = torch.stack(
            [
                torch.randperm(batch_size, generator=generator)
                for _ in range(num_permutations)
            ]
        ).to(device=probs.device)
        permuted_labels = labels[perms]  # [P, N]

        counts = probs.new_zeros(num_permutations, num_groups)
        counts.scatter_add_(1, permuted_labels, probs.new_ones(permuted_labels.shape))

        sums = probs.new_zeros(num_permutations, num_groups, latent_dim)
        sums.scatter_add_(
            1,
            permuted_labels.unsqueeze(-1).expand(-1, -1, latent_dim),
            probs.unsqueeze(0).expand(num_permutations, -1, -1),
        )

        theta_groups = sums / counts.clamp_min(1.0).unsqueeze(-1)
        h_groups = self._entropy_from_theta(theta_groups).sum(dim=-1)  # [P, G]
        h_conditional = (counts / batch_size * h_groups).sum(dim=-1)  # [P]

        h_marginal = self._entropy_from_theta(probs.mean(dim=0)).sum()

        mi = h_marginal - h_conditional
        mi = torch.where(torch.isnan(mi), mi.new_zeros(()), mi)
        return mi.to(dtype=torch.float32)

    @torch.no_grad()
    def analytic_null_floor(
        self,
        latent: torch.Tensor,
        sensitive: torch.Tensor,
    ) -> torch.Tensor:
        """Second-order approximation of E[MI] under independence of L and S.

        Expanding the plug-in estimator to second order in the per-bin
        deviations of theta gives, with G occupied bins and N events,

            E[MI_null] ~= (G - 1) / (2 N ln 2) * sum_j Var(p_j) / (theta_j (1 - theta_j))

        The ratio Var(p_j) / (theta_j (1 - theta_j)) lies in [0, 1]: it is 0 for a
        unit whose probability is constant over events and 1 for a saturated,
        deterministic unit. The floor therefore grows with how sharp and
        informative the latent is, independent of S. The approximation degrades
        when bins are small or theta_j is close to 0 or 1; the permutation
        estimate is the reference, this is a cheap cross-check.

        :return: float32 scalar tensor.
        """
        probs, _, num_groups = self._diagnostic_inputs(latent, sensitive)
        batch_size = probs.shape[0]

        theta = probs.mean(dim=0)
        variance = probs.var(dim=0, unbiased=False)
        denominator = (theta * (1.0 - theta)).clamp_min(self.eps)
        ratio = (variance / denominator).clamp(max=1.0)

        prefactor = (num_groups - 1) / (2.0 * batch_size * math.log(2.0))
        floor = prefactor * ratio.sum()
        return floor.to(dtype=torch.float32)

    def _diagnostic_inputs(
        self,
        latent: torch.Tensor,
        sensitive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Detached probabilities [N, D], contiguous labels [N], #occupied bins."""
        if latent.ndim < 2:
            raise ValueError(
                f"Expected latent shape [batch, latent_dim, ...], got {tuple(latent.shape)}."
            )

        work_dtype = self._work_dtype(latent)
        latent = torch.flatten(latent.detach(), start_dim=1).to(dtype=work_dtype)
        sensitive = self._prepare_sensitive(
            sensitive=sensitive,
            batch_size=latent.shape[0],
            device=latent.device,
        )

        probs = self._bernoulli_probs(latent).to(dtype=work_dtype)
        # Map arbitrary label values onto 0..G-1 so empty bins cost nothing.
        _, labels = torch.unique(sensitive, sorted=True, return_inverse=True)
        num_groups = int(labels.max().item()) + 1 if labels.numel() else 0
        return probs, labels, num_groups

    def _work_dtype(self, latent: torch.Tensor) -> torch.dtype:
        # hepinfo casts y_pred to float64 before the entropy computation.
        # MPS does not support float64 well, so keep the previous MPS fallback.
        if self.use_float64 and latent.device.type != "mps":
            return torch.float64
        return torch.float32

    def _prepare_sensitive(
        self,
        sensitive: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if sensitive.ndim == 0:
            raise ValueError("Sensitive variable must have a batch dimension.")

        if sensitive.shape[0] != batch_size:
            raise ValueError(
                f"Sensitive first dimension ({sensitive.shape[0]}) must match "
                f"batch size ({batch_size}). Got shape {tuple(sensitive.shape)}."
            )

        sensitive_flat = sensitive.detach().reshape(batch_size, -1)

        if sensitive_flat.shape[1] != 1:
            raise ValueError(
                "Sensitive must contain exactly one scalar label/bin per event. "
                f"Got shape {tuple(sensitive.shape)}, which flattens to "
                f"{tuple(sensitive_flat.shape)}."
            )

        return sensitive_flat[:, 0].to(device=device, dtype=torch.long)

    def _bernoulli_probs(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.temperature * latent)

    def _log2(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + x.new_tensor(self.eps)) / torch.log(x.new_tensor(2.0))

    def _h_bernoulli(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.numel() == 0:
            return latent.new_zeros(())

        theta = self._bernoulli_probs(latent).mean(dim=0)

        return self._entropy_from_theta(theta).sum()

    def _entropy_from_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """Element-wise binary entropy in bits, same eps handling as hepinfo."""
        return -(
            (1.0 - theta) * self._log2(1.0 - theta)
            + theta * self._log2(theta)
        )

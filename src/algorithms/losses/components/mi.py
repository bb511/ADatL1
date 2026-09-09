"""Mutual-information loss for Bernoulli bottleneck autoencoders.

Ported from the Keras / TensorFlow implementation of
``mutual_information_bernoulli_loss``.

The loss estimates the mutual information I(L; S) between the latent
Bernoulli activations L and a sensitive / class attribute S via:

    I(L; S) = H(L) − Σ_v  p(S=v) · H(L | S=v)

where H is the binary entropy of Bernoulli random variables whose
parameter θ is estimated as the sample mean of the activation
probabilities.

In the *unsupervised* variant used by ``MIAwareAE`` there is no
explicit sensitive attribute.  Instead the loss is called on the
bottleneck probabilities directly and only the marginal entropy
H(L) is returned (equivalent to γ · H(L) acting as a capacity
regulariser).
"""

from __future__ import annotations

import torch

from src.algorithms.losses.components import ADLoss


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

_LOG2 = torch.log(torch.tensor(2.0)).item()  # ln(2)
_EPS = 1e-20  # numerical guard for log


def _log2(x: torch.Tensor) -> torch.Tensor:
    """Numerically safe base-2 logarithm."""
    return torch.log(x + _EPS) / _LOG2


def binary_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Element-wise binary (Bernoulli) entropy in bits.

    H(p) = −(1−p) log₂(1−p) − p log₂(p)

    :param probs: Tensor of Bernoulli probabilities in [0, 1].
    :returns: Tensor of the same shape with per-element entropy.
    """
    return -(1 - probs) * _log2(1 - probs) - probs * _log2(probs)


# ──────────────────────────────────────────────────────────────────────
# Unsupervised MI loss (marginal entropy only)
# ──────────────────────────────────────────────────────────────────────

class BernoulliBottleneckMILoss(ADLoss):
    """Marginal binary entropy of the bottleneck activations.

    When used without a sensitive attribute this reduces to

        loss = Σ_j  H_Bernoulli( mean_i(p_{i,j}) )

    i.e. the sum (or mean, depending on *reduction*) of per-neuron
    entropies computed from the batch-averaged activation probability.
    Maximising this encourages the bottleneck to use its full capacity.
    """

    name: str = "mi"

    def __init__(
        self,
        scale: float = 1.0,
        reduction: str = "sum",
    ) -> None:
        super().__init__(scale=scale, reduction=reduction)

    # Expose the helper so callers (e.g. MIAwareAE) can compute
    # per-element entropy for logging without re-implementing it.
    @staticmethod
    def binary_entropy(probs: torch.Tensor) -> torch.Tensor:
        return binary_entropy(probs)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute marginal entropy of Bernoulli bottleneck activations.

        :param probs: (B, D) Bernoulli probabilities from the bottleneck.
        :returns: Scalar loss (sum or mean over latent dimensions).
        """
        # θ_j = E_i[p_{i,j}]  — batch-mean per latent dimension.
        theta = probs.mean(dim=0)  # (D,)

        # Per-neuron entropy in bits.
        h = binary_entropy(theta)  # (D,)

        return self.scale * self.reduce(h)


# ──────────────────────────────────────────────────────────────────────
# Supervised MI loss (conditional entropy subtracted)
# ──────────────────────────────────────────────────────────────────────

class SupervisedBernoulliMILoss(ADLoss):
    """Mutual information I(L; S) between latent activations and a
    discrete sensitive attribute, estimated via Bernoulli entropy.

    This is the direct PyTorch port of the original Keras
    ``mutual_information_bernoulli_loss``.

    I(L; S) = H(L) − Σ_v  p(S=v) · H(L | S=v)
    """

    name: str = "supervised_mi"

    def __init__(
        self,
        scale: float = 1.0,
        reduction: str = "sum",
    ) -> None:
        super().__init__(scale=scale, reduction=reduction)

    def forward(
        self,
        probs: torch.Tensor,
        sensitive: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MI between bottleneck probs and a sensitive attribute.

        :param probs: (B, D) Bernoulli activation probabilities.
        :param sensitive: (B,) or (B, 1) integer-valued sensitive attribute.
        :returns: Scalar MI estimate.
        """
        probs = probs.double()
        sensitive = sensitive.view(-1).long()

        # ── H(L) ────────────────────────────────────────────────────
        theta_all = probs.mean(dim=0)
        h_total = binary_entropy(theta_all).sum()  # scalar

        # ── H(L | S=v) for each unique value v ─────────────────────
        unique_vals = sensitive.unique()
        weighted_cond = probs.new_tensor(0.0, dtype=torch.float64)

        for v in unique_vals:
            mask = sensitive == v
            probs_v = probs[mask]                        # (n_v, D)
            theta_v = probs_v.mean(dim=0)                # (D,)
            h_v = binary_entropy(theta_v).sum()          # scalar
            frac = probs_v.shape[0] / probs.shape[0]     # p(S=v)
            weighted_cond = weighted_cond + frac * h_v

        mi = h_total - weighted_cond

        # Hotfix: clamp NaN (can occur when a class has 0 samples).
        mi = torch.where(torch.isnan(mi), mi.new_tensor(0.0), mi)

        return self.scale * mi.float()

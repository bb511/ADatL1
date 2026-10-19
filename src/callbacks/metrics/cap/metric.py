# Approximation Capacity implementation. Based on the work of Victor Jimenez.
import math
from typing import Any, Callable, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from .binary import get_energy_fn as get_energy_fn_binary
from .binary import get_normalizer_fn as get_normalizer_fn_binary
from .kernel import posterior_agreement_curve


class ApproximationCapacity(Metric):
    """Approximation Capacity (CAP) Metric for classification tasks.

    The metric maximizes the mutual information between two sets of classification
    scores (e.g. log-probabilities, reconstruction losses, ...) parametrizing their
    distribution over the hypothesis space (i.e. the set of possible clustering
    assignments) using Gibbs distributions of inverse temperature `beta`.

        CAP ∝ log(Z_12 / (Z_1 * Z_2))

    Where Z_i are partition functions over the set of clustering assignments, computed
    using different energy functions and constraints.

    The maximum over beta is taken on `n_beta` logarithmically spaced points between
    `beta_min` and `beta_max`, where the mean kernel is evaluated exactly over all
    pairs in float64. After `update`, `betas` and `curve` hold the grid and the mean
    kernel on it, `beta_star` the maximising beta, and `at_bound` whether that is
    `beta_max`, in which case the maximum over all beta may lie beyond the grid.

    Functions that are imported:
    - Energy: Determines the probability of a specific clustering assignment.
    - Normalization: Normalizes model outputs (logits, loss...) to [0,1] range.
    - Regularization: Enforces constraints on the hypothesis space to implicitly
        delimit a feasible clustering assignments. For instance, by limiting
        the number of observations assigned to a specific cluster.

    The beta grid is a non-persistent buffer that follows the metric's device, while
    `curve`, `beta_star` and `at_bound` are kept on the CPU after each update.

    :param normalization_type: Method to normalize reconstruction losses to [0,1].
        Options: "none", "minmax", "sigmoid", "sigmoid_robust", "softmax", "rank",
        "rank_mid", "log_sigmoid" (default: "minmax")
    :param energy_type: Energy function for Gibbs distribution over anomaly assignments.
        Options: "baseline", "focal", "exponential", "margin", "contrastive", "adaptive"
    :param regularization_type: Method to enforce anomaly count constraints.
        Options: "none", "threshold_max", "threshold_mean", "threshold_zero",
    :param binary: Accepted for config compatibility and unused.
    :param beta_min: Smallest inverse temperature of the grid.
    :param beta_max: Largest inverse temperature of the grid.
    :param n_beta: Number of grid points.
    """

    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        normalization_type: Optional[str] = "minmax",
        normalization_params: Optional[dict] = None,
        energy_type: Optional[str] = "baseline",
        energy_params: Optional[dict] = None,
        regularization_type: Optional[str] = "none",
        regularization_params: Optional[dict] = None,
        binary: Optional[bool] = True,
        beta_min: float = 0.01,
        beta_max: float = 1000.0,
        n_beta: int = 200,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ):
        if not 0.0 < beta_min <= beta_max:
            raise ValueError("The beta grid needs 0 < beta_min <= beta_max.")

        super().__init__(
            dist_sync_on_step=False,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        # Parameters for energy and regularization functions:
        self.binary = binary
        self.normalization_type = normalization_type
        self.normalization_params = normalization_params or {}
        self.energy_type = energy_type
        # Copied, since update writes the mean and std of the scores into it.
        self.energy_params = dict(energy_params or {})
        self.regularization_type = regularization_type
        self.regularization_params = regularization_params or {}

        betas = torch.logspace(
            math.log10(beta_min), math.log10(beta_max), n_beta, dtype=torch.float64
        )
        self.register_buffer("betas", betas, persistent=False)
        self.curve = None
        self.beta_star = None
        self.at_bound = None

        # Initialize CAP:
        self.add_state(
            "cap", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )

        # Normalizer of the loss:
        self.normalizer_fn = self._get_normalizer_fn()

    def _get_normalizer_fn(self):
        return get_normalizer_fn_binary(
            self.normalization_type, self.normalization_params
        )

    def _get_energy_fn(self):
        return get_energy_fn_binary(self.energy_type, self.energy_params)

    def _energy_gaps(self, prob1: Tensor, prob2: Tensor) -> tuple[Tensor, Tensor]:
        """Return the energy gaps D_k = E_k(1) - E_k(0) of the normalised pairs."""
        combined = torch.cat([prob1, prob2], dim=0)
        self.energy_params.update(
            {
                "mean": combined.mean().item(),
                "std": combined.std().item(),
            }
        )
        energy_fn = self._get_energy_fn()
        ones = torch.ones_like(prob1)
        zeros = torch.zeros_like(prob1)
        gap1 = energy_fn(prob1, ones) - energy_fn(prob1, zeros)
        gap2 = energy_fn(prob2, ones) - energy_fn(prob2, zeros)
        return gap1, gap2

    def update(self, logits1: Tensor, logits2: Tensor, **kwargs):
        """Evaluate the mean kernel on the beta grid and keep its maximum."""
        with torch.no_grad():
            prob1, prob2 = self.normalizer_fn(logits1, logits2)
            gap1, gap2 = self._energy_gaps(
                prob1.to(torch.float64), prob2.to(torch.float64)
            )
            curve = posterior_agreement_curve(gap1, gap2, self.betas.to(gap1.device))

        self.curve = curve.cpu()
        best = int(torch.argmax(self.curve))
        cap = self.curve[best].item()
        # On a curve made NaN by a non-finite score, argmax returns the first NaN.
        self.beta_star = self.betas[best].item() if math.isfinite(cap) else math.nan
        self.at_bound = best == len(self.betas) - 1
        self.cap.fill_(cap)

    def compute(self) -> Tensor:
        """Return the computed CAP value."""
        return self.cap.item()


class PosteriorConsistency(ApproximationCapacity):
    """Consistency-only CAP: the beta-free limit of CAP's posterior log-cosine.

    CAP is, per paired sample, the log overlap of the two-state Gibbs posteriors
    `p_k = softmax(-beta * E_k(.))` induced by the two score sets. Normalising
    that inner product splits it exactly into a consistency and a confidence
    term,

        CAP_i = log <p1, p2> = log cos(p1, p2) + log|p1| + log|p2|,

    where the norm term depends only on how peaked each posterior is and is
    blind to whether the two agree. That is the informative half. The log-cosine
    is the consistency half, but it cannot serve as an objective on its own: it
    is identically zero at beta = 0, so any beta chosen to maximise it collapses
    onto the uninformative solution where every model ties at zero.

    Expanding around beta = 0 removes the ambiguity. Writing
    `D_k = E_k(1) - E_k(0)` for the per-sample energy gap,

        log cos(p1, p2) = -(beta**2 / 8) * sum_i (D1_i - D2_i)**2 + O(beta**4),

    so beta enters only through a common prefactor and drops out of any ranking
    of models. This metric reports that beta-free limit, up to that constant,

        consistency = -mean_i (D1_i - D2_i)**2,

    which is <= 0, is maximised (zero only when the two paired score sets induce
    identical energy gaps), and requires no inner optimisation at all.

    The normalisation and energy front-end is inherited from CAP unchanged, so
    the same `cap_metric_config` block can be reused verbatim. The `beta_min`,
    `beta_max` and `n_beta` entries of that config are accepted for drop-in
    compatibility and have no effect here; `curve`, `beta_star` and `at_bound`
    stay None.
    """

    def update(self, logits1: Tensor, logits2: Tensor, **kwargs):
        """Compute the beta-free consistency limit over the paired scores."""
        with torch.no_grad():
            # In the dtype of the normalised scores rather than CAP's float64, so that
            # the consistency values of earlier runs are reproduced bit for bit.
            gap1, gap2 = self._energy_gaps(*self.normalizer_fn(logits1, logits2))
            self.cap.fill_((-torch.mean((gap1 - gap2) ** 2)).item())

    def compute(self) -> Tensor:
        """Return the computed consistency value."""
        return self.cap.item()

import torch
from torchmetrics import Metric


def compute_tie_weight(
    scores: torch.Tensor, threshold: torch.Tensor, fpr: float
) -> torch.Tensor:
    """Return the weight of the scores equal to the threshold that makes the count of
    scores passing it equal fpr * scores.numel(), clipped to [0, 1].

    The weight is meaningful only when the threshold is one of the scores.
    """
    n_above = (scores > threshold).sum().double()
    n_tied = (scores == threshold).sum().double()
    return ((fpr * scores.numel() - n_above) / n_tied).clamp(0.0, 1.0)


def count_exceedances(
    scores: torch.Tensor, threshold: torch.Tensor, tie_weight: torch.Tensor | float
) -> torch.Tensor:
    """Count the scores above the threshold plus tie_weight times those equal to it."""
    n_tied = (scores == threshold).sum().double()
    return (scores > threshold).sum() + tie_weight * n_tied


class AnomalyRate(Metric):
    """Counts exceedances above a threshold at a target false positive rate.

    This metric can operate in two modes:

    - If `base_rate` is provided, `target_rate` is interpreted as a physical rate
      and converted to an FPR via `target_rate / base_rate`.
    - If `base_rate` is `None`, `target_rate` is interpreted directly as an FPR.

    Scores above the threshold count 1 and scores equal to it count `tie_weight`,
    which `set_threshold` fixes so that the count on the background scores is
    FPR * N; a constant score therefore has efficiency equal to the FPR.

    `rate()` returns:
    - the physical rate if `base_rate` is provided
    - the empirical exceedance probability otherwise

    `efficiency()` always returns the empirical exceedance fraction.
    `compute(quantity)` returns either, synchronised across processes and uncached.
    """

    def __init__(self, target_rate: float, base_rate: float | None):
        super().__init__(compute_with_cache=False)
        self.target_rate = float(target_rate)
        self.base_rate = None if base_rate is None else float(base_rate)

        self.add_state(
            "ntriggered",
            default=torch.tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "nsamples",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state("threshold", default=torch.tensor(float("nan")))
        self.add_state("tie_weight", default=torch.tensor(1.0, dtype=torch.float64))

    def _compute_target_fpr(self) -> float:
        """Compute the target false positive rate."""
        if self.base_rate is None:
            fpr = self.target_rate
        else:
            if self.base_rate <= 0:
                raise ValueError("base_rate must be positive.")
            fpr = self.target_rate / self.base_rate

        if not (0.0 < fpr < 1.0):
            raise ValueError(f"Computed FPR must be in (0, 1), got {fpr}.")

        return fpr

    def set_threshold(self, bkg_score: torch.Tensor) -> None:
        """Set threshold and tie weight corresponding to the target FPR."""
        fpr = self._compute_target_fpr()
        q = 1.0 - fpr
        bkg_score = bkg_score.float()
        # With distinct scores, 1 + floor(fpr (N - 1)) of them lie at or above the
        # threshold, give or take one from torch.quantile's float32 rank; scores
        # equal to it count with the tie weight, which makes the count fpr N unless
        # the clip to [0, 1] binds.
        thr = torch.quantile(bkg_score, q, interpolation="higher")
        self.threshold.copy_(thr)
        self.tie_weight.copy_(compute_tie_weight(bkg_score, thr, fpr))

    def apply_threshold(
        self, threshold: torch.Tensor | float, tie_weight: torch.Tensor | float = 1.0
    ) -> None:
        """Store an externally computed threshold and the tie weight that goes with it.

        The default tie weight of 1 accepts every score equal to the threshold.
        """
        threshold = torch.as_tensor(
            threshold,
            device=self.threshold.device,
            dtype=self.threshold.dtype,
        )
        self.threshold.copy_(threshold)
        self.tie_weight.copy_(torch.as_tensor(tie_weight))

    def update(self, anomaly_score: torch.Tensor) -> None:
        """Update exceedance counts from anomaly scores."""
        if torch.isnan(self.threshold).item():
            raise RuntimeError(
                "Threshold has not been set. Call set_threshold() before update()."
            )

        anomaly_score = anomaly_score.float().view(-1)
        self.ntriggered += count_exceedances(
            anomaly_score, self.threshold, self.tie_weight
        )
        self.nsamples += anomaly_score.numel()

    def efficiency(self) -> torch.Tensor:
        """Return the fraction of the scores seen so far that pass the threshold."""
        if self.nsamples == 0:
            raise RuntimeError("Cannot compute metric with zero samples.")

        return (self.ntriggered / self.nsamples).float()

    def rate(self) -> torch.Tensor:
        """Return the physical rate, or the efficiency when base_rate is None."""
        if self.base_rate is None:
            return self.efficiency()
        return self.efficiency() * self.base_rate

    def compute(self, quantity: str) -> torch.Tensor:
        """Compute either rate or efficiency."""
        if quantity == "efficiency":
            return self.efficiency()

        if quantity == "rate":
            return self.rate()

        raise ValueError(f"{quantity} is not a valid quantity to compute!")

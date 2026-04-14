import torch
from torchmetrics import Metric


class AnomalyRate(Metric):
    """Counts exceedances above a threshold at a target false positive rate.

    This metric can operate in two modes:

    - If `base_rate` is provided, `target_rate` is interpreted as a physical rate
      and converted to an FPR via `target_rate / base_rate`.
    - If `base_rate` is `None`, `target_rate` is interpreted directly as an FPR.

    `compute("rate")` returns:
    - the physical rate if `base_rate` is provided
    - the empirical exceedance probability otherwise

    `compute("efficiency")` always returns the empirical exceedance fraction.
    """

    def __init__(self, target_rate: float, base_rate: float | None):
        super().__init__()
        self.target_rate = float(target_rate)
        self.base_rate = None if base_rate is None else float(base_rate)

        self.add_state(
            "ntriggered",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "nsamples",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state("threshold", default=torch.tensor(float("nan")))

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
        """Set threshold corresponding to the target FPR."""
        fpr = self._compute_target_fpr()
        q = 1.0 - fpr
        thr = torch.quantile(
            bkg_score.float(),
            q,
            interpolation="higher",
        ).to(self.threshold.device)
        self.threshold.copy_(thr)

    def apply_threshold(self, threshold: torch.Tensor | float) -> None:
        """Store an externally computed threshold."""
        threshold = torch.as_tensor(
            threshold,
            device=self.threshold.device,
            dtype=self.threshold.dtype,
        )
        self.threshold.copy_(threshold)

    def update(self, anomaly_score: torch.Tensor) -> None:
        """Update exceedance counts from anomaly scores."""
        if torch.isnan(self.threshold).item():
            raise RuntimeError(
                "Threshold has not been set. Call set_threshold() before update()."
            )

        anomaly_score = anomaly_score.float().view(-1)
        ntriggered = (anomaly_score >= self.threshold).sum()
        self.ntriggered += ntriggered
        self.nsamples += anomaly_score.numel()

    def compute(self, quantity: str) -> torch.Tensor:
        """Compute either rate or efficiency."""
        if self.nsamples == 0:
            raise RuntimeError("Cannot compute metric with zero samples.")

        efficiency = self.ntriggered.float() / self.nsamples

        if quantity == "efficiency":
            return efficiency

        if quantity == "rate":
            if self.base_rate is None:
                return efficiency
            return efficiency * self.base_rate

        raise ValueError(f"{quantity} is not a valid quantity to compute!")

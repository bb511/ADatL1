import torch
import numpy as np
from torchmetrics import Metric


class AnomalyRate(Metric):
    """Calculates the how many anomalies are detected given a certain trigger rate.

    :target_rate: Float specifying the target rate of anomalies.
    :bc_rate: Float containing the bunch crossing rate in kHz. This is the
        revolution frequency of the LHC times the number of proton bunches in the
        LHC tunnel, which gives the total rate of events that are processed by
        the L1 trigger.
    """

    def __init__(self, target_rate: float, bc_rate: float):
        super().__init__()
        self.target_rate = target_rate
        self.bc_rate = bc_rate

        self.add_state("ntriggered", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("nsamples", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def set_threshold(self, bkg_score: torch.Tensor) -> None:
        """Get the score threshold for a certain rate to determine anomalies.

        Set a threshold for which you will achieve a certain rate of background events,
        i.e., set the threshold at a particular FPR.

        :bkg_score: Torch tensor containing scores for background samples.
        """
        self.threshold = np.percentile(
            bkg_score, 100 - (self.target_rate / self.bc_rate) * 100
        )

    def update(self, anomaly_score: torch.Tensor) -> None:
        """The anomaly score can be defined in a number of ways. See model code."""
        ntriggered = len(np.where(anomaly_score > self.threshold)[0])
        self.ntriggered += ntriggered
        self.nsamples += anomaly_score.numel()

    def compute(self) -> torch.Tensor:
        return self.ntriggered.float() * self.bc_rate / self.nsamples

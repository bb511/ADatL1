from typing import List, Tuple

import torch
import numpy as np
from torchmetrics import Metric


class AnomalyRate(Metric):
    def __init__(
        self,
        target_rates: List[float],
        bc_rate: float,
    ):
        """Calculates the how many anomalies are detected given a certain trigger rate.

        :target_rate: List of floats for the hypothetical trigger rates to use.
            This expected in kHz.
        :bc_rate: Float containing the bunch crossing rate in kHz. This is the
            revolution frequency of the LHC times the number of proton bunches in the
            LHC tunnel, which gives the total rate of events that are processed by
            the L1 trigger.
        """
        super().__init__()
        self.target_rates = target_rates
        self.bc_rate = bc_rate

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def set_threshold(self, mu_bkg: torch.Tensor) -> None:
        """Get the latent space threshold for a certain rate to determine anomalies.

        :mu_bkg: Torch tensor containing mean of background data samples in the latent
            space of the variational autoencoder model. The training set is taken
            conventionally as background for this application.
        """
        bkg_score = mu_bkg.pow(2)
        self.thresholds = {}
        for rate in self.target_rates:
            self.thresholds[rate] = np.percentile(bkg_score, 100 - (rate/self.bc_rate)*100)

    def update(self, mu: torch.Tensor) -> None:
        """The anomaly score is the mu^2 of the sample in the VAE latent space."""
        anomaly_score = mu.pow(2)


    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total

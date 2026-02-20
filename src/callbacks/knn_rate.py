# Callback that computes the AUPRC based on kNN output.
from collections import defaultdict
from pathlib import Path
import pickle

import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
from src.callbacks.metrics.rate import AnomalyRate

from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


class KNNRate(Callback):
    """Runs a kNN algorithm on the outputs of the network and computes AUPRC.

    Expects that there is a main_val dataloader that contains zero-bias data. This
    dataloader is used to compute the fake positives in the precision.
    The other dataloaders are treated as signal no matter what they actually are.
    The AUPRC is computed under these assumptions.

    :param output_name: String specifying the name of the dictionary entry where the
        output to run the kNN on is found. Every model outputs a dictionary
        during the model_step.
    :param k: Int specifying the number of centroids for the kNN algorithm.
    :param reference_sample_size: Int specifying the size of the reference sample to
        compute the knn anomaly score with respect to.
    :param ds: List of strings specifying which data sets to evaluate the kNN on.
    """

    def __init__(
        self,
        output_name: str,
        target_rates: list[int],
        bc_rate: int,
        k: int = 50,
        reference_sample_size: int = 40_000,
        bkg_sample_size: int = 200_000,
    ):
        super().__init__()
        self.device = None

        self.output_name = output_name
        self.k = k
        self.target_rates = target_rates
        self.reference_sample_size = reference_sample_size
        self.bkg_sample_size = bkg_sample_size
        self.bc_rate = bc_rate

        self.n_bkg_scored = 0

        self.knnrate_summary = defaultdict(lambda: defaultdict(float))

    def on_validation_start(self, trainer, pl_module):
        """Check if 'main_val' labeled dataloader is among the test dataloaders."""
        self.device = pl_module.device

        dset_names = list(trainer.val_dataloaders.keys())
        first_val_dset_key = list(trainer.val_dataloaders.keys())[0]
        if first_val_dset_key != "main_val":
            raise ValueError(
                "KNN Rate callback requires main_val first in the val dict!"
            )

    def on_validation_epoch_start(self, trainer, pl_module):
        """Clear the results dictionaries."""
        self.mainval_score_data = []
        self.rates = {}
        self.z_ref = []

        self.mainval_rate_computed = 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ):
        """Run a kNN on the output of the neural network.

        Use a given amount of samples as a reference bkg data set to compute the kNN
        scores with respect to.
        Use all other samples as bkg in the BinaryAveragePrecision.
        Finally, compute BinaryAveragePrecision for all the signal data sets.
        """
        self.dataset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]

        _, _, _, labels = batch
        labels = self._check_labels(labels, self.dataset_name)
        z = outputs[self.output_name].detach()

        if self.dataset_name == "main_val":
            batch_size = labels.size(0)
            nsamples = (batch_idx + 1) * batch_size
            if nsamples <= self.reference_sample_size:
                self.z_ref.append(z)
            elif nsamples - self.reference_sample_size <= self.bkg_sample_size:
                self._add_bkg_preds(z, labels)
            elif self.mainval_rate_computed == 0:
                self.mainval_score_data = torch.cat(self.mainval_score_data, dim=0)
                self._compute_mainval_rate()
                self.mainval_rate_computed = 1

            return

        scores = self._knn_scores(z)
        self._initialize_rate_metric()
        self._compute_batch_rate(scores)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Log the auprc computed on each of the data sets."""
        if self.n_bkg_scored == 0:
            raise ValueError(
                f"No background negatives were scored. "
                f"reference_sample_size={self.reference_sample_size} may be too large "
                f"for main_test."
            )

        for rate_name, rate in self.rates.items():
            pl_module.log_dict(
                {f"val/{rate_name}": rate.compute("rate")},
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    def _add_bkg_preds(self, z_bkg_batch: torch.Tensor, labels: torch.Tensor):
        """Compute kNN prediction using zero_bias data.

        Treat these predictions as kNN predictions on normal, non-anomalous data.
        """
        if isinstance(self.z_ref, list):
            self.z_ref = torch.cat(self.z_ref, dim=0)[: self.reference_sample_size]
            self.z_ref = torch.nn.functional.normalize(self.z_ref, dim=1)

        scores = self._knn_scores(z_bkg_batch)
        self.mainval_score_data.append(scores)
        self.n_bkg_scored += labels.numel()

    def _check_labels(self, labels: torch.Tensor, dataset_name: str):
        """Checks if the labels are as expected depending on the data set.

        The labels of the main_val data set are supposed to all be 0s.
        The labels of any of the other data sets, if the are not 1, are converted to
        be all 1s. All other data sets are treated as signals.
        """
        if dataset_name == "main_val":
            if torch.count_nonzero(labels) > 0:
                raise ValueError(
                    "KNN_AUPRC callback needs 'main_val' loader labels to all be 0s."
                )
            return labels

        if (labels == 0).any():
            raise ValueError(
                "KNN_AUPRC callback needs all labels to be not 0 for the other data "
                "except for main_val. Check this is the case."
            )

        return (labels != 0).to(labels.dtype).detach()

    @torch.no_grad()
    def _knn_scores(self, z_query: torch.Tensor):
        """Compute the k-nearest neighbour average cosine similarity.

        L2 normalise the data.
        Take the z_query data and compute the cosine similarity of each sample with
        respect to a reference data set z_ref. Then, choose the top k most similar and
        average over them to get an anomaly score per sample.
        """
        z_query = torch.nn.functional.normalize(z_query, dim=1)
        sim = z_query @ self.z_ref.T  # [Bq, Nref]
        topk_sim, _ = torch.topk(sim, self.k, largest=True, dim=1)
        knn_dist = 1.0 - topk_sim
        score = knn_dist[:, -1]

        return score

    def _compute_mainval_rate(self):
        """Computes the desired rates on the main validation data set.

        This is a sanity check. The threshold computed on the mainval and applied to the
        mainval data should return the rate for which this threshold was computed.
        """
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.bc_rate).to(self.device)
            rate.set_threshold(self.mainval_score_data)
            rate.update(self.mainval_score_data)
            rate_name = f"{self.dataset_name}/knnradius_rate{target_rate}kHz"
            self.rates.update({rate_name: rate})

    def _initialize_rate_metric(self):
        """Initializes the rate metric for a dataset for each given target rate.

        The anomaly rate metric object is initialised. Then, the threshold is computed
        given the main_val metric distribution. This threshold is then used to compute
        the rate on the other data sets differing from main_val.
        """
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.bc_rate).to(self.device)
            rate.set_threshold(self.mainval_score_data)
            rate_name = f"{self.dataset_name}/knnradius_rate{target_rate}kHz"
            self.rates.update({rate_name: rate})

    def _compute_batch_rate(self, scores: torch.Tensor):
        """Done after knowing the rate thresholds.

        For all the other validation data sets, except the main one, this calculates
        which events pass the rate threshold for each batch and updates the total.
        """
        for target_rate in self.target_rates:
            rate_name = f"{self.dataset_name}/knnradius_rate{target_rate}kHz"
            self.rates[rate_name].update(scores)

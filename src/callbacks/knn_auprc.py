# Callback that computes the AUPRC based on kNN output.
from collections import defaultdict
from pathlib import Path
import pickle

import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
from torchmetrics.classification import BinaryAveragePrecision

from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


class KNNAUPRC(Callback):
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
    :param skip_ds: List of strings specifying which data sets to skip evaluating
        the kNN on.
    :param log_raw_mlflow: Boolean to decide whether to log the raw plots produced by
        this callback to mlflow artifacts. Default is True. An html gallery of all the
        plots is made by default, so if this is set to false, one can still view the
        plots produced with this callback in the gallery.
    """

    def __init__(
        self,
        output_name: str,
        k: int = 10,
        reference_sample_size: int = 10_000,
        skip_ds: list[str] = [],
    ):
        super().__init__()
        self.device = None
        self.k = k
        self.output_name = output_name
        self.skip_ds = skip_ds
        self.reference_sample_size = reference_sample_size

    def on_validation_start(self, trainer, pl_module):
        """Check if 'main_val' labeled dataloader is among the val dataloaders."""
        dset_names = list(trainer.val_dataloaders.keys())
        if not "main_val" in dset_names:
            raise ValueError("ROC callback needs main_val data set in the data dict!")

    def on_validation_epoch_start(self, trainer, pl_module):
        """Clear the results dictionaries."""
        self.device = pl_module.device
        self.auprcs = {}
        self.z_ref = []

        dset_names = list(trainer.val_dataloaders.keys())
        for ds_name in dset_names:
            if ds_name == "main_val" or ds_name in self.skip_ds:
                # Skip the background, no point in computing AUPRC for bkg only.
                continue
            self.auprcs[f"{ds_name}"] = BinaryAveragePrecision().to(self.device)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ):
        """Run a kNN on the output of the neural network.

        Use a given amount of samples as a reference bkg data set to compute the kNN
        scores with respect to.
        Use all other samples as bkg in the BinaryAveragePrecision.
        Finally, compute BinaryAveragePrecision for all the signal data sets.
        """
        dataset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dataset_name in self.skip_ds:
            return

        _, _, labels = batch
        labels = self._check_labels(labels, dataset_name)
        z = outputs[self.output_name].detach()

        if dataset_name == "main_val":
            batch_size = labels.size(0)
            num_batches = trainer.num_val_batches[dataloader_idx]
            if num_batches <= 1:
                raise ValueError("KNN AUPRC callback needs > 1 batch for main_val")
            if (batch_idx + 1)*batch_size <= self.reference_sample_size:
                self.z_ref.append(z)
            else:
                self._add_bkg_preds_to_all(z, labels)

            return

        scores = self._knn_scores(z)
        self.auprcs[f"{dataset_name}"].update(scores.float(), labels)

    def _add_bkg_preds_to_all(self, z_bkg_batch: torch.Tensor, labels: torch.Tensor):
        """Compute kNN prediction using zero_bias data.

        Treat these predictions as kNN predictions on normal, non-anomalous data.
        """
        if isinstance(self.z_ref, list):
            self.z_ref = torch.cat(self.z_ref, dim=0)[:self.reference_sample_size]
            self.z_ref = torch.nn.functional.normalize(self.z_ref, dim=1)

        scores = self._knn_scores(z_bkg_batch)
        for dataset_name in self.auprcs.keys():
            self.auprcs[f"{dataset_name}"].update(scores.float(), labels)

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
        sim = z_query @ self.z_ref.T           # [Bq, Nref]
        topk_sim, _ = torch.topk(sim, self.k, largest=True, dim=1)
        knn_dist = 1.0 - topk_sim

        return knn_dist.mean(dim=1)       # [Bq]

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Log the auprc computed on each of the data sets."""
        for ds_name, auprc in self.auprcs.items():
            pl_module.log_dict(
                {f"val/{ds_name}/knn_auprc": auprc.compute()},
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
            )

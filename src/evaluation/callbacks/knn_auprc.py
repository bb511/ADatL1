# Callback that computes the ROC curves for each signal given a checkpoint.
from collections import defaultdict
from pathlib import Path
import pickle

import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
from torchmetrics.classification import BinaryAveragePrecision

from src.evaluation.callbacks import utils
from src.plot import horizontal_bar
from src.plot import matrix


class KNNAUPRC(Callback):
    """Runs a kNN algorithm on the outputs of the network and computes AUPRC.

    Expects that there is a main_test dataloader that contains zero-bias data. This
    dataloader is used to compute the fake positives in the precision.
    The other dataloaders are treated as signal no matter what they actually are.
    The AUPRC is computed under these assumptions..

    :param output_name: String specifying the name of the dictionary entry where the
        output to run the kNN on is found. Every model outputs a dictionary
        during the model_step.
    :param k: Int specifying the number of centroids for the kNN algorithm.
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
        log_raw_mlflow: bool = True
    ):
        super().__init__()
        self.device = None
        self.k = k
        self.output_name = output_name
        self.reference_sample_size = reference_sample_size
        self.log_raw_mlflow = log_raw_mlflow
        self.auprcs_summary = defaultdict(float)

    def on_test_start(self, trainer, pl_module):
        """Check if 'main_test' labeled dataloader is among the test dataloaders."""
        dset_names = list(trainer.test_dataloaders.keys())
        if not "main_test" in dset_names:
            raise ValueError("ROC callback needs main_test data set in the data dict!")

    def on_test_epoch_start(self, trainer, pl_module):
        """Clear the results dictionaries."""
        self.device = pl_module.device
        self.auprcs = {}
        self.z_ref = []

        dset_names = list(trainer.test_dataloaders.keys())
        for ds_name in dset_names:
            if ds_name == "main_test":
                # Skip the background, no point in computing ROC for bkg only.
                continue
            self.auprcs[f"{ds_name}"] = BinaryAveragePrecision().to(self.device)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ):
        """Run a kNN on the output of the neural network."""
        dataset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        _, _, labels = batch
        labels = self._check_labels(labels, dataset_name)
        z = outputs[self.output_name].detach()

        if dataset_name == "main_test":
            num_batches = trainer.num_test_batches[dataloader_idx]
            if num_batches <= 1:
                raise ValueError("KNN AUPRC callback needs > 1 batch for main_test")

            if (batch_idx + 1) <= num_batches/2:
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

        The labels of the main_test data set are supposed to all be 0s.
        The labels of any of the other data sets, if the are not 1, are converted to
        be all 1s. All other data sets are treated as signals.
        """
        if dataset_name == "main_test":
            if torch.count_nonzero(labels) > 0:
                raise ValueError(
                    "KNN_AUPRC callback needs 'main_test' loader labels to all be 0s."
                )
            return labels

        if (labels == 0).any():
            raise ValueError(
                "KNN_AUPRC callback needs all labels to be not 0 for the other data "
                "except for main_test. Check this is the case."
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

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Log the auprc computed on each of the data sets in the artifacts sec."""
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        plot_folder = ckpts_dir / "plots" / ckpt_name / "knn_auprcs"
        plot_folder.mkdir(parents=True, exist_ok=True)

        auprcs_per_dataset = {
            ds_name: auprc.compute().float().item()
            for ds_name, auprc in self.auprcs.items()
        }

        xlabel = f"AUPRC {self.output_name}"
        ylabel = ' '
        horizontal_bar.plot_yright(
            auprcs_per_dataset, auprcs_per_dataset, xlabel, ylabel, plot_folder
        )
        self._store_summary(auprcs_per_dataset, ckpt_name)

        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name,
            "knn_auprcs",
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=f"knn_auprcs_{self.output_name.replace('/', '_')}"
        )

    def _get_dsname(self, roc_name: str):
        """Retrieves the data set name from string specifying rate name."""
        dataset_name = roc_name.split("/")[0]
        return dataset_name

    def _store_summary(self, auprcs_per_dataset: dict, ckpt_name: str):
        """Store a summary statistic for the losses of one checkpoint.

        Here, we store losses computed on all the test data sets. At the end, this is
        plotted as a matrix.
        """
        ckpt_ds_name = utils.misc.get_ckpt_ds_name(ckpt_name)
        self.auprcs_summary[ckpt_ds_name] = auprcs_per_dataset

    def plot_summary(self, trainer, root_folder: Path):
        """Plot the summary metrics accummulated in eff_summary and reset this attr."""
        plot_folder = root_folder / "plots"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)

        matrix.plot(self.auprcs_summary, self.output_name, plot_folder)
        utils.mlflow.log_plots_to_mlflow(trainer, None, "losses", plot_folder)

    def clear_crit_summary(self):
        self.auprcs_summary.clear()

    def get_optimized_metric(self, ckpt_ds: str = None):
        """Get one number that one should optimize on this callback.

        Here, the auprc mean**2/var over test datasets is computed for each checkpoint.
        Then, the maximum is computed over checkpoints if no ckpt_ds is provided.
        Otherwise, just get the mean auprc over specified ckpt_ds.
        """
        eps = 1e-14
        if ckpt_ds:
            ckpt_auprcs = list(self.auprcs_summary[ckpt_ds].values())
            ckpt_auprcs_mean = np.mean(ckpt_auprcs)
            ckpt_auprcs_var = np.var(ckpt_auprcs)
            optimized_metric = ckpt_auprcs_mean ** 2 / (ckpt_auprcs_var + eps)
            return ckpt_ds, optimized_metric

        ckpt_optim_metrics = []
        for cktp_ds in self.auprcs_summary.keys():
            ckpt_auprcs = list(self.auprcs_summary[ckpt_ds].values())
            ckpt_auprcs_mean = np.mean(ckpt_auprcs)
            ckpt_auprcs_var = np.var(ckpt_auprcs)
            optimized_metric = ckpt_auprcs_mean ** 2 / (ckpt_auprcs_var + eps)
            ckpt_optim_metrics.append(optimized_metric)

        optimized_metric = max(ckpt_optim_metrics)
        return ckpt_ds, optimized_metric

    def _cache_summary(self, cache_folder: Path):
        """Cache the summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            plain_dict = utils.misc.to_plain_dict(self.auprcs_summary)
            pickle.dump(plain_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

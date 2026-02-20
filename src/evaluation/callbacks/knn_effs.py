# Callback that computes the ROC curves for each signal given a checkpoint.
from collections import defaultdict
from pathlib import Path
import pickle

import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
from src.evaluation.callbacks.metrics.rate import AnomalyRate

from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


class KNNEffs(Callback):
    """Runs a kNN algorithm on the outputs of the network and computes rate.

    Expects that there is a main_test dataloader that contains zero-bias data. This
    dataloader is used to compute the fake positives in the precision.
    The other dataloaders are either signal or background simulation.

    :param output_name: String specifying the name of the dictionary entry where the
        output to run the kNN on is found. Every model outputs a dictionary
        during the model_step.
    :param k: Int specifying the number of centroids for the kNN algorithm.
    :param reference_sample_size: Int specifying the size of the reference sample to
        compute the knn anomaly score with respect to.
    :param bkg_sample_size: Integer specifying the number of background samples for
        kNN evaluatoin.
    :param ds: List of strings with the data set names to compute the kNN on.
    :param log_raw_mlflow: Boolean to decide whether to log the raw plots produced by
        this callback to mlflow artifacts. Default is True. An html gallery of all the
        plots is made by default, so if this is set to false, one can still view the
        plots produced with this callback in the gallery.
    """

    def __init__(
        self,
        output_name: str,
        ds: list[str],
        target_rates: list[int],
        bc_rate: float,
        k: int = 50,
        reference_sample_size: int = 40_000,
        bkg_sample_size: int = 200_000,
        log_raw_mlflow: bool = True,
    ):
        super().__init__()
        self.device = None

        self.output_name = output_name
        self.ds = ds
        self.k = k
        self.target_rates = target_rates
        self.bc_rate = bc_rate
        self.reference_sample_size = reference_sample_size
        self.bkg_sample_size = bkg_sample_size
        self.log_raw_mlflow = log_raw_mlflow

        self.n_bkg_scored = 0

        self.knneffs_summary = defaultdict(lambda: defaultdict(float))

    def on_test_start(self, trainer, pl_module):
        """Check if 'main_test' labeled dataloader is among the test dataloaders."""
        self.device = pl_module.device

        first_test_dset_key = list(trainer.test_dataloaders.keys())[0]
        if first_test_dset_key != "main_test":
            raise ValueError(
                "KNN callback needs main_test data set to be the first "
                "in the data dictionary!"
            )

    def on_test_epoch_start(self, trainer, pl_module):
        """Clear the results dictionaries."""
        self.main_rate = {}
        self.sig_rates = {}
        self.bkg_rates = {}
        self.maintest_score_data = []
        self.z_ref = []

        self.maintest_rate_computed = 0

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ):
        """Run a kNN on the output of the neural network.

        Use a given amount of samples as a reference bkg data set to compute the kNN
        scores with respect to.
        Use all other samples as bkg in the BinaryAveragePrecision.
        Finally, compute BinaryAveragePrecision for all the signal data sets.
        """
        self.dataset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if self.dataset_name != 'main_test' and not self.dataset_name in self.ds:
            return

        _, _, _, labels = batch
        labels = self._check_labels(labels, self.dataset_name)
        z = outputs[self.output_name].detach()

        if self.dataset_name == "main_test":
            batch_size = labels.size(0)
            nsamples = (batch_idx + 1) * batch_size
            if nsamples <= self.reference_sample_size:
                self.z_ref.append(z)
            elif nsamples - self.reference_sample_size <= self.bkg_sample_size:
                self._add_bkg_preds(z, labels)
            elif self.maintest_rate_computed == 0:
                self.maintest_score_data = torch.cat(self.maintest_score_data, dim=0)
                self._compute_maintest_rate()
                self.maintest_rate_computed = 1

            return

        scores = self._knn_scores(z)
        self._initialize_rate_metric(labels)
        self._compute_batch_rate(scores, labels)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Log the auprc computed on each of the data sets in the artifacts sec."""
        self._check_bkg_negatives()
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        plot_folder = ckpts_dir / "plots" / ckpt_name / "knn_auprcs"
        plot_folder.mkdir(parents=True, exist_ok=True)

        for trate in self.target_rates:
            # Construct rate per data set for specific metric name and target rate.
            sig_effs = self._compute_eff(trate, self.sig_rates)
            bkg_effs = self._compute_eff(trate, self.bkg_rates)
            main_eff = self._compute_eff(trate, self.main_rate)
            effs = sig_effs | bkg_effs | main_eff

            trate_name = f"{trate} kHz"
            ascore = f"anomaly score: knn radius"
            xlabel = f"eff at threshold: {trate_name}\n{ascore}"
            self._plot(effs, xlabel, plot_folder, percent=True)
            self._store_summary(sig_effs, bkg_effs, ckpt_name, trate)

        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name,
            "knn_effs",
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=f"knn_effs_{self.output_name.replace('/', '_')}",
        )

    def plot_summary(self, trainer, root_folder: Path):
        """Plot the summary metrics accummulated in eff_summary and reset this attr."""
        plot_folder = root_folder / "plots" / "knneffs_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)

        for target_rate in self.knneffs_summary.keys():
            # Get the summary metric per checkpoint.
            smet = self.knneffs_summary[target_rate]

            # Configure plot.
            trate = f"{target_rate} kHz"
            ascore = f"anomaly score: knn radius"
            xlabel = f"25% CVaR at threshold: {trate}\n{ascore}"
            self._plot(smet, xlabel, plot_folder)

        utils.mlflow.log_plots_to_mlflow(trainer, None, "knn_effs", plot_folder)

    def clear_crit_summary(self):
        self.knneffs_summary.clear()

    def get_optimized_metric(self, target_rate: float):
        """Get one number that one should optimize on this callback.

        Here, it's the maximum of the summary metric across checkpoints corresponding
        to a certain checkpointing criterion.
        """
        if not target_rate in list(self.knneffs_summary.keys()):
            raise ValueError(
                f"KNN rate callback for metric {self.output_name} did not calculate "
                f"rate at rate {target_rate}. Choose {self.knneffs_summary.keys()}."
            )

        metric_across_ckpts = self.knneffs_summary[target_rate]
        max_ckpt_ds = max(metric_across_ckpts, key=metric_across_ckpts.get)
        max_metric_value = metric_across_ckpts[max_ckpt_ds]
        return max_ckpt_ds, max_metric_value

    def _compute_eff(self, relevant_target_rate: float, rate_dict: dict):
        """Compute the efficiency in given rate dictionary."""
        rates = {
            self._get_dsname(rate_name): val.compute("efficiency").item()
            for rate_name, val in rate_dict.items()
            if str(relevant_target_rate) in rate_name
        }
        return rates

    def _plot(self, data: dict, xlabel: str, plot_folder: Path, percent: bool = False):
        """Plot the efficiency per data set for the knn radius at target rate."""
        ylabel = " "
        horizontal_bar.plot_yright(data, data, xlabel, ylabel, plot_folder, percent)

    def _store_summary(self, sig_rates: dict, bkg_rates: dict, ckpt: str, trate: float):
        """Store the summary statistic for the rate for one checkpoint.

        We compute the CVaR robustness metric by collecting the TPR values across all
        simulated anomaly types and averaging the worst fraction of them (e.g., the
        lowest 20%). Concretely, we select the smallest subset of TPR values
        corresponding to the chosen tail level and compute their mean. This tail-average
        summarizes detection performance on the most challenging anomaly scenarios.
        """
        sig_data = np.fromiter(sig_rates.values(), dtype=float)
        bkg_data = np.fromiter(bkg_rates.values(), dtype=float)

        bkg_mean = bkg_data.mean() if bkg_data.size else 0.0

        alpha = 0.25
        if sig_data.size:
            k = max(1, int(np.ceil(alpha * sig_data.size)))
            worst = np.partition(sig_data, k - 1)[:k]
            sig_cvar = worst.mean()
        else:
            sig_cvar = 0.0

        summary_metric = 1e3 * (sig_cvar - bkg_mean)

        ckpt_ds = utils.misc.get_ckpt_ds_name(ckpt)
        self.knneffs_summary[trate][ckpt_ds] = summary_metric

    def _cache_summary(self, cache_folder: Path):
        """Cache the summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            plain_dict = utils.misc.to_plain_dict(self.knneffs_summary)
            pickle.dump(plain_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _get_dsname(self, rate_name: str):
        """Retrieves the data set name from string specifying rate name."""
        dataset_name = rate_name.split("/")[0]
        return dataset_name

    def _add_bkg_preds(self, z_bkg_batch: torch.Tensor, labels: torch.Tensor):
        """Compute kNN prediction using zero_bias data.

        Treat these predictions as kNN predictions on normal, non-anomalous data.
        These are the negative examples, scored with respect to z_ref, a reference
        sample of normal data.
        """
        if isinstance(self.z_ref, list):
            self.z_ref = torch.cat(self.z_ref, dim=0)[: self.reference_sample_size]
            self.z_ref = torch.nn.functional.normalize(self.z_ref, dim=1)

        scores = self._knn_scores(z_bkg_batch)
        self.maintest_score_data.append(scores)
        self.n_bkg_scored += labels.numel()

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

    def _compute_maintest_rate(self):
        """Computes the desired rates on the main test data set.

        This is a sanity check. The threshold computed on the maintest and applied to the
        maintest data should return the rate for which this threshold was computed.
        """
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.bc_rate).to(self.device)
            rate.set_threshold(self.maintest_score_data)
            rate.update(self.maintest_score_data)
            self.main_rate.update({f"{self.dataset_name}/{target_rate}": rate})

    def _initialize_rate_metric(self, labels: torch.Tensor):
        """Initialise the rate metric for signal or bkg data.

        The anomaly rate metric object is initialised. Then, the threshold is computed
        given the main_test metric distribution. This threshold is then used to compute
        the rate on the other data sets differing from main_test.
        """
        if bool(torch.all(labels >= 1)):
            self._initialize_sig_rate_metric()
        elif bool(torch.all(labels <= -1)):
            self._initialize_bkg_rate_metric()
        else:
            raise ValueError(
                "The efficiency test callback requires that the labels of the signal "
                "data are all >= 1 and labels of background data are all <= -1."
            )

    def _initialize_sig_rate_metric(self):
        """Initializes the rate metric for a sig dataset for each given target rate."""
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.bc_rate).to(self.device)
            rate.set_threshold(self.maintest_score_data)
            self.sig_rates.update({f"{self.dataset_name}/{target_rate}": rate})

    def _initialize_bkg_rate_metric(self):
        """Initializes the rate metric for a bkg dataset for each given target rate."""
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.bc_rate).to(self.device)
            rate.set_threshold(self.maintest_score_data)
            self.bkg_rates.update({f"{self.dataset_name}/{target_rate}": rate})

    def _compute_batch_rate(self, scores: torch.Tensor, labels: torch.Tensor):
        """Done after knowing the rate thresholds.

        For all the other test data sets, except the main one, this calculates
        which events pass the rate threshold for each batch and updates the total.
        """
        if bool(torch.all(labels >= 1)):
            self._compute_sig_batch_rate(scores)
        elif bool(torch.all(labels <= -1)):
            self._compute_bkg_batch_rate(scores)
        else:
            raise ValueError(
                "The efficiency test callback requires that the labels of the signal "
                "data are all >= 1 and labels of background data are all <= -1."
            )

    def _compute_sig_batch_rate(self, scores: torch.Tensor):
        """Compute batch rate of a signal dataset."""
        for target_rate in self.target_rates:
            rate_name = f"{self.dataset_name}/{target_rate}"
            self.sig_rates[rate_name].update(scores)

    def _compute_bkg_batch_rate(self, scores: torch.Tensor):
        """Compute batch rate of a background data set."""
        for target_rate in self.target_rates:
            rate_name = f"{self.dataset_name}/{target_rate}"
            self.bkg_rates[rate_name].update(scores)

    def _check_labels(self, labels: torch.Tensor, dataset_name: str):
        """Checks if the labels are as expected depending on the data set.

        The labels of the main_test data set are supposed to all be 0s.
        The labels of any of the other data sets, if they are not 1, are converted to
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

    def _check_bkg_negatives(self):
        if self.n_bkg_scored == 0:
            raise ValueError(
                f"No background negatives were scored. "
                f"reference_sample_size={self.reference_sample_size} may be too large "
                f"for main_test."
            )


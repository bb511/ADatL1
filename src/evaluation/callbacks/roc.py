# Callback that computes the ROC curves for each signal given a checkpoint.
from collections import defaultdict
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback
from torchmetrics.classification import BinaryROC, BinaryAUROC

from src.evaluation.callbacks import utils
from src.plot import roc


class ROCs(Callback):
    """Calculates the ROC curve.

    Expects that there is a main_test dataloader that contains zero-bias data. This
    dataloader is used to compute the FPRs. The other dataloaders are treated as signal
    no matter what they actually are. The ROC curves are computed thus.

    :param metric_names: String specifying the metric name that is used as anomaly
        score. Needs to be in the outdict returned by the model.
    :param log_raw_mlflow: Boolean to decide whether to log the raw plots produced by
        this callback to mlflow artifacts. Default is True. An html gallery of all the
        plots is made by default, so if this is set to false, one can still view the
        plots produced with this callback in the gallery.
    """

    def __init__(self, metric_name: str, log_raw_mlflow: bool = True):
        super().__init__()
        self.device = None
        self.log_raw_mlflow = log_raw_mlflow
        self.metric_name = metric_name

    def on_test_start(self, trainer, pl_module):
        """Check if 'main_test' labeled dataloader is among the test dataloaders."""
        dset_names = list(trainer.test_dataloaders.keys())
        if not "main_test" in dset_names:
            raise ValueError("ROC callback needs main_test data set in the data dict!")

    def on_test_epoch_start(self, trainer, pl_module):
        """Clear the results dictionaries."""
        self.device = pl_module.device
        self.rocs = {}
        self.aurocs = {}

        dset_names = list(trainer.test_dataloaders.keys())
        for ds_name in dset_names:
            if ds_name == "main_test":
                # Skip the background, no point in computing ROC for bkg only.
                continue
            self.rocs[f"{ds_name}"] = BinaryROC().to(self.device)
            self.aurocs[f"{ds_name}"] = BinaryAUROC().to(self.device)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ):
        """Compute the number of true positives for the given number of thresholds."""
        dataset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        _, _, labels = batch
        labels = self._check_labels(labels, dataset_name)

        scores = outputs[self.metric_name].detach()
        if dataset_name == "main_test":
            self.rocs[name].update(scores, labels)
            self.aurocs[name].update(scores, labels)
            continue

        self.rocs[f"{dataset_name}"].update(scores, labels)
        self.aurocs[f"{dataset_name}"].update(scores, labels)

    def _check_labels(self, labels: torch.Tensor, dataset_name: str):
        """Checks if the labels are as expected depending on the data set.

        The labels of the main_test data set are supposed to all be 0s.
        The labels of any of the other data sets, if the are not 1, are converted to
        be all 1s. All other data sets are treated as signals.
        """
        if dataset_name == "main_test":
            if torch.count_nonzero(labels) > 0:
                raise ValueError(
                    "ROC callback needs 'main_test' dataloader labels to all be 0s."
                )
            return labels

        if (labels == 0).any():
            raise ValueError(
                "ROC callback needs all labels to be not 0 for the other data "
                "except for main_test. Check this is the case."
            )

        return (labels != 0).to(labels.dtype).detach()

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Log the anomaly rates computed on each of the data sets."""
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        plot_folder = ckpts_dir / "plots" / ckpt_name / "rocs"
        plot_folder.mkdir(parents=True, exist_ok=True)

        # Get the ROCs per data set.
        roc_per_dataset = {
            ds_name: self._convert2numpy(val.compute())
            for ds_name, val in self.rocs.items()
        }
        auroc_per_dataset = {
            ds_name: self._convert2numpy(val.compute())
            for ds_name, val in self.aurocs.items()
        }

        roc.plot(roc_per_dataset, auroc_per_dataset, self.metric_name, plot_folder)
        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name,
            "rocs",
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=f'rocs_{self.metric_name}'
        )

    def _convert2numpy(self, arr: tuple[torch.Tensor] | torch.Tensor):
        """Convert torch tensor or list of torch tensors to numpy arrays."""
        if isinstance(arr, tuple):
            arr = [tens.item() for tens in arr]
        elif isinstance(arr, torch.Tensor):
            arr = arr.item()
        else:
            raise ValueError(
                "ROC callback cannot convert torch tensor to numpy array. The given "
                f"array for conversion is of type {type(arr)}."
            )

        return arr

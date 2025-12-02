# Callback that computes the anomaly rate during training.
from collections import defaultdict
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger, Logger
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from torchmetrics.classification import BinaryROC, BinaryAUROC

from src.evaluation.callbacks import utils
from src.plot import roc


class ROCs(Callback):
    """Calculates the ROC curve.

    Expects that there is a main_test dataloader that contains zero-bias data. This
    dataloader is used to compute the FPRs. The other dataloaders are treated as signal
    no matter what they actually are. The ROC curves are computed thus.

    :metric_names: List of strings specifying which model metrics to use in computing
        the ROC curve, i.e., which model metrics to use as anomaly score.
    """

    def __init__(self, metric_names: list[str]):
        super().__init__()
        self.metric_names = metric_names

        self.rocs = {}
        self.aurocs = {}

    def on_test_start(self, trainer, pl_module):
        """Check if 'main_test' labeled dataloader is among the test dataloaders."""
        dset_names = list(trainer.test_dataloaders.keys())
        if not 'main_test' in dset_names:
            raise ValueError("ROC callback needs main_test data set in the data dict!")

    def on_test_epoch_start(self, trainer, pl_module):
        """Clear the results dictionaries."""
        dset_names = list(trainer.test_dataloaders.keys())
        for ds_name in dset_names:
            if ds_name == 'main_test':
                # Skip the background, no point in computing ROC for bkg only.
                continue
            for metric_name in self.metric_names:
                metric_name = metric_name.replace('/', '_')
                self.rocs[f"{ds_name}/{metric_name}"] = BinaryROC()
                self.aurocs[f"{ds_name}/{metric_name}"] = BinaryAUROC()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ):
        """Compute the number of true positives for the given number of thresholds."""
        dataset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        _, labels = batch
        labels = self._check_labels(labels, dataset_name)
        labels = labels

        for metric_name in self.metric_names:
            scores = outputs[metric_name]
            metric_name = metric_name.replace('/', '_')
            if dataset_name == 'main_test':
                self._add_bkg_to_all(scores, labels, metric_name)
                continue

            self.rocs[f"{dataset_name}/{metric_name}"].update(scores, labels)
            self.aurocs[f"{dataset_name}/{metric_name}"].update(scores, labels)

    def _add_bkg_to_all(self, scores: torch.Tensor, labels: torch.Tensor, mname: str):
        """Add the background scores to all entries in the ROCs dictionary."""
        for name in self.rocs.keys():
            if not name.endswith(f'/{mname}'):
                continue
            self.rocs[name].update(scores, labels)
            self.aurocs[name].update(scores, labels)

    def _check_labels(self, labels: torch.Tensor, dataset_name: str):
        """Checks if the labels are as expected depending on the data set.

        The labels of the main_test data set are supposed to all be 0s.
        The labels of any of the other data sets, if the are not 1, are converted to
        be all 1s. All other data sets are treated as signals.
        """
        if dataset_name == 'main_test':
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

        return (labels != 0).to(labels.dtype)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Log the anomaly rates computed on each of the data sets."""
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        plot_folder = ckpts_dir / 'plots' / ckpt_name / 'rocs'
        plot_folder.mkdir(parents=True, exist_ok=True)

        for metric_name in self.metric_names:
            # Get the ROCs per data set.
            metric_name = metric_name.replace('/', '_')
            roc_per_dataset = {
                self._get_dsname(roc_name): self._convert2numpy(val.compute())
                for roc_name, val in self.rocs.items()
                if roc_name.endswith(f'/{metric_name}')
            }
            auroc_per_dataset = {
                self._get_dsname(auroc_name): self._convert2numpy(val.compute())
                for auroc_name, val in self.aurocs.items()
                if auroc_name.endswith(f'/{metric_name}')
            }

            roc.plot(roc_per_dataset, auroc_per_dataset, metric_name, plot_folder)
            self._log_plots_to_mlflow(trainer, ckpt_name, plot_folder)

    def _convert2numpy(self, arr: tuple[torch.Tensor] | torch.Tensor):
        """Convert torch tensor or list of torch tensors to numpy arrays."""
        if isinstance(arr, tuple):
            arr = [tens.detach().to('cpu').numpy() for tens in arr]
        elif isinstance(arr, torch.Tensor):
            arr = arr.detach().to('cpu').numpy()
        else:
            raise ValueError(
                "ROC callback cannot convert torch tensor to numpy array. The given "
                f"array for conversion is of type {type(arr)}."
            )

        return arr

    def _get_dsname(self, rate_name: str):
        """Retrieves the data set name from string specifying rate name."""
        dataset_name = rate_name.split('/')[0]
        return dataset_name

    def _log_plots_to_mlflow(self, trainer, ckpt_name: str, plot_folder: Path):
        """Logs the plots generated by this callback to MLFlow."""
        mlflow_logger = utils.mlflow.get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return

        run_id = mlflow_logger.run_id
        arti_ckpt_dir = self._resolve_arti_dir(trainer, ckpt_name)

        # Log each image in the given plot_folder as an artifact.
        img_paths = sorted(plot_folder.glob('*.jpg'))
        for img_path in img_paths:
            mlflow_logger.experiment.log_artifact(
                run_id=run_id,
                local_path=str(img_path),
                artifact_path=str(arti_ckpt_dir),
            )

        # Generate an html gallery at the parent directory level.
        gallery_dir = arti_ckpt_dir.parent
        html_gallery = utils.mlflow.build_html(
            mlflow_logger, plot_folder, gallery_dir, arti_ckpt_dir
        )
        mlflow_logger.experiment.log_text(
            run_id, html_gallery, artifact_file=gallery_dir / 'index.html'
        )

    def _resolve_arti_dir(self, trainer, ckpt_name: str):
        """Resolve the artifacts directory where the plots will be stored in mlflow."""
        arti_dir = Path()
        if trainer.strat_name:
            arti_dir = arti_dir / trainer.strat_name / 'rocs'
        if trainer.metric_name:
            arti_dir = arti_dir / trainer.metric_name
        if trainer.criterion_name:
            arti_dir = arti_dir / trainer.criterion_name

        if arti_dir == Path():
            arti_dir = arti_dir / 'rocs'

        if ckpt_name in arti_dir.parts:
            return arti_dir

        return arti_dir / ckpt_name

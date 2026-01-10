# Callback that computes the loss of a given checkpoint on all test data sets.
from pathlib import Path
from collections import defaultdict
import pickle

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningDataModule

from src.evaluation.callbacks import utils
from src.plot import overlaid_hist


class ReconstructionPlots(Callback):
    """Plot the reconstructed data distributions overlaid with the input ones.

    :param nbatches: Int of number of batches to plot.
    :param output_name: String with the name of the key containing the output in the
        output dictionary of the model.
    :param datamodule: The datamodule that was used during the training of the model.
    :param ckpt_dataset: String specifying the dataset name of the checkpoints that
        this callback should be evaluated on.
    :param datasets: List of strings with the dataset names that the checkpoint should
        be evaluated on and the plots should be made for.
    :param log_raw_mlflow: Boolean to decide whether to log the raw plots produced by
        this callback to mlflow artifacts. Default is True. An html gallery of all the
        plots is made by default, so if this is set to false, one can still view the
        plots produced with this callback in the gallery.
    """

    def __init__(
        self,
        nbatches: int,
        output_name: str,
        datamodule: LightningDataModule,
        ckpt_dataset: str,
        datasets: list[str] = [],
        log_raw_mlflow: bool = True
    ):
        self.nbatches = nbatches
        self.output_name = output_name
        self.log_raw_mlflow = log_raw_mlflow
        self.ckpt_dataset = ckpt_dataset
        self.datasets = datasets
        self.object_feature_map = datamodule.loader.object_feature_map

    def on_test_epoch_start(self, trainer, pl_module):
        """Initialise dictionary used to accummulate the initial data and output.

        Additionally, get the name of the dataset that the checkpoint that is currently
        evaluated was made on.
        """
        self.input_accumulator = []
        self.output_accumulator = []
        self.mask_accumulator = []
        ckpt_name = Path(pl_module._ckpt_path).stem
        self.ckpt_ds_name = utils.misc.get_ckpt_ds_name(ckpt_name)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Accumulate output for nbatches and then plot overlaid histogram."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if self.ckpt_dataset != self.ckpt_ds_name:
            return
        if not dset_name in self.datasets:
            return
        if (batch_idx + 1) > self.nbatches:
            return

        x, m, _ = batch
        x = torch.flatten(x, start_dim=1)
        if hasattr(pl_module, 'features'):
            x = pl_module.features(x)

        m = torch.flatten(m, start_dim=1)

        if (batch_idx + 1) < self.nbatches:
            self.input_accumulator.append(x)
            self.mask_accumulator.append(m)
            self.output_accumulator.append(outputs[self.output_name])
        else:
            self.input_accumulator.append(x)
            self.mask_accumulator.append(m)
            self.output_accumulator.append(outputs[self.output_name])

            self.input_accumulator = torch.cat(self.input_accumulator, dim=0)
            self.output_accumulator = torch.cat(self.output_accumulator, dim=0)
            self.mask_accumulator = torch.cat(self.mask_accumulator, dim=0)

            if hasattr(pl_module, 'features') and not isinstance(pl_module.features, torch.nn.Identity):
                self._on_dataset_end_features(trainer, pl_module, dset_name)
            else:
                self._on_dataset_end(trainer, pl_module, dset_name)

            self.input_accumulator = []
            self.mask_accumulator = []
            self.output_accumulator = []

    def _on_dataset_end(self, trainer, pl_module, dataset_name: str):
        """Plot the overlaid histogram for each feature.

        This is done when accummulation is completed for one data set.
        """
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        plot_folder = ckpts_dir / "plots" / ckpt_name / "recos" / dataset_name
        plot_folder.mkdir(parents=True, exist_ok=True)

        for object_name, feature_map in self.object_feature_map.items():
            for feat_name, feat_idxs in feature_map.items():
                input_feat = self.input_accumulator[:, feat_idxs]
                output_feat = self.output_accumulator[:, feat_idxs]

                mask_feat = self.mask_accumulator[:, feat_idxs]
                input_feat = input_feat[mask_feat]
                output_feat = output_feat[mask_feat]

                # Detach and move to CPU for plotting
                input_feat = input_feat.detach().flatten().float().cpu().numpy()
                output_feat = output_feat.detach().flatten().float().cpu().numpy()
                overlaid_hist.plot_1d(
                    x1=input_feat, 
                    x2=output_feat, 
                    obj_name=object_name,
                    feat_name=feat_name,
                    save_dir=plot_folder,
                    label1='input',
                    label2='reco'
                )

        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name,
            "recos",
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=f"{dataset_name}_reco"
        )

    def _on_dataset_end_features(self, trainer, pl_module, dataset_name: str):
        """Plot the overlaid histogram for each feature.

        This is done when accummulation is completed for one data set.
        """
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        plot_folder = ckpts_dir / "plots" / ckpt_name / "recos" / dataset_name
        plot_folder.mkdir(parents=True, exist_ok=True)

        for feat_idx in range(self.input_accumulator.size(-1)):
            input_feat = self.input_accumulator[:, feat_idx]
            output_feat = self.output_accumulator[:, feat_idx]

            # Detach and move to CPU for plotting
            input_feat = input_feat.detach().flatten().float().cpu().numpy()
            output_feat = output_feat.detach().flatten().float().cpu().numpy()
            overlaid_hist.plot_1d(
                x1=input_feat,
                x2=output_feat,
                obj_name='noobject',
                feat_name=f'feat_{feat_idx}',
                save_dir=plot_folder,
                label1='input',
                label2='reco'
            )

        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name,
            "recos",
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=f"{dataset_name}_reco"
        )

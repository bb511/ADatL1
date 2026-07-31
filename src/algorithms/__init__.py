from typing import Optional
import inspect

import torch
from torch import nn, optim

from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.memory import garbage_collection_cuda


class ADLightningModule(LightningModule):
    """Base class for AD@L1 LightningModules."""

    def __init__(
        self,
        model: nn.Module = None,
        optimizer: optim.Optimizer = None,
        scheduler: Optional[DictConfig] = None,
        save_hyperparameters: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss"])

        self.model = model
        self._log_sum = {}
        self._log_nsteps = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Override with the forward pass."""
        return self.model(x)

    def model_step(self, batch) -> dict[str, torch.Tensor]:
        """Run one model step. Must be implemented by subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement `model_step`."
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        outdict = self.model_step(batch)
        self._log_dict(outdict, dataloader_idx=0)
        return outdict

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ):
        outdict = self.model_step(batch)
        self._log_dict(outdict, dataloader_idx=dataloader_idx)
        return outdict

    def test_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0):
        outdict = self.model_step(batch)
        return outdict

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Log metrics and clean up memory.

        For some reason on_train_epoch_end executes after on_validation_epoch_end
        so I have to do this hack to do things right at the end of the training.
        """
        is_last = (batch_idx + 1) == self.trainer.num_training_batches
        if is_last:
            self._log_on_epoch_end("train")
            garbage_collection_cuda()

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        """Log quantities at the end of the last validation epoch."""
        is_last = (batch_idx + 1) == self.trainer.num_val_batches[dataloader_idx]
        if is_last:
            self._log_on_epoch_end("val", dataloader_idx=dataloader_idx)

    def on_train_epoch_end(self):
        """Clean up memory after finishing one training epoch and validation."""
        garbage_collection_cuda()

    def on_validation_epoch_end(self):
        """Log the epochs so the mlflow plotting is not buggy, clean memory."""
        self.log("epoch_idx", float(self.current_epoch), on_epoch=True, on_step=False)

    def on_test_epoch_end(self):
        """Clean up memory."""
        garbage_collection_cuda()

    def outlog(self, outdict: dict) -> dict:
        """Override with the values you want to log."""
        return outdict

    def configure_optimizers(self) -> dict:
        """Configure the optimiser that goes with the pl_module."""
        optimizer = LightningOptimizer(self.hparams.optimizer(params=self.parameters()))

        if self.hparams.scheduler:
            scheduler_dict = self._set_up_scheduler(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

        return {"optimizer": optimizer}

    def _log_dict(self, outdict: dict, dataloader_idx: int):
        """Compile the dictionary with loss/metric values to log during training."""
        outdict = self.outlog(outdict)
        if dataloader_idx not in self._log_sum:
            self._log_sum[dataloader_idx] = {}
            self._log_nsteps[dataloader_idx] = 0

        for mname, mvalue in outdict.items():
            if isinstance(mvalue, (int, float)):
                self._log_sum[dataloader_idx][mname] = self._log_sum[
                    dataloader_idx
                ].get(mname, 0.0) + float(mvalue)
            elif torch.is_tensor(mvalue) and mvalue.ndim == 0:
                self._log_sum[dataloader_idx][mname] = self._log_sum[
                    dataloader_idx
                ].get(mname, 0.0) + float(mvalue.detach())

        self._log_nsteps[dataloader_idx] += 1

    def _log_on_epoch_end(self, stage: str, dataloader_idx: int = 0):
        """Log metrics at the end of the epoch instead of every step (default)."""
        nsteps = max(self._log_nsteps[dataloader_idx], 1)
        logs = self._log_sum[dataloader_idx]
        if stage == "train":
            logs = {f"train/{k}": v / nsteps for k, v in logs.items()}
        else:
            datasets = list(getattr(self.trainer, f"{stage}_dataloaders").keys())
            dataset_name = datasets[dataloader_idx]
            logs = {f"{stage}/{dataset_name}/{k}": v / nsteps for k, v in logs.items()}

        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=False,  # set True only for a few key metrics if needed
            add_dataloader_idx=False,
        )

        self._log_sum[dataloader_idx] = {}
        self._log_nsteps[dataloader_idx] = 0

    def _set_up_scheduler(self, optimizer: optim.Optimizer) -> dict:
        """Configure a scheduler for the optimiser, that can be used for lr etc..."""
        scheduler_fn = self.hparams.scheduler.scheduler
        param_names = inspect.signature(scheduler_fn).parameters
        kwargs = {}
        if "total_steps" in param_names:
            kwargs["total_steps"] = int(self.trainer.estimated_stepping_batches)
        elif "T_max" in param_names:
            kwargs["T_max"] = int(self.trainer.estimated_stepping_batches)

        scheduler = scheduler_fn(optimizer=optimizer, **kwargs)
        scheduler_dict = OmegaConf.to_container(self.hparams.scheduler, resolve=True)
        scheduler_dict.update({"scheduler": scheduler})

        return scheduler_dict

    def _split_by_type_from_flat(
        self,
        x_flat: torch.Tensor,
        m_flat: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Build per-type tensors from flattened inputs using object_feature_map."""
        object_feature_map = getattr(self, "object_feature_map", None)
        if object_feature_map is None:
            raise RuntimeError(
                "object_feature_map not found on module. "
                "Make sure inject_object_feature_map(self) was called in "
                "on_fit_start/on_test_start."
            )

        if m_flat is None:
            m_flat = torch.ones_like(x_flat, dtype=x_flat.dtype, device=x_flat.device)

        x_by_type = {}
        m_by_type = {}

        for obj_name, feature_map in object_feature_map.items():
            feat_names = list(feature_map.keys())
            feat_indices = [feature_map[feat_name] for feat_name in feat_names]

            n_obj = len(feat_indices[0])
            n_feat = len(feat_indices)

            if not all(len(idxs) == n_obj for idxs in feat_indices):
                raise ValueError(
                    f"Feature map for '{obj_name}' has inconsistent object counts."
                )

            obj_features = []
            obj_masks = []

            for obj_idx in range(n_obj):
                this_obj_feat_idx = [
                    feat_indices[f_idx][obj_idx] for f_idx in range(n_feat)
                ]
                idx_tensor = torch.tensor(
                    this_obj_feat_idx, device=x_flat.device, dtype=torch.long
                )

                x_obj = torch.index_select(x_flat, dim=1, index=idx_tensor)
                m_obj = torch.index_select(m_flat, dim=1, index=idx_tensor)

                obj_features.append(x_obj)
                obj_masks.append(m_obj.all(dim=1))

            x_by_type[obj_name] = torch.stack(obj_features, dim=1)
            m_by_type[obj_name] = torch.stack(obj_masks, dim=1)

        return x_by_type, m_by_type

    @property
    def target_fpr(self) -> float:
        return self.compute_target_fpr()

    def compute_target_fpr(self) -> float:
        """Target FPR: target_rate/base_rate if base_rate is set, else target_rate."""
        target_rate = getattr(self.hparams, "target_rate", None)
        base_rate = getattr(self.hparams, "base_rate", None)

        if target_rate is None:
            raise ValueError("target_rate must be defined in hparams.")

        if base_rate is None:
            fpr = float(target_rate)
        else:
            if base_rate <= 0:
                raise ValueError("base_rate must be positive.")
            fpr = float(target_rate) / float(base_rate)

        if not (0.0 < fpr < 1.0):
            raise ValueError(f"Computed FPR must be in (0,1), got {fpr}")

        return fpr

    def compute_threshold_quantile(self) -> float:
        return 1.0 - self.compute_target_fpr()

    def compute_operational_ascore(self, ascore: torch.Tensor) -> float:
        """Anomaly score at the operational point, i.e. at the target FPR.

        A single quantile is too noisy when the operational tail holds fewer than 10
        events, so below that the top-k mean is used instead.
        """
        n = ascore.numel()
        k = max(1, int(self.target_fpr * n))
        if k < 10:
            k_eff = min(max(10, k), n)
            return torch.topk(ascore, k_eff).values.mean().item()

        return torch.quantile(ascore, 1.0 - self.target_fpr).item()

    def compute_score_quantiles(self, ascore: torch.Tensor) -> tuple[float, float]:
        """Median and 99th percentile of the anomaly score distribution."""
        quantiles = torch.tensor([0.5, 0.99], device=ascore.device)
        q50, q99 = torch.quantile(ascore, quantiles).tolist()

        return q50, q99

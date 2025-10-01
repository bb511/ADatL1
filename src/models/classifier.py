from typing import Optional, Tuple

import torch
from torch import nn
from retry import retry

import torch.nn.functional as F
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from src.models import L1ADLightningModule
from src.models.losses.classifier import BinaryCrossEntropy


class Classifier(L1ADLightningModule):
    def __init__(
        self,
        classifier: nn.Module,
        features: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "classifier", "loss"]
        )

        self.classifier = classifier
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        classifier_logits = self.classifier(x)
        return classifier_logits

    def on_train_start(self):
        weight = self.trainer.datamodule.get_pos_weight()
        self.train_loss = BinaryCrossEntropy(weight, self.trainer.model.loss.reduction)

    def model_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        classifier_logits = self.forward(x)
        classifier_loss = self.loss(score=classifier_logits.flatten(), label=y)


        return {
            # Used for backpropagation:
            "loss": classifier_loss.mean(),
            # Used for logging:
            f"loss/{self.loss.name}/mean": classifier_loss.mean(),
            # Used for callbacks:
            f"loss/{self.loss.name}/full": classifier_loss,
            f"score": classifier_logits.sigmoid_()
        }

    def _filter_log_dict(self, outdict: dict) -> dict:
        """Override with the values you want to log."""
        return {
            "loss": outdict.get("loss"),
            "loss_bce": outdict.get(f"loss/{self.loss.name}/mean"),
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        classifier_logits = self.forward(batch[0].flatten(start_dim=1).to(dtype=torch.float32))
        classifier_loss = self.train_loss(score=classifier_logits.flatten(), label=batch[1])

        del classifier_logits

        outdict = {
            # Used for backpropagation:
            "loss": classifier_loss.mean(),
            # Used for logging:
            f"loss/{self.loss.name}/mean": classifier_loss.mean(),
            # Used for callbacks:
            f"loss/{self.loss.name}/full": classifier_loss,
        }

        # Decide what to log:
        self.log_dict(
            self._log_dict(outdict, "train", dataloader_idx=0),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        return outdict

    @retry((Exception), tries=3, delay=3, backoff=0)
    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = 0
    ):
        outdict = self.model_step(
            batch[0].flatten(start_dim=1).to(dtype=torch.float32), batch[1]
        )

        self.log_dict(
            self._log_dict(outdict, "val", dataloader_idx=dataloader_idx),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        return outdict

    def test_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = 0
    ):
        outdict = self.model_step(
            batch[0].flatten(start_dim=1).to(dtype=torch.float32), batch[1]
        )

        # Decide what to log:
        self.log_dict(
            self._log_dict(outdict, "test", dataloader_idx=dataloader_idx),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,  # !!
            add_dataloader_idx=False,
        )
        return outdict

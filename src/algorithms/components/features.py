# Import model to use its output as features for training another model.
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn


class FeaturesFromCkpt(nn.Module):
    """Load a checkpoint and use component of model to generate features.

    :param litmodule_cls: LightningModule of the checkpointed model.
    :param ckpt_path: String specifying the path to the checkpoint to import.
    :param attr: String specifying the name of the component that should be used to
        generate the features. For example, in VICreg we don't need the whole algorithm,
        with its augmentation modules and projector module, we can just use self.model
        to generate the features that are used downstream.
    """

    def __init__(self, litmodule_cls: LightningModule, ckpt_path: str, attr: str):
        super().__init__()

        lm = litmodule_cls.load_from_checkpoint(ckpt_path, map_location="cpu")
        features = getattr(lm, attr)

        for p in features.parameters():
            p.requires_grad = False
        features.eval()

        self.features = features

    def forward(self, x):
        with torch.no_grad():
            return self.features(x)

    def train(self, mode: bool = True):
        super().train(mode)
        self.features.eval()
        return self

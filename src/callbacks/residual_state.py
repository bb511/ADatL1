from contextlib import contextmanager

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from sklearn.covariance import OAS
from torch.utils.data import DataLoader, Dataset

from src.data.utils import unpack_batch


@contextmanager
def _clean_pass(module: LightningModule):
    """Run in eval mode, without gradients, on a forked RNG; restore every mode.

    The fork keeps the training random stream the same with and without this callback,
    since the VAE encoder samples a latent even when only its mean is used.
    """
    modes = [submodule.training for submodule in module.modules()]
    devices = [module.device] if module.device.type == "cuda" else []
    module.eval()
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            yield
    finally:
        for submodule, mode in zip(module.modules(), modes):
            submodule.training = mode


def _batch_residuals(module: LightningModule, batch) -> torch.Tensor:
    b = unpack_batch(batch)
    if b.mask is not None:
        raise ValueError("The residual fit does not support masked inputs.")
    x = torch.flatten(b.x, start_dim=1).to(module.device)
    return module.residuals(x).double().cpu()


def _training_residuals(
    module: LightningModule, dataset: Dataset, batch_size: int
) -> torch.Tensor:
    """Residuals of every row of the dataset in its order, shape (rows, features)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with _clean_pass(module):
        return torch.cat([_batch_residuals(module, batch) for batch in loader])


class ResidualStateCallback(Callback):
    """Fit the state of the whitened residual score before every validation epoch.

    The state is the mean and the OAS-shrunk precision of the residuals of
    trainer.datamodule.data_train, all of whose rows are normal, computed with the
    weights of the epoch just trained. It is passed to the module's set_residual_state,
    and the module must implement residuals(x). The fit also runs during the sanity
    check, since a module that declares the residual score raises on an unfitted state.

    :param batch_size: Batch size of the pass over the training rows.
    """

    def __init__(self, batch_size: int = 8192):
        super().__init__()
        self.batch_size = batch_size

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        data_train = trainer.datamodule.data_train
        residuals = _training_residuals(pl_module, data_train, self.batch_size)
        oas = OAS(store_precision=True, assume_centered=False).fit(residuals.numpy())
        location = torch.from_numpy(oas.location_)
        pl_module.set_residual_state(location, torch.from_numpy(oas.precision_))

"""Tests for train-normal-fitted residual OAS CAP scoring."""

from types import SimpleNamespace

import torch
from sklearn.covariance import OAS
from torch import nn

from src.algorithms.ae import AE
from src.callbacks.cap import ResidualOASStateCallback


def test_residual_oas_state_uses_clean_final_weight_training_pass() -> None:
    """The callback must reproduce sklearn OAS state from clean residuals."""
    callback = ResidualOASStateCallback()
    train_x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]], dtype=torch.float32)
    train_reconstruction = torch.tensor(
        [[0.1, 0.0], [0.8, 0.1], [0.2, 1.7], [1.6, 1.2]], dtype=torch.float32
    )
    batches = [
        {"x": train_x[indices], "y": torch.zeros(2)} for indices in (slice(0, 2), slice(2, 4))
    ]
    trainer = SimpleNamespace(
        world_size=1,
        sanity_checking=False,
        train_dataloader=batches,
        val_dataloaders={"normal": None, "reference_normal": None},
    )
    reconstruction_batches = iter(train_reconstruction.split(2))
    state = {}
    module = SimpleNamespace(device=torch.device("cpu"))
    module.eval = lambda: None
    module.forward = lambda x: (torch.zeros(len(x), 1), next(reconstruction_batches))
    module.set_residual_oas_state = lambda location, precision: state.update(
        location=location, precision=precision
    )
    callback.on_validation_epoch_start(trainer, module)

    expected = OAS().fit((train_x - train_reconstruction).double().numpy())
    torch.testing.assert_close(state["location"], torch.from_numpy(expected.location_))
    torch.testing.assert_close(state["precision"], torch.from_numpy(expected.precision_))


def test_ae_residual_oas_score_matches_sklearn_mahalanobis() -> None:
    """AE scoring must apply the installed dimension-normalized precision."""
    model = AE(
        encoder=nn.Linear(2, 1),
        decoder=nn.Linear(1, 2),
        target_rate=0.01,
    )
    train_residual = torch.tensor([[-0.1, 0.0], [0.2, -0.1], [-0.2, 0.3], [0.4, -0.2]])
    estimator = OAS().fit(train_residual.double().numpy())
    model.set_residual_oas_state(
        torch.from_numpy(estimator.location_), torch.from_numpy(estimator.precision_)
    )
    target = torch.tensor([[0.5, 0.5], [1.5, 1.0]])
    reconstruction = torch.tensor([[0.4, 0.3], [1.1, 1.1]])
    residual = (target - reconstruction).numpy()

    expected = estimator.mahalanobis(residual) / residual.shape[1]
    torch.testing.assert_close(
        model.residual_oas_score(target, reconstruction),
        torch.tensor(expected, dtype=target.dtype),
    )

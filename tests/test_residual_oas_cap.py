"""Tests for train-normal-fitted residual OAS CAP scoring."""

from types import SimpleNamespace

import torch
from sklearn.covariance import OAS

from src.callbacks.cap import ResidualOASCAPCallback


def test_residual_oas_cap_fits_training_residuals_and_scores_validation() -> None:
    """The callback must reproduce sklearn OAS Mahalanobis scores."""
    callback = ResidualOASCAPCallback(
        output_name="ascore/full",
        dataset_1="normal",
        dataset_2="reference_normal",
        pairing_type="cdf",
        cap_metric_config={"n_epochs": 0},
    )
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
    module = SimpleNamespace(
        device=torch.device("cpu"),
        forward=lambda x: (torch.zeros(len(x), 1), next(reconstruction_batches)),
    )
    callback.on_fit_start(trainer, module)
    callback.on_validation_epoch_start(trainer, module)

    expected = OAS().fit((train_x - train_reconstruction).double().numpy())
    torch.testing.assert_close(
        callback._oas_location, torch.from_numpy(expected.location_).double()
    )
    torch.testing.assert_close(
        callback._oas_precision, torch.from_numpy(expected.precision_).double()
    )

    validation_x = torch.tensor([[0.5, 0.5], [1.5, 1.0]])
    validation_reconstruction = torch.tensor([[0.4, 0.3], [1.1, 1.1]])
    callback.on_validation_batch_end(
        trainer,
        None,
        {"reconstructed_data": validation_reconstruction},
        {"x": validation_x, "y": torch.zeros(2)},
        0,
        dataloader_idx=0,
    )

    residual = (validation_x - validation_reconstruction).double().numpy()
    expected_score = expected.mahalanobis(residual) / residual.shape[1]
    torch.testing.assert_close(
        callback.dataset_1_scores[0], torch.from_numpy(expected_score).double()
    )

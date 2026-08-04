"""Tests for train-normal-fitted residual OAS CAP scoring."""

from types import SimpleNamespace

import numpy as np
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


def test_residual_oas_state_excludes_structurally_padded_coordinates() -> None:
    """Padding must be neutral, including coordinates absent from every event."""
    callback = ResidualOASStateCallback()
    train_x = torch.tensor([[1.0, 999.0, -999.0], [2.0, 3.0, 999.0]])
    train_mask = torch.tensor([[True, False, False], [True, True, False]])
    trainer = SimpleNamespace(
        sanity_checking=False,
        world_size=1,
        train_dataloader=[{"x": train_x, "mask": train_mask, "y": torch.zeros(2)}],
    )
    module = nn.Module()
    module.device = torch.device("cpu")
    module.forward = lambda x: (x[:, :1], torch.zeros_like(x))
    state = {}
    module.set_residual_oas_state = lambda location, precision: state.update(
        location=location, precision=precision
    )

    callback.on_validation_epoch_start(trainer, module)

    completed = np.array([[1.0, 3.0], [2.0, 3.0]])
    expected = OAS().fit(completed)
    torch.testing.assert_close(
        state["location"], torch.tensor([1.5, 3.0, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(state["precision"][:2, :2], torch.from_numpy(expected.precision_))
    torch.testing.assert_close(state["precision"][2], torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(state["precision"][:, 2], torch.zeros(3, dtype=torch.float64))


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


def test_ae_residual_oas_score_ignores_padded_decoder_residuals() -> None:
    """A nonexistent object's decoder output must contribute no anomaly energy."""
    model = AE(encoder=nn.Linear(2, 1), decoder=nn.Linear(1, 2))
    model.set_residual_oas_state(torch.zeros(2), torch.eye(2))
    target = torch.tensor([[2.0, 0.0]])
    reconstruction = torch.tensor([[0.0, 1000.0]])
    mask = torch.tensor([[True, False]])

    score = model.residual_oas_score(target, reconstruction, mask)

    torch.testing.assert_close(score, torch.tensor([4.0]))


def test_ae_canonical_score_is_residual_mahalanobis() -> None:
    """All generic AE consumers must see Mahalanobis, with MSE only diagnostic."""
    model = AE(
        encoder=nn.Linear(2, 1, bias=False),
        decoder=nn.Linear(1, 2, bias=False),
        anomaly_score="residual_oas",
        target_rate=0.25,
    )
    with torch.no_grad():
        model.encoder.weight.copy_(torch.tensor([[1.0, 0.0]]))
        model.decoder.weight.copy_(torch.tensor([[0.0], [0.0]]))
    model.set_residual_oas_state(
        torch.tensor([0.25, -0.5]),
        torch.tensor([[4.0, 0.0], [0.0, 1.0]]),
    )
    batch = {"x": torch.tensor([[1.0, 2.0], [2.0, 1.0]]), "y": torch.zeros(2)}

    output = model.model_step(batch)
    expected = model.residual_oas_score(batch["x"], torch.zeros_like(batch["x"]))

    torch.testing.assert_close(output["ascore/full"], expected)
    torch.testing.assert_close(output["ascore/residual_oas"], expected)
    assert not torch.equal(output["ascore/full"], output["ascore/mse"])


def test_residual_oas_state_respects_max_samples() -> None:
    """OAS fitting must stop after its configured deterministic sample budget."""
    callback = ResidualOASStateCallback(max_samples=3)
    batches = [
        {"x": torch.tensor([[0.0, 0.0], [1.0, 0.0]]), "y": torch.zeros(2)},
        {"x": torch.tensor([[0.0, 2.0], [2.0, 1.0]]), "y": torch.zeros(2)},
    ]
    seen = []
    module = SimpleNamespace(device=torch.device("cpu"))
    module.eval = lambda: None
    module.forward = lambda x: (torch.zeros(len(x), 1), seen.append(x.clone()) or x * 0.5)
    module.set_residual_oas_state = lambda location, precision: None
    trainer = SimpleNamespace(world_size=1, sanity_checking=False, train_dataloader=batches)

    callback.on_validation_epoch_start(trainer, module)

    assert callback.fit_samples == 3
    assert sum(len(x) for x in seen) == 3

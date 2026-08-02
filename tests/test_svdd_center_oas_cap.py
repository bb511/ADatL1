"""Tests for train-normal-fitted center-OAS SVDD scoring."""

from types import SimpleNamespace

import torch
from sklearn.covariance import OAS
from torch import nn

from src.algorithms.svdd import DeepSVDD
from src.callbacks.cap import SVDDCenterOASStateCallback


def test_svdd_center_oas_state_uses_clean_final_weight_pass() -> None:
    """The callback must fit centered OAS on complete train-normal embeddings."""
    callback = SVDDCenterOASStateCallback()
    embeddings = torch.tensor(
        [[0.2, 0.4], [0.7, -0.1], [-0.3, 0.8], [0.9, 0.2]], dtype=torch.float32
    )
    batches = [
        {"x": torch.zeros(2, 3), "y": torch.zeros(2)},
        {"x": torch.ones(2, 3), "y": torch.zeros(2)},
    ]
    trainer = SimpleNamespace(world_size=1, sanity_checking=False, train_dataloader=batches)
    embedding_batches = iter(embeddings.split(2))
    state = {}
    module = SimpleNamespace(
        device=torch.device("cpu"),
        center=torch.tensor([0.1, 0.2]),
        center_initialized=True,
    )
    module.eval = lambda: None
    module.forward = lambda x: next(embedding_batches)
    module.set_center_oas_state = lambda precision: state.update(precision=precision)
    callback.on_validation_epoch_start(trainer, module)

    expected = OAS(store_precision=True, assume_centered=True).fit(
        (embeddings - module.center).double().numpy()
    )
    torch.testing.assert_close(state["precision"], torch.from_numpy(expected.precision_))


def test_svdd_center_oas_score_matches_centered_mahalanobis() -> None:
    """SVDD scoring must apply dimension-normalized centered precision."""
    model = DeepSVDD(
        encoder=nn.Linear(3, 2, bias=False),
        target_rate=0.01,
        enforce_architecture_constraints=True,
    )
    model.center = torch.tensor([0.1, -0.2])
    train_z = torch.tensor([[0.2, 0.0], [0.5, -0.4], [-0.1, 0.3], [0.8, -0.1]])
    estimator = OAS(store_precision=True, assume_centered=True).fit(
        (train_z - model.center).double().numpy()
    )
    model.set_center_oas_state(torch.from_numpy(estimator.precision_))
    values = torch.tensor([[0.4, 0.2], [-0.2, -0.5]])

    expected = estimator.mahalanobis((values - model.center).numpy()) / values.shape[1]
    torch.testing.assert_close(
        model.center_oas_score(values), torch.tensor(expected, dtype=values.dtype)
    )

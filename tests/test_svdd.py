from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.components.encoder import Encoder, ImageEncoder
from src.algorithms.losses.svdd import SVDDLoss
from src.algorithms.svdd import DeepSVDD


def _encoder(*, bias: bool = False) -> Encoder:
    return Encoder(in_dim=3, nodes=[4, 2], activation="relu", bias=bias)


def _model() -> DeepSVDD:
    return DeepSVDD(
        encoder=_encoder(),
        center_eps=0.0,
        optimizer=partial(torch.optim.Adam, lr=1e-3),
    )


def test_svdd_architectures_can_be_made_bias_free() -> None:
    tabular = _encoder()
    image = ImageEncoder(
        in_channels=3,
        nodes=[4, 2],
        input_size=(8, 8),
        strides=[2],
        bias=False,
        batchnorm=False,
    )

    for encoder in (tabular, image):
        affine_layers = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        )
        assert all(
            module.bias is None
            for module in encoder.modules()
            if isinstance(module, affine_layers)
        )


def test_center_uses_all_training_batches_and_is_fixed() -> None:
    model = _model()
    x = torch.tensor(
        [
            [1.0, 0.0, -1.0],
            [2.0, 1.0, 0.0],
            [-1.0, 2.0, 1.0],
            [4.0, -2.0, 3.0],
        ]
    )
    loader = DataLoader(TensorDataset(x, torch.zeros(len(x))), batch_size=2)
    expected = model.encoder(x).mean(dim=0)

    model.initialize_center(loader)

    assert torch.allclose(model.center, expected)
    center_before = model.center.clone()
    model.eval()
    model.model_step(next(iter(loader)))
    assert torch.equal(model.center, center_before)


def test_one_class_loss_is_exact_distance() -> None:
    distances = torch.tensor([0.25, 1.0, 4.0])
    loss = SVDDLoss(objective="one_class")
    assert torch.equal(loss(distances), distances)

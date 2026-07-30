from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.components.encoder import Encoder, ImageEncoder
from src.algorithms.losses.svdd import SVDDLoss
from src.algorithms.svdd import DeepSVDD


def _encoder(*, bias: bool = False) -> Encoder:
    return Encoder(in_dim=3, nodes=[4, 2], activation="relu", bias=bias)


def _model(**kwargs) -> DeepSVDD:
    return DeepSVDD(
        encoder=kwargs.pop("encoder", _encoder()),
        center_eps=kwargs.pop("center_eps", 0.0),
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        **kwargs,
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


def test_center_checkpoint_round_trip_resizes_empty_buffer() -> None:
    source = _model()
    source.center = torch.tensor([0.25, -0.5])
    target = _model()

    target.load_state_dict(source.state_dict(), strict=True)

    assert torch.equal(target.center, source.center)


def test_one_class_loss_is_exact_distance() -> None:
    distances = torch.tensor([0.25, 1.0, 4.0])
    loss = SVDDLoss(objective="one_class")
    assert torch.equal(loss(distances), distances)


def test_soft_boundary_loss_is_stateless() -> None:
    distances = torch.tensor([0.25, 1.0, 4.0])
    radius = torch.tensor(1.0)
    loss = SVDDLoss(objective="soft_boundary", nu=0.5)

    actual = loss(distances, radius)

    expected = torch.tensor([1.0, 1.0, 7.0])
    assert torch.equal(actual, expected)
    assert radius.item() == 1.0
    assert not tuple(loss.buffers())


def test_network_regularization_uses_weights_not_latents() -> None:
    model = _model(network_weight_decay=0.2)
    expected = 0.1 * sum(
        parameter.square().sum() for parameter in model.encoder.parameters() if parameter.ndim > 1
    )

    actual = model._network_regularization()

    assert torch.allclose(actual, expected)


def test_existing_ckpt_parameter_loads_matching_ae_encoder_weights(tmp_path) -> None:
    source = _encoder()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.125)
    checkpoint = tmp_path / "ae.ckpt"
    state_dict = {
        f"encoder.{key}": value for key, value in source.state_dict().items()
    }
    state_dict["decoder.unused"] = torch.tensor(1.0)
    torch.save(
        {"state_dict": state_dict},
        checkpoint,
    )

    target = _model(ckpt=str(checkpoint))
    target._load_checkpoint()

    for source_parameter, target_parameter in zip(
        source.parameters(), target.encoder.parameters()
    ):
        assert torch.equal(source_parameter, target_parameter)

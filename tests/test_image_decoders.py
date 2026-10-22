import pytest
import torch
from torch import nn

from src.algorithms.components.decoder import ImageDecoder
from src.algorithms.components.encoder import ImageVariationalEncoder

ACTIVATIONS = {"relu": "ReLU", "gelu": "GELU", "silu": "SiLU"}

# Pool defaults: the CIFAR-10 AE decoder and the CIFAR-10 and RobustAD VAE. The
# activations carry no parameters, so the counts are those of the affine maps alone.
PARAMETER_COUNTS = {
    "cifar10_ae_decoder": (ImageDecoder, ([128, 64, 32], 3, (32, 32), [2, 2]), 1149891),
    "cifar10_vae_encoder": (
        ImageVariationalEncoder,
        (3, (32, 32), [32, 64, 128, 64], [2, 2]),
        560320,
    ),
    "cifar10_vae_decoder": (ImageDecoder, ([64, 64, 32], 3, (32, 32), [2, 2]), 322499),
    "robustad_vae_encoder": (
        ImageVariationalEncoder,
        (3, (32, 32), [16, 32, 64], [2]),
        135776,
    ),
    "robustad_vae_decoder": (ImageDecoder, ([64, 16], 3, (32, 32), [2]), 1074627),
}


def _call_order(module: nn.Module, x: torch.Tensor) -> list[str]:
    calls = []
    leaves = [m for m in module.modules() if not list(m.children())]
    handles = [
        leaf.register_forward_hook(lambda m, i, o: calls.append(type(m).__name__))
        for leaf in leaves
        if not isinstance(leaf, nn.Identity)
    ]
    module(x)
    for handle in handles:
        handle.remove()
    return calls


@pytest.mark.parametrize("activation", list(ACTIVATIONS))
@pytest.mark.parametrize("strides", [[2], [2, 2], [1, 2]])
def test_decoder_activation_after_every_hidden_map(strides, activation):
    nodes = [8] + [4] * len(strides)
    decoder = ImageDecoder(nodes, 3, (8, 8), strides, activation=activation)
    act = ACTIVATIONS[activation]
    expected = ["Linear", act] + ["ConvTranspose2d", act] * len(strides) + ["Conv2d"]
    assert _call_order(decoder, torch.randn(2, 8)) == expected


def test_decoder_output_activation_only_when_configured():
    decoder = ImageDecoder([8, 4], 3, (8, 8), [2], final_activation="sigmoid")
    assert _call_order(decoder, torch.randn(2, 8))[-2:] == ["Conv2d", "Sigmoid"]


@pytest.mark.parametrize("activation", list(ACTIVATIONS))
@pytest.mark.parametrize("nodes, strides", [([4, 8, 16], [2]), ([4, 8, 16, 8], [2, 2])])
def test_variational_encoder_activation_before_heads(nodes, strides, activation):
    encoder = ImageVariationalEncoder(3, (8, 8), nodes, strides, activation=activation)
    act = ACTIVATIONS[activation]
    expected = ["Conv2d", act] * len(strides) + ["Linear", act, "Linear", "Linear"]
    assert _call_order(encoder, torch.randn(2, 3, 8, 8)) == expected


@pytest.mark.parametrize("case", list(PARAMETER_COUNTS))
def test_parameter_counts_unchanged(case):
    cls, args, n_params = PARAMETER_COUNTS[case]
    assert sum(p.numel() for p in cls(*args).parameters()) == n_params

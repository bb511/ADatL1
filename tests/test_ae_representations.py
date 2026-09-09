import torch
from torch import nn

from src.algorithms.ae import AE


def make_model() -> AE:
    return AE(
        encoder=nn.Identity(),
        decoder=nn.Identity(),
        input_noise_std=0.0,
        mi_sensitive_num_bins=2,
    )

def test_forward_with_representations_exposes_named_tensors() -> None:
    model = make_model()
    model.eval()

    x = torch.tensor(
        [
            [1.0, -1.0],
            [0.25, -0.25],
        ]
    )

    representations = model.forward_with_representations(x)

    assert set(representations) == {
        "latent_logits",
        "latent_sample",
        "reconstructed_data",
    }

    torch.testing.assert_close(representations["latent_logits"], x)

    expected_sample = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    torch.testing.assert_close(
        representations["latent_sample"],
        expected_sample,
    )
    torch.testing.assert_close(
        representations["reconstructed_data"],
        expected_sample,
    )

def test_forward_preserves_existing_return_contract() -> None:
    model = make_model()
    model.eval()

    x = torch.tensor([[1.0, -1.0]])

    latent_logits, reconstructed_data = model(x)
    representations = model.forward_with_representations(x)

    torch.testing.assert_close(
        latent_logits,
        representations["latent_logits"],
    )
    torch.testing.assert_close(
        reconstructed_data,
        representations["reconstructed_data"],
    )

def test_eval_latent_sample_is_deterministic_and_binary() -> None:
    model = make_model()
    model.eval()

    x = torch.tensor(
        [
            [2.0, -2.0],
            [0.1, -0.1],
        ]
    )

    first = model.forward_with_representations(x)["latent_sample"]
    second = model.forward_with_representations(x)["latent_sample"]

    torch.testing.assert_close(first, second)
    assert set(torch.unique(first).tolist()) <= {0.0, 1.0}

def test_forward_with_representations_preserves_gradients() -> None:
    model = make_model()
    model.eval()

    x = torch.tensor(
        [[1.0, -1.0]],
        requires_grad=True,
    )

    representations = model.forward_with_representations(x)
    representations["reconstructed_data"].sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
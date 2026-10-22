from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.components.decoder import Decoder
from src.algorithms.components.encoder import DeepSetsEncoder, VariationalEncoder
from src.algorithms.dsae import DeepSetsAE
from src.algorithms.vae import VAE
from tests.test_score_dump import assert_identical

DIM = 5
# Two object types in nine flat features: the FET, whose eta is never filled, and two
# jets.
FEATURE_MAP = {
    "FET": {"et": [0], "eta": [1], "phi": [2]},
    "jets": {"et": [3, 6], "eta": [4, 7], "phi": [5, 8]},
}
OBJECTS = {"object_dims": {"FET": 3, "jets": 3}, "pooling": "sum"}


def _vae(eval_mean: bool) -> VAE:
    torch.manual_seed(0)
    encoder = VariationalEncoder(in_dim=DIM, nodes=[8, 6, 2])
    decoder = Decoder(nodes=[2, 8], out_dim=DIM)
    module = VAE(
        encoder, decoder, target_rate=0.01, base_rate=None, eval_mean=eval_mean
    )
    module._setup_kl_annealing(0.0, 1)
    return module


def _decode_the_sample(module, x):
    # VAE.forward before eval_mean existed.
    z_mean, z_log_var, z = module.encoder(module.features(x))
    return z_mean, z_log_var, z, module.decoder(z)


def _step(module, seed: int = 2) -> tuple[dict, torch.Tensor]:
    x = torch.randn(32, DIM, generator=torch.Generator().manual_seed(1))
    torch.manual_seed(seed)
    with torch.no_grad():
        outputs = module.model_step((x, torch.zeros(len(x))))
    return outputs, torch.get_rng_state()


@pytest.mark.parametrize("training", [True, False])
def test_vae_without_eval_mean_is_unchanged(monkeypatch, training):
    module = _vae(eval_mean=False).train(training)
    current, _ = _step(module)
    monkeypatch.setattr(VAE, "forward", _decode_the_sample)
    previous, _ = _step(module)
    assert_identical(current, previous)


def test_vae_eval_mean_decodes_the_posterior_mean_in_eval_only():
    steps = {}
    for eval_mean in (False, True):
        module = _vae(eval_mean)
        for training in (True, False):
            steps[eval_mean, training] = _step(module.train(training))
    for part in range(2):
        assert_identical(steps[False, True][part], steps[True, True][part])

    (off, off_rng), (on, on_rng) = steps[False, False], steps[True, False]
    assert torch.equal(off_rng, on_rng)
    assert torch.equal(off["ascore/full"], on["ascore/full"])
    assert not torch.equal(off["loss/full"], on["loss/full"])
    assert torch.equal(on["loss/full"], _step(module, seed=3)[0]["loss/full"])
    x = torch.randn(32, DIM, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = module.decoder(module.encoder(x)[0])
    assert torch.equal(on["reconstructed_data"], expected)


def test_an_object_is_present_when_any_feature_is_unmasked():
    module = SimpleNamespace(object_feature_map=FEATURE_MAP)
    x = torch.arange(18.0).reshape(2, 9)
    # Event 0: the FET without eta, the first jet whole, the second padded; event 1:
    # nothing.
    mask = torch.tensor([[1, 0, 1, 1, 1, 1, 0, 0, 0], [0] * 9], dtype=torch.float)
    x_by_type, m_by_type = ADLightningModule._split_by_type_from_flat(module, x, mask)
    assert m_by_type["FET"].tolist() == [[True], [False]]
    assert m_by_type["jets"].tolist() == [[True, False], [False, False]]
    assert x_by_type["jets"][0, 1].tolist() == [6.0, 7.0, 8.0]
    _, m_by_type = ADLightningModule._split_by_type_from_flat(module, x, None)
    assert m_by_type["jets"].all() and m_by_type["FET"].all()


def _deep_sets(**kwargs) -> DeepSetsEncoder:
    torch.manual_seed(0)
    phi_nodes = {"FET": [8], "jets": [16, 8]}
    return DeepSetsEncoder(
        object_phi_nodes=phi_nodes, rho_nodes=[32, 16], **OBJECTS, **kwargs
    )


def test_rho_final_activation_drops_only_the_last_activation():
    default, linear = _deep_sets(), _deep_sets(rho_final_activation=False)
    assert isinstance(default.rho.net[-1], nn.ReLU)
    assert isinstance(linear.rho.net[-1], nn.Linear)
    assert_identical(default.state_dict(), linear.state_dict())

    gen = torch.Generator().manual_seed(1)
    x = {name: torch.randn(64, 2, 3, generator=gen) for name in ("FET", "jets")}
    m = {name: torch.ones(64, 2, dtype=torch.bool) for name in ("FET", "jets")}
    with torch.no_grad():
        z = linear(x, m)
        assert (z < 0).any()
        assert torch.equal(torch.relu(z), default(x, m))


def test_dsae_trains_the_fet_network_with_its_eta_masked():
    encoder = _deep_sets(rho_final_activation=False)
    module = DeepSetsAE(encoder, Decoder(nodes=[16, 32], out_dim=9), base_rate=None)
    module.object_feature_map = FEATURE_MAP
    x = torch.randn(64, 9, generator=torch.Generator().manual_seed(1))
    mask = torch.ones(64, 9)
    mask[:, FEATURE_MAP["FET"]["eta"]] = 0.0
    module.model_step((x, mask, None, torch.zeros(len(x))))["loss"].backward()
    for parameter in module.encoder.phis["FET"].parameters():
        assert parameter.grad.abs().sum() > 0

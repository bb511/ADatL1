from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.covariance import OAS
from torch.utils.data import TensorDataset

from src.algorithms.ae import AE
from src.algorithms.components.decoder import Decoder
from src.algorithms.components.encoder import Encoder, VariationalEncoder
from src.algorithms.vae import VAE
from src.callbacks.residual_state import ResidualStateCallback

DIM = 5
RESIDUAL_KEYS = (
    "residual_oas_location",
    "residual_oas_precision",
    "residual_oas_ready",
)


def _ae(**kwargs) -> AE:
    torch.manual_seed(0)
    encoder = Encoder(in_dim=DIM, nodes=[8, 3])
    decoder = Decoder(nodes=[3, 8], out_dim=DIM)
    return AE(encoder, decoder, target_rate=0.01, base_rate=None, **kwargs)


def _vae(**kwargs) -> VAE:
    torch.manual_seed(0)
    encoder = VariationalEncoder(in_dim=DIM, nodes=[8, 6, 2])
    decoder = Decoder(nodes=[2, 8], out_dim=DIM)
    module = VAE(encoder, decoder, target_rate=0.01, base_rate=None, **kwargs)
    module._setup_kl_annealing(0.0, 1)
    return module


FACTORIES = {"ae": _ae, "vae": _vae}
NATIVE = {"ae": "mse", "vae": "kl"}


def _data(n: int, seed: int) -> torch.Tensor:
    mixing = torch.randn(DIM, DIM, generator=torch.Generator().manual_seed(0))
    noise = torch.randn(n, DIM, generator=torch.Generator().manual_seed(seed))
    return noise @ mixing


def _batch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x, torch.zeros(len(x))


def _fit(module, x_train: torch.Tensor, batch_size: int = 64) -> None:
    data_train = TensorDataset(*_batch(x_train))
    trainer = SimpleNamespace(datamodule=SimpleNamespace(data_train=data_train))
    callback = ResidualStateCallback(batch_size=batch_size)
    callback.on_validation_epoch_start(trainer, module)


def _step(module, x: torch.Tensor) -> dict:
    with torch.no_grad():
        return module.model_step(_batch(x))


@pytest.mark.parametrize("name", FACTORIES)
@pytest.mark.parametrize("declared", ["native", "residual_oas"])
def test_scores_are_per_event(name, declared):
    score = NATIVE[name] if declared == "native" else declared
    module = FACTORIES[name](anomaly_score=score)
    _fit(module, _data(300, 1))
    module.eval()
    out = _step(module, _data(40, 2))
    for key in ("ascore/full", f"ascore/{NATIVE[name]}", "ascore/residual_oas"):
        assert out[key].shape == (40,)
    assert torch.equal(out["ascore/full"], out[f"ascore/{score}"])


@pytest.mark.parametrize("name", FACTORIES)
def test_residual_score_matches_sklearn_reference(name):
    module = FACTORIES[name](anomaly_score="residual_oas")
    x_train, x_val = _data(500, 1), _data(60, 2)
    _fit(module, x_train)
    module.eval()
    with torch.no_grad():
        residual_train = module.residuals(x_train).double().numpy()
        residual_val = module.residuals(x_val).double().numpy()
    oas = OAS(store_precision=True, assume_centered=False).fit(residual_train)
    centred = residual_val - oas.location_
    reference = np.einsum("bi,ij,bj->b", centred, oas.precision_, centred) / DIM

    location = module.residual_oas_location.double().numpy()
    precision = module.residual_oas_precision.double().numpy()
    np.testing.assert_allclose(location, oas.location_, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(precision, oas.precision_, rtol=1e-5, atol=1e-6)
    score = _step(module, x_val)["ascore/full"].double().numpy()
    np.testing.assert_allclose(score, reference, rtol=1e-5)


@pytest.mark.parametrize("name", FACTORIES)
def test_state_dict_without_residual_state_loads_unfitted(name):
    module = FACTORIES[name](anomaly_score="residual_oas")
    _fit(module, _data(300, 1))
    state = {k: v for k, v in module.state_dict().items() if k not in RESIDUAL_KEYS}
    module.load_state_dict(state, strict=True)
    assert not bool(module.residual_oas_ready)
    assert torch.equal(module.residual_oas_precision, torch.eye(DIM))

    module.eval()
    with pytest.raises(RuntimeError, match="fitted residual state"):
        _step(module, _data(20, 2))


@pytest.mark.parametrize("name", FACTORIES)
def test_callback_restores_modes_and_rng(name):
    module = FACTORIES[name](anomaly_score="residual_oas")
    module.train()
    module.decoder.eval()
    modes = [submodule.training for submodule in module.modules()]
    rng = torch.get_rng_state()
    _fit(module, _data(300, 1))
    assert [submodule.training for submodule in module.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng)

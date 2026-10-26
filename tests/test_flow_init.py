import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.algorithms.components.flow import RealNVP
from src.algorithms.realnvp import RealNVP as RealNVPModule

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
OVERRIDES = ["paths.raw_data_dir=/x", "run_name=dbg"]


def _flow(identity_init: bool, n_hidden_layers: int = 2) -> RealNVP:
    torch.manual_seed(0)
    return RealNVP(
        input_dim=12,
        n_flows=4,
        hidden_dim=16,
        n_hidden_layers=n_hidden_layers,
        identity_init=identity_init,
    )


def _module(mask_aware: bool) -> RealNVPModule:
    return RealNVPModule(flow=_flow(False), target_rate=0.01, mask_aware=mask_aware)


def _events() -> tuple[torch.Tensor, torch.Tensor]:
    # The first 16 events are fully present; padding is zero, as in the L1AD data.
    g = torch.Generator().manual_seed(1)
    present = torch.rand(64, 12, generator=g) < 0.6
    present[:16] = True
    x = torch.randn(64, 12, generator=g) * 3.0 * present
    return x, present


def _batch(x: torch.Tensor, present: torch.Tensor) -> tuple:
    return x, present, torch.zeros(len(x)), torch.zeros(len(x))


def _maskless_log_prob(flow: RealNVP, x: torch.Tensor) -> torch.Tensor:
    # The coupling and base density op for op as at a10184b, before the data mask.
    log_det = torch.zeros(x.shape[0])
    for layer in flow.flows:
        x_masked = x * layer.mask
        shift, log_scale = torch.chunk(layer.net(x_masked), 2, dim=1)
        inv_mask = 1.0 - layer.mask
        shift = shift * inv_mask
        log_scale = torch.tanh(log_scale) * layer.scale_clamp
        log_scale = log_scale * inv_mask
        x = x_masked + inv_mask * (x * torch.exp(log_scale) + shift)
        log_det += log_scale.sum(dim=1)
    log_pz = -0.5 * (
        ((x - flow.base_mean) ** 2) / torch.exp(2 * flow.base_log_std)
        + 2 * flow.base_log_std
        + math.log(2 * math.pi)
    ).sum(dim=1)
    return log_pz + log_det


@pytest.mark.parametrize("n_hidden_layers", [1, 2])
def test_identity_init_is_the_identity_map(n_hidden_layers):
    flow = _flow(True, n_hidden_layers).eval()
    x = torch.randn(64, 12) * 3.0

    with torch.no_grad():
        z, log_det = flow.encode(x)
        log_prob = flow.log_prob(x)

    gaussian = -0.5 * (x**2 + math.log(2 * math.pi)).sum(dim=1)
    assert torch.equal(z, x)
    assert torch.equal(log_det, torch.zeros(64))
    torch.testing.assert_close(log_prob, gaussian)


@pytest.mark.parametrize("n_hidden_layers", [1, 2])
def test_identity_init_leaves_the_other_parameters_unchanged(n_hidden_layers):
    on = _flow(True, n_hidden_layers).state_dict()
    off = _flow(False, n_hidden_layers).state_dict()
    last = f"net.{2 * n_hidden_layers}."

    assert on.keys() == off.keys()
    for name in on:
        if last in name:
            assert not on[name].any()
            assert off[name].any()
        else:
            assert torch.equal(on[name], off[name]), name


@pytest.mark.parametrize("with_mask", [True, False])
def test_mask_aware_off_reproduces_the_maskless_flow(with_mask):
    module = _module(False).eval()
    x, present = _events()
    batch = _batch(x, present) if with_mask else (x, torch.zeros(len(x)))

    with torch.no_grad():
        out = module.model_step(batch)
        log_prob = module.flow.log_prob(x)
        reference = _maskless_log_prob(module.flow, x)

    assert torch.equal(log_prob, reference)
    assert torch.equal(out["ascore/full"], -reference)
    assert torch.equal(out["loss"], (-reference).mean())


def test_mask_aware_score_ignores_the_padded_values():
    module = _module(True).eval()
    x, present = _events()
    noisy = torch.where(present, x, torch.randn(x.shape) * 5.0)

    with torch.no_grad():
        clean = module.model_step(_batch(x, present))["ascore/full"]
        perturbed = module.model_step(_batch(noisy, present))["ascore/full"]

    assert torch.equal(perturbed, clean)


def test_mask_aware_on_fully_present_events_equals_off():
    x, present = _events()
    full = present.all(dim=1)

    with torch.no_grad():
        on = _module(True).eval().model_step(_batch(x, present))["ascore/full"]
        off = _module(False).eval().model_step(_batch(x, present))["ascore/full"]

    assert torch.equal(on[full], off[full])
    assert not torch.isclose(on[~full], off[~full]).any()


def test_mask_aware_padding_passes_through_and_decode_inverts_encode():
    # In double precision: the untrained flow reaches |z| ~ 5e4, where float32 loses
    # the inverse to cancellation, with or without the data mask.
    flow = _flow(False).double().eval()
    x, present = _events()
    noisy = torch.where(present, x, torch.randn(x.shape)).double()

    with torch.no_grad():
        z, log_det = flow.encode(noisy, present.double())
        back, inverse_log_det = flow.decode(z, present.double())

    assert torch.equal(z[~present], noisy[~present])
    torch.testing.assert_close(back, noisy)
    torch.testing.assert_close(inverse_log_det, -log_det)


@pytest.mark.parametrize("n_hidden_layers", [1, 2])
def test_identity_init_with_mask_is_the_gaussian_of_the_present_coordinates(
    n_hidden_layers,
):
    flow = _flow(True, n_hidden_layers).eval()
    x, present = _events()

    with torch.no_grad():
        log_prob = flow.log_prob(x, present.float())

    gaussian = -0.5 * ((x**2 + math.log(2 * math.pi)) * present).sum(dim=1)
    torch.testing.assert_close(log_prob, gaussian)


def test_mask_aware_needs_the_mask_in_the_batch():
    x, _ = _events()
    with pytest.raises(ValueError, match="data mask"):
        _module(True).model_step((x, torch.zeros(len(x))))


@pytest.mark.parametrize(
    "experiment, identity_init, mask_aware",
    [
        ("cifar10/realnvp", True, False),
        ("mscct/realnvp", True, False),
        ("physics/realnvp", True, True),
        ("cchamber/realnvp", True, False),
        ("cchamber_wt/realnvp", True, False),
    ],
)
def test_identity_init_and_mask_aware_per_experiment(
    experiment, identity_init, mask_aware
):
    overrides = [f"experiment={experiment}", *OVERRIDES]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        cfg = compose(config_name="train", overrides=overrides)
    assert cfg.algorithm.flow.get("identity_init", False) is identity_init
    assert cfg.algorithm.get("mask_aware", False) is mask_aware

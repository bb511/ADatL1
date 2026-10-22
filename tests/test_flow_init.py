import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.algorithms.components.flow import RealNVP

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


@pytest.mark.parametrize(
    "experiment, expected",
    [
        ("cifar10/realnvp", True),
        ("robustad/realnvp", True),
        ("physics/realnvp", False),
        ("cchamber/realnvp", False),
        ("cchamber_wt/realnvp", False),
    ],
)
def test_only_the_image_experiments_set_identity_init(experiment, expected):
    overrides = [f"experiment={experiment}", *OVERRIDES]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        cfg = compose(config_name="train", overrides=overrides)
    assert cfg.algorithm.flow.get("identity_init", False) is expected

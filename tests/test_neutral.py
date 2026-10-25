import math
from functools import partial
from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import IntervalSweep
from omegaconf import OmegaConf

from src.algorithms.components.encoder import Encoder
from src.algorithms.neutral import TRANSFORM_TYPES, NeuTraLAD, _contrastive_loss
from src.utils.omegaconf import register_resolvers

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TUNNELS = ("cchamber", "cchamber_wt")
DIM = 10


def _model(n_transforms: int = 11, transform_type: str = "residual") -> NeuTraLAD:
    torch.manual_seed(0)
    return NeuTraLAD(
        encoder=Encoder(in_dim=DIM, nodes=[32, 16, 8]),
        in_dim=DIM,
        n_transforms=n_transforms,
        transform_type=transform_type,
        temperature=0.1,
        target_rate=0.01,
        optimizer=partial(torch.optim.AdamW, lr=1e-2, weight_decay=1e-4),
    )


def _x(n: int = 64, seed: int = 1) -> torch.Tensor:
    return torch.randn(n, DIM, generator=torch.Generator().manual_seed(seed))


def _batch(n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    return _x(n), torch.zeros(n)


def _equation_2(z: torch.Tensor, temperature: float) -> torch.Tensor:
    # Qiu et al., eq. 2, term by term, divided by K log K.
    n_transforms = z.shape[1] - 1

    def h(event, k, j):
        cosine = torch.cosine_similarity(event[k], event[j], dim=0)
        return torch.exp(cosine / temperature)

    transformed = range(1, n_transforms + 1)
    losses = []
    for event in z:
        loss = 0.0
        for k in transformed:
            negatives = sum(h(event, k, j) for j in transformed if j != k)
            loss -= torch.log(h(event, k, 0) / (h(event, k, 0) + negatives))
        losses.append(loss / (n_transforms * math.log(n_transforms)))
    return torch.stack(losses)


@pytest.mark.parametrize("n_transforms", [2, 6, 8, 11, 16])
def test_identity_transformations_score_one(n_transforms):
    model = _model(n_transforms)
    with torch.no_grad():
        model.transforms.w_out.zero_()
        scores = model.model_step(_batch())["ascore/full"]
    assert torch.allclose(scores, torch.ones_like(scores), rtol=0, atol=1e-5)


@pytest.mark.parametrize("n_transforms", [2, 6, 16])
@pytest.mark.parametrize("temperature", [0.05, 1.0])
def test_identical_embeddings_give_one(n_transforms, temperature):
    z = torch.randn(5, 1, 8, dtype=torch.float64).expand(5, n_transforms + 1, 8)
    loss = _contrastive_loss(z, temperature)
    assert torch.allclose(loss, torch.ones_like(loss), rtol=0, atol=1e-12)


@pytest.mark.parametrize("temperature", [0.05, 0.25, 1.0])
def test_loss_is_equation_2(temperature):
    generator = torch.Generator().manual_seed(2)
    z = torch.randn(4, 7, 5, dtype=torch.float64, generator=generator)
    expected = _equation_2(z, temperature)
    assert torch.allclose(_contrastive_loss(z, temperature), expected, rtol=1e-10)


def test_score_is_the_per_event_loss_and_loss_its_mean():
    outputs = _model().model_step(_batch())
    assert outputs["ascore/full"].shape == (64,)
    assert outputs["loss"].shape == ()
    assert torch.equal(outputs["ascore/full"], outputs["loss/full"])
    assert torch.allclose(outputs["loss"], outputs["loss/full"].mean())
    assert outputs["loss"].requires_grad
    assert not outputs["ascore/full"].requires_grad
    assert outputs["ascore/full"].isfinite().all()


@pytest.mark.parametrize("transform_type", TRANSFORM_TYPES)
def test_views_hold_the_event_and_one_view_per_transformation(transform_type):
    model = _model(n_transforms=6, transform_type=transform_type)
    x = _x(8)
    views = model.views(x)
    assert views.shape == (8, 7, DIM)
    assert torch.equal(views[:, 0], x)
    for k in range(1, 7):
        assert not torch.equal(views[:, k], x)
    assert model.transforms.w_in.shape == (6, DIM, 50)
    assert model.transforms.w_out.shape == (6, 50, DIM)


def test_two_scorings_are_identical():
    model = _model().eval()
    with torch.no_grad():
        first = model.model_step(_batch())["ascore/full"]
        torch.manual_seed(123)
        second = model.model_step(_batch())["ascore/full"]
    assert torch.equal(first, second)


@pytest.mark.parametrize("transform_type", TRANSFORM_TYPES)
def test_padded_entries_keep_their_value_in_every_view(transform_type):
    model = _model(n_transforms=6, transform_type=transform_type)
    present = torch.rand(16, DIM, generator=torch.Generator().manual_seed(3)) > 0.4
    # A nonzero padded value, which the multiplicative mask would otherwise rescale.
    x = torch.where(present, _x(16), torch.full((16, DIM), -1.5))
    views = model.views(x, present)
    for k in range(1, 7):
        assert torch.equal(views[:, k][~present], x[~present])
        assert not torch.equal(views[:, k][present], x[present])


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
def test_the_batch_mask_enters_the_score(mask_dtype):
    model = _model().eval()
    x = _x(16)
    present = torch.rand(16, DIM, generator=torch.Generator().manual_seed(4)) > 0.4
    batch = (x, present.to(mask_dtype), torch.zeros(16), torch.zeros(16))
    with torch.no_grad():
        scores = model.model_step(batch)["ascore/full"]
        assert torch.equal(scores, model(x, present))
        assert not torch.equal(scores, model(x))


@pytest.mark.parametrize("transform_type", TRANSFORM_TYPES)
def test_a_few_optimiser_steps_lower_the_loss(transform_type):
    model = _model(transform_type=transform_type)
    optimizer = model.hparams.optimizer(params=model.parameters())
    batch = _batch(256)
    initial = model.model_step(batch)["loss"].item()
    for _ in range(50):
        optimizer.zero_grad()
        model.model_step(batch)["loss"].backward()
        optimizer.step()
    assert model.model_step(batch)["loss"].item() < 0.9 * initial


@pytest.mark.parametrize(
    "invalid",
    [{"n_transforms": 1}, {"temperature": 0.0}, {"transform_type": "forward"}],
)
def test_invalid_hyperparameters_raise(invalid):
    kwargs = {"n_transforms": 11, "temperature": 0.1, "transform_type": "residual"}
    with pytest.raises(ValueError):
        NeuTraLAD(Encoder(in_dim=DIM, nodes=[8, 4]), in_dim=DIM, **kwargs | invalid)


@pytest.fixture(scope="module")
def experiments() -> dict:
    if not OmegaConf.has_resolver("short_hash"):
        register_resolvers()
    overrides = ["paths.raw_data_dir=/x", "run_name=dbg"]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        return {
            tunnel: compose(
                config_name="train",
                overrides=[f"experiment={tunnel}/neutral", *overrides],
            )
            for tunnel in TUNNELS
        }


def _space(tunnel: str) -> dict:
    path = CONFIGS / "hparams_search" / tunnel / "neutral_optuna.yaml"
    params = OmegaConf.load(path).hydra.sweeper.params
    parser = OverridesParser.create()
    return {
        key: parser.parse_override(f"{key}={value}").value()
        for key, value in params.items()
    }


@pytest.mark.parametrize("tunnel", TUNNELS)
def test_the_module_is_rebuilt_from_its_config(experiments, tunnel):
    algorithm = experiments[tunnel].algorithm
    trained, rebuilt = (hydra.utils.instantiate(algorithm) for _ in range(2))
    rebuilt.load_state_dict(trained.state_dict(), strict=True)
    x = _x(32)
    with torch.no_grad():
        assert torch.equal(rebuilt.eval()(x), trained.eval()(x))
    assert isinstance(rebuilt, NeuTraLAD)
    assert rebuilt.transforms.w_in.shape == (11, DIM, 50)


@pytest.mark.parametrize("tunnel", TUNNELS)
def test_every_default_lies_inside_the_space(experiments, tunnel):
    space = _space(tunnel)
    assert space == _space(TUNNELS[0])
    for key, sweep in space.items():
        default = OmegaConf.select(experiments[tunnel], key)
        if OmegaConf.is_config(default):
            default = OmegaConf.to_container(default)
        if isinstance(sweep, IntervalSweep):
            assert sweep.start <= default <= sweep.end, key
        else:
            assert default in sweep.list, key

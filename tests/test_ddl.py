from functools import partial

import pytest
import torch

from src.algorithms.components.ddl import (
    DenoiserMLP,
    ImageDenoiser,
    alpha_bars,
    fixed_noise,
    score_timesteps,
)
from src.algorithms.ddl import DDL
from src.algorithms.image_ddl import ImageDDL
from src.callbacks.score_dump import ScoreDumpCallback
from src.evaluation.replay import ReplayModule
from tests.test_fit import _served
from tests.test_score_dump import KEYS, N_FEATURES, ToyDataModule, fit, load

DIM = 6
IMAGE = (3, 8, 8)


def _ddl(dim: int = DIM, seed: int = 0, **kwargs) -> DDL:
    torch.manual_seed(seed)
    denoiser = DenoiserMLP(dim, width=16, depth=2, time_dim=8)
    kwargs = {"n_steps": 50, "n_score_steps": 10, "target_rate": 0.1, **kwargs}
    return DDL(denoiser, dim, **kwargs)


def _image_ddl(seed: int = 0, **kwargs) -> ImageDDL:
    torch.manual_seed(seed)
    denoiser = ImageDenoiser(3, nodes=[4, 8], time_dim=8)
    kwargs = {"n_steps": 50, "n_score_steps": 10, "target_rate": 0.1, **kwargs}
    return ImageDDL(denoiser, IMAGE, **kwargs)


def _inputs(image: bool, n: int = 40, seed: int = 1) -> torch.Tensor:
    shape = (n, *IMAGE) if image else (n, DIM)
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _model(image: bool, **kwargs):
    return (_image_ddl(**kwargs) if image else _ddl(**kwargs)), _inputs(image)


def _labelled(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x, torch.zeros(len(x), dtype=torch.long)


def _scores(module, x: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
    module.eval()
    with torch.no_grad():
        parts = x.split(batch_size or len(x))
        return torch.cat(
            [module.model_step(_labelled(p))["ascore/full"] for p in parts]
        )


def _masked(n: int = 40) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(2)
    present = torch.rand(n, DIM, generator=gen) < 0.6
    present[:, 0] = True
    return torch.randn(n, DIM, generator=gen) * present, present


def _masked_batch(x: torch.Tensor, present: torch.Tensor) -> tuple:
    return x, present, torch.zeros(len(x)), torch.zeros(len(x), dtype=torch.long)


def test_schedule_keeps_the_sum_of_beta_whatever_the_number_of_steps():
    for n_steps in (100, 200, 500, 1000):
        alpha_bar = alpha_bars(n_steps, 0.02)
        previous = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])
        betas = 1 - alpha_bar / previous
        assert betas.sum().item() == pytest.approx(500 * (1e-4 + 0.02))
    assert torch.all(alpha_bars(100, 0.05).diff() < 0)
    with pytest.raises(ValueError):
        alpha_bars(10, 0.2)


def test_score_timesteps_are_strided_and_cover_every_step_at_full_count():
    assert score_timesteps(100, 10).tolist() == list(range(5, 100, 10))
    assert score_timesteps(7, 7).tolist() == list(range(7))
    with pytest.raises(ValueError):
        score_timesteps(10, 11)


def test_fixed_noise_ignores_the_torch_generator():
    torch.manual_seed(5)
    first = fixed_noise(4, (3, 2), seed=9)
    torch.manual_seed(6)
    assert torch.equal(first, fixed_noise(4, (3, 2), seed=9))
    assert not torch.equal(first, fixed_noise(4, (3, 2), seed=10))


@pytest.mark.parametrize("image", [False, True])
def test_score_is_deterministic_and_the_same_whatever_the_batching(image):
    module, x = _model(image)
    whole = _scores(module, x)
    assert torch.equal(whole, _scores(module, x))
    torch.testing.assert_close(_scores(module, x, batch_size=7), whole)
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(_scores(module, x[order]), whole[order])
    assert len(whole.unique()) == len(x)


@pytest.mark.parametrize("image", [False, True])
def test_eval_outputs_are_one_score(image):
    module, x = _model(image)
    module.eval()
    with torch.no_grad():
        outputs = module.model_step(_labelled(x))
    for key in ("fit/full", "loss/full"):
        assert torch.equal(outputs[key], outputs["ascore/full"])
    assert outputs["loss"].item() == pytest.approx(outputs["ascore/full"].mean().item())


@pytest.mark.parametrize("image", [False, True])
def test_training_loss_is_the_denoising_loss_at_a_random_timestep(image):
    module, x = _model(image)
    module.train()
    torch.manual_seed(3)
    with torch.no_grad():
        outputs = module.model_step(_labelled(x))
    torch.manual_seed(3)
    timesteps = torch.randint(0, 50, (len(x),))
    with torch.no_grad():
        expected = _denoising_loss(module, x, timesteps, torch.randn_like(x))
    torch.testing.assert_close(outputs["loss/full"], expected)
    assert torch.equal(outputs["fit/full"], outputs["loss/full"])


def _denoising_loss(module, x, timesteps, noise) -> torch.Tensor:
    if isinstance(module, ImageDDL):
        return module.denoising_loss(x, timesteps, noise)
    return module.denoising_loss(x, None, timesteps, noise)


@pytest.mark.parametrize("image", [False, True])
def test_score_equals_the_training_loss_in_expectation(image):
    # With every timestep scored, the score over noise draws and the training loss over
    # timesteps and noise share one expectation; 4 standard errors apart at most.
    module, x = _model(image, n_steps=5, n_score_steps=5, beta_end=0.002)
    x = x[:1]
    module.train()
    with torch.no_grad():
        losses = module.model_step(_labelled(x.expand(20000, *x.shape[1:])))
    draws = [_scores(_score_variant(image, seed), x) for seed in range(400)]
    draws, losses = torch.cat(draws).double(), losses["loss/full"].double()
    error = (draws.var() / len(draws) + losses.var() / len(losses)).sqrt()
    assert abs(draws.mean() - losses.mean()) < 4 * error


def _score_variant(image: bool, noise_seed: int):
    kwargs = {"n_steps": 5, "n_score_steps": 5, "beta_end": 0.002}
    kwargs["noise_seed"] = noise_seed
    return _image_ddl(**kwargs) if image else _ddl(**kwargs)


def test_padded_features_keep_their_value_and_leave_the_loss():
    module, (x, present) = _ddl(), _masked()
    score = _masked_scores(module, x, present)
    expected = torch.zeros(len(x))
    with torch.no_grad():
        for step, noise in zip(module.score_steps.tolist(), module.score_noise):
            a = module.sqrt_alpha_bar[step] ** 2
            x_t = torch.where(present, a.sqrt() * x + (1 - a).sqrt() * noise, x)
            eps = module.denoiser(x_t, torch.full((len(x),), step))
            residual = (noise - eps) ** 2 * present
            expected += residual.sum(dim=1) / present.sum(dim=1)
    assert not present.all()
    torch.testing.assert_close(score, expected / len(module.score_noise))


def test_the_noise_on_padded_features_changes_nothing():
    module, (x, present) = _ddl(), _masked()
    first = _masked_scores(module, x, present)
    module.score_noise.add_(torch.randn_like(module.score_noise) * ~present[0])
    torch.testing.assert_close(_masked_scores(module, x[:1], present[:1]), first[:1])
    torch.manual_seed(4)
    module.train()
    batch = _masked_batch(x, present)
    with torch.no_grad():
        loss = module.model_step(batch)["loss/full"]
    torch.manual_seed(4)
    timesteps = torch.randint(0, 50, (len(x),))
    noise = torch.randn_like(x)
    noise_on_padding = noise + torch.randn_like(x) * ~present
    with torch.no_grad():
        again = module.denoising_loss(x, present, timesteps, noise_on_padding)
    torch.testing.assert_close(loss, again)


def _masked_scores(module, x, present) -> torch.Tensor:
    module.eval()
    with torch.no_grad():
        return module.model_step(_masked_batch(x, present))["ascore/full"]


def test_constructor_rejects_more_score_steps_than_steps():
    with pytest.raises(ValueError):
        _ddl(n_steps=10, n_score_steps=20)


def test_weights_alone_fill_the_state_dict():
    module = _ddl()
    buffers = {"score_noise", "score_steps", "sqrt_alpha_bar"}
    assert not buffers & module.state_dict().keys()
    rebuilt = _ddl(seed=1)
    rebuilt.load_state_dict(module.state_dict(), strict=True)
    x = _inputs(False)
    assert torch.equal(_scores(rebuilt, x), _scores(module, x))


def _toy_ddl() -> DDL:
    return _ddl(dim=N_FEATURES, optimizer=partial(torch.optim.SGD, lr=0.05))


def test_replay_rescores_the_training_time_values(tmp_path):
    path = tmp_path / "scores"
    sizes = {"aux_seed": 7, "aux_size": 50, "aux_batch_size": 16}
    dump = ScoreDumpCallback(str(path), str(tmp_path), KEYS, **sizes)
    fit(tmp_path, [dump], module=_toy_ddl())
    meta, record = load(path / "meta.pt"), load(path / "epoch_0001.pt")
    stored = record["val"]["normal"]["fit/full"]
    del record["val"]["normal"]["fit/full"]
    replay = ReplayModule(_toy_ddl(), meta)
    replay.load_epoch(record, tmp_path / "epoch_0001.pt")
    served = _served(replay, "val")["normal"]
    rescored = torch.cat([outputs["fit/full"] for outputs in served])
    torch.testing.assert_close(rescored, stored, rtol=0, atol=0)
    torch.testing.assert_close(stored, record["val"]["normal"]["ascore/full"])

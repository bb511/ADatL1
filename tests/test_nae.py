import math
from functools import partial

import pytest
import torch
from pytorch_lightning import Callback, Trainer
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.components.decoder import Decoder
from src.algorithms.components.encoder import Encoder
from src.algorithms.nae import (
    NAE,
    _identity,
    _langevin,
    _log_gaussian_proposal,
    _log_projected_normal,
    _mala,
    _project,
)


def _model(in_dim: int = 6, nodes: tuple = (8, 4), **overrides) -> NAE:
    torch.manual_seed(0)
    settings = dict(
        pretrain_epochs=1,
        temperature=1e-2,
        neg_energy_weight=1.0,
        latent_steps=3,
        latent_step_size=1e-3,
        input_steps=4,
        input_step_size=1e-3,
        input_grad_clip=0.1,
        input_bound=10.0,
        buffer_size=50,
        target_rate=0.01,
        base_rate=None,
        optimizer=partial(torch.optim.Adam, lr=1e-3),
    )
    settings.update(overrides)
    return NAE(
        encoder=Encoder(in_dim=in_dim, nodes=list(nodes)),
        decoder=Decoder(nodes=list(nodes)[::-1], out_dim=in_dim),
        **settings,
    )


def _batch(n: int = 32, in_dim: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return torch.randn(n, in_dim, generator=generator), torch.zeros(n, dtype=torch.long)


def _physics_batch(n: int = 32, in_dim: int = 6) -> tuple:
    x, y = _batch(n, in_dim)
    mask = torch.ones_like(x)
    mask[: n // 2, in_dim // 2 :] = 0.0
    return x * mask, mask, torch.zeros(n), y


def _scores(model: NAE, batch) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model.model_step(batch)["ascore/full"].clone()


def test_energy_is_the_per_dimension_mse():
    model = _model()
    x, _ = _batch()
    _, reconstruction = model(x)
    expected = ((x - reconstruction) ** 2).mean(dim=1)
    torch.testing.assert_close(model.energy(x), expected, rtol=0, atol=0)

    out = model.model_step((x, torch.zeros(len(x))))
    torch.testing.assert_close(out["ascore/full"], expected.detach())
    torch.testing.assert_close(out["loss/full"], expected.detach())


def test_code_lies_on_the_unit_sphere():
    model = _model(nodes=(8, 5))
    z, _ = model(_batch()[0] * 100.0)
    torch.testing.assert_close(z.norm(dim=1), torch.ones(len(z)))
    assert z.shape[1] == 5


def test_masked_energy_averages_the_present_entries():
    model = _model()
    x, mask, _, _ = _physics_batch()
    _, reconstruction = model(x)
    squared = (x - reconstruction) ** 2
    expected = (squared * mask).sum(dim=1) / mask.sum(dim=1)
    torch.testing.assert_close(model.energy(x, mask), expected)


@pytest.mark.parametrize("metropolis", [False, True])
def test_masked_negatives_keep_the_absent_entries_of_their_event(metropolis):
    model = _model(
        latent_step_size=1e-2,
        input_step_size=1e-3,
        input_grad_clip=None,
        metropolis=metropolis,
    )
    model.train()
    x, mask, _, _ = _physics_batch()
    negative, decoded, _ = model.sample_negatives(x, mask)

    absent = mask == 0
    assert torch.equal(negative[absent], x[absent])
    assert torch.equal(decoded[absent], x[absent])
    assert not torch.equal(negative[~absent], decoded[~absent])


def test_scores_depend_on_the_weights_alone():
    model = _model(pretrain_epochs=0)
    batch = _batch()
    first = _scores(model, batch)
    assert torch.equal(first, _scores(model, batch))

    model.train()
    model.model_step(batch)
    assert model._buffer_filled == len(batch[0])
    assert torch.equal(first, _scores(model, batch))

    assert not any("buffer" in key for key in model.state_dict())
    fresh = _model(pretrain_epochs=0, latent_steps=7)
    fresh.load_state_dict(model.state_dict())
    assert torch.equal(first, _scores(fresh, batch))


@pytest.mark.parametrize("metropolis", [False, True])
def test_negatives_are_detached_and_bounded(metropolis):
    model = _model(
        pretrain_epochs=0, input_bound=0.5, input_step_size=1.0, metropolis=metropolis
    )
    model.train()
    x, _ = _batch()
    negative, decoded, _ = model.sample_negatives(x)
    assert not negative.requires_grad and not decoded.requires_grad
    assert negative.abs().max() <= 0.5


def test_replay_buffer_is_a_ring():
    model = _model(pretrain_epochs=0, buffer_size=50, replay_ratio=1.0)
    model.train()
    for _ in range(3):
        model.sample_negatives(_batch(n=32)[0])
    assert model._buffer_filled == 50
    assert model._buffer_next == (3 * 32) % 50
    torch.testing.assert_close(model._buffer.norm(dim=1), torch.ones(50))


def _gaussian_energy(x, mean, std, temperature):
    return temperature * 0.5 * ((x - mean) / std).square().sum(dim=1)


def _linear_energy(z, direction, scale):
    return -scale * z @ direction


def test_mala_samples_a_gaussian_that_ula_biases():
    mean, std = torch.tensor([1.0, -2.0]), torch.tensor([0.5, 2.0])
    temperature, step_size, steps = 0.1, 0.3, 600
    energy = partial(_gaussian_energy, mean=mean, std=std, temperature=temperature)
    log_q = partial(_log_gaussian_proposal, step_size=step_size)
    start = torch.zeros(4000, 2)

    torch.manual_seed(0)
    end, acceptance = _mala(
        start, energy, steps, step_size, temperature, log_q, _identity
    )
    assert 0.0 < acceptance < 1.0
    assert ((end.mean(dim=0) - mean).abs() < 0.1 * std).all()
    assert ((end.var(dim=0) / std**2 - 1.0).abs() < 0.1).all()

    # The unadjusted chain's variance is 2 lambda/(1 - (1 - lambda/s^2)^2), 2.5 s^2
    # in the narrow direction.
    torch.manual_seed(0)
    ula = _langevin(start, energy, steps, step_size, temperature, _identity)
    assert ula.var(dim=0)[0] / std[0] ** 2 > 2.0


def test_mala_on_the_sphere_samples_von_mises_fisher():
    # At this step the drift, step_size kappa, exceeds the radius: a chain at the
    # antipode must still find its way back.
    kappa, temperature, step_size = 4.0, 0.5, 0.3
    direction = torch.tensor([0.0, 0.0, 1.0])
    energy = partial(_linear_energy, direction=direction, scale=temperature * kappa)
    log_q = partial(_log_projected_normal, step_size=step_size)
    torch.manual_seed(0)
    start = _project(torch.randn(4000, 3))
    end, acceptance = _mala(start, energy, 300, step_size, temperature, log_q, _project)

    assert 0.0 < acceptance < 1.0
    torch.testing.assert_close(end.norm(dim=1), torch.ones(len(end)))
    # The mean resultant length of the von Mises-Fisher density on S^2.
    expected = 1.0 / math.tanh(kappa) - 1.0 / kappa
    assert abs((end @ direction).mean().item() - expected) < 0.02


@pytest.mark.parametrize("step_size", [0.02, 0.5])
@pytest.mark.parametrize("dim", [3, 8])
def test_projected_normal_density_integrates_to_one(dim, step_size):
    theta = torch.linspace(0.0, math.pi, 4001, dtype=torch.float64)
    u = torch.zeros(len(theta), dim, dtype=torch.float64)
    u[:, 0], u[:, 1] = theta.cos(), theta.sin()
    mean = torch.zeros_like(u)
    mean[:, 0] = 1.1
    density = _log_projected_normal(u, mean, step_size).exp()
    sphere = 2.0 * math.pi ** ((dim - 1) / 2) / math.gamma((dim - 1) / 2)
    total = sphere * torch.trapezoid(density * theta.sin() ** (dim - 2), theta)
    assert abs(total.item() - 1.0) < 1e-4


class _PhaseRecorder(Callback):
    def __init__(self):
        self.phases: dict[int, set[bool]] = {}
        self.lrs: dict[int, set[float]] = {}
        self.optimizers: set[int] = set()
        self.losses: list[float] = []
        self.acceptances: list[float] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        drew = "energy/negative" in outputs
        self.phases.setdefault(trainer.current_epoch, set()).add(drew)
        optimizer = trainer.optimizers[0]
        lr = optimizer.param_groups[0]["lr"]
        self.lrs.setdefault(trainer.current_epoch, set()).add(lr)
        self.optimizers.add(id(optimizer))
        values = [outputs["loss"], outputs["energy/positive"]]
        if drew:
            values += [outputs["energy/negative"], outputs["energy/decoded"]]
        self.losses += [float(v) for v in values]
        accept = [v for k, v in outputs.items() if k.startswith("sample/accept_")]
        self.acceptances += [float(v) for v in accept]


@pytest.mark.parametrize("metropolis", [False, True])
@pytest.mark.parametrize("mask", [False, True])
def test_phase_and_learning_rate_switch_at_pretrain_epochs_and_run_finite(
    mask, metropolis
):
    model = _model(pretrain_epochs=2, energy_lr=1e-5, metropolis=metropolis)
    batch = _physics_batch(n=64) if mask else _batch(n=64)
    loader = DataLoader(TensorDataset(*batch), batch_size=16)
    recorder = _PhaseRecorder()
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=4,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[recorder],
    )
    trainer.fit(model, train_dataloaders=loader)

    assert recorder.phases == {0: {False}, 1: {False}, 2: {True}, 3: {True}}
    assert recorder.lrs == {0: {1e-3}, 1: {1e-3}, 2: {1e-5}, 3: {1e-5}}
    assert len(recorder.optimizers) == 1
    assert all(torch.isfinite(torch.tensor(recorder.losses)))
    assert model._buffer_filled == 50
    assert len(recorder.acceptances) == (2 * 2 * 4 if metropolis else 0)
    assert all(0.0 <= a <= 1.0 for a in recorder.acceptances)
    assert ("train/accept_input" in trainer.callback_metrics) == metropolis

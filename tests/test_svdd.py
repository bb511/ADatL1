import copy
from functools import partial
from pathlib import Path

import hydra
import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.components.encoder import Encoder, ImageEncoder
from src.algorithms.image_svdd import ImageDeepSVDD
from src.algorithms.svdd import DeepSVDD, _away_from_zero
from src.utils.omegaconf import register_resolvers

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENTS = [
    f"{domain}/svdd" for domain in ("cchamber", "cchamber_wt", "cifar10", "mscct")
] + ["physics/svdd"]
DIM, LATENT = 6, 3


def _x(n: int = 96, seed: int = 1) -> torch.Tensor:
    return torch.randn(n, DIM, generator=torch.Generator().manual_seed(seed)) + 1.0


def _batches(x: torch.Tensor, size: int = 32) -> list[tuple[torch.Tensor, ...]]:
    return [(chunk, torch.zeros(len(chunk))) for chunk in x.split(size)]


def _model(**kwargs) -> DeepSVDD:
    torch.manual_seed(0)
    encoder = Encoder(in_dim=DIM, nodes=[16, 8, LATENT], bias=False)
    optimizer = partial(torch.optim.Adam, lr=1e-2)
    return DeepSVDD(encoder=encoder, target_rate=0.01, optimizer=optimizer, **kwargs)


class _Data(pl.LightningDataModule):
    def __init__(self, x: torch.Tensor):
        super().__init__()
        self.x = x

    def train_dataloader(self) -> DataLoader:
        return DataLoader(TensorDataset(self.x, torch.zeros(len(self.x))), 32)

    def val_dataloader(self) -> dict[str, DataLoader]:
        # Two sets, since Lightning treats a one-entry dict as a single loader.
        return {"normal": self.train_dataloader(), "copy": self.train_dataloader()}


class _LastOutputs(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.distances = outputs["ascore/full"].clone()


def _fit(model: DeepSVDD, epochs: int = 3, callbacks: list | None = None) -> None:
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=callbacks or [],
    )
    trainer.fit(model, datamodule=_Data(_x()))


def test_coordinates_nearer_zero_than_eps_move_to_eps():
    center = torch.tensor([-0.05, 0.0, 0.03, 0.2, -0.3, -0.1])
    expected = torch.tensor([-0.1, 0.1, 0.1, 0.2, -0.3, -0.1])
    assert torch.equal(_away_from_zero(center, 0.1), expected)
    assert torch.equal(_away_from_zero(center, 0.0), center)


def test_centre_is_the_mean_representation_over_every_batch():
    model, x = _model(center_eps=0.0), _x()
    model.initialize_center(_batches(x))
    with torch.no_grad():
        expected = model(x).mean(dim=0)
    torch.testing.assert_close(model.center, expected)


def test_center_max_batches_limits_the_pass():
    model, x = _model(center_eps=0.0, center_max_batches=1), _x()
    model.initialize_center(_batches(x))
    with torch.no_grad():
        expected = model(x[:32]).mean(dim=0)
    torch.testing.assert_close(model.center, expected)


def test_fit_fixes_the_centre_from_the_untrained_network_and_keeps_it():
    model = _model()
    untrained = copy.deepcopy(model)
    untrained.initialize_center(_batches(_x()))
    with torch.no_grad():
        before = untrained.model_step(_batches(_x())[0])["ascore/full"].mean()
    _fit(model)
    assert torch.equal(model.center, untrained.center)
    with torch.no_grad():
        after = model.model_step(_batches(_x())[0])["ascore/full"].mean()
    assert after < before


def test_one_class_loss_is_equation_4():
    model = _model(objective="one_class", network_weight_decay=0.3)
    model.initialize_center(_batches(_x()))
    batch = _batches(_x(seed=2))[0]
    outputs = model.model_step(batch)
    with torch.no_grad():
        z = model(batch[0])
    distance = ((z - model.center) ** 2).sum(dim=1)
    squares = sum((p**2).sum() for p in model.encoder.parameters())
    torch.testing.assert_close(outputs["loss"], distance.mean() + 0.15 * squares)
    torch.testing.assert_close(outputs["ascore/full"], distance)
    torch.testing.assert_close(outputs["loss/full"], distance)
    torch.testing.assert_close(outputs["fit/full"], distance / LATENT)
    torch.testing.assert_close(outputs["loss/reg_scaled/mean"], 0.15 * squares)


def test_soft_boundary_loss_is_equation_3():
    model = _model(objective="soft_boundary", nu=0.2, network_weight_decay=0.0)
    model.initialize_center(_batches(_x()))
    model.radius.fill_(0.5)
    batch = _batches(_x(seed=2))[0]
    outputs = model.model_step(batch)
    distance = outputs["ascore/full"]
    expected = 0.25 + torch.clamp(distance - 0.25, min=0.0).mean() / 0.2
    torch.testing.assert_close(outputs["loss"], expected)
    assert (outputs["loss/full"] >= 0.25).all()


def test_radius_waits_for_the_warm_up():
    model = _model(objective="soft_boundary", radius_warmup_epochs=3)
    _fit(model, epochs=3)
    assert model.radius.item() == 0.0


def test_radius_is_the_quantile_of_the_last_batch_after_the_warm_up():
    model = _model(objective="soft_boundary", nu=0.25, radius_warmup_epochs=1)
    last = _LastOutputs()
    _fit(model, epochs=2, callbacks=[last])
    expected = torch.quantile(last.distances.sqrt(), 0.75)
    assert model.radius.item() > 0.0
    torch.testing.assert_close(model.radius, expected)


def test_a_fresh_module_loads_a_stored_state():
    trained = _model()
    _fit(trained, epochs=1)
    state = trained.state_dict()
    assert set(state) == {"center", "radius"} | {
        f"encoder.net.net.{i}.weight" for i in (0, 3, 6)
    }
    fresh = _model()
    fresh.load_state_dict(state, strict=True)
    batch = _batches(_x(seed=3))[0]
    trained.eval(), fresh.eval()
    with torch.no_grad():
        assert torch.equal(
            fresh.model_step(batch)["ascore/full"],
            trained.model_step(batch)["ascore/full"],
        )


def test_model_step_refuses_an_unset_centre():
    with pytest.raises(RuntimeError, match="centre"):
        _model().model_step(_batches(_x())[0])


def test_the_flag_chooses_the_objective_without_a_name():
    assert _model().objective == "one_class"
    assert _model(soft_boundary=True).objective == "soft_boundary"
    assert _model(objective="soft_boundary").objective == "soft_boundary"


@pytest.mark.parametrize(
    "invalid",
    [
        {"objective": "hard_boundary"},
        {"objective": "one_class", "soft_boundary": True},
        {"nu": 0.0},
        {"nu": 1.5},
        {"center_init_method": "zeros"},
        {"center_eps": -0.1},
        {"center_max_batches": 0},
        {"radius_warmup_epochs": -1},
    ],
)
def test_invalid_hyperparameters_raise(invalid):
    with pytest.raises(ValueError):
        _model(**invalid)


def test_image_module_scores_unflattened_images():
    torch.manual_seed(0)
    encoder = ImageEncoder(3, [4, 8, LATENT], (8, 8), [2, 2], bias=False)
    model = ImageDeepSVDD(encoder=encoder)
    images = torch.randn(10, 3, 8, 8, generator=torch.Generator().manual_seed(1))
    batches = [(images, torch.zeros(10))]
    model.initialize_center(batches)
    with torch.no_grad():
        outputs = model.eval().model_step(batches[0])
        expected = ((encoder(images) - model.center) ** 2).sum(dim=1)
    torch.testing.assert_close(outputs["ascore/full"], expected)
    assert model.compute_target_fpr() == 0.01


@pytest.fixture(scope="module")
def algorithms() -> dict:
    if not OmegaConf.has_resolver("short_hash"):
        register_resolvers()
    overrides = ["paths.raw_data_dir=/x", "run_name=dbg"]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        return {
            name: compose(
                config_name="train", overrides=[f"experiment={name}", *overrides]
            )
            for name in EXPERIMENTS
        }


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_experiment_builds_a_bias_free_network_without_batchnorm(
    algorithms, experiment
):
    model = hydra.utils.instantiate(algorithms[experiment].algorithm)
    assert isinstance(model, DeepSVDD) and model.objective == "one_class"
    assert not [name for name, _ in model.named_parameters() if "bias" in name]
    batchnorm = nn.modules.batchnorm._BatchNorm
    assert not [m for m in model.modules() if isinstance(m, batchnorm)]

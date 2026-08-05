"""Scientific equivalence tests for segmented retrain RNG continuation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, IterableDataset

from src.callbacks.exact_retrain_rng import ExactRetrainRNGState


class _ShuffledIterable(IterableDataset):
    def __init__(self, values: torch.Tensor, shuffler: torch.Generator) -> None:
        self.values = values
        self.shuffler = shuffler

    def __iter__(self):
        for index in torch.randperm(len(self.values), generator=self.shuffler):
            yield self.values[index]


class _DataModule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.shuffler = torch.Generator().manual_seed(917)
        self.values = torch.linspace(-1.0, 1.0, 12).reshape(-1, 1)

    def train_dataloader(self):
        return DataLoader(
            _ShuffledIterable(self.values, self.shuffler),
            batch_size=None,
            num_workers=0,
        )


class _StochasticModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.7]]))

    def training_step(self, batch, batch_idx):
        python_scale = random.uniform(0.8, 1.2)  # nosec B311 -- stochastic test path
        numpy_shift = float(np.random.uniform(-0.1, 0.1))
        torch_noise = torch.randn_like(batch) * 0.03
        prediction = self.weight * batch * python_scale + numpy_shift + torch_noise
        return prediction.square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.04, momentum=0.3)


class _ResumeStartupRNGConsumer(Callback):
    """Model a lazy logger/setup path that runs only in a resumed process."""

    def on_train_start(self, trainer, pl_module) -> None:
        if trainer.ckpt_path is not None:
            random.random()  # nosec B311 -- deliberate lifecycle perturbation
            np.random.random()


def _trainer(max_epochs: int, callbacks: list, root: Path) -> Trainer:
    return Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=False,
        default_root_dir=root,
        deterministic=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


def test_segmented_rng_and_iterable_shuffle_match_uninterrupted_training(tmp_path: Path) -> None:
    """A fresh process at a segment boundary follows the uninterrupted trajectory."""
    seed_everything(123, workers=True)
    uninterrupted = _StochasticModel()
    _trainer(
        4,
        [ExactRetrainRNGState()],
        tmp_path / "uninterrupted",
    ).fit(uninterrupted, datamodule=_DataModule())

    seed_everything(123, workers=True)
    first = _StochasticModel()
    checkpointing = ModelCheckpoint(
        dirpath=tmp_path / "segmented",
        save_last=True,
        save_top_k=0,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    _trainer(2, [ExactRetrainRNGState(), checkpointing], tmp_path / "first").fit(
        first, datamodule=_DataModule()
    )
    checkpoint = tmp_path / "segmented" / "last.ckpt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert ExactRetrainRNGState().state_key in payload["callbacks"]

    # Reproduce a new allocation: process seeding and object construction happen
    # again, then the exact callback restores boundary state before iteration.
    seed_everything(123, workers=True)
    resumed = _StochasticModel()
    _trainer(
        4,
        [ExactRetrainRNGState(), _ResumeStartupRNGConsumer()],
        tmp_path / "resumed",
    ).fit(
        resumed,
        datamodule=_DataModule(),
        ckpt_path=checkpoint,
    )

    assert torch.equal(uninterrupted.weight, resumed.weight)


def test_exact_rng_state_rejects_missing_datamodule_generator() -> None:
    callback = ExactRetrainRNGState(require_datamodule_shuffler=True)
    trainer = type(
        "TrainerStub",
        (),
        {"datamodule": object(), "world_size": 1, "ckpt_path": None},
    )()
    try:
        callback.on_fit_start(trainer, None)
    except RuntimeError as error:
        assert "datamodule.shuffler" in str(error)
    else:
        raise AssertionError("Missing datamodule shuffler was accepted.")


def test_resume_without_exact_rng_callback_state_fails_closed() -> None:
    callback = ExactRetrainRNGState(require_datamodule_shuffler=False)
    trainer = type(
        "TrainerStub",
        (),
        {"datamodule": None, "world_size": 1, "ckpt_path": "boundary.ckpt"},
    )()
    try:
        callback.on_fit_start(trainer, None)
    except RuntimeError as error:
        assert "no exact retrain RNG state" in str(error)
    else:
        raise AssertionError("Resume without exact RNG callback state was accepted.")


def test_corrupt_numpy_or_shuffler_state_is_rejected() -> None:
    callback = ExactRetrainRNGState()
    state = {
        "schema_version": 1,
        "python": random.getstate(),
        "numpy": {},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [
            torch.get_rng_state()
            for _ in range(torch.cuda.device_count() if torch.cuda.is_available() else 0)
        ],
        "datamodule_shuffler": None,
    }
    try:
        callback.load_state_dict(state)
    except RuntimeError as error:
        assert "NumPy RNG" in str(error)
    else:
        raise AssertionError("Corrupt NumPy RNG state was accepted.")
    numpy_state = np.random.get_state()
    state["numpy"] = {
        "algorithm": numpy_state[0],
        "keys": torch.from_numpy(numpy_state[1].copy()),
        "position": numpy_state[2],
        "has_gauss": numpy_state[3],
        "cached_gaussian": numpy_state[4],
    }
    try:
        callback.load_state_dict(state)
    except RuntimeError as error:
        assert "shuffler state" in str(error)
    else:
        raise AssertionError("Missing shuffler RNG state was accepted.")

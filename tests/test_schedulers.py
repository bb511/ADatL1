from functools import partial
from types import SimpleNamespace

import torch
from omegaconf import DictConfig
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.schedulers.cdw import CosineWithWarmup


def _module(total_steps: int | None) -> ADLightningModule:
    scheduler = {
        "interval": "step",
        "frequency": 1,
        "scheduler": partial(
            CosineWithWarmup,
            warmup_ratio=0.05,
            min_lr_ratio=0.05,
            warmup_start_ratio=1e-3,
        ),
    }
    if total_steps is not None:
        scheduler["total_steps"] = total_steps
    module = ADLightningModule(
        model=nn.Linear(2, 2),
        scheduler=DictConfig(scheduler, flags={"allow_objects": True}),
    )
    module._trainer = SimpleNamespace(estimated_stepping_batches=100)
    return module


def test_scheduler_can_preserve_a_preregistered_training_horizon() -> None:
    module = _module(total_steps=1600)
    optimizer = torch.optim.AdamW(module.parameters(), lr=2e-4)

    configured = module._set_up_scheduler(optimizer)

    assert configured["scheduler"].total_steps == 1600
    assert "total_steps" not in configured


def test_scheduler_defaults_to_lightning_estimated_steps() -> None:
    module = _module(total_steps=None)
    optimizer = torch.optim.AdamW(module.parameters(), lr=2e-4)

    configured = module._set_up_scheduler(optimizer)

    assert configured["scheduler"].total_steps == 100

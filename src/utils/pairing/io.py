from __future__ import annotations

import os
from pathlib import Path

import hydra
import rootutils
import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.omegaconf import register_resolvers


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def compose_config(
    config_dir: str | Path = "configs",
    config_name: str = "train",
    overrides: list[str] | None = None,
) -> DictConfig:
    config_dir = Path(config_dir)
    if not config_dir.is_absolute():
        config_dir = repo_root() / config_dir

    GlobalHydra.instance().clear()
    register_resolvers()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name=config_name, overrides=overrides or [])


def load_encoder_run(
    ckpt_path: str | Path,
    *,
    config_dir: str | Path = "configs",
    config_name: str = "train",
    overrides: list[str] | None = None,
    stage: str = "validate",
    device: str = "cpu",
):
    load_dotenv()
    os.environ.setdefault("KERAS_BACKEND", "torch")
    if stage not in {"validate", "test"}:
        raise ValueError("stage must be 'validate' or 'test'.")
    checkpoint = Path(ckpt_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Encoder checkpoint does not exist: {checkpoint}")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

    cfg = compose_config(config_dir=config_dir, config_name=config_name, overrides=overrides)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup(stage)

    model = hydra.utils.instantiate(cfg.algorithm)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Checkpoint does not contain a Lightning state_dict: {checkpoint}")
    state = payload["state_dict"]
    model.load_state_dict(state, strict=True)
    if hasattr(model, "setup_pairing"):
        model.setup_pairing(datamodule, setup_lorentz=True)
    model.to(device)
    model.eval()
    return cfg, datamodule, model

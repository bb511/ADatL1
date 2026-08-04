"""Frozen JetCLR inference helpers for canonical CAP pair-table production."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import hydra
import torch
import torch.nn.functional as F

from src.utils.pairing.io import compose_config


def load_frozen_encoder(
    checkpoint: Path,
    object_feature_map: Mapping,
    *,
    config_dir: Path,
    config_name: str,
    overrides: list[str],
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate the configured encoder and strictly restore checkpoint weights."""
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"JetCLR checkpoint does not exist: {checkpoint}")
    cfg = compose_config(config_dir=config_dir, config_name=config_name, overrides=overrides)
    encoder = hydra.utils.instantiate(cfg.algorithm.model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint has no Lightning state_dict: {checkpoint}")
    encoder_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    if not encoder_state:
        raise ValueError(f"Checkpoint contains no model.* encoder weights: {checkpoint}")
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.set_object_feature_map(dict(object_feature_map))
    return encoder.to(device).eval()


@torch.inference_mode()
def encode_in_batches(
    encoder: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode an ordered cache tensor and return finite L2-normalized CPU embeddings."""
    if int(batch_size) <= 0:
        raise ValueError("JetCLR inference batch size must be positive.")
    if x.shape != mask.shape:
        raise ValueError(f"JetCLR data/mask shape mismatch: {x.shape} vs {mask.shape}.")
    chunks = []
    for start in range(0, x.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), x.shape[0])
        embedding = encoder(
            x[start:stop].to(device, non_blocking=False),
            mask[start:stop].to(device, non_blocking=False),
        ).float()
        embedding = F.normalize(embedding, dim=1)
        if not torch.isfinite(embedding).all():
            raise ValueError(f"Non-finite JetCLR embeddings at rows [{start}, {stop}).")
        chunks.append(embedding.cpu())
    if not chunks:
        raise ValueError("Cannot encode an empty JetCLR source tensor.")
    return torch.cat(chunks, dim=0).contiguous()

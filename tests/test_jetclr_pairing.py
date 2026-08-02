from pathlib import Path

import hydra
import pytest
import torch

from src.utils.pairing.io import compose_config
from src.utils.pairing.jetclr import encode_in_batches, load_frozen_encoder
from src.utils.pairing.table import sha256_file


class _MaskAwareEncoder(torch.nn.Module):
    """Small deterministic encoder that exposes whether masks are forwarded."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return two mask-sensitive summary coordinates."""
        return torch.stack(((x * mask).sum(dim=1), (x * mask).square().sum(dim=1)), dim=1)


def test_encode_in_batches_preserves_order_masks_and_normalizes() -> None:
    """Batched inference must retain row order, masks, and unit normalization."""
    x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ],
        dtype=torch.bool,
    )
    encoder = _MaskAwareEncoder()

    actual = encode_in_batches(encoder, x, mask, batch_size=2, device=torch.device("cpu"))
    expected = torch.nn.functional.normalize(encoder(x, mask), dim=1)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.linalg.vector_norm(actual, dim=1), torch.ones(6))


def test_encode_in_batches_rejects_invalid_outputs() -> None:
    """Non-finite encoder outputs must fail before matching."""

    class _InvalidEncoder(torch.nn.Module):
        """Encoder fixture returning invalid values."""

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """Return a deliberately invalid embedding."""
            return torch.full((x.shape[0], 2), float("nan"))

    with pytest.raises(ValueError, match="Non-finite"):
        encode_in_batches(
            _InvalidEncoder(),
            torch.ones(2, 3),
            torch.ones(2, 3, dtype=torch.bool),
            batch_size=1,
            device=torch.device("cpu"),
        )


def test_load_frozen_encoder_strictly_restores_model_weights(tmp_path: Path) -> None:
    """The inference encoder must be restored strictly and switched to eval mode."""
    cfg = compose_config(
        config_dir=Path("configs"),
        config_name="train",
        overrides=["experiment=physics/jetclr_aad_best"],
    )
    reference = hydra.utils.instantiate(cfg.algorithm.model)
    checkpoint = tmp_path / "encoder.ckpt"
    torch.save(
        {"state_dict": {f"model.{name}": value for name, value in reference.state_dict().items()}},
        checkpoint,
    )
    feature_map = {
        "jets": {"Et": [0], "eta": [1], "phi": [2]},
    }

    restored = load_frozen_encoder(
        checkpoint,
        feature_map,
        config_dir=Path("configs"),
        config_name="train",
        overrides=["experiment=physics/jetclr_aad_best"],
        device=torch.device("cpu"),
    )

    assert restored.object_feature_map == feature_map
    assert not restored.training
    assert sha256_file(checkpoint)


def test_physics_pairing_config_resolves_jetclr_tables() -> None:
    """Selecting JetCLR must resolve the canonical validation and test filenames."""
    cfg = compose_config(
        config_dir=Path("configs"),
        config_name="train",
        overrides=[
            "experiment=physics/vae_background_pairing",
            "physics_pairing.strategy=jetclr",
        ],
    )

    assert str(cfg.physics_pairing.validation_table).endswith("validate_jetclr_cap_n163840.pt")
    assert str(cfg.physics_pairing.test_table).endswith("test_jetclr_cap_n163840.pt")

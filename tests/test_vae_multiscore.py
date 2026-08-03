"""Tests for AE-initialized, multi-score VAE research support."""

from __future__ import annotations

import math

import torch

from scripts import cchamber_vae_multiscore_campaign as campaign
from src.algorithms.components.decoder import Decoder
from src.algorithms.components.encoder import Encoder, VariationalEncoder
from src.algorithms.vae import VAE


def _dense_modules():
    ae_encoder = Encoder(in_dim=11, nodes=[24, 8], activation="relu")
    ae_decoder = Decoder(nodes=[8, 24], out_dim=11, activation="relu")
    vae_encoder = VariationalEncoder(in_dim=11, nodes=[24, 8], activation="relu")
    vae_decoder = Decoder(nodes=[8, 24], out_dim=11, activation="relu")
    return ae_encoder, ae_decoder, vae_encoder, vae_decoder


def test_ae_initialization_copies_trunk_mean_and_decoder(tmp_path) -> None:
    """Dense AE parameters seed the corresponding VAE components exactly."""
    ae_encoder, ae_decoder, vae_encoder, vae_decoder = _dense_modules()
    state = {
        **{f"encoder.{name}": value for name, value in ae_encoder.state_dict().items()},
        **{f"decoder.{name}": value for name, value in ae_decoder.state_dict().items()},
    }
    checkpoint = tmp_path / "ae.ckpt"
    torch.save({"state_dict": state}, checkpoint)

    model = VAE(
        encoder=vae_encoder,
        decoder=vae_decoder,
        pretrained_ae_ckpt=str(checkpoint),
        initial_log_variance=math.log(0.01),
    )

    torch.testing.assert_close(model.encoder.net.net[0].weight, ae_encoder.net.net[0].weight)
    torch.testing.assert_close(model.encoder.z_mean.weight, ae_encoder.net.net[3].weight)
    torch.testing.assert_close(model.encoder.z_mean.bias, ae_encoder.net.net[3].bias)
    torch.testing.assert_close(model.decoder.net[0].weight, ae_decoder.net[0].weight)
    torch.testing.assert_close(
        model.encoder.z_log_var.weight, torch.zeros_like(model.encoder.z_log_var.weight)
    )
    torch.testing.assert_close(
        model.encoder.z_log_var.bias,
        torch.full_like(model.encoder.z_log_var.bias, math.log(0.01)),
    )


def test_residual_scores_match_prespecified_formulas() -> None:
    """VAE residual scores implement MSE, diagonal energy, and OAS energy."""
    _, _, encoder, decoder = _dense_modules()
    model = VAE(encoder=encoder, decoder=decoder)
    model.set_residual_score_state(
        mean=torch.tensor([1.0, 1.0] + [0.0] * 9),
        variance=torch.tensor([4.0, 1.0] + [1.0] * 9),
        oas_location=torch.tensor([1.0, 1.0] + [0.0] * 9),
        oas_precision=torch.eye(11),
    )
    residual = torch.tensor([[1.0, 2.0] + [0.0] * 9, [3.0, 4.0] + [0.0] * 9])
    target = residual
    reconstruction = torch.zeros_like(target)

    mse, diagonal, oas = model.residual_scores(target, reconstruction)

    torch.testing.assert_close(mse, torch.tensor([5.0 / 11.0, 25.0 / 11.0]))
    torch.testing.assert_close(diagonal, torch.tensor([1.0 / 11.0, 10.0 / 11.0]))
    torch.testing.assert_close(oas, torch.tensor([1.0 / 11.0, 13.0 / 11.0]))


def test_campaign_covers_every_score_selector_on_shared_grid() -> None:
    """The frozen search surface is exactly four scores by four selectors."""
    assert len(campaign.GRID) == 16
    assert len(campaign.BRANCHES) == 16
    assert set(campaign.BRANCHES) == {
        f"{score}__{selector}" for score in campaign.SCORES for selector in campaign.SELECTORS
    }
    assert set(campaign.MONITORS) == set(campaign.BRANCHES)
    assert campaign.EXPECTED_TRAJECTORIES == 48

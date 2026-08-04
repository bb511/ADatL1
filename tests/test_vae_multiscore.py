"""Tests for AE-initialized, multi-score VAE research support."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn

from scripts import cchamber_vae_multiscore_campaign as campaign
from scripts import cchamber_vae_multiscore_evaluate_safe as safe_evaluation
from src.algorithms.components.decoder import Decoder
from src.algorithms.components.encoder import Encoder, VariationalEncoder
from src.algorithms.vae import VAE
from src.callbacks.cap import VAEResidualScoreStateCallback


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


def test_vae_can_route_residual_oas_as_its_canonical_score() -> None:
    """Generic metric consumers must receive OAS when the experiment selects it."""
    _, _, encoder, decoder = _dense_modules()
    model = VAE(encoder=encoder, decoder=decoder, anomaly_score="residual_oas")
    model.set_residual_score_state(
        mean=torch.zeros(11),
        variance=torch.ones(11),
        oas_location=torch.zeros(11),
        oas_precision=torch.eye(11),
    )
    batch = {"x": torch.ones(3, 11), "y": torch.zeros(3)}

    output = model.model_step(batch)

    torch.testing.assert_close(output["ascore/full"], output["ascore/residual_oas"])
    assert not torch.equal(output["ascore/full"], output["ascore/kl_raw"])


def test_vae_residual_oas_ignores_padded_decoder_residuals() -> None:
    """VAE residual scores must use the same observed-feature support as masked MSE."""
    _, _, encoder, decoder = _dense_modules()
    model = VAE(encoder=encoder, decoder=decoder)
    model.set_residual_score_state(
        mean=torch.zeros(11),
        variance=torch.ones(11),
        oas_location=torch.zeros(11),
        oas_precision=torch.eye(11),
    )
    target = torch.tensor([[2.0] + [0.0] * 10])
    reconstruction = torch.tensor([[0.0] + [1000.0] * 10])
    mask = torch.tensor([[True] + [False] * 10])

    mse, diagonal, oas = model.residual_scores(target, reconstruction, mask)

    expected = torch.tensor([4.0])
    torch.testing.assert_close(mse, expected)
    torch.testing.assert_close(diagonal, expected)
    torch.testing.assert_close(oas, expected)


def test_vae_residual_state_respects_max_samples() -> None:
    """Physics OAS fitting must remain bounded on the large training dataset."""
    callback = VAEResidualScoreStateCallback(max_samples=3)
    batches = [
        {"x": torch.tensor([[0.0, 0.0], [1.0, 0.0]]), "y": torch.zeros(2)},
        {"x": torch.tensor([[2.0, 0.0], [3.0, 0.0]]), "y": torch.zeros(2)},
    ]
    trainer = SimpleNamespace(
        sanity_checking=False,
        world_size=1,
        train_dataloader=batches,
    )
    module = nn.Module()
    module.deterministic_reconstruction = lambda x: torch.zeros_like(x)
    module.set_residual_score_state = lambda *args: None
    module.device = torch.device("cpu")

    callback.on_validation_epoch_start(trainer, module)

    assert callback.fit_samples == 3


def test_campaign_covers_every_score_selector_on_shared_grid() -> None:
    """The frozen search surface is exactly four scores by four selectors."""
    assert len(campaign.GRID) == 16
    assert len(campaign.BRANCHES) == 16
    assert set(campaign.BRANCHES) == {
        f"{score}__{selector}" for score in campaign.SCORES for selector in campaign.SELECTORS
    }
    assert set(campaign.MONITORS) == set(campaign.BRANCHES)
    assert campaign.EXPECTED_TRAJECTORIES == 48


def test_safe_evaluation_moves_model_before_inference_mode(monkeypatch, tmp_path) -> None:
    """The evaluation adapter performs the first device move outside inference mode."""
    inference_states: list[bool] = []

    class FakeModel:
        def eval(self):
            return self

        def to(self, _device):
            inference_states.append(torch.is_inference_mode_enabled())
            return self

    @torch.inference_mode()
    def inference_scores(model, _loader, _score_name, device):
        model.eval().to(device)
        return torch.ones(1).numpy()

    frozen = SimpleNamespace(_scores=inference_scores)

    def evaluate(root, trajectory_index):
        frozen._scores(FakeModel(), [], "residual_oas", torch.device("cpu"))
        assert trajectory_index == 7
        return root / "evaluation" / "007.json"

    frozen.evaluate = evaluate
    monkeypatch.setattr(safe_evaluation, "_frozen_module", lambda: frozen)

    result = safe_evaluation.evaluate(tmp_path, 7)

    assert result == tmp_path / "evaluation" / "007.json"
    assert inference_states == [False, True]

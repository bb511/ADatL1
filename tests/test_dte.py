from __future__ import annotations

from functools import partial
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from src.algorithms.components.dte import DTEPredictor
from src.algorithms.dte import DTE
from src.data.components.normalization import L1DataNormalizer


def _model(
    *,
    in_dim: int = 4,
    n_steps: int = 10,
    n_bins: int = 3,
    dropout: float = 0.0,
) -> DTE:
    return DTE(
        predictor=DTEPredictor(
            in_dim=in_dim,
            hidden_dims=[8, 6],
            out_dim=n_bins,
            dropout=dropout,
        ),
        n_steps=n_steps,
        n_bins=n_bins,
        beta_start=0.0,
        beta_end=0.1,
        target_rate=0.25,
        base_rate=None,
        optimizer=partial(torch.optim.AdamW, lr=1e-3),
        scheduler=None,
    )


def test_dte_schedule_and_time_bins() -> None:
    model = _model()

    assert model.betas.shape == (10,)
    assert model.betas[0].item() == 0.0
    assert model.betas[-1].item() == pytest.approx(0.1)
    assert model.noise_scales[0].item() == 0.0
    assert torch.all(model.noise_scales[1:] > model.noise_scales[:-1])

    timesteps = torch.arange(10)
    assert model.time_to_bin(timesteps).tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_dte_corruption_respects_explicit_padding_mask() -> None:
    model = _model()
    x = torch.tensor([[0.0, 1.0, 0.0, 2.0]])
    mask = torch.tensor([[True, True, False, False]])
    timestep = torch.tensor([9])
    noise = torch.ones_like(x)

    corrupted = model.corrupt(x, timestep, mask=mask, noise=noise)
    scale = model.noise_scales[9]

    torch.testing.assert_close(corrupted[0, :2], x[0, :2] + scale)
    torch.testing.assert_close(corrupted[0, 2:], x[0, 2:])
    assert corrupted[0, 0] != 0.0  # An active zero is not mistaken for padding.


def test_dte_expected_bin_score_matches_hand_calculation() -> None:
    model = _model(n_bins=3)
    logits = torch.tensor([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]])
    expected = torch.softmax(logits, dim=1) @ torch.arange(3, dtype=torch.float32)

    torch.testing.assert_close(model.anomaly_score(logits), expected)


def test_dte_rejects_an_autoencoder_encoder() -> None:
    with pytest.raises(ValueError, match="does not accept an encoder"):
        DTE(
            predictor=DTEPredictor(
                in_dim=4,
                hidden_dims=[8],
                out_dim=3,
                dropout=0.0,
            ),
            encoder={"in_dim": 4},
        )


def test_dte_model_step_has_finite_scores_and_predictor_gradients() -> None:
    model = _model(dropout=0.1)
    model.train()
    x = torch.randn(12, 4)
    mask = torch.ones_like(x, dtype=torch.bool)
    batch = (
        x,
        mask,
        torch.zeros(12, dtype=torch.bool),
        torch.zeros(12, dtype=torch.long),
    )

    output = model.model_step(batch)
    assert output["ascore/full"].shape == (12,)
    assert output["loss/full"].shape == (12,)
    assert torch.isfinite(output["ascore/full"]).all()
    assert torch.isfinite(output["loss/full"]).all()

    output["loss"].backward()
    parameters = [
        parameter for parameter in model.predictor.parameters() if parameter.requires_grad
    ]
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)


def test_dte_clean_scores_are_deterministic_and_survive_state_dict_round_trip() -> None:
    model = _model(dropout=0.3).eval()
    x = torch.randn(9, 4)
    mask = torch.ones_like(x, dtype=torch.bool)
    batch = (
        x,
        mask,
        torch.zeros(9, dtype=torch.bool),
        torch.zeros(9, dtype=torch.long),
    )

    score_1 = model.model_step(batch)["ascore/full"]
    score_2 = model.model_step(batch)["ascore/full"]
    torch.testing.assert_close(score_1, score_2)

    restored = _model(dropout=0.3).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(
        score_1,
        restored.model_step(batch)["ascore/full"],
    )
    torch.testing.assert_close(model.noise_scales, restored.noise_scales)


def test_standard_normalizer_uses_training_mean_and_scale() -> None:
    train = ak.Array({"Et": [[1.0, 3.0], [5.0]], "eta": [[-1.0, 1.0], [3.0]]})
    normalizer = L1DataNormalizer(name="standard", hyperparams={})
    normalizer.fit(train, "jets")
    normalized = normalizer.norm(train, "jets")

    et = ak.to_numpy(ak.flatten(normalized["Et"]))
    eta = ak.to_numpy(ak.flatten(normalized["eta"]))
    assert np.mean(et) == pytest.approx(0.0, abs=1e-7)
    assert np.std(et) == pytest.approx(1.0)
    assert np.mean(eta) == pytest.approx(0.0, abs=1e-7)
    assert np.std(eta) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("experiment", "in_dim"),
    [
        ("synthetic/dte", 8),
        ("physics/dte_agnostic", 117),
    ],
)
def test_dte_experiment_configs_compose_and_instantiate(
    experiment: str,
    in_dim: int,
) -> None:
    with initialize_config_dir(
        config_dir=str(Path("configs").resolve()),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}"],
        )

    model = instantiate(cfg.algorithm)
    assert isinstance(model, DTE)
    assert model.predictor.in_dim == in_dim
    batch_size = 4
    output = model.model_step(
        (
            torch.randn(batch_size, in_dim),
            torch.ones(batch_size, in_dim, dtype=torch.bool),
            torch.zeros(batch_size, dtype=torch.bool),
            torch.zeros(batch_size, dtype=torch.long),
        )
    )
    assert output["ascore/full"].shape == (batch_size,)
    assert torch.isfinite(output["ascore/full"]).all()
    if "physics" in experiment:
        assert cfg.algorithm.encoder is None
        assert cfg.data.data_normalizer.name == "standard"
        assert cfg.evaluation.callbacks.reco is None

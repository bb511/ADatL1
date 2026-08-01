from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.algorithms.components.augmentation import (
    FastDetectorSmearing,
    FastFeatureBlur,
    FastLorentzRotation,
    FastObjectDropout,
)
from src.algorithms.components.jet_encoder import ObjectTransformerEncoder
from src.algorithms.jetclr import JetCLR
from src.algorithms.losses import contrastive as contrastive_module
from src.algorithms.losses.contrastive import NTXentLoss
from src.evaluation.callbacks.embedding_anomaly import EmbeddingAnomalyMetrics
from src.evaluation.callbacks.pairing_diagnostics import PairingDiagnostics


class _IdentityNormalizer:
    def __init__(self, n_features: int):
        self.scale_tensor = torch.ones(n_features)
        self.shift_tensor = torch.zeros(n_features)

    def denorm_1d_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def norm_1d_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _feature_map() -> dict:
    return {
        "FET": {"Et": [0], "eta": [1], "phi": [2]},
        "jets": {"Et": [3, 6], "eta": [4, 7], "phi": [5, 8]},
    }


def test_encoder_keeps_fet_with_padded_eta() -> None:
    encoder = ObjectTransformerEncoder(
        object_types=["FET", "jets"],
        d_model=8,
        out_dim=8,
        n_heads=2,
        n_layers=1,
        dim_feedforward=16,
        dropout=0.0,
    )
    encoder.set_object_feature_map(_feature_map())
    x = torch.randn(4, 9)
    mask = torch.ones_like(x)
    mask[:, 1] = 0

    _, token_mask, _ = encoder._tokens_from_flat(x, mask)

    assert token_mask[:, 0].all()


def test_encoder_is_invariant_to_same_type_object_permutations() -> None:
    torch.manual_seed(7)
    encoder = ObjectTransformerEncoder(
        object_types=["FET", "jets"],
        d_model=8,
        out_dim=8,
        n_heads=2,
        n_layers=1,
        dim_feedforward=16,
        dropout=0.0,
    ).eval()
    encoder.set_object_feature_map(_feature_map())
    x = torch.randn(3, 9)
    mask = torch.ones_like(x)
    mask[:, 1] = 0
    permuted = x.clone()
    permuted[:, 3:6] = x[:, 6:9]
    permuted[:, 6:9] = x[:, 3:6]

    torch.testing.assert_close(encoder(x, mask), encoder(permuted, mask))


def test_object_dropout_removes_whole_objects_and_protects_fet() -> None:
    augmenter = FastObjectDropout(
        prob=1.0,
        object_prob=1.0,
        protected_object_types=["FET"],
    )
    augmenter.rng.set_seed(11)
    augmenter.set_object_feature_map(_feature_map())
    x = torch.arange(18, dtype=torch.float32).reshape(2, 9)
    mask = torch.ones_like(x)

    x_out, mask_out = augmenter(x, mask)

    torch.testing.assert_close(x_out[:, :3], x[:, :3])
    torch.testing.assert_close(mask_out[:, :3], mask[:, :3])
    assert torch.count_nonzero(x_out[:, 3:]) == 0
    assert torch.count_nonzero(mask_out[:, 3:]) == 0


def test_detector_smearing_respects_padding_and_seed() -> None:
    x = torch.zeros(8, 4)
    mask = torch.ones_like(x)
    mask[:, -1] = 0
    resolution = torch.tensor([0.5, 1.0, 2.0, 4.0])

    def make_augmenter() -> FastDetectorSmearing:
        augmenter = FastDetectorSmearing(
            prob=1.0,
            strength=1.0,
            normalizer=_IdentityNormalizer(4),
            resolution_tensor=resolution,
        )
        augmenter.rng.set_seed(13)
        return augmenter

    first, first_mask = make_augmenter()(x, mask)
    second, _ = make_augmenter()(x, mask)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first_mask, mask)
    assert torch.count_nonzero(first[:, :3]) > 0
    assert torch.count_nonzero(first[:, -1]) == 0


def test_evaluation_augmentation_reset_is_independent_of_rng_history() -> None:
    blur_pair = nn.ModuleDict(
        {
            "1": FastFeatureBlur(prob=1.0, magnitude=1.0, strength=0.5),
            "2": FastFeatureBlur(prob=1.0, magnitude=1.0, strength=0.5),
        }
    )
    module = SimpleNamespace(
        feat_blurs=blur_pair,
        detector_smears=nn.ModuleDict({"1": nn.Identity(), "2": nn.Identity()}),
        obj_masks=nn.ModuleDict({"1": nn.Identity(), "2": nn.Identity()}),
        lorentz_rot=nn.ModuleDict({"1": nn.Identity(), "2": nn.Identity()}),
    )
    x = torch.zeros(8, 4)

    JetCLR._reset_augmentation_rng(module, 101)
    expected = blur_pair["1"](x)
    blur_pair["1"](x)
    blur_pair["1"](x)
    JetCLR._reset_augmentation_rng(module, 101)
    actual = blur_pair["1"](x)

    torch.testing.assert_close(actual, expected)


def test_detector_smearing_sanitizes_energy_and_wraps_phi() -> None:
    x = torch.tensor([[0.0, 3.13], [0.0, -3.13]])
    mask = torch.ones_like(x)
    augmenter = FastDetectorSmearing(
        prob=1.0,
        strength=5.0,
        normalizer=_IdentityNormalizer(2),
        resolution_tensor=torch.ones(2),
        nonnegative_mask=torch.tensor([True, False]),
        periodic_scale_tensor=torch.tensor([0.0, 1.0]),
    )
    augmenter.rng.set_seed(17)

    smeared, _ = augmenter(x, mask)

    assert (smeared[:, 0] >= 0).all()
    assert (smeared[:, 1] >= -torch.pi).all()
    assert (smeared[:, 1] < torch.pi).all()


def test_phi_rotation_preserves_circular_separations() -> None:
    x = torch.tensor([[0.2, 3.0], [-2.9, 1.7]])
    mask = torch.ones_like(x)
    augmenter = FastLorentzRotation(
        prob=1.0,
        normalizer=_IdentityNormalizer(2),
        phi_mask=torch.tensor([True, True]),
        l1_scale_phi=torch.ones(2),
    )
    augmenter.rng.set_seed(19)

    rotated, _ = augmenter(x, mask)
    original_delta = torch.atan2(torch.sin(x[:, 0] - x[:, 1]), torch.cos(x[:, 0] - x[:, 1]))
    rotated_delta = torch.atan2(
        torch.sin(rotated[:, 0] - rotated[:, 1]),
        torch.cos(rotated[:, 0] - rotated[:, 1]),
    )

    torch.testing.assert_close(original_delta, rotated_delta)
    assert (rotated >= -torch.pi).all()
    assert (rotated < torch.pi).all()


def test_ntxent_prefers_matching_pairs() -> None:
    z1 = torch.eye(8)
    z2 = z1.clone()
    mismatched = torch.roll(z2, shifts=1, dims=0)
    loss = NTXentLoss(temperature=0.1, gather_distributed=False)

    assert loss(z1, z2) < loss(z1, mismatched)


def test_ntxent_uses_rank_offset_for_distributed_negatives(monkeypatch) -> None:
    monkeypatch.setattr(contrastive_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(contrastive_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(contrastive_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(contrastive_module.dist, "get_rank", lambda: 1)

    def fake_size_gather(outputs, value) -> None:
        for output in outputs:
            output.copy_(value)

    monkeypatch.setattr(contrastive_module.dist, "all_gather", fake_size_gather)
    monkeypatch.setattr(
        contrastive_module,
        "all_gather",
        lambda value: (torch.flip(value, dims=(0,)), value),
    )
    z = torch.eye(4)

    loss = NTXentLoss(temperature=0.1, gather_distributed=True)(z, z)

    assert torch.isfinite(loss)


def test_embedding_diagnostics_detect_rank_collapse() -> None:
    generator = torch.Generator().manual_seed(17)
    healthy = torch.randn(512, 16, generator=generator)
    collapsed = torch.ones(512, 16)

    healthy_stats = PairingDiagnostics._embedding_statistics(healthy)
    collapsed_stats = PairingDiagnostics._embedding_statistics(collapsed)

    assert healthy_stats["collapse_pass"]
    assert not collapsed_stats["collapse_pass"]
    assert "inactive_dimensions" in collapsed_stats["collapse_failures"]


def test_masked_smd_ignores_padding_values() -> None:
    x1 = torch.tensor([[1.0, 100.0], [3.0, 200.0], [5.0, 300.0]])
    x2 = torch.tensor([[1.0, -100.0], [3.0, -200.0], [5.0, -300.0]])
    mask = torch.tensor([[1, 0], [1, 0], [1, 0]], dtype=torch.bool)

    smd = PairingDiagnostics._masked_value_smd(x1, mask, x2, mask)

    assert smd[0] == pytest.approx(0.0)
    assert torch.isnan(smd[1])


def test_embedding_anomaly_knn_scores_separated_queries() -> None:
    callback = EmbeddingAnomalyMetrics(reference_size=8, max_query_events=8, k=2)
    reference = torch.tensor([[1.0, 0.0]]).repeat(8, 1)
    callback.reference = torch.nn.functional.normalize(reference, dim=1)

    normal_score = callback._score(torch.tensor([[1.0, 0.01], [1.0, -0.01]]))
    anomaly_score = callback._score(torch.tensor([[0.0, 1.0], [0.0, -1.0]]))

    assert anomaly_score.min() > normal_score.max()

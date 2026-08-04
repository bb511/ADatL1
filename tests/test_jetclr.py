from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from src.algorithms import jetclr as jetclr_module
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


class _CountingProjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.linear(x)


class _ZeroContrastiveLoss(nn.Module):
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return 0.0 * (z1.sum() + z2.sum())


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


def test_sum_pooling_is_permutation_invariant_and_has_expected_shape() -> None:
    torch.manual_seed(8)
    encoder = ObjectTransformerEncoder(
        object_types=["FET", "jets"],
        d_model=8,
        out_dim=6,
        n_heads=2,
        n_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        pooling="sum",
    ).eval()
    encoder.set_object_feature_map(_feature_map())
    x = torch.randn(3, 9)
    mask = torch.ones_like(x)
    permuted = x.clone()
    permuted[:, 3:6] = x[:, 6:9]
    permuted[:, 6:9] = x[:, 3:6]

    output = encoder(x, mask)

    assert output.shape == (3, 6)
    torch.testing.assert_close(output, encoder(permuted, mask))


@pytest.mark.parametrize("pooling", ["mean", "sum"])
def test_non_cls_pooling_excludes_masked_objects(pooling: str) -> None:
    torch.manual_seed(9)
    encoder = ObjectTransformerEncoder(
        object_types=["FET", "jets"],
        d_model=8,
        out_dim=8,
        n_heads=2,
        n_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        pooling=pooling,
    ).eval()
    encoder.set_object_feature_map(_feature_map())
    x = torch.randn(2, 9)
    mask = torch.ones_like(x)
    mask[:, 6:9] = 0
    changed_padding = x.clone()
    changed_padding[:, 6:9] = 1e6

    torch.testing.assert_close(encoder(x, mask), encoder(changed_padding, mask))


def test_encoder_fidelity_options_preserve_current_defaults() -> None:
    encoder = ObjectTransformerEncoder(d_model=8, out_dim=8, n_heads=2, n_layers=1)

    assert encoder.pooling == "cls"
    assert encoder.encoder.layers[0].norm_first is True
    assert isinstance(encoder.norm, nn.LayerNorm)
    assert encoder.cls_token.shape == (1, 1, 8)

    post_norm_encoder = ObjectTransformerEncoder(
        d_model=8,
        out_dim=8,
        n_heads=2,
        n_layers=1,
        pooling="sum",
        norm_first=False,
        post_pool_norm=False,
    )
    assert post_norm_encoder.encoder.layers[0].norm_first is False
    assert isinstance(post_norm_encoder.norm, nn.Identity)
    assert post_norm_encoder.cls_token.shape == (1, 1, 8)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"pooling": "max"}, ValueError),
        ({"norm_first": 1}, TypeError),
        ({"post_pool_norm": 0}, TypeError),
    ],
)
def test_encoder_rejects_invalid_fidelity_options(kwargs: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        ObjectTransformerEncoder(d_model=8, n_heads=2, **kwargs)


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


def test_jetclr_cosine_scheduler_runs_each_optimizer_step() -> None:
    config = OmegaConf.load(Path(__file__).parents[1] / "configs/algorithm/jetclr.yaml")

    assert config.scheduler.interval == "step"
    assert config.scheduler.frequency == 1
    assert config.encoder_variance_weight == 0.0
    assert config.encoder_covariance_weight == 0.0


def test_default_jetclr_loss_is_exact_ntxent_without_vicreg_work(monkeypatch) -> None:
    module = JetCLR(
        model=nn.Linear(4, 3, bias=False),
        projector=nn.Linear(3, 3, bias=False),
        loss=NTXentLoss(temperature=0.1, gather_distributed=False),
        optimizer=None,
        diagnosis_metrics=False,
    )

    def unexpected_vicreg(*args) -> None:
        raise AssertionError("default JetCLR must not compute VICReg statistics")

    monkeypatch.setattr(module, "_encoder_vicreg_terms", unexpected_vicreg)
    x = torch.randn(8, 4)
    outputs = module.model_step((x, torch.ones_like(x), torch.zeros(8), torch.zeros(8)))

    assert torch.equal(outputs["loss"].detach(), outputs["loss/ntxent"])
    assert outputs["loss/encoder_variance"] == 0.0
    assert outputs["loss/encoder_covariance"] == 0.0


def test_encoder_vicreg_penalizes_collapsed_representations_more() -> None:
    module = JetCLR(
        model=nn.Identity(),
        projector=nn.Identity(),
        loss=_ZeroContrastiveLoss(),
        optimizer=None,
        diagnosis_metrics=False,
        encoder_variance_weight=1.0,
    )
    generator = torch.Generator().manual_seed(31)
    healthy = torch.randn(4096, 8, generator=generator)
    collapsed = torch.zeros_like(healthy)

    healthy_variance, _ = module._encoder_vicreg_terms(healthy, healthy)
    collapsed_variance, _ = module._encoder_vicreg_terms(collapsed, collapsed)

    assert collapsed_variance > healthy_variance
    assert collapsed_variance > 0.9


def test_encoder_vicreg_covariance_is_normalized_by_dimension() -> None:
    correlated = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])

    penalty = JetCLR._covariance_penalty(correlated)

    # Unbiased covariance has two off-diagonal entries equal to two:
    # (2^2 + 2^2) / feature_dim = 4.
    assert penalty == pytest.approx(4.0)


def test_encoder_vicreg_gradients_reach_encoder() -> None:
    encoder = nn.Linear(4, 3, bias=False)
    module = JetCLR(
        model=encoder,
        projector=nn.Linear(3, 3, bias=False),
        loss=_ZeroContrastiveLoss(),
        optimizer=None,
        diagnosis_metrics=False,
        encoder_variance_weight=1.0,
        encoder_covariance_weight=0.1,
    )
    x = 0.1 * torch.randn(32, 4)

    outputs = module.model_step((x, torch.ones_like(x), torch.zeros(32), torch.zeros(32)))
    outputs["loss"].backward()

    assert encoder.weight.grad is not None
    assert encoder.weight.grad.abs().sum() > 0.0
    assert outputs["loss/encoder_covariance"] >= 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"encoder_variance_weight": -1.0},
        {"encoder_covariance_weight": -1.0},
        {"encoder_variance_weight": 0.0, "encoder_covariance_weight": 1.0},
    ],
)
def test_encoder_vicreg_rejects_invalid_weights(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        JetCLR(
            model=nn.Identity(),
            projector=nn.Identity(),
            loss=_ZeroContrastiveLoss(),
            optimizer=None,
            **kwargs,
        )


def test_encoder_vicreg_ddp_gather_is_differentiable(monkeypatch) -> None:
    monkeypatch.setattr(jetclr_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(jetclr_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(jetclr_module.dist, "get_world_size", lambda: 2)

    def equal_size_gather(outputs, value) -> None:
        for output in outputs:
            output.copy_(value)

    monkeypatch.setattr(jetclr_module.dist, "all_gather", equal_size_gather)
    monkeypatch.setattr(
        jetclr_module,
        "differentiable_all_gather",
        lambda value: (value, value + 1.0),
    )
    local = torch.randn(4, 3, requires_grad=True)

    gathered = JetCLR._gather_encoder_representation(local)
    gathered.sum().backward()

    assert gathered.shape == (8, 3)
    assert local.grad is not None


def test_encoder_vicreg_ddp_rejects_variable_batch_sizes(monkeypatch) -> None:
    monkeypatch.setattr(jetclr_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(jetclr_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(jetclr_module.dist, "get_world_size", lambda: 2)

    def variable_size_gather(outputs, value) -> None:
        outputs[0].fill_(value.item())
        outputs[1].fill_(value.item() + 1)

    monkeypatch.setattr(jetclr_module.dist, "all_gather", variable_size_gather)

    with pytest.raises(RuntimeError, match="equal per-rank batch sizes.*drop_last=True"):
        JetCLR._gather_encoder_representation(torch.randn(4, 3))


def test_clean_projector_output_is_evaluation_only() -> None:
    projector = _CountingProjector()
    module = JetCLR(
        model=nn.Linear(4, 3, bias=False),
        projector=projector,
        loss=NTXentLoss(temperature=0.1, gather_distributed=False),
        optimizer=None,
        diagnosis_metrics=False,
    )
    x = torch.randn(8, 4)
    batch = (x, torch.ones_like(x), torch.zeros(8), torch.zeros(8))

    module.train()
    train_outputs = module.model_step(batch)

    assert projector.calls == 2
    assert "jetclr_clean_proj_data" not in train_outputs

    module.eval()
    eval_outputs = module.model_step(batch)

    assert projector.calls == 5
    expected = projector.linear(module.model(x))
    torch.testing.assert_close(eval_outputs["jetclr_clean_proj_data"], expected)


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


def test_embedding_rank_gate_is_calibrated_independently_of_output_width() -> None:
    generator = torch.Generator().manual_seed(23)
    latent = torch.randn(1024, 8, generator=generator)
    embedding = latent @ torch.randn(8, 128, generator=generator)

    calibrated = PairingDiagnostics._embedding_statistics(embedding)
    width_fraction_gate = PairingDiagnostics._embedding_statistics(
        embedding,
        min_effective_rank=32.0,
        min_participation_rank=25.6,
    )

    assert calibrated["collapse_pass"]
    assert not width_fraction_gate["collapse_pass"]
    assert calibrated["collapse_min_effective_rank"] == 6.0


def test_pairing_diagnostics_records_zero_pairs_as_ineligible() -> None:
    callback = PairingDiagnostics(k=1, caliper_quantile=0.0)
    z1 = torch.eye(4)
    z2 = -torch.eye(4)
    raw = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    mask = torch.ones_like(raw, dtype=torch.bool)
    callback.reps = {"normal": [z1], "reference_normal": [z2]}
    callback.raw = {"normal": [raw], "reference_normal": [raw.clone()]}
    callback.raw_mask = {"normal": [mask], "reference_normal": [mask.clone()]}
    callback.closure_1 = [z1]
    callback.closure_2 = [z1.clone()]
    callback.projector_reps = [torch.randn(4, 8)]

    metrics = callback._compute_metrics()

    assert metrics["mnn_pairs"] == 0
    assert metrics["mnn_coverage"] == 0.0
    assert metrics["raw_selection_score"] == 0.0
    assert metrics["selection_score"] == 0.0
    assert metrics["collapse_pass"] is False
    assert "no_mutual_nearest_pairs" in metrics["collapse_failures"]
    assert metrics["pair_distance_mean"] is None
    assert "projector_embedding_effective_rank" in metrics
    assert "projector_collapse_pass" in metrics
    assert metrics["projector_collapse_min_effective_rank"] == 6.0
    json.dumps(metrics, allow_nan=False)


def test_pairing_diagnostics_fails_clearly_when_projector_output_is_missing() -> None:
    callback = PairingDiagnostics(max_events_per_dataset=4)
    callback.on_test_epoch_start(None, None)
    trainer = SimpleNamespace(test_dataloaders={"normal": object()})
    batch = (
        torch.randn(4, 3),
        torch.ones(4, 3),
        torch.zeros(4),
        torch.zeros(4),
    )
    outputs = {
        "pairing_rep_data": torch.randn(4, 8),
        "pairing_view1_data": torch.randn(4, 8),
        "pairing_view2_data": torch.randn(4, 8),
    }

    with pytest.raises(KeyError, match="jetclr_clean_proj_data.*missing"):
        callback.on_test_batch_end(trainer, None, outputs, batch, 0)


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


def test_embedding_anomaly_knn_promotes_mixed_precision_reference() -> None:
    callback = EmbeddingAnomalyMetrics(reference_size=2, max_query_events=2, k=1)
    callback.reference = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16), dim=1
    )

    scores = callback._score(torch.tensor([[1.0, 0.0]], dtype=torch.float32))

    assert scores.dtype == torch.float32
    assert scores.item() == pytest.approx(0.0)

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.data.utils import unpack_batch
from src.evaluation.leakage_probe import (
    ProbeExtractionError,
    ProbeRepresentationSet,
    extract_probe_split,
)


class FakeNormalizer:
    scale = 10.0
    shift = 100.0


def make_batch(
    offset: float = 0.0,
    *,
    constant_target: bool = False,
) -> tuple[torch.Tensor, ...]:
    x = torch.tensor(
        [
            [1.0 + offset, -1.0 - offset],
            [2.0 + offset, -2.0 - offset],
        ],
        dtype=torch.float32,
    )
    mask = torch.ones_like(x, dtype=torch.bool)
    l1bit = torch.zeros(x.shape[0], dtype=torch.bool)
    labels = torch.zeros(x.shape[0], dtype=torch.float32)

    if constant_target:
        target_values = torch.zeros(x.shape[0], dtype=torch.float32)
    else:
        target_values = torch.tensor(
            [offset, offset + 1.0],
            dtype=torch.float32,
        )

    control_x = torch.stack(
        [
            target_values,
            torch.full_like(target_values, 999.0),
        ],
        dim=1,
    )
    control_mask = torch.ones_like(control_x, dtype=torch.bool)

    return (
        x,
        mask,
        l1bit,
        labels,
        control_x,
        control_mask,
    )


class FakeProbeDataModule:
    def __init__(
        self,
        batches,
        *,
        sensitive_in_model_input: bool = False,
    ) -> None:
        self._batches = list(batches)
        self.normalizer = FakeNormalizer()

        if sensitive_in_model_input:
            self.object_feature_map = {
                "FET": {
                    "Et": [0],
                },
                "Jet": {
                    "pt": [1],
                },
            }
        else:
            self.object_feature_map = {
                "Jet": {
                    "pt": [0, 1],
                }
            }

        self.control_object_feature_map = {
            "FET": {
                "Et": [0],
            },
            "Jet": {
                "pt": [1],
            },
        }

        self.active_split = None
        self.setup_calls = []
        self.release_calls = 0

    def setup_probe_split(self, split: str) -> None:
        if self.active_split is not None:
            raise RuntimeError("A probe split is already active.")

        self.active_split = split
        self.setup_calls.append(split)

    def probe_dataloader(self):
        if self.active_split is None:
            raise RuntimeError("No probe split is active.")

        return self._batches

    def release_probe_split(self) -> None:
        self.active_split = None
        self.release_calls += 1


class RecordingProbeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.inference_mode_flags = []
        self.normalizers = []
        self.denormalization_flags = []

        self.object_feature_map = None
        self.control_object_feature_map = None

    def _assert_sensitive_not_in_model_input(self) -> None:
        if self.object_feature_map is None:
            raise RuntimeError("object_feature_map is missing.")

        fet_features = self.object_feature_map.get("FET", {})

        if "Et" in fet_features:
            raise RuntimeError(
                "Sensitive variable FET.Et is present in model input."
            )

    def forward_with_representations(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.inference_mode_flags.append(
            torch.is_inference_mode_enabled()
        )

        latent_logits = x * self.scale
        latent_sample = (latent_logits >= 0).to(dtype=x.dtype)
        reconstructed_data = x + 0.5

        return {
            "latent_logits": latent_logits,
            "latent_sample": latent_sample,
            "reconstructed_data": reconstructed_data,
        }

    def extract_sensitive_values(
        self,
        batch,
        *,
        use_denormalized: bool | None = None,
        normalizer=None,
    ) -> torch.Tensor:
        self.normalizers.append(normalizer)
        self.denormalization_flags.append(use_denormalized)

        batch_view = unpack_batch(batch)
        target = batch_view.control_x[:, 0]

        if use_denormalized:
            target = target * normalizer.scale + normalizer.shift

        return target


class NonDeterministicSampleModel(RecordingProbeModel):
    def __init__(self) -> None:
        super().__init__()
        self.hard_sample_calls = 0

    def forward_with_representations(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        result = super().forward_with_representations(x)

        self.hard_sample_calls += 1
        fill_value = float(self.hard_sample_calls % 2)
        result["latent_sample"] = torch.full_like(
            result["latent_sample"],
            fill_value,
        )

        return result


class NonBinarySampleModel(RecordingProbeModel):
    def forward_with_representations(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        result = super().forward_with_representations(x)
        result["latent_sample"] = torch.full_like(
            result["latent_sample"],
            0.5,
        )
        return result


class NonFiniteRepresentationModel(RecordingProbeModel):
    def forward_with_representations(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        result = super().forward_with_representations(x)
        result["latent_logits"] = result["latent_logits"].clone()
        result["latent_logits"][0, 0] = torch.inf
        return result


class ShortTargetModel(RecordingProbeModel):
    def extract_sensitive_values(
        self,
        batch,
        *,
        use_denormalized: bool | None = None,
        normalizer=None,
    ) -> torch.Tensor:
        target = super().extract_sensitive_values(
            batch,
            use_denormalized=use_denormalized,
            normalizer=normalizer,
        )
        return target[:-1]


class WrongReconstructionWidthModel(RecordingProbeModel):
    def forward_with_representations(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        result = super().forward_with_representations(x)
        result["reconstructed_data"] = result[
            "reconstructed_data"
        ][:, :1]
        return result


def test_extract_probe_split_collects_contract_arrays() -> None:
    batches = [
        make_batch(offset=0.0),
        make_batch(offset=2.0),
    ]
    datamodule = FakeProbeDataModule(batches)
    model = RecordingProbeModel()
    model.train()

    result = extract_probe_split(
        model,
        datamodule,
        "train",
        device="cpu",
    )

    assert isinstance(result, ProbeRepresentationSet)
    assert result.split == "train"
    assert result.n_events == 4
    assert result.sample_seed == 12345
    assert result.max_samples is None
    assert len(result.manifest_hash) == 64

    expected_x = torch.cat(
        [batch[0] for batch in batches],
        dim=0,
    ).numpy()

    np.testing.assert_allclose(
        result.latent_logits,
        expected_x * 2.0,
    )
    np.testing.assert_allclose(
        result.latent_sample,
        (expected_x >= 0).astype(np.float32),
    )
    np.testing.assert_allclose(
        result.reconstructed_data,
        expected_x + 0.5,
    )
    np.testing.assert_allclose(
        result.sensitive_target,
        np.array(
            [100.0, 110.0, 120.0, 130.0],
            dtype=np.float32,
        ),
    )

    assert not model.training
    assert all(
        not parameter.requires_grad
        for parameter in model.parameters()
    )
    assert model.inference_mode_flags
    assert all(model.inference_mode_flags)
    assert model.denormalization_flags == [True, True]
    assert all(
        normalizer is datamodule.normalizer
        for normalizer in model.normalizers
    )

    assert datamodule.setup_calls == ["train"]
    assert datamodule.release_calls == 1
    assert datamodule.active_split is None


def test_manifest_hash_is_reproducible_for_same_event_positions() -> None:
    first = extract_probe_split(
        RecordingProbeModel(),
        FakeProbeDataModule(
            [
                make_batch(offset=0.0),
                make_batch(offset=2.0),
            ]
        ),
        "valid",
    )
    second = extract_probe_split(
        RecordingProbeModel(),
        FakeProbeDataModule(
            [
                make_batch(offset=0.0),
                make_batch(offset=2.0),
            ]
        ),
        "valid",
    )

    assert first.n_events == second.n_events
    assert first.manifest_hash == second.manifest_hash


@pytest.mark.parametrize(
    ("model", "expected_reason"),
    [
        (
            NonDeterministicSampleModel(),
            "latent_sample_not_deterministic",
        ),
        (
            NonBinarySampleModel(),
            "latent_sample_not_binary",
        ),
        (
            NonFiniteRepresentationModel(),
            "non_finite_latent_logits",
        ),
        (
            ShortTargetModel(),
            "representation_target_row_mismatch",
        ),
        (
            WrongReconstructionWidthModel(),
            "reconstruction_shape_mismatch",
        ),
    ],
)
def test_invalid_extractions_fail_and_release_split(
    model: nn.Module,
    expected_reason: str,
) -> None:
    datamodule = FakeProbeDataModule(
        [make_batch(offset=0.0)]
    )

    with pytest.raises(ProbeExtractionError) as error:
        extract_probe_split(
            model,
            datamodule,
            "train",
        )

    assert error.value.reason == expected_reason
    assert datamodule.release_calls == 1
    assert datamodule.active_split is None


def test_empty_probe_split_is_rejected_and_released() -> None:
    datamodule = FakeProbeDataModule([])

    with pytest.raises(ProbeExtractionError) as error:
        extract_probe_split(
            RecordingProbeModel(),
            datamodule,
            "valid",
        )

    assert error.value.reason == "empty_split"
    assert datamodule.release_calls == 1
    assert datamodule.active_split is None


def test_constant_target_is_rejected_and_released() -> None:
    datamodule = FakeProbeDataModule(
        [
            make_batch(
                offset=0.0,
                constant_target=True,
            )
        ]
    )

    with pytest.raises(ProbeExtractionError) as error:
        extract_probe_split(
            RecordingProbeModel(),
            datamodule,
            "train",
        )

    assert error.value.reason == "constant_target"
    assert datamodule.release_calls == 1
    assert datamodule.active_split is None


def test_sensitive_feature_in_model_input_is_rejected() -> None:
    datamodule = FakeProbeDataModule(
        [make_batch(offset=0.0)],
        sensitive_in_model_input=True,
    )

    with pytest.raises(ProbeExtractionError) as error:
        extract_probe_split(
            RecordingProbeModel(),
            datamodule,
            "train",
        )

    assert error.value.reason == "sensitive_feature_in_input"
    assert datamodule.release_calls == 1
    assert datamodule.active_split is None
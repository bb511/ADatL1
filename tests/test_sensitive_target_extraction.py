import pytest
import torch

from src.data.components.normalization import L1DataNormalizer
from src.data.sensitive_binning import FixedQuantileSensitiveBinner

def make_feature_map() -> dict:
    return {
        "FET": {
            "Et": [0],
        }
    }


def make_normalizer() -> L1DataNormalizer:
    normalizer = L1DataNormalizer(
        name="robust",
        hyperparams={"percentiles": [0.05, 0.95]},
    )
    normalizer.norm_params = {
        "FET": {
            "Et": {
                "shift": 100.0,
                "scale": 20.0,
            }
        }
    }
    return normalizer

def test_sensitive_extraction_preserves_configured_normalized_behavior() -> None:
    binner = FixedQuantileSensitiveBinner(
        variable="FET.Et",
        num_bins=2,
        use_denormalized=False,
    )

    x = torch.tensor([[-1.0], [0.0], [2.0]])

    values = binner.extract_values(
        x=x,
        object_feature_map=make_feature_map(),
        normalizer=make_normalizer(),
    )

    torch.testing.assert_close(
        values,
        torch.tensor([-1.0, 0.0, 2.0]),
    )

def test_sensitive_extraction_can_request_physical_values() -> None:
    binner = FixedQuantileSensitiveBinner(
        variable="FET.Et",
        num_bins=2,
        use_denormalized=False,
    )

    x = torch.tensor([[-1.0], [0.0], [2.0]])
    original_x = x.clone()

    values = binner.extract_values(
        x=x,
        object_feature_map=make_feature_map(),
        normalizer=make_normalizer(),
        use_denormalized=True,
    )

    torch.testing.assert_close(
        values,
        torch.tensor([80.0, 100.0, 140.0]),
    )

    # Denormalization must not mutate the batch tensor.
    torch.testing.assert_close(x, original_x)

def test_sensitive_extraction_can_explicitly_request_normalized_values() -> None:
    binner = FixedQuantileSensitiveBinner(
        variable="FET.Et",
        num_bins=2,
        use_denormalized=True,
    )

    x = torch.tensor([[-1.0], [0.0], [2.0]])

    values = binner.extract_values(
        x=x,
        object_feature_map=make_feature_map(),
        normalizer=make_normalizer(),
        use_denormalized=False,
    )

    torch.testing.assert_close(
        values,
        torch.tensor([-1.0, 0.0, 2.0]),
    )

def test_requested_denormalization_requires_normalizer() -> None:
    binner = FixedQuantileSensitiveBinner(
        variable="FET.Et",
        num_bins=2,
        use_denormalized=False,
    )

    with pytest.raises(RuntimeError, match="normalizer is missing"):
        binner.extract_values(
            x=torch.tensor([[0.0]]),
            object_feature_map=make_feature_map(),
            normalizer=None,
            use_denormalized=True,
        )

def test_denormalized_sensitive_extraction_returns_one_value_per_event() -> None:
    binner = FixedQuantileSensitiveBinner(
        variable="FET.Et",
        num_bins=2,
        reduction="first",
        use_denormalized=False,
    )

    x = torch.tensor([[-1.0], [0.0], [2.0]])
    mask = torch.ones_like(x, dtype=torch.bool)

    values = binner.extract_values(
        x=x,
        mask=mask,
        object_feature_map=make_feature_map(),
        normalizer=make_normalizer(),
        use_denormalized=True,
    )

    assert values.shape == (3,)
    torch.testing.assert_close(
        values,
        torch.tensor([80.0, 100.0, 140.0]),
    )


from types import SimpleNamespace
from unittest.mock import Mock

from torch import nn

from src.algorithms.ae import AE


def test_ae_forwards_denormalization_override() -> None:
    model = AE(
        encoder=nn.Identity(),
        decoder=nn.Identity(),
        mi_sensitive_num_bins=2,
    )

    feature_map = make_feature_map()
    normalizer = make_normalizer()
    expected = torch.tensor([80.0, 100.0])

    extract_values = Mock(return_value=expected)
    model.sensitive_binner.extract_values = extract_values
    model.control_object_feature_map = feature_map
    model.object_feature_map = {"input": {"feature": [0]}}
    model._trainer = SimpleNamespace(
        datamodule=SimpleNamespace(normalizer=normalizer)
    )

    x = torch.zeros((2, 1))
    mask = torch.ones_like(x, dtype=torch.bool)
    l1bit = torch.zeros(2, dtype=torch.bool)
    labels = torch.zeros(2)
    control_x = torch.tensor([[-1.0], [0.0]])
    control_mask = torch.ones_like(control_x, dtype=torch.bool)

    batch = (
        x,
        mask,
        l1bit,
        labels,
        control_x,
        control_mask,
    )

    result = model.extract_sensitive_values(
        batch,
        use_denormalized=True,
    )

    torch.testing.assert_close(result, expected)

    extract_values.assert_called_once_with(
        x=control_x,
        mask=control_mask,
        object_feature_map=feature_map,
        normalizer=normalizer,
        use_denormalized=True,
    )

def test_ae_accepts_explicit_normalizer_without_trainer() -> None:
    model = AE(
        encoder=nn.Identity(),
        decoder=nn.Identity(),
        mi_sensitive_num_bins=2,
    )

    model.object_feature_map = {
        "input": {
            "feature": [0],
        }
    }
    model.control_object_feature_map = make_feature_map()
    model._trainer = None

    x = torch.zeros((2, 1))
    mask = torch.ones_like(x, dtype=torch.bool)
    l1bit = torch.zeros(2, dtype=torch.bool)
    labels = torch.zeros(2)

    control_x = torch.tensor(
        [
            [-1.0],
            [2.0],
        ]
    )
    control_mask = torch.ones_like(
        control_x,
        dtype=torch.bool,
    )

    batch = (
        x,
        mask,
        l1bit,
        labels,
        control_x,
        control_mask,
    )

    values = model.extract_sensitive_values(
        batch,
        use_denormalized=True,
        normalizer=make_normalizer(),
    )

    torch.testing.assert_close(
        values,
        torch.tensor([80.0, 140.0]),
    )
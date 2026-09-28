"""All FET.* features are sensitive: excluded from the AE input, guarded, and
kept out of the mean correlation C (which stays defined on FET.Et only)."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.algorithms.ae import AE
from src.data.L1AD_datamodule import L1ADDataModule
from src.data.feature_refs import label_matches_any, resolve_feature_refs
from src.evaluation.callbacks.correlation_matrix import CorrelationMatrixCallback

# Same layout as the mlready object_feature_map.json: FET first, features sorted.
RAW_MAP = {
    "FET": {"Et": [0], "eta": [1], "phi": [2]},
    "egammas": {"Et": [3, 6], "eta": [4, 7], "phi": [5, 8]},
    "jets": {"Et": [9], "eta": [10], "phi": [11]},
}


def test_wildcard_resolves_all_object_features_case_insensitively() -> None:
    resolved = resolve_feature_refs(RAW_MAP, ["fet.*"])
    assert [(obj, feat) for obj, feat, _ in resolved] == [
        ("FET", "Et"),
        ("FET", "eta"),
        ("FET", "phi"),
    ]
    assert label_matches_any("FET.phi", ["FET.*"])
    assert not label_matches_any("jets.Et", ["FET.*"])


def test_unmatched_reference_raises_only_when_strict() -> None:
    with pytest.raises(KeyError):
        resolve_feature_refs(RAW_MAP, ["MET.*"])
    assert resolve_feature_refs(RAW_MAP, ["MET.*"], strict=False) == []
    with pytest.raises(ValueError):
        resolve_feature_refs(RAW_MAP, ["FET"])


def _datamodule(excluded: list[str]) -> L1ADDataModule:
    dm = object.__new__(L1ADDataModule)
    dm.model_input_exclude_features = excluded
    dm.control_object_feature_map = None
    dm.object_feature_map = None
    dm._model_excluded_indices = set()
    dm._model_keep_indices = None
    dm._configure_feature_views(RAW_MAP)
    return dm


def test_datamodule_removes_every_fet_feature_and_keeps_contiguous_view() -> None:
    dm = _datamodule(["FET.*"])

    assert dm._model_excluded_indices == {0, 1, 2}
    assert dm._model_keep_indices == list(range(3, 12))
    assert dm._keep_indices_are_contiguous_run()
    assert "FET" not in dm.object_feature_map
    assert dm.object_feature_map["egammas"]["Et"] == [0, 3]
    assert dm.control_object_feature_map["FET"] == RAW_MAP["FET"]


def _guard_module(object_feature_map: dict, sensitive: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        forbid_sensitive_variable_in_input=True,
        object_feature_map=object_feature_map,
        sensitive_input_features=sensitive,
    )


def test_guard_rejects_any_leaked_fet_feature() -> None:
    leaky = _datamodule(["FET.Et"]).object_feature_map  # FET.eta/phi remain
    with pytest.raises(RuntimeError, match="FET.phi"):
        AE._assert_sensitive_not_in_model_input(
            _guard_module(leaky, ["FET.Et", "FET.*"])
        )

    clean = _datamodule(["FET.*"]).object_feature_map
    AE._assert_sensitive_not_in_model_input(_guard_module(clean, ["FET.Et", "FET.*"]))


def test_mean_correlation_excludes_sensitive_group() -> None:
    corr = pd.DataFrame(
        [[1.0, 0.9, 0.5, 0.1], [0.9, 1.0, 0.0, 0.0], [0.5, 0.0, 1.0, 0.0], [0.1, 0.0, 0.0, 1.0]],
        index=["FET.Et", "FET.phi", "jets.Et", "muons.Et"],
        columns=["FET.Et", "FET.phi", "jets.Et", "muons.Et"],
    )
    variables = list(corr.columns)

    grouped = CorrelationMatrixCallback(
        variables=variables, sensitive_variable="FET.Et", sensitive_group=["FET.*"]
    )
    mean, n_other = grouped._mean_sensitive_variable_correlation(corr)
    assert n_other == 2
    assert mean == pytest.approx(0.3)

    ungrouped = CorrelationMatrixCallback(variables=variables, sensitive_variable="FET.Et")
    mean, n_other = ungrouped._mean_sensitive_variable_correlation(corr)
    assert n_other == 3
    assert mean == pytest.approx(0.5)

"""Focused tests for the curated Causal Chamber report layer."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import cchamber_final_report


def test_presentation_model_order_excludes_ae() -> None:
    """The first two presentation figures use the frozen requested model order."""
    assert cchamber_final_report.PRESENTATION_MODELS == ("svdd", "vae", "realnvp")
    assert "ae" not in cchamber_final_report.PRESENTATION_MODELS
    assert "ae" in cchamber_final_report.MODELS


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a monotonic three-strength synthetic report bundle."""
    rows = []
    for seed in (1001, 1002):
        for index, intervention in enumerate(("weak", "mid", "strong"), start=1):
            rows.extend(
                [
                    {
                        "model": "ae",
                        "strategy": "cap_metadata_nearest",
                        "metric": "auprc",
                        "seed": seed,
                        "intervention": intervention,
                        "value": 0.1 * index + 0.001 * (seed - 1001),
                    },
                    {
                        "model": "ae",
                        "strategy": "cap_random",
                        "metric": "auprc",
                        "seed": seed,
                        "intervention": intervention,
                        "value": 0.05 * index + 0.001 * (seed - 1001),
                    },
                ]
            )
    physical = pd.DataFrame(
        {
            "intervention": ["weak", "mid", "strong"],
            "target": ["red", "red", "red"],
            "strength": ["weak", "mid", "strong"],
            "semantic_family": ["color_led"] * 3,
            "system_group": ["process"] * 3,
            "biased_energy_distance": [0.1, 0.2, 0.3],
        }
    )
    official = pd.DataFrame(
        {
            "model": ["ae"] * 3,
            "metric": ["auprc"] * 3,
            "intervention": ["weak", "mid", "strong"],
            "contrast_id": ["metadata_vs_random"] * 3,
            "strategy_left": ["cap_metadata_nearest"] * 3,
            "strategy_right": ["cap_random"] * 3,
            "mean_difference": [0.05, 0.10, 0.15],
        }
    )
    return pd.DataFrame(rows), physical, official


def test_physical_associations_are_seed_first_and_stratified() -> None:
    """Associations average seeds first and retain physical strata."""
    results, physical, official = _inputs()
    intervention, family, target, gain_family = cchamber_final_report.physical_associations(
        results, physical, official
    )

    weak = intervention[
        (intervention["strategy"] == "cap_metadata_nearest")
        & (intervention["intervention"] == "weak")
    ]["mean_performance"].item()
    assert weak == pytest.approx(0.1005)
    assert set(family["semantic_family"]) == {"color_led"}
    assert set(target["target"]) == {"red"}
    assert family["spearman_rho"].dropna().eq(1.0).all()
    assert target["spearman_rho"].dropna().eq(1.0).all()
    assert gain_family["spearman_rho"].item() == pytest.approx(1.0)
    assert gain_family["n_interventions"].item() == 3


def test_physical_associations_reject_disagreement_with_frozen_analysis() -> None:
    """Recomputed gains must agree with the frozen confirmatory table."""
    results, physical, official = _inputs()
    official.loc[0, "mean_difference"] = 999.0

    with pytest.raises(ValueError, match="disagree"):
        cchamber_final_report.physical_associations(results, physical, official)

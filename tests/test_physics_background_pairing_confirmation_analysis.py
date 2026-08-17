"""Focused contracts for paired physics confirmation endpoint analysis."""

from __future__ import annotations

import copy

import pytest

from src.evaluation.physics_background_pairing_analysis import DOWNSTREAM_DATASETS
from src.evaluation.physics_background_pairing_confirmation_analysis import (
    _content_hash,
    build_confirmation_endpoint_analysis,
)

MODELS = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
SELECTORS = (
    ("cap", "flat_physical"),
    ("cap", "physics_summary"),
    ("cap", "typed_sliced_wasserstein"),
    ("cap", "jetclr"),
    ("cap", "cdf"),
    ("wasserstein", None),
    ("drift", None),
)
SEEDS = (271828, 314159, 161803)


def _fixture() -> tuple[dict, list[dict]]:
    """Build a complete 42-cell, three-seed, two-endpoint synthetic matrix."""
    cells, rows = [], []
    for model_index, model in enumerate(MODELS):
        for selector_index, (selector, pairing) in enumerate(SELECTORS):
            cell_index = len(cells)
            objective = f"native__cap__{pairing}" if selector == "cap" else f"native__{selector}"
            cell = {
                "cell_index": cell_index,
                "configuration_index": cell_index % 28,
                "model": model,
                "score": "native",
                "selector": selector,
                "pairing": pairing,
                "logical_objective": objective,
                "trial_number": 1000 + cell_index,
            }
            cells.append(cell)
            selector_effect = {
                ("cap", "flat_physical"): 0.03,
                ("cap", "physics_summary"): 0.02,
                ("cap", "typed_sliced_wasserstein"): 0.01,
                ("cap", "jetclr"): -0.01,
                ("cap", "cdf"): 0.00,
                ("wasserstein", None): -0.02,
                ("drift", None): -0.04,
            }[(selector, pairing)]
            for seed_index, seed in enumerate(SEEDS):
                checkpoint = f"{cell_index:02d}-{seed}.ckpt"
                digest = f"{cell_index * 10 + seed_index:064x}"
                for anomaly_index, anomaly in enumerate(DOWNSTREAM_DATASETS):
                    for endpoint in ("efficiency_250hz", "auprc"):
                        base = 0.20 if endpoint == "efficiency_250hz" else 0.50
                        value = (
                            base
                            + selector_effect
                            + 0.001 * model_index
                            + 0.0001 * anomaly_index
                            + 0.00001 * seed_index
                        )
                        rows.append(
                            {
                                **cell,
                                "seed": seed,
                                "pairing": pairing or "not_applicable",
                                "checkpoint_path": checkpoint,
                                "checkpoint_sha256": digest,
                                "anomaly": anomaly,
                                "endpoint": endpoint,
                                "value": value,
                            }
                        )
    matrix = {
        "schema_version": 1,
        "kind": "physics_background_native_confirmation_matrix",
        "training_seeds": list(SEEDS),
        "logical_cells": cells,
        "configurations": [{"configuration_index": index} for index in range(28)],
    }
    matrix["content_sha256"] = _content_hash(matrix)
    return matrix, rows


def test_complete_confirmation_matrix_produces_exact_paired_tables() -> None:
    """The complete matrix expands to every predeclared table cardinality."""
    result = build_confirmation_endpoint_analysis(
        *_fixture(), bootstrap_replicates=1_000, bootstrap_seed=7
    )
    method = result["method"]
    assert method["endpoint_rows"] == 42 * 3 * 20 * 2 == 5040
    assert method["endpoint_summaries"] == 42 * 3 * 2 == 252
    assert method["controlled_paired_rows"] == 6 * 2 * 3 * 20 * 2 == 1440
    assert method["pairing_paired_rows"] == 6 * 4 * 3 * 20 * 2 == 2880
    assert method["comparison_summaries"] == 72 + 12 == 84
    assert len(result["endpoints_summary"]) == 252
    assert {row["contrast"] for row in result["selector_comparison_paired"]} == {
        "cdf_cap_minus_wasserstein",
        "cdf_cap_minus_drift",
    }


def test_deltas_use_same_seed_anomaly_and_have_expected_direction() -> None:
    """Controlled and pairing deltas preserve paired identities and sign."""
    result = build_confirmation_endpoint_analysis(
        *_fixture(), bootstrap_replicates=1_000, bootstrap_seed=11
    )
    controlled = result["selector_comparison_paired"]
    assert {round(row["delta"], 8) for row in controlled if "wasserstein" in row["contrast"]} == {
        0.02
    }
    assert {round(row["delta"], 8) for row in controlled if "drift" in row["contrast"]} == {0.04}
    flat = [row for row in result["pairing_vs_cdf_paired"] if row["pairing"] == "flat_physical"]
    assert {round(row["delta"], 8) for row in flat} == {0.03}
    summary = next(
        row
        for row in result["comparison_summary"]
        if row["scope"] == "model"
        and row["model"] == "ae"
        and row["endpoint"] == "efficiency_250hz"
        and row["contrast"] == "flat_physical_minus_cdf"
    )
    assert summary["mean_delta"] == pytest.approx(0.03)
    assert summary["classification"] == "meaningful_improvement"
    assert summary["seed_positive"] == 3 and summary["anomaly_positive"] == 20


def test_bootstrap_is_deterministic_and_macro_is_equal_model() -> None:
    """Repeated analysis has identical intervals and fixed-model macro effects."""
    fixture = _fixture()
    first = build_confirmation_endpoint_analysis(
        *fixture, bootstrap_replicates=1_000, bootstrap_seed=23
    )
    second = build_confirmation_endpoint_analysis(
        *fixture, bootstrap_replicates=1_000, bootstrap_seed=23
    )
    assert first["comparison_summary"] == second["comparison_summary"]
    macro = next(
        row
        for row in first["comparison_summary"]
        if row["scope"] == "macro"
        and row["endpoint"] == "auprc"
        and row["contrast"] == "cdf_cap_minus_wasserstein"
    )
    assert macro["mean_delta"] == pytest.approx(0.02)


def test_incomplete_or_cross_endpoint_checkpoint_mismatch_fails_closed() -> None:
    """Missing observations and endpoints attached to different checkpoints are fatal."""
    matrix, rows = _fixture()
    with pytest.raises(ValueError, match="coverage incomplete"):
        build_confirmation_endpoint_analysis(matrix, rows[:-1], bootstrap_replicates=1_000)
    changed = copy.deepcopy(rows)
    changed[1]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="same selected checkpoint"):
        build_confirmation_endpoint_analysis(matrix, changed, bootstrap_replicates=1_000)


def test_matrix_hash_and_row_cell_identity_are_authenticated() -> None:
    """A stale matrix or endpoint-to-cell reassignment cannot enter analysis."""
    matrix, rows = _fixture()
    stale = copy.deepcopy(matrix)
    stale["training_seeds"][0] += 1
    with pytest.raises(ValueError, match="content hash differs"):
        build_confirmation_endpoint_analysis(stale, rows, bootstrap_replicates=1_000)
    changed = copy.deepcopy(rows)
    changed[0]["trial_number"] += 1
    with pytest.raises(ValueError, match="trial_number differs"):
        build_confirmation_endpoint_analysis(matrix, changed, bootstrap_replicates=1_000)

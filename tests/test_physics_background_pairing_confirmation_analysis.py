"""Focused contracts for paired physics confirmation endpoint analysis."""

from __future__ import annotations

import copy
import json

import pytest

from src.evaluation.physics_background_pairing_analysis import DOWNSTREAM_DATASETS
from src.evaluation.physics_background_pairing_confirmation_analysis import (
    _content_hash,
    _write_json,
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
RESERVE_SEEDS = (141421, 173205)


def test_analysis_receipt_is_atomic_idempotent_and_immutable(tmp_path) -> None:
    """Analysis receipts can be replayed exactly but cannot be silently replaced."""
    path = tmp_path / "confirmation_endpoint_analysis.json"
    payload = {"schema_version": 1, "value": "sealed"}
    _write_json(path, payload)
    _write_json(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="different analysis receipt"):
        _write_json(path, {**payload, "value": "changed"})


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
    assert method["across_seed_summaries"] == 42 * 2 == 84
    assert method["controlled_paired_rows"] == 6 * 2 * 3 * 20 * 2 == 1440
    assert method["pairing_paired_rows"] == 6 * 4 * 3 * 20 * 2 == 2880
    assert method["comparison_summaries"] == 72 + 12 == 84
    assert len(result["endpoints_summary"]) == 252
    assert len(result["endpoints_across_seeds"]) == 84
    assert len(result["reserve_seed_decisions"]) == 6
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


def test_across_seed_robustness_summarizes_anomaly_means() -> None:
    """Worst/CVaR average seeds first instead of flattening seed observations."""
    matrix, rows = _fixture()
    target = [
        row for row in rows if row["cell_index"] == 0 and row["endpoint"] == "efficiency_250hz"
    ]
    for row in target:
        if row["anomaly"] == DOWNSTREAM_DATASETS[0]:
            row["value"] = 0.0 if row["seed"] == SEEDS[0] else 1.0
    anomaly_means = sorted(
        sum(row["value"] for row in target if row["anomaly"] == anomaly) / len(SEEDS)
        for anomaly in DOWNSTREAM_DATASETS
    )
    result = build_confirmation_endpoint_analysis(
        matrix, rows, bootstrap_replicates=1_000, bootstrap_seed=27
    )
    summary = next(
        row
        for row in result["endpoints_across_seeds"]
        if row["cell_index"] == 0 and row["endpoint"] == "efficiency_250hz"
    )
    assert summary["worst"] == pytest.approx(anomaly_means[0])
    assert summary["bottom_quartile_cvar"] == pytest.approx(sum(anomaly_means[:5]) / 5)
    assert summary["median"] == pytest.approx((anomaly_means[9] + anomaly_means[10]) / 2)


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


def test_oas_diagnostics_use_the_same_ae_vae_checkpoints() -> None:
    """Masked OAS is paired with native endpoints without selecting new checkpoints."""
    matrix, native = _fixture()
    oas = []
    for row in native:
        if row["model"] not in {"ae", "vae"}:
            continue
        oas.append({**row, "score": "residual_oas", "value": row["value"] + 0.01})
    result = build_confirmation_endpoint_analysis(
        matrix,
        native,
        oas_endpoint_rows=oas,
        bootstrap_replicates=1_000,
        bootstrap_seed=29,
    )
    assert result["method"]["oas_endpoint_rows"] == 2 * 7 * 3 * 20 * 2 == 1680
    assert result["method"]["oas_paired_rows"] == 1680
    assert result["method"]["oas_comparison_summaries"] == 2 * 7 * 2 == 28
    assert {round(row["delta"], 8) for row in result["oas_vs_native_paired"]} == {0.01}
    changed = copy.deepcopy(oas)
    changed[0]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="same selected checkpoint"):
        build_confirmation_endpoint_analysis(
            matrix,
            native,
            oas_endpoint_rows=changed,
            bootstrap_replicates=1_000,
        )


def test_only_dsvae_extended_uses_mixed_seed_counts_and_primary_trigger() -> None:
    """A DSVAE-only extension has exact dynamic counts and three-seed macro inference."""
    matrix, rows = _fixture()
    dsvae_cells = {row["cell_index"] for row in matrix["logical_cells"] if row["model"] == "dsvae"}
    for row in rows:
        if (
            row["cell_index"] not in dsvae_cells
            or row["selector"] == "cap"
            and row["pairing"] == "cdf"
        ):
            continue
        row["value"] += {SEEDS[0]: -0.03, SEEDS[1]: 0.0, SEEDS[2]: 0.03}[row["seed"]]
    templates = {
        (row["cell_index"], row["anomaly"], row["endpoint"]): row
        for row in rows
        if row["cell_index"] in dsvae_cells and row["seed"] == SEEDS[1]
    }
    for seed_index, seed in enumerate(RESERVE_SEEDS):
        for template in templates.values():
            rows.append(
                {
                    **template,
                    "seed": seed,
                    "checkpoint_path": f"{template['cell_index']:02d}-{seed}.ckpt",
                    "checkpoint_sha256": f"{9000 + template['cell_index'] * 10 + seed_index:064x}",
                }
            )
    seeds_by_model = {
        model: SEEDS + RESERVE_SEEDS if model == "dsvae" else SEEDS for model in MODELS
    }
    result = build_confirmation_endpoint_analysis(
        matrix,
        rows,
        training_seeds_by_model=seeds_by_model,
        bootstrap_replicates=1_000,
        bootstrap_seed=31,
    )
    method = result["method"]
    assert method["endpoint_rows"] == 5600
    assert method["endpoint_summaries"] == 280
    assert method["controlled_paired_rows"] == 1600
    assert method["pairing_paired_rows"] == 3200
    assert method["across_seed_summaries"] == 84
    assert method["comparison_summaries"] == 84
    dsvae_summary = next(
        row
        for row in result["comparison_summary"]
        if row["scope"] == "model"
        and row["model"] == "dsvae"
        and row["endpoint"] == "efficiency_250hz"
    )
    macro = next(row for row in result["comparison_summary"] if row["scope"] == "macro")
    assert dsvae_summary["seeds"] == 5
    assert macro["seeds"] == 3
    decision = next(row for row in result["reserve_seed_decisions"] if row["model"] == "dsvae")
    assert decision["triggered"] is True
    assert decision["decision"] == "extension_complete"

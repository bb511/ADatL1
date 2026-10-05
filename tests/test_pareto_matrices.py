"""Smoke tests for the Phase 4b gamma x bins matrices."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from src.evaluation.pareto_matrices import (  # noqa: E402
    METRICS,
    STATUS_FILENAME,
    collapse_rule_text,
    write_pareto_matrices,
)


def _candidates(architectures=("h64_32",)) -> pd.DataFrame:
    rows = []
    for arch in architectures:
        for gamma in (0.0, 0.1, 0.5):
            for bins in (10, 50):
                cid = f"study__gamma-{gamma}__bins-{bins}__arch-{arch}"
                rejected = gamma == 0.5 and bins == 50
                front = gamma == 0.1
                rows.append({
                    "configuration_id": cid, "autoencoder_seed": 180524, "architecture_id": arch,
                    "mi_gamma": gamma, "mi_sensitive_num_bins": bins,
                    "feasible": not rejected, "is_pareto_front": front,
                    "pareto_rank": (1 if bins == 10 else 2) if front else float("nan"),
                    **{col: 0.1 + gamma + bins / 1000 for col, *_ in METRICS if col != "redundancy_bits"},
                })
    return pd.DataFrame(rows)


def _study_root(tmp_path: Path, table: pd.DataFrame) -> Path:
    phase3 = tmp_path / "Pareto-Front-Test" / "phase3"
    phase3.mkdir(parents=True)
    table.to_csv(phase3 / "pareto_candidates.csv", index=False)
    (phase3 / "pareto_selection.json").write_text(json.dumps(
        {"selected_configuration_id": "study__gamma-0.1__bins-10__arch-h64_32"}))
    return phase3


def test_one_png_per_metric_plus_status(tmp_path):
    phase3 = _study_root(tmp_path, _candidates())
    out = tmp_path / "matrices"
    written = write_pareto_matrices(phase3 / "pareto_candidates.csv", out,
                                    selection=phase3 / "pareto_selection.json")
    names = {p.name for p in written}
    assert STATUS_FILENAME in names
    assert len(written) == len(METRICS) + 1
    assert all(p.is_file() and p.stat().st_size > 10_000 for p in written)


def test_architectures_get_one_directory_each(tmp_path):
    phase3 = _study_root(tmp_path, _candidates(("h64_32", "h128_64")))
    out = tmp_path / "matrices"
    written = write_pareto_matrices(phase3 / "pareto_candidates.csv", out)
    assert {p.parent.name for p in written} == {"h64_32", "h128_64"}


def test_collapse_rule_read_from_resolved_config(tmp_path):
    run_dir = tmp_path / "checkpoints" / "Exp" / "Run01"
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text(
        "pareto_study:\n  collapse_constraint:\n    rule:\n"
        "      minimum_joint_code_entropy_bits: 1.0\n"
        "      minimum_fraction_of_paired_gamma_zero_joint_entropy: 0.5\n"
        "      paired_reference: same_architecture_and_bins\n")
    study_map = tmp_path / "study_map.yaml"
    # Paths of another machine: found again under the local checkpoints root.
    study_map.write_text("runs:\n- manifest_path: /eos/x/checkpoints/Exp/Run01/resolved_config.yaml\n"
                         "  checkpoint_run_dir: /eos/x/checkpoints/Exp/Run01\n")
    text = collapse_rule_text(study_map, tmp_path / "checkpoints")
    assert text == "H(L) < 1 bit or < 0.5 × H(L) of the γ=0 run with the same bins"

"""Regression checks for the Section 3 analytical artifact pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_analytical_smoke_profile_generates_consistent_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "section3"
    subprocess.run(
        [
            sys.executable,
            "src/analytical.py",
            "--profile",
            "smoke",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["profile"] == "smoke"
    assert metadata["seed"] == 123
    assert metadata["n_pairs"] == 800
    assert set(metadata["artifacts"]) == {
        path.name for path in output_dir.iterdir() if path.name != "metadata.json"
    }
    assert all((output_dir / name).stat().st_size > 0 for name in metadata["artifacts"])

    channel = pd.read_csv(output_dir / "channel_reliability.csv")
    assert np.all(np.diff(channel["cap_theory"]) >= -1e-12)
    assert np.all(np.diff(channel["tpr_theory"]) > 0.0)

    direction = pd.read_csv(output_dir / "linear_direction_sweep.csv")
    assert direction.loc[direction["cap_theory"].idxmax(), "angle_deg"] == 0.0
    assert direction.loc[direction["tpr_theory"].idxmax(), "angle_deg"] == 0.0

    alignment = pd.read_csv(output_dir / "alignment_assumption_check.csv")
    aligned = alignment[alignment["case"] == "aligned"]
    nuisance = alignment[alignment["case"] == "nuisance-dominated"]
    assert aligned.loc[aligned["cap_theory"].idxmax(), "angle_deg"] == 0.0
    assert nuisance.loc[nuisance["cap_theory"].idxmax(), "angle_deg"] == 90.0
    assert nuisance.loc[nuisance["tpr_theory"].idxmax(), "angle_deg"] == 0.0

    ratios = pd.read_csv(output_dir / "alignment_ratio_sweep.csv")
    assert set(ratios["selected_type"]) == {"anomaly", "tie", "nuisance"}
    assert ratios.loc[ratios["lambda_ratio_u_over_z"] == 1.0, "selected_type"].item() == "tie"

    selectors = pd.read_csv(output_dir / "marginal_shift_selector_sweep.csv")
    assert np.allclose(selectors["cap_selected_angle_min_deg"], 0.0)
    assert np.allclose(selectors["cap_selected_angle_max_deg"], 0.0)
    zero_shift = selectors.iloc[0]
    assert zero_shift["w1_selected_angle_min_deg"] == 90.0
    assert zero_shift["threshold_selected_angle_min_deg"] == 90.0

    controls = pd.read_csv(output_dir / "score_family_summary.csv").set_index("score")
    oracle = controls.loc["linear_oracle_w_star"]
    reversed_oracle = controls.loc["linear_negative_oracle"]
    assert oracle["cap_empirical"] == pytest.approx(reversed_oracle["cap_empirical"], abs=1e-12)
    assert oracle["tpr_population"] > 0.5
    assert reversed_oracle["tpr_population"] < metadata["fpr"]
    assert controls.loc["constant_collapse", "cap_empirical"] == pytest.approx(0.0, abs=1e-12)

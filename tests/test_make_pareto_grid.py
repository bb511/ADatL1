"""scripts/physics/make_pareto_grid.py: one γ = 0 / 50-bin baseline per architecture."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.evaluation.pareto_baseline import GAMMA_ZERO_BASELINE_BINS

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "physics" / "make_pareto_grid.py"


@pytest.fixture(scope="module")
def grid_module():
    spec = importlib.util.spec_from_file_location("make_pareto_grid", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_study_grid_has_one_gamma_zero_run_at_50_bins(grid_module):
    rows = grid_module.build_rows(grid_module.DEFAULT_GRID, "01")
    baselines = [row for row in rows if float(row["gamma"]) == 0.0]
    assert [(row["bins"], row["architecture_id"]) for row in baselines] == [
        (GAMMA_ZERO_BASELINE_BINS, "h64_32")]
    assert baselines[0]["run_name"] == "Seed180524_Gamma_0_Bins_50_architecture_h64_32_Run01"
    # Runs at other bin counts need no baseline of their own.
    assert {row["bins"] for row in rows if float(row["gamma"]) != 0.0} - {50}


@pytest.mark.parametrize("bins", [[10, 50], 30])
def test_anything_but_a_single_50_bin_baseline_is_refused(grid_module, tmp_path, bins):
    grid = OmegaConf.to_container(OmegaConf.load(grid_module.DEFAULT_GRID), resolve=False)
    grid["search_space"]["gamma_zero_baseline"]["mi_sensitive_num_bins"] = bins
    path = tmp_path / "grid.yaml"
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(grid)))
    with pytest.raises(SystemExit, match="single value 50"):
        grid_module.build_rows(path, "01")

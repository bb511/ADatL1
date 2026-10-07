"""Smoke tests for the Phase 4b gamma x bins matrices."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from src.evaluation.pareto_matrices import (  # noqa: E402
    BLUE,
    METRICS,
    STATUS_FILENAME,
    collapse_rule_text,
    metric_colormap,
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
    # Pareto-Front-261002's per-bin pairing is gone: always the single baseline.
    assert text == "H(L) < 1 bit or < 0.5 × H(L) of the γ=0 baseline"


def test_subtitle_quotes_the_single_50_bin_baseline():
    from src.evaluation.pareto_matrices import _baseline_text

    table = _candidates()  # γ = 0 at 10 bins: 0.11, at 50 bins: 0.15
    assert _baseline_text(table, "leakage_worst", "{:.2f}") == "baseline γ=0: 0.15"
    without = table[~((table["mi_gamma"] == 0) & (table["mi_sensitive_num_bins"] == 50))]
    assert _baseline_text(without, "leakage_worst", "{:.2f}") == "baseline γ=0: n/a"


def test_every_metric_has_a_best_end():
    assert {better for _, _, _, better, _ in METRICS} == {"lower", "higher"}


@pytest.mark.parametrize("better", ["lower", "higher"])
def test_best_end_is_pale_blue_and_worst_end_dark_blue(better):
    from matplotlib.colors import to_rgba

    cmap = metric_colormap(better)
    best, worst = (0.0, 1.0) if better == "lower" else (1.0, 0.0)
    assert cmap(best) == pytest.approx(to_rgba(BLUE[0]))
    assert cmap(worst) == pytest.approx(to_rgba(BLUE[-1]))


@pytest.fixture
def saved_figures(monkeypatch):
    """Figures of every savefig call, inspected after drawing."""
    from matplotlib.figure import Figure

    figures = {}
    original = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        result = original(self, fname, *args, **kwargs)
        figures[Path(fname).name] = {
            "axes": [
                {
                    "position": ax.get_position(original=False).bounds,
                    "yticks": list(ax.get_yticks()),
                    "yticklabels": [t.get_text() for t in ax.get_yticklabels()],
                    "texts": [t.get_text() for t in ax.texts],
                    "cells": {
                        (round(p.get_x() - 0.04), round(p.get_y() - 0.04)): p.get_facecolor()
                        for p in ax.patches
                        if abs(p.get_width() - 0.92) < 1e-9 and p.get_facecolor()[3] > 0
                    },
                }
                for ax in self.axes
            ],
            "colorbars": [ax._colorbar for ax in self.axes if hasattr(ax, "_colorbar")],
        }
        return result

    monkeypatch.setattr(Figure, "savefig", savefig)
    return figures


def test_colour_scale_looks_like_the_correlation_matrices(tmp_path, saved_figures):
    from matplotlib.colors import to_rgba

    phase3 = _study_root(tmp_path, _candidates())
    write_pareto_matrices(phase3 / "pareto_candidates.csv", tmp_path / "matrices")

    # Values are 0.1 + gamma + bins/1000 on gammas (0, 0.1, 0.5) x bins (10, 50).
    # Feasible min 0.11 at (gamma 0, 10 bins) = cell (0, 0), feasible max 0.61 at
    # (gamma 0.5, 10 bins) = cell (0, 2). The rejected (0.5, 50) cell holds 0.65.
    for name, higher_is_better in (("01_leakage_worst.png", False),
                                   ("04_median_efficiency.png", True)):
        figure = saved_figures[name]
        matrix_ax, bar_ax = figure["axes"]
        (colorbar,) = figure["colorbars"]
        # Attached right of the matrix, same height, no extension arrows, framed.
        mx, my, mw, mh = matrix_ax["position"]
        bx, by, bw, bh = bar_ax["position"]
        assert by == pytest.approx(my) and bh == pytest.approx(mh)
        assert mx + mw < bx < mx + mw + 0.05
        assert colorbar.extend == "neither"
        assert colorbar.outline.get_visible()
        assert to_rgba(colorbar.outline.get_edgecolor()) == to_rgba("#0b0b0b")
        # Normalised to the min and max of the feasible cells, both written on the scale.
        assert (colorbar.norm.vmin, colorbar.norm.vmax) == pytest.approx((0.11, 0.61))
        assert bar_ax["yticks"][0] == pytest.approx(0.11)
        assert bar_ax["yticks"][-1] == pytest.approx(0.61)
        labels = bar_ax["yticklabels"]
        if name.startswith("04"):  # printed x1e3
            assert (labels[0], labels[-1]) == ("110.00", "610.00")
        else:
            assert (labels[0], labels[-1]) == ("0.110", "0.610")
        assert len(labels) >= 3 and len(set(labels)) == len(labels)
        assert not {"best", "worst"} & set(bar_ax["texts"])
        # Best end pale, worst end dark; the rejected cell is white.
        low, high = matrix_ax["cells"][(0, 0)], matrix_ax["cells"][(0, 2)]
        best, worst = (high, low) if higher_is_better else (low, high)
        assert best == pytest.approx(to_rgba(BLUE[0]))
        assert worst == pytest.approx(to_rgba(BLUE[-1]))
        assert matrix_ax["cells"][(1, 2)] == pytest.approx(to_rgba("#ffffff"))


def test_colorbar_ticks_keep_both_ends_and_unique_labels():
    from src.evaluation.pareto_matrices import colorbar_ticks, tick_labels

    ticks = colorbar_ticks(0.1053, 0.1142)
    assert ticks[0] == 0.1053 and ticks[-1] == 0.1142
    assert all(0.1053 < t < 0.1142 for t in ticks[1:-1])
    labels = tick_labels(ticks, "{:.3f}")
    assert labels[0].startswith("0.105") and labels[-1].startswith("0.114")
    assert len(set(labels)) == len(labels)
    assert colorbar_ticks(2.0, 2.0) == [2.0]

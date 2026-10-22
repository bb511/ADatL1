"""Synthetic acceptance tests for the Phase 4 figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.pareto.plots import (
    FIGURE_FILENAMES,
    _gamma_colors,
    ParetoPlotError,
    front_parallel_coordinates,
    write_pareto_figures,
)
from src.evaluation.pareto.selection import select_pareto_front


def _row(
    configuration_id: str,
    *,
    gamma: float,
    bins: int,
    architecture: str,
    leakage: float,
    correlation: float,
    efficiency: float,
    valid: bool = True,
    feasible: bool = True,
) -> dict[str, object]:
    row: dict[str, object] = {
        "study_id": "study",
        "protocol_version": "fet-et-four-probe-v10",
        "configuration_id": configuration_id,
        "mi_gamma": gamma,
        "mi_sensitive_num_bins": bins,
        "architecture_id": architecture,
        "configuration_valid": valid,
        "feasible": feasible,
        "rejection_reasons": "" if valid and feasible else "paired_constraints_failed",
    }
    for column, value in (
        ("leakage_worst", leakage),
        ("residual_correlation", correlation),
        ("median_efficiency", efficiency),
    ):
        row[column] = value
    return row


def _study() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for index, architecture in enumerate(("h64_32", "h128_64", "h64_64_32")):
        for step, gamma in enumerate((0.0, 0.05, 0.1, 0.15, 0.2, 0.25)):
            rows.append(
                _row(
                    f"{architecture}-g{gamma}",
                    gamma=gamma,
                    bins=(40, 50, 60)[step % 3],
                    architecture=architecture,
                    leakage=0.12 - 0.02 * step + 0.004 * index,
                    correlation=0.30 - 0.03 * step,
                    efficiency=0.90 - 0.05 * step,
                )
            )
    rows.append(
        _row("collapsed", gamma=0.25, bins=60, architecture="h64_32",
             leakage=0.01, correlation=0.01, efficiency=0.99, feasible=False)
    )
    candidates, front, _ = select_pareto_front(pd.DataFrame(rows))
    return candidates, front


def test_every_figure_is_written(tmp_path: Path) -> None:
    candidates, front = _study()
    written = write_pareto_figures(candidates, front, output_dir=tmp_path)

    assert set(written) == set(FIGURE_FILENAMES)
    for name, path in written.items():
        assert path.name == FIGURE_FILENAMES[name]
        assert path.is_file() and path.stat().st_size > 0


def test_infeasible_configuration_is_not_treated_as_a_competitor(tmp_path: Path) -> None:
    """A latent-collapsed run must never be ringed as front material."""

    candidates, front = _study()
    collapsed = candidates.loc[candidates["configuration_id"] == "collapsed"].iloc[0]

    assert collapsed["selection_status"] == "infeasible"
    assert not bool(collapsed["is_pareto_front"])
    assert "collapsed" not in set(front["configuration_id"])
    write_pareto_figures(candidates, front, output_dir=tmp_path)


def test_eleven_gamma_levels_over_four_decades_are_drawn(tmp_path: Path) -> None:
    """The study grid: gamma from 0.01 to 100 and bins from 10 to 500."""

    gammas = (0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 5, 10, 50, 100)
    bins = (10, 20, 30, 40, 50, 60, 80, 150, 300, 500)
    rows = [_row("baseline", gamma=0.0, bins=50, architecture="h64_32",
                 leakage=0.2, correlation=0.3, efficiency=0.9)]
    rows += [
        _row(f"g{gamma}-b{nbins}", gamma=gamma, bins=nbins, architecture="h64_32",
             leakage=0.2 / (1 + gamma) + 0.0001 * nbins, correlation=0.3 / (1 + gamma),
             efficiency=0.9 - 0.02 * np.log10(gamma * 100 + 1))
        for gamma in gammas for nbins in bins
    ]
    candidates, front, _ = select_pareto_front(pd.DataFrame(rows))
    written = write_pareto_figures(candidates, front, output_dir=tmp_path)

    assert set(written) == set(FIGURE_FILENAMES)
    colors = _gamma_colors(candidates)
    assert len(colors) == len(gammas) and len(set(colors.values())) == len(gammas)

def test_empty_front_refuses_the_parallel_coordinates_figure(tmp_path: Path) -> None:
    candidates, _ = _study()
    with pytest.raises(ParetoPlotError, match="empty"):
        front_parallel_coordinates(
            candidates.iloc[0:0], output_path=tmp_path / "front.png"
        )


def test_missing_column_is_reported_rather_than_crashing_matplotlib(tmp_path: Path) -> None:
    candidates, front = _study()
    with pytest.raises(ParetoPlotError, match="missing columns"):
        write_pareto_figures(
            candidates.drop(columns=["median_efficiency"]), front, output_dir=tmp_path
        )


def _figures_on_save(monkeypatch) -> dict[str, object]:
    """Every figure as it is saved, keyed by file name."""

    from matplotlib.figure import Figure

    saved: dict[str, object] = {}
    original = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        saved[Path(fname).name] = {
            "axes": [
                {
                    "xscale": ax.get_xscale(),
                    "xlabel": ax.get_xlabel(),
                    "ylabel": ax.get_ylabel(),
                    "xticklabels": [t.get_text() for t in ax.get_xticklabels()],
                    "collections": [
                        {
                            "n": len(c.get_offsets()),
                            "facecolors": {tuple(np.round(f, 4)) for f in c.get_facecolors()},
                            "has_array": c.get_array() is not None,
                        }
                        for c in ax.collections
                    ],
                }
                for ax in self.axes
            ],
        }
        return original(self, fname, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)
    return saved


def test_pairs_are_one_colour_without_front_or_architecture_marks(tmp_path, monkeypatch):
    from matplotlib.colors import to_rgba

    from src.evaluation.pareto.plots import POINT_COLOR

    saved = _figures_on_save(monkeypatch)
    candidates, front = _study()
    write_pareto_figures(candidates, front, output_dir=tmp_path)

    pairs = {name: fig for name, fig in saved.items()
             if name.startswith("pareto_") and "coloured" not in name and "parallel" not in name}
    assert set(pairs) == {
        "pareto_L_vs_E.png",
        "pareto_L_vs_1_minus_eff.png", "pareto_L_vs_1_minus_eff_log.png",
        "pareto_E_vs_1_minus_eff.png", "pareto_E_vs_1_minus_eff_log.png",
    }
    n_eligible = int((candidates["configuration_valid"] & candidates["feasible"]).sum())
    for name, figure in pairs.items():
        (axis,) = figure["axes"]  # one panel: no architecture facets
        filled = [c for c in axis["collections"] if c["facecolors"] and c["n"] == n_eligible]
        assert len(filled) == 1, name  # no front rings or other overlays
        assert filled[0]["facecolors"] == {tuple(np.round(to_rgba(POINT_COLOR), 4))}, name
        # The collapsed run is the only other mark: hollow.
        others = [c for c in axis["collections"] if c is not filled[0]]
        assert [c["n"] for c in others] == [1] and others[0]["facecolors"] == set(), name


def test_efficiency_is_drawn_as_one_minus_eff_linear_and_log(tmp_path, monkeypatch):
    saved = _figures_on_save(monkeypatch)
    candidates, front = _study()
    write_pareto_figures(candidates, front, output_dir=tmp_path)

    for stem in ("pareto_L_vs_1_minus_eff", "pareto_E_vs_1_minus_eff"):
        (linear,) = saved[f"{stem}.png"]["axes"]
        (log,) = saved[f"{stem}_log.png"]["axes"]
        assert linear["xscale"] == "linear" and log["xscale"] == "log"
        assert linear["xlabel"].startswith("$1 - ")
        assert log["xlabel"].endswith("(log scale)")
        # Readable ticks even though the data span less than a decade.
        assert len([t for t in log["xticklabels"] if t]) >= 2


def test_log_axis_labels_values_close_to_one(tmp_path, monkeypatch):
    """1 - eff within 0.1 % of 1, as in the real studies: ticks must still show."""

    saved = _figures_on_save(monkeypatch)
    rows = [
        _row(f"g{gamma}", gamma=gamma, bins=50, architecture="h64_32",
             leakage=0.11 - 0.001 * step, correlation=0.10 - 0.005 * step,
             efficiency=0.00477 + 0.0001 * step)
        for step, gamma in enumerate((0.0, 0.01, 0.1, 0.5, 0.8))
    ]
    candidates, front, _ = select_pareto_front(pd.DataFrame(rows))
    write_pareto_figures(candidates, front, output_dir=tmp_path)

    (log,) = saved["pareto_L_vs_1_minus_eff_log.png"]["axes"]
    labels = [t for t in log["xticklabels"] if t]
    assert len(labels) >= 2
    assert all(0.99 < float(t) < 1.0 for t in labels)


def test_three_objective_figure_puts_l_up_e_across_and_efficiency_in_colour(tmp_path, monkeypatch):
    saved = _figures_on_save(monkeypatch)
    candidates, front = _study()
    write_pareto_figures(candidates, front, output_dir=tmp_path)

    matrix_ax, colorbar_ax = saved["pareto_L_vs_E_coloured_by_eff.png"]["axes"]
    assert matrix_ax["ylabel"].startswith("Leakage")
    assert matrix_ax["xlabel"].startswith("Residual correlation")
    assert any(c["has_array"] for c in matrix_ax["collections"])  # coloured by a value
    assert colorbar_ax["ylabel"].startswith("Signal efficiency")


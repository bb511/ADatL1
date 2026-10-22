"""Loss figures: the legend sits below the x-axis."""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from src.plot import scatter


def _capture_legends(monkeypatch) -> list[dict]:
    """Record where each plot_lines legend ends up, before the figure is closed."""
    captured = []
    original = scatter._legend_below_xaxis

    def wrapper(fig, ax, handles, labels, max_columns):
        legend = original(fig, ax, handles, labels, max_columns)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        captured.append(
            {
                "legend": legend.get_window_extent(renderer),
                "xaxis": ax.xaxis.get_tightbbox(renderer),
                "axes": ax.get_window_extent(renderer),
                "texts": [text.get_text() for text in legend.get_texts()],
                "ncols": legend._ncols,
            }
        )
        return legend

    monkeypatch.setattr(scatter, "_legend_below_xaxis", wrapper)
    return captured


def test_plot_lines_legend_sits_below_xaxis_title(tmp_path: Path, monkeypatch) -> None:
    captured = _capture_legends(monkeypatch)
    initial_figures = tuple(plt.get_fignums())

    scatter.plot_lines(
        data={"Loss_mi * gamma": {1: 0.5, 2: 0.4}, "Loss_reco": {1: 1.0, 2: 0.8}},
        xlabel="Epoch",
        ylabel="Loss",
        title="Unnormalized training losses",
        save_dir=tmp_path,
        filename="lines.png",
        right_axis_data={"Loss_total": {1: 1.5, 2: 1.2}},
        right_ylabel="Loss_total",
    )

    (box,) = captured
    assert box["legend"].y1 <= box["xaxis"].y0  # under the tick labels and "Epoch"
    assert box["legend"].y1 < box["axes"].y0  # outside the axes
    assert box["texts"] == ["Loss_mi * gamma", "Loss_reco", "Loss_total"]
    assert box["ncols"] == 3
    assert tuple(plt.get_fignums()) == initial_figures


def test_plot_lines_legend_fills_rows_first(tmp_path: Path, monkeypatch) -> None:
    captured = _capture_legends(monkeypatch)
    labels = ["a", "b", "c", "d", "e"]

    scatter.plot_lines(
        data={label: {1: index, 2: index + 1} for index, label in enumerate(labels)},
        xlabel="Epoch",
        ylabel="MI [bits]",
        title="five series",
        save_dir=tmp_path,
    )

    # Matplotlib fills columns first: column 1 = a, d; column 2 = b, e; column 3 = c,
    # so the rows read a b c / d e.
    assert captured[0]["texts"] == ["a", "d", "b", "e", "c"]
    assert captured[0]["ncols"] == 3


def test_plot_lines_rejects_zero_legend_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="legend_max_columns"):
        scatter.plot_lines(
            data={"a": {1: 1.0}},
            xlabel="x",
            ylabel="y",
            title="t",
            save_dir=tmp_path,
            legend_max_columns=0,
        )

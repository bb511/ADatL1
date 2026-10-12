"""src/plot/horizontal_bar.plot_comparison: two models and their change."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from src.plot import horizontal_bar  # noqa: E402


@pytest.fixture
def figure(monkeypatch):
    """The figure plot_comparison closes, kept for inspection."""
    kept = []
    close = plt.close

    def keep(fig=None):
        kept.append(fig)
        close(fig)

    monkeypatch.setattr(plt, "close", keep)
    return kept


def test_layout_labels_and_colours(tmp_path, figure):
    path = horizontal_bar.plot_comparison(
        {"normal": 0.1, "sigA": 0.0}, {"sigA": 0.5, "normal": 0.12, "only_run": 1.0},
        "efficiency at threshold: operational\nanomaly score: ascore/full", tmp_path,
        reference_label=r"$\gamma$ = 0, bins = 50", run_label=r"$\gamma$ = 0.1, bins = 40",
        title="Signal efficiency\nA vs B", percent=True,
    )
    assert path == tmp_path / "efficiency_at_threshold_operational__anomaly_score_ascorefull.jpg"
    assert path.stat().st_size > 0

    fig = figure[0]
    ax, ax_change = fig.axes[:2]
    # Keys of both models only, in the run's order, first at the bottom.
    assert [t.get_text() for t in ax.get_yticklabels()] == ["sigA", "normal"]
    colours = {to_hex(bar.get_facecolor()) for bar in ax.patches}
    assert colours == {horizontal_bar.REFERENCE_COLOR, horizontal_bar.RUN_COLOR}
    assert {to_hex(bar.get_facecolor()) for bar in ax_change.patches} == {horizontal_bar.CHANGE_COLOR}
    assert ax_change.get_title() == "Change"
    assert [round(bar.get_width(), 6) for bar in ax_change.patches] == [0.0, 20.0]
    texts = [t.get_text() for t in ax_change.texts]
    assert texts == ["n/a", "+20.0%"]  # sigA: the reference is 0
    assert {"12.00%", "10.00%", "50.00%", "0.00%"} <= {t.get_text() for t in ax.texts}
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend == [r"$\gamma$ = 0, bins = 50", r"$\gamma$ = 0.1, bins = 40"]
    assert fig._suptitle.get_text() == "Signal efficiency\nA vs B"


def test_filename_override_and_plain_values(tmp_path, figure):
    path = horizontal_bar.plot_comparison(
        {"loss_total": 0.58}, {"loss_total": 0.41}, "ascore/operational", tmp_path / "sub",
        reference_label="a", run_label="b", filename="ascore_operational")
    assert path == tmp_path / "sub" / "ascore_operational.jpg"
    ax, ax_change = figure[0].axes[:2]
    assert {"0.58", "0.41"} <= {t.get_text() for t in ax.texts}
    assert [t.get_text() for t in ax_change.texts] == ["-29.3%"]

"""FET.Et row highlight in every correlation-matrix plot."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.plot import correlation_matrix as corr_plot  # noqa: E402
from src.plot import matrix  # noqa: E402

LABELS = ["jets.phi", "FET.Et", "jets.Et", "muons.Et"]
GREEN = mcolors.to_rgba(corr_plot.DECORRELATED_TEXT_COLOR)


def _correlation(values) -> pd.DataFrame:
    return pd.DataFrame(values, index=LABELS, columns=LABELS, dtype=float)


CORR = _correlation(
    [
        [1.0, 0.05, 0.3, 0.02],
        [0.05, 1.0, -0.1, 0.4],
        [0.3, -0.1, 1.0, -0.6],
        [0.02, 0.4, -0.6, 1.0],
    ]
)


@pytest.fixture
def captured_axes(monkeypatch):
    """Capture the matrix axes of every figure saved by ``matrix.plot``."""
    captured = []
    original_savefig = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        ax = self.axes[0]
        captured.append(
            {
                "patches": [p for p in ax.patches if isinstance(p, Rectangle)],
                "texts": {
                    (round(t.get_position()[0]), round(t.get_position()[1])): t
                    for t in ax.texts
                },
            }
        )
        return original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)
    return captured


def _is_green(text) -> bool:
    return mcolors.to_rgba(text.get_color()) == GREEN


def test_matrix_plot_outlines_row_and_greens_small_entries(tmp_path, captured_axes):
    matrix.plot(
        data=CORR.to_dict(orient="index"),
        value_name="t",
        save_dir=tmp_path,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        filename="m.png",
        outline_row="FET.Et",
        text_highlight_max_abs=0.1,
        text_highlight_color=corr_plot.DECORRELATED_TEXT_COLOR,
    )

    (axes,) = captured_axes
    (border,) = axes["patches"]
    assert border.get_xy() == (-0.5, 0.5)
    assert border.get_width() == len(LABELS)
    assert border.get_height() == 1
    assert border.get_linewidth() == 3.0
    assert mcolors.to_rgba(border.get_edgecolor()) == mcolors.to_rgba("black")
    assert not border.get_fill()

    # Row 1 is FET.Et: |0.05| and |-0.1| pass, 1.0 and 0.4 do not.
    green_cells = {pos for pos, text in axes["texts"].items() if _is_green(text)}
    assert green_cells == {(0, 1), (2, 1)}
    assert axes["texts"][(0, 1)].get_fontweight() == "bold"
    # Small values outside the FET.Et row stay uncoloured.
    assert not _is_green(axes["texts"][(3, 0)])
    assert (tmp_path / "m.png").is_file()


def test_matrix_plot_without_highlight_or_missing_row_draws_no_border(
    tmp_path, captured_axes
):
    data = CORR.to_dict(orient="index")
    matrix.plot(data=data, value_name="t", save_dir=tmp_path, filename="a.png")
    matrix.plot(
        data=data,
        value_name="t",
        save_dir=tmp_path,
        filename="b.png",
        outline_row="taus.Et",
        text_highlight_max_abs=0.1,
    )
    for axes in captured_axes:
        assert axes["patches"] == []
        assert not any(_is_green(text) for text in axes["texts"].values())


def test_matrix_plot_matches_outline_row_case_insensitively(tmp_path, captured_axes):
    matrix.plot(
        data=CORR.to_dict(orient="index"),
        value_name="t",
        save_dir=tmp_path,
        filename="m.png",
        outline_row="fet.et",
    )
    (border,) = captured_axes[0]["patches"]
    assert border.get_xy() == (-0.5, 0.5)


def test_matrix_plot_greens_named_columns_of_the_outlined_row(tmp_path, captured_axes):
    matrix.plot(
        data=CORR.to_dict(orient="index"),
        value_name="t",
        save_dir=tmp_path,
        filename="m.png",
        outline_row="FET.Et",
        text_highlight_columns=["jets.phi", "muons.Et", "not.there"],
        text_highlight_color=corr_plot.DECORRELATED_TEXT_COLOR,
    )
    texts = captured_axes[0]["texts"]
    assert {pos for pos, text in texts.items() if _is_green(text)} == {(0, 1), (3, 1)}


def test_decorrelated_columns_uses_abs_r_of_the_reference_row():
    assert corr_plot.decorrelated_columns(CORR) == ["jets.phi", "jets.Et"]
    assert corr_plot.decorrelated_columns(CORR, "fet.et") == ["jets.phi", "jets.Et"]
    assert corr_plot.decorrelated_columns(CORR, "taus.Et") == []
    assert corr_plot.decorrelated_columns(None) == []


@pytest.mark.parametrize(
    "reference, expected",
    [(None, []), (CORR, ["jets.phi", "jets.Et"]), (CORR * 0.1, LABELS)],
)
def test_variants_frame_fet_et_and_green_from_the_reference(
    tmp_path, monkeypatch, reference, expected
):
    calls = []
    monkeypatch.setattr("src.plot.matrix.plot", lambda **kwargs: calls.append(kwargs))

    corr_plot.write_correlation_matrix_variants(
        CORR,
        plot_folder=tmp_path,
        stem="x",
        title="t",
        decorrelation_reference=reference,
    )

    assert [call["filename"] for call in calls] == ["x.png", "x_et_only.png"]
    assert [call["figure_scale"] for call in calls] == [1.0, 0.6]
    assert list(calls[1]["data"]) == ["FET.Et", "jets.Et", "muons.Et"]
    for call in calls:
        assert call["outline_row"] == "FET.Et"
        assert call["text_highlight_columns"] == expected
        assert (call["cmap"], call["vmin"], call["vmax"]) == ("coolwarm", -1.0, 1.0)


def _walsh_table(fet_et):
    """Pairwise uncorrelated columns (Pearson and Spearman r = 0)."""
    a = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
    b = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float)
    c = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
    return {"jets.phi": a, "FET.Et": fet_et(b, c), "jets.Et": c, "muons.Et": a * b * c}


def test_callback_greens_from_own_matrix_and_from_reconstruction_for_changes(
    tmp_path, monkeypatch
):
    callback_module = pytest.importorskip("src.evaluation.callbacks.correlation_matrix")
    calls = []
    monkeypatch.setattr("src.plot.matrix.plot", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        "src.evaluation.callbacks.correlation_matrix.utils.mlflow.log_plots_to_mlflow",
        lambda *args, **kwargs: None,
    )
    # Before training FET.Et == jets.Et; after training FET.Et is uncorrelated.
    before = _walsh_table(lambda b, c: c)
    after = _walsh_table(lambda b, c: b)
    callback = callback_module.CorrelationMatrixCallback(
        variables=LABELS,
        correlation_methods=["pearson", "spearman"],
        sensitive_variable="FET.Et",
        write_source_tables=False,
    )
    callback._active = True
    callback._resolved_variables = [{"label": label} for label in LABELS]
    callback._buffers = {"normal": {"input": [before], "reconstruction": [after]}}
    callback._event_counts = {"normal": 8}
    monkeypatch.setattr(callback, "_write_metadata", lambda *args, **kwargs: None)

    callback.on_test_epoch_end(
        trainer=SimpleNamespace(split="val"),
        pl_module=SimpleNamespace(_ckpt_path=tmp_path / "loss_total.ckpt"),
    )

    assert len(calls) == 2 * (2 + 2 + 3 * 2)
    for call in calls:
        assert call["outline_row"] == "FET.Et"
        if call["filename"].startswith("input_"):
            assert call["text_highlight_columns"] == ["jets.phi", "muons.Et"]
        else:
            # reconstruction_* and every abs_reconstruction_minus_input_* variant
            assert call["text_highlight_columns"] == ["jets.phi", "jets.Et", "muons.Et"]


def test_legacy_delta_plot_frames_fet_et(tmp_path, captured_axes):
    from src.analysis.correlation_matrix import (
        CorrelationMatrixPlotter,
        CorrelationMatrixSpecs,
    )

    plotter = CorrelationMatrixPlotter(CorrelationMatrixSpecs(tmp_path, tmp_path))
    plotter._plot_heatmap(
        CORR - CORR.T * 0.5,
        save_path=tmp_path / "d.png",
        title="t",
        decorrelation_reference=CORR,
    )

    texts = captured_axes[0]["texts"]
    assert {pos for pos, text in texts.items() if _is_green(text)} == {(0, 1), (2, 1)}
    (border,) = captured_axes[0]["patches"]
    assert border.get_xy() == (-0.5, 0.5)
    assert border.get_width() == len(LABELS)


# --------------------------------------------------------------------------------
# redo_correlation_matrix_plots.py
# --------------------------------------------------------------------------------

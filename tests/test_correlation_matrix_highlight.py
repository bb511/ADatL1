"""FET.Et row highlight in every correlation-matrix plot, and the redo script."""

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

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


def _blank_png(path: Path) -> None:
    Image.new("RGB", (4, 4), "white").save(path)


def test_redo_redraws_existing_pngs_only_and_refreshes_primary_galleries(tmp_path):
    from src.analysis.scripts import redo_correlation_matrix_plots as redo

    checkpoints = tmp_path / "checkpoints"
    run = checkpoints / "Exp" / "RunA"
    plot_dir = run / "plots/val/loss_total/correlation_matrix/normal/Pearson"
    plot_dir.mkdir(parents=True)
    CORR.to_csv(plot_dir / "input_pearson_correlation_matrix.csv")
    (CORR * 0.5).to_csv(plot_dir / "reconstruction_pearson_correlation_matrix.csv")
    existing = [
        "input_pearson_correlation_matrix.png",
        "reconstruction_pearson_correlation_matrix_et_only.png",
        "abs_reconstruction_minus_input_pearson_correlation_matrix_sorted_by_decrease.png",
    ]
    for name in existing:
        _blank_png(plot_dir / name)
    _blank_png(plot_dir / "unrelated.png")
    orphan_dir = checkpoints / "Exp/RunB/plots/val/last/correlation_matrix/normal"
    orphan_dir.mkdir(parents=True)
    _blank_png(orphan_dir / "input_pearson_correlation_matrix.png")

    jobs = redo.discover_plot_jobs(checkpoints)
    results, not_started = redo.redraw_all(jobs)

    assert not_started == 0
    assert sorted((r.png.name, r.status) for r in results if r.png.parent == plot_dir) == [
        (name, "redrawn") for name in sorted(existing)
    ]
    assert [r.status for r in results if r.png.parent == orphan_dir] == ["no_source"]
    for name in existing:
        assert Image.open(plot_dir / name).size[0] > 100
    assert Image.open(plot_dir / "unrelated.png").size == (4, 4)
    assert Image.open(orphan_dir / "input_pearson_correlation_matrix.png").size == (4, 4)
    assert sorted(p.name for p in plot_dir.iterdir()) == sorted(
        existing
        + [
            "unrelated.png",
            "input_pearson_correlation_matrix.csv",
            "reconstruction_pearson_correlation_matrix.csv",
        ]
    )

    # MLflow store: one primary run, one other attempt with the same checkpoint.
    mlruns = tmp_path / "mlruns"
    experiment = mlruns / "123"
    experiment.mkdir(parents=True)
    (experiment / "meta.yaml").write_text("name: Exp\n")
    old_card = (
        "<div class='card'><img loading='lazy' src='data:image/png;base64,OLD' "
        "alt='{stem}' onclick='openLightbox(this.src)'><div class='caption'>{stem}"
        "</div></div>"
    )
    stems = [Path(name).stem for name in existing] + ["unrelated"]
    page = "<html>" + "".join(old_card.format(stem=stem) for stem in stems) + "</html>"
    galleries = {}
    roles = {
        "a" * 32: "primary training run of checkpoint",
        "b" * 32: "other training attempt (checkpoint belongs to a)",
    }
    for run_id, role in roles.items():
        run_dir = experiment / run_id
        (run_dir / "tags").mkdir(parents=True)
        (run_dir / "meta.yaml").write_text("lifecycle_stage: active\n")
        (run_dir / "tags" / "mlflow.runName").write_text("RunA")
        (run_dir / "tags" / "link.role").write_text(role)
        (run_dir / "tags" / "link.checkpoint_dir").write_text("/elsewhere/Exp/RunA")
        gallery = run_dir / "artifacts/val/loss_total/correlation_matrix"
        gallery = gallery / "normal_correlation_matrix_pearson.html"
        gallery.parent.mkdir(parents=True)
        gallery.write_text(page)
        galleries[role.split()[0]] = gallery

    gallery_results = redo.refresh_galleries(
        mlruns,
        checkpoints,
        [r.png for r in results if r.status == "redrawn"],
        thumbnail=lambda png: f"data:image/png;base64,NEW-{png.stem}",
    )

    by_path = {r.gallery: r for r in gallery_results}
    assert by_path[galleries["primary"]].status == "refreshed"
    assert by_path[galleries["other"]].status == "skipped"
    refreshed = galleries["primary"].read_text()
    for stem in stems[:-1]:
        assert f"NEW-{stem}" in refreshed
    assert refreshed.count("base64,OLD") == 1  # unrelated.png keeps its thumbnail
    assert galleries["other"].read_text() == page


def test_redo_skip_redrawn_after_resumes(tmp_path):
    from src.analysis.scripts import redo_correlation_matrix_plots as redo

    plot_dir = tmp_path / "correlation_matrix" / "normal"
    plot_dir.mkdir(parents=True)
    CORR.to_csv(plot_dir / "input_pearson_correlation_matrix.csv")
    _blank_png(plot_dir / "input_pearson_correlation_matrix.png")

    jobs = redo.discover_plot_jobs(tmp_path)
    results, _ = redo.redraw_all(jobs, skip_redrawn_after=0.0)

    assert [r.status for r in results] == ["skipped_recent"]
    assert Image.open(plot_dir / "input_pearson_correlation_matrix.png").size == (4, 4)


def test_classify_png_recognises_every_variant():
    from src.analysis.scripts.redo_correlation_matrix_plots import classify_png

    space = classify_png(Path("reconstruction_spearman_correlation_matrix_et_only.png"))
    assert (space.kind, space.space, space.method, space.suffix) == (
        "space",
        "reconstruction",
        "spearman",
        "_et_only",
    )
    change = classify_png(
        Path("abs_reconstruction_minus_input_pearson_correlation_matrix_sorted_by_increase.png")
    )
    assert (change.kind, change.method, change.direction, change.suffix) == (
        "change",
        "pearson",
        "increase",
        "",
    )
    legacy = classify_png(Path("abs_correlation_delta_spearman_20260710_100411.png"))
    assert (legacy.kind, legacy.method) == ("legacy", "spearman")
    assert classify_png(Path("x_abs_correlation_delta.png")).method == "pearson"
    assert classify_png(Path("training_losses.png")) is None


def test_redo_prefers_the_csv_with_the_pngs_own_stem(tmp_path, monkeypatch):
    """Older folders hold one CSV per variant, NaN rows included; draw exactly that."""
    from src.analysis.scripts import redo_correlation_matrix_plots as redo

    plot_dir = tmp_path / "correlation_matrix" / "normal"
    plot_dir.mkdir(parents=True)
    stem = "abs_reconstruction_minus_input_pearson_correlation_matrix"
    plotted = CORR.copy()
    plotted.loc["jets.phi", :] = np.nan
    plotted.loc[:, "jets.phi"] = np.nan
    plotted.to_csv(plot_dir / f"{stem}.csv")
    _blank_png(plot_dir / f"{stem}.png")
    (CORR * 0.1).to_csv(plot_dir / "reconstruction_pearson_correlation_matrix.csv")
    calls = []

    def fake_plot(**kwargs):
        calls.append(kwargs)
        _blank_png(kwargs["save_dir"] / kwargs["filename"])

    monkeypatch.setattr("src.plot.matrix.plot", fake_plot)

    results, _ = redo.redraw_all(redo.discover_plot_jobs(tmp_path))

    assert [r.status for r in results] == ["redrawn"]
    (call,) = calls
    assert list(call["data"]) == LABELS
    assert np.isnan(call["data"]["jets.phi"]["FET.Et"])
    assert call["text_highlight_columns"] == LABELS  # from the reconstruction CSV
    assert call["outline_row"] == "FET.Et"
    assert not list(plot_dir.glob(".*"))


def test_redo_change_png_needs_the_reconstruction_matrix(tmp_path):
    from src.analysis.scripts import redo_correlation_matrix_plots as redo

    plot_dir = tmp_path / "correlation_matrix" / "normal"
    plot_dir.mkdir(parents=True)
    stem = "abs_reconstruction_minus_input_pearson_correlation_matrix_et_only"
    redo.corr_plot.et_only(CORR).to_csv(plot_dir / f"{stem}.csv")
    _blank_png(plot_dir / f"{stem}.png")

    results, _ = redo.redraw_all(redo.discover_plot_jobs(tmp_path))

    assert [r.status for r in results] == ["no_source"]
    assert Image.open(plot_dir / f"{stem}.png").size == (4, 4)

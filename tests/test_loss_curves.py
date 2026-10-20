"""Loss figures: legend below the x-axis, and the redraw script on a fake store."""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from src.analysis.scripts import redo_loss_plots
from src.plot import loss_curves, scatter


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


def _write_metric(run_dir: Path, key: str, values: list[float]) -> None:
    path = run_dir / "metrics" / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{1000 + step} {value} {step * 10}\n" for step, value in enumerate(values))
    )


def _fake_run(tmp_path: Path, with_links: bool) -> tuple[Path, Path, Path]:
    """Checkpoint run folder + MLflow file store with one matching training run."""
    mlruns = tmp_path / "mlruns"
    experiment = mlruns / "123"
    experiment.mkdir(parents=True)
    (experiment / "meta.yaml").write_text("name: Study-X\nlifecycle_stage: active\n")
    run_dir = experiment / "abc"
    run_dir.mkdir()
    (run_dir / "meta.yaml").write_text(
        "run_name: Seed1_Gamma_0.2_Run01\nlifecycle_stage: active\n"
    )
    for key in (
        "train/loss",
        "train/loss_mi",
        "train/loss_reco",
        "train/loss_mi_permuted",
        "train/loss_mi_floor_analytic",
        "val/normal/loss_mi",
        "val/normal/loss_mi_permuted",
    ):
        _write_metric(run_dir, key, [0.3, 0.2, 0.25])
    (run_dir / "artifacts" / "losses").mkdir(parents=True)

    checkpoint = tmp_path / "checkpoints" / "Study-X" / "Seed1_Gamma_0.2_Run01"
    (checkpoint / "plots" / "losses").mkdir(parents=True)
    (checkpoint / "resolved_config.yaml").write_text("algorithm:\n  mi_gamma: 0.2\n")
    if with_links:
        (checkpoint / "links.yaml").write_text(
            "mlflow:\n  experiment_id: '123'\n  run_id: abc\n"
        )
    return checkpoint, mlruns, run_dir


@pytest.mark.parametrize("with_links", [True, False])
def test_redo_loss_plots_overwrites_checkpoint_and_artifacts(
    tmp_path: Path, with_links: bool
) -> None:
    checkpoint, mlruns, run_dir = _fake_run(tmp_path, with_links)

    (result,) = redo_loss_plots.redraw_all(
        redo_loss_plots.discover_checkpoint_dirs([checkpoint.parent]), mlruns_root=mlruns
    )

    assert result.status == "ok", result.message
    assert "gamma=0.2" in result.message
    names = {
        loss_curves.TRAINING_LOSSES_FILE,
        loss_curves.UNNORMALIZED_FILE,
        loss_curves.MI_NULL_FILE,
    }
    for folder in (checkpoint / "plots" / "losses", run_dir / "artifacts" / "losses"):
        assert {path.name for path in folder.glob("*.png")} == names


def test_redo_loss_plots_skips_ambiguous_run_names(tmp_path: Path) -> None:
    checkpoint, mlruns, run_dir = _fake_run(tmp_path, with_links=False)
    twin = run_dir.parent / "def"
    twin.mkdir()
    (twin / "meta.yaml").write_text((run_dir / "meta.yaml").read_text())
    _write_metric(twin, "train/loss", [0.1])

    result = redo_loss_plots.redraw_run(checkpoint, mlruns)

    assert result.status == "skipped"
    assert "2 active runs" in result.message
    assert not list((checkpoint / "plots" / "losses").glob("*.png"))


def test_history_by_step_keeps_latest_point_per_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    path = run_dir / "metrics" / "train" / "loss"
    path.parent.mkdir(parents=True)
    path.write_text("100 0.5 10\n200 0.4 10\n300 0.3 20\n")

    assert redo_loss_plots.read_history(run_dir, "train/loss") == {10: 0.4, 20: 0.3}

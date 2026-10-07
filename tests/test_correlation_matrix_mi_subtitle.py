"""MI hyperparameters (γ, requested and effective bins) below every correlation-matrix title."""

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

from src.analysis.run_mi_hyperparameters import (  # noqa: E402
    MiHyperparameters,
    read_mi_hyperparameters,
    run_dir_of,
)
from src.plot import matrix  # noqa: E402

LABELS = ["jets.phi", "FET.Et", "jets.Et"]
CORR = pd.DataFrame(
    [[1.0, 0.05, 0.3], [0.05, 1.0, -0.4], [0.3, -0.4, 1.0]], index=LABELS, columns=LABELS
)
TEXT = "MI: γ = 0.1 · requested bins = 80 · effective bins = 60"


def test_text_shows_every_value_and_na_for_missing_ones():
    assert MiHyperparameters(0.1, 80, 60).text() == TEXT
    assert MiHyperparameters(0.25, 50, None).text() == (
        "MI: γ = 0.25 · requested bins = 50 · effective bins = n/a"
    )
    assert not MiHyperparameters().known


def test_run_dir_is_the_parent_of_plots():
    path = Path("checkpoints/Exp/Run/plots/val/loss_total/correlation_matrix/normal/x.png")
    assert run_dir_of(path) == Path("checkpoints/Exp/Run")
    assert run_dir_of(Path("elsewhere/x.png")) is None


def _bin_widths(run_dir: Path, n: int) -> None:
    for epoch in ("epoch_0000", "epoch_0199"):
        folder = run_dir / "plots/mi_diagnostics/data" / epoch
        folder.mkdir(parents=True)
        pd.DataFrame({"bin_id": np.arange(n)}).to_csv(
            folder / f"mi_bin_widths_{epoch.replace('_', '')}.csv", index=False
        )


def test_reads_resolved_config_and_bin_width_csv(tmp_path):
    run = tmp_path / "checkpoints/Exp/Run"
    run.mkdir(parents=True)
    (run / "resolved_config.yaml").write_text(
        "algorithm:\n  mi_gamma: 0.1\n  mi_sensitive_num_bins: 80\n"
    )
    _bin_widths(run, 60)
    assert read_mi_hyperparameters(run, tmp_path) == MiHyperparameters(0.1, 80, 60)


def test_falls_back_to_mlflow_params_and_the_training_log(tmp_path):
    run = tmp_path / "checkpoints/Exp/Run"
    run.mkdir(parents=True)
    (run / "links.yaml").write_text(
        "mlflow:\n  experiment_id: '1'\n  run_id: abc\n"
        "hydra:\n  train: logs/train/runs/first\n  other: [logs/train/runs/second]\n"
    )
    params = tmp_path / "logs/mlflow/mlruns/1/abc/params/algorithm"
    params.mkdir(parents=True)
    (params / "mi_gamma").write_text("0.25")
    (params / "mi_sensitive_num_bins").write_text("50")
    log = tmp_path / "logs/train/runs/second"
    log.mkdir(parents=True)
    (log / "train.log").write_text("[MI] Requested bins: 50\n[MI] Effective bins: 48\n")
    assert read_mi_hyperparameters(run, tmp_path) == MiHyperparameters(0.25, 50, 48)


def test_nothing_on_disk_gives_an_empty_record(tmp_path):
    run = tmp_path / "Run"
    run.mkdir()
    assert read_mi_hyperparameters(run, tmp_path) == MiHyperparameters()


def test_matrix_plot_prints_the_subtitle_between_title_and_matrix(tmp_path, monkeypatch):
    seen = {}
    original = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        ax = self.axes[0]
        seen["texts"] = [t.get_text() for t in ax.texts]
        seen["title"] = ax.get_title()
        renderer = self.canvas.get_renderer()
        seen["title_y"] = ax.title.get_window_extent(renderer).y0
        sub = next(t for t in ax.texts if t.get_text() == TEXT)
        seen["sub_box"] = sub.get_window_extent(renderer)
        seen["ax_top"] = ax.get_window_extent(renderer).y1
        return original(self, fname, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)
    matrix.plot(data=CORR.to_dict(orient="index"), value_name="Title", save_dir=tmp_path,
                filename="m.png", subtitle=TEXT)

    assert seen["title"] == "Title" and TEXT in seen["texts"]
    assert seen["ax_top"] < seen["sub_box"].y0 < seen["sub_box"].y1 <= seen["title_y"]


def test_callback_passes_the_models_mi_hyperparameters(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    callback_module = pytest.importorskip("src.evaluation.callbacks.correlation_matrix")
    calls = []
    monkeypatch.setattr("src.plot.matrix.plot", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        "src.evaluation.callbacks.correlation_matrix.utils.mlflow.log_plots_to_mlflow",
        lambda *args, **kwargs: None,
    )
    rng = np.random.default_rng(0)
    table = {label: rng.normal(size=16) for label in LABELS}
    callback = callback_module.CorrelationMatrixCallback(
        variables=LABELS, correlation_methods=["pearson", "spearman"],
        sensitive_variable="FET.Et", write_source_tables=False,
    )
    callback._active = True
    callback._resolved_variables = [{"label": label} for label in LABELS]
    callback._buffers = {"normal": {"input": [table], "reconstruction": [table]}}
    callback._event_counts = {"normal": 16}
    monkeypatch.setattr(callback, "_write_metadata", lambda *args, **kwargs: None)
    module = SimpleNamespace(
        _ckpt_path=tmp_path / "loss_total.ckpt",
        mi_gamma=0.1,
        sensitive_binner=SimpleNamespace(num_bins=80, bin_edges=torch.zeros(59)),
    )

    callback.on_test_epoch_end(trainer=SimpleNamespace(split="val"), pl_module=module)

    assert calls and {call["subtitle"] for call in calls} == {TEXT}


def test_callback_without_mi_model_draws_no_subtitle():
    callback_module = pytest.importorskip("src.evaluation.callbacks.correlation_matrix")
    mi = callback_module.CorrelationMatrixCallback._mi_hyperparameters(SimpleNamespace())
    assert not mi.known


def test_redo_puts_the_runs_mi_hyperparameters_below_the_title(tmp_path, monkeypatch):
    from src.analysis.scripts import redo_correlation_matrix_plots as redo

    run = tmp_path / "checkpoints/Exp/Run"
    plot_dir = run / "plots/val/loss_total/correlation_matrix/normal/Pearson"
    plot_dir.mkdir(parents=True)
    (run / "resolved_config.yaml").write_text(
        "algorithm:\n  mi_gamma: 0.1\n  mi_sensitive_num_bins: 80\n"
    )
    _bin_widths(run, 60)
    CORR.to_csv(plot_dir / "input_pearson_correlation_matrix.csv")
    CORR.to_csv(plot_dir / "reconstruction_pearson_correlation_matrix.csv")
    for name in ("input_pearson_correlation_matrix.png",
                 "abs_reconstruction_minus_input_pearson_correlation_matrix.png"):
        Image.new("RGB", (4, 4), "white").save(plot_dir / name)
    calls = []

    def fake_plot(**kwargs):
        calls.append(kwargs)
        Image.new("RGB", (4, 4), "white").save(kwargs["save_dir"] / kwargs["filename"])

    monkeypatch.setattr("src.plot.matrix.plot", fake_plot)
    results, _ = redo.redraw_all(redo.discover_plot_jobs(tmp_path / "checkpoints"))

    assert [r.status for r in results] == ["redrawn", "redrawn"]
    assert [call["subtitle"] for call in calls] == [TEXT, TEXT]

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from matplotlib.figure import Figure

import src.callbacks.binning as binning_callback_module
from src.callbacks.binning import BinningDiagnosticsCallback
from src.data.sensitive_binning import FixedQuantileSensitiveBinner
from src.plot.histogram import (
    plot_categorical_bin_counts,
    plot_fixed_bin_widths,
    plot_minibatch_scalar_histogram,
)


class _DiagnosticModule:
    def __init__(self) -> None:
        self.sensitive_binner = FixedQuantileSensitiveBinner(
            variable="FET.Et",
            num_bins=3,
        )
        self.sensitive_binner.bin_edges = torch.tensor([1.0, 2.0])
        self.sensitive_binner.fit_stats = {
            "num_bins_effective": 3,
            "counts": [50, 30, 20],
            "min": 0.0,
            "max": 4.0,
        }
        self.extraction_calls = 0

    def extract_sensitive_values(self, batch) -> torch.Tensor:
        self.extraction_calls += 1
        return batch["values"]


def _trainer(tmp_path: Path, epoch: int = 0):
    return SimpleNamespace(
        current_epoch=epoch,
        global_step=7,
        default_root_dir=tmp_path,
        is_global_zero=True,
        datamodule=SimpleNamespace(batch_size_per_device=5),
    )


def test_plotting_helpers_save_png_and_close_figures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    initial_figures = tuple(plt.get_fignums())
    captured_figures = []
    original_savefig = Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        captured_figures.append(figure)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)

    occupancy_path = plot_categorical_bin_counts(
        counts=[4, 0, 2, 3, 5, 1, 4, 2, 3, 1, 4, 2],
        expected_counts=[2.0] * 12,
        save_path=tmp_path / "occupancy.png",
        title="MI bin occupancy: batch = 100",
        metadata={
            "Epoch": 0,
            "Global step": 100,
            "Effective bins": 12,
            "Minibatch size": 31,
        },
    )
    bin_widths_path = plot_fixed_bin_widths(
        widths=[0.25, 0.5, 1.0],
        save_path=tmp_path / "bin_widths.png",
        title="Fixed MI bin widths: epoch = 0",
    )
    diversity_path = plot_minibatch_scalar_histogram(
        values=[3, 4, 4, 5],
        save_path=tmp_path / "diversity.png",
        title="Raw FET.Et diversity: epoch 0",
    )

    assert occupancy_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert bin_widths_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert diversity_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert tuple(plt.get_fignums()) == initial_figures

    occupancy_axis = captured_figures[0].axes[0]
    assert occupancy_axis.get_title() == "MI bin occupancy: batch = 100"
    assert occupancy_axis.xaxis.label.get_position()[0] == 0.5
    assert occupancy_axis.yaxis.label.get_position()[1] == 0.5
    assert len(occupancy_axis.tables) == 1
    np.testing.assert_array_equal(occupancy_axis.get_xticks(), [0, 5, 10])
    assert occupancy_axis.get_ylim()[1] >= 5.0 * 1.22

    bin_widths_axis = captured_figures[1].axes[0]
    assert bin_widths_axis.get_title() == "Fixed MI bin widths: epoch = 0"
    assert bin_widths_axis.get_xlabel() == "Bin ID"
    assert bin_widths_axis.get_ylabel() == "Bin width ΔFET.Et"
    np.testing.assert_array_equal(bin_widths_axis.get_xticks(), [0])
    np.testing.assert_allclose(
        bin_widths_axis.patches[0].get_data().values,
        [0.25, 0.5, 1.0],
    )

    diversity_axis = captured_figures[2].axes[0]
    bar_centers = [
        patch.get_x() + patch.get_width() / 2
        for patch in diversity_axis.patches
    ]
    bar_heights = [patch.get_height() for patch in diversity_axis.patches]
    np.testing.assert_array_equal(
        bar_centers,
        [0, 1, 2, 3],
    )
    np.testing.assert_array_equal(
        bar_heights,
        [3, 4, 4, 5],
    )
    assert diversity_axis.get_xlabel() == "Minibatch number"
    assert diversity_axis.get_ylabel() == "Number of unique FET.Et values"
    assert diversity_axis.xaxis.label.get_position()[0] == 0.5
    assert diversity_axis.yaxis.label.get_position()[1] == 0.5
    np.testing.assert_allclose(diversity_axis.get_ylim(), [2.0, 6.0])


def test_categorical_counts_can_cap_and_annotate_clipped_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_figures = []
    original_savefig = Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        captured_figures.append(figure)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)

    output_path = plot_categorical_bin_counts(
        counts=[2, 24, 5],
        save_path=tmp_path / "capped.png",
        title="Unique FET.Et values per MI bin: batch = 0",
        y_axis_max=20,
        annotate_clipped_values=True,
        integer_y_ticks=True,
    )

    assert output_path.is_file()
    axis = captured_figures[0].axes[0]
    np.testing.assert_allclose(axis.get_ylim(), [0, 20])
    assert [text.get_text() for text in axis.texts] == ["24"]
    assert axis.texts[0].get_position() == (1, 19.2)
    assert axis.texts[0].get_color() == "black"
    assert np.all(np.mod(axis.get_yticks(), 1) == 0)


def test_callback_collects_every_batch_and_plots_selected_batches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    occupancy_calls = []
    bin_width_calls = []
    histogram_calls = []

    def capture_occupancy(counts, save_path, **kwargs):
        occupancy_calls.append((np.asarray(counts), Path(save_path), kwargs))

    def capture_histogram(values, save_path, **kwargs):
        histogram_calls.append((list(values), Path(save_path), kwargs))

    def capture_bin_widths(widths, save_path, **kwargs):
        bin_width_calls.append((np.asarray(widths), Path(save_path), kwargs))

    monkeypatch.setattr(
        binning_callback_module,
        "plot_categorical_bin_counts",
        capture_occupancy,
    )
    monkeypatch.setattr(
        binning_callback_module,
        "plot_minibatch_scalar_histogram",
        capture_histogram,
    )
    monkeypatch.setattr(
        binning_callback_module,
        "plot_fixed_bin_widths",
        capture_bin_widths,
    )

    callback = BinningDiagnosticsCallback(
        enabled=True,
        epochs=[0],
        batch_indices=[0],
    )
    trainer = _trainer(tmp_path)
    module = _DiagnosticModule()

    callback.on_train_epoch_start(trainer, module)
    callback.on_train_batch_end(
        trainer,
        module,
        outputs=None,
        batch={"values": torch.tensor([0.2, 0.2, 2.5, 2.5, 2.5])},
        batch_idx=0,
    )
    trainer.global_step = 8
    callback.on_train_batch_end(
        trainer,
        module,
        outputs=None,
        batch={"values": torch.tensor([0.1, float("nan")])},
        batch_idx=1,
    )
    callback.on_train_epoch_end(trainer, module)

    assert module.extraction_calls == 2
    assert len(bin_width_calls) == 1
    fitted_widths, bin_width_path, bin_width_kwargs = bin_width_calls[0]
    np.testing.assert_array_equal(fitted_widths, [1.0, 1.0, 2.0])
    assert bin_width_path.name == "mi_bin_widths_epoch0000.png"
    assert bin_width_kwargs["title"] == "Fixed MI bin widths: epoch = 0"
    assert bin_width_kwargs["ylabel"] == "Bin width ΔFET.Et"

    assert len(occupancy_calls) == 2
    observed, occupancy_path, occupancy_kwargs = occupancy_calls[0]
    np.testing.assert_array_equal(observed, [2, 0, 3])
    np.testing.assert_allclose(
        occupancy_kwargs["expected_counts"],
        [2.5, 1.5, 1.0],
    )
    assert occupancy_kwargs["title"] == "MI bin occupancy: batch = 0"
    assert occupancy_kwargs["metadata"] == {
        "Epoch": 0,
        "Global step": 7,
        "Effective bins": 3,
        "Minibatch size": 5,
    }
    assert occupancy_path.name == "mi_bin_occupancy_batch_0.png"
    assert occupancy_path.parent == tmp_path / "mi_diagnostics"

    unique_per_bin, unique_path, unique_kwargs = occupancy_calls[1]
    np.testing.assert_array_equal(unique_per_bin, [1, 0, 1])
    assert unique_path.name == "mi_bin_unique_values_batch_0.png"
    assert (
        unique_kwargs["title"]
        == "Unique FET.Et values per MI bin: batch = 0"
    )
    assert (
        unique_kwargs["observed_label"]
        == "Unique finite FET.Et values in minibatch"
    )
    assert "expected_counts" not in unique_kwargs
    assert unique_kwargs["ylabel"] == "Number of unique FET.Et values"
    assert unique_kwargs["metadata"] == occupancy_kwargs["metadata"]
    assert unique_kwargs["y_axis_max"] == 20
    assert unique_kwargs["annotate_clipped_values"] is True
    assert unique_kwargs["integer_y_ticks"] is True

    assert len(histogram_calls) == 1
    unique_counts, histogram_path, histogram_kwargs = histogram_calls[0]
    assert unique_counts == [2, 1]
    assert histogram_kwargs["title"] == "Raw FET.Et diversity: epoch 0"
    assert histogram_kwargs["xlabel"] == "Minibatch number"
    assert histogram_kwargs["ylabel"] == "Number of unique FET.Et values"
    assert "epoch0000_batches000002_nominal_n000005" in histogram_path.name
    assert callback._unique_counts == []


def test_callback_is_inactive_outside_its_schedule(tmp_path: Path) -> None:
    callback = BinningDiagnosticsCallback(
        enabled=True,
        epochs=[1],
        batch_indices=[0],
    )
    trainer = _trainer(tmp_path, epoch=0)
    module = _DiagnosticModule()

    callback.on_train_epoch_start(trainer, module)
    callback.on_train_batch_end(
        trainer,
        module,
        outputs=None,
        batch={"values": torch.tensor([0.2])},
        batch_idx=0,
    )
    callback.on_train_epoch_end(trainer, module)

    assert module.extraction_calls == 0
    assert not (tmp_path / "mi_diagnostics").exists()


def test_callback_saves_plots_in_checkpoint_tree_and_logs_mlflow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _Experiment:
        def __init__(self) -> None:
            self.artifacts = []
            self.texts = []

        def log_artifact(self, **kwargs) -> None:
            self.artifacts.append(kwargs)

        def log_text(self, **kwargs) -> None:
            self.texts.append(kwargs)

    experiment = _Experiment()
    mlflow_logger = SimpleNamespace(run_id="run-123", experiment=experiment)
    checkpoint_root = tmp_path / "checkpoints" / "physics_ae_models" / "run-123"
    callback = BinningDiagnosticsCallback(
        enabled=True,
        epochs=[0],
        batch_indices=[0],
        output_root_dir=checkpoint_root / "plots",
        log_to_mlflow=True,
        mlflow_artifact_path="mi_diagnostics",
    )
    monkeypatch.setattr(
        callback,
        "_get_mlflow_logger",
        lambda trainer: mlflow_logger,
    )
    trainer = _trainer(tmp_path)
    module = _DiagnosticModule()

    callback.on_train_epoch_start(trainer, module)
    callback.on_train_batch_end(
        trainer,
        module,
        outputs=None,
        batch={"values": torch.tensor([0.2, 0.2, 2.5, 2.5, 2.5])},
        batch_idx=0,
    )
    callback.on_train_epoch_end(trainer, module)

    run_plots = sorted(
        (checkpoint_root / "plots" / "mi_diagnostics").glob("*.png")
    )
    assert len(run_plots) == 4

    assert len(experiment.artifacts) == 4
    assert all(
        Path(call["local_path"]).parent
        == checkpoint_root / "plots" / "mi_diagnostics"
        for call in experiment.artifacts
    )
    assert all(
        call["artifact_path"] == "mi_diagnostics"
        for call in experiment.artifacts
    )
    assert experiment.texts[0]["artifact_file"] == "mi_diagnostics/index.html"
    assert "MI Binning Diagnostics" in experiment.texts[0]["text"]
    assert "data:image/png;base64," in experiment.texts[0]["text"]


def test_transform_values_matches_existing_transform_path() -> None:
    binner = FixedQuantileSensitiveBinner(variable="FET.Et", num_bins=3)
    binner.bin_edges = torch.tensor([1.0, 2.0])
    x = torch.tensor([[0.5], [1.5], [2.5]])
    feature_map = {"FET": {"Et": [0]}}

    values = binner.extract_values(x=x, object_feature_map=feature_map)

    torch.testing.assert_close(
        binner.transform_values(values),
        binner.transform(x=x, object_feature_map=feature_map),
    )


def test_physics_ae_enables_binning_diagnostics_only_for_that_experiment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))

    with initialize(version_base="1.3", config_path="../configs"):
        physics_ae = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=physics/ae",
                "run_name=diagnostics-test",
            ],
        )
        default = compose(
            config_name="train.yaml",
            overrides=["data=basis"],
        )

    assert physics_ae.callbacks.binning.enabled is True
    assert list(physics_ae.callbacks.binning.epochs) == [0]
    assert list(physics_ae.callbacks.binning.batch_indices) == [0, 100, 400, 764]
    output_root = Path(physics_ae.callbacks.binning.output_root_dir)
    assert output_root.name == "plots"
    assert output_root.parent.name == "diagnostics-test"
    assert output_root.parent.parent.name == "physics_ae_models"
    assert physics_ae.callbacks.binning.output_subdir == "mi_diagnostics"
    assert physics_ae.callbacks.binning.get("checkpoint_root_dir") is None
    assert physics_ae.callbacks.binning.log_to_mlflow is True
    assert physics_ae.callbacks.binning.mlflow_artifact_path == "mi_diagnostics"
    assert isinstance(
        instantiate(physics_ae.callbacks.binning),
        BinningDiagnosticsCallback,
    )
    assert "binning" not in default.callbacks

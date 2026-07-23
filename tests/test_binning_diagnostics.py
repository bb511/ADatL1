from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

import src.callbacks.binning as binning_callback_module
from src.callbacks.binning import BinningDiagnosticsCallback
from src.data.sensitive_binning import FixedQuantileSensitiveBinner
from src.plot.histogram import (
    plot_categorical_bin_counts,
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


def test_plotting_helpers_save_png_and_close_figures(tmp_path: Path) -> None:
    initial_figures = tuple(plt.get_fignums())

    occupancy_path = plot_categorical_bin_counts(
        counts=[4, 0, 2],
        expected_counts=[2.0, 2.0, 2.0],
        save_path=tmp_path / "occupancy.png",
        title="Occupancy",
    )
    diversity_path = plot_minibatch_scalar_histogram(
        values=[3, 4, 4, 5],
        save_path=tmp_path / "diversity.png",
        title="Diversity",
        xlabel="Unique finite values",
    )

    assert occupancy_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert diversity_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert tuple(plt.get_fignums()) == initial_figures


def test_callback_collects_every_batch_and_plots_selected_batches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    occupancy_calls = []
    histogram_calls = []

    def capture_occupancy(counts, save_path, **kwargs):
        occupancy_calls.append((np.asarray(counts), Path(save_path), kwargs))

    def capture_histogram(values, save_path, **kwargs):
        histogram_calls.append((list(values), Path(save_path), kwargs))

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
    assert len(occupancy_calls) == 1
    observed, occupancy_path, occupancy_kwargs = occupancy_calls[0]
    np.testing.assert_array_equal(observed, [2, 0, 3])
    np.testing.assert_allclose(
        occupancy_kwargs["expected_counts"],
        [2.5, 1.5, 1.0],
    )
    assert "epoch0000_batch000000_step000000007_bins0003_n000005" in (
        occupancy_path.name
    )
    assert occupancy_path.parent == tmp_path / "mi_diagnostics"

    assert len(histogram_calls) == 1
    unique_counts, histogram_path, histogram_kwargs = histogram_calls[0]
    assert unique_counts == [2, 1]
    assert "epoch=0, minibatches=2, nominal minibatch size=5" in (
        histogram_kwargs["title"]
    )
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


def test_physics_ae_enables_binning_diagnostics_only_for_that_experiment() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        physics_ae = compose(
            config_name="train.yaml",
            overrides=["experiment=physics/ae"],
        )
        default = compose(
            config_name="train.yaml",
            overrides=["data=basis"],
        )

    assert physics_ae.callbacks.binning.enabled is True
    assert list(physics_ae.callbacks.binning.epochs) == [0]
    assert list(physics_ae.callbacks.binning.batch_indices) == [0, 100, 400]
    assert physics_ae.callbacks.binning.output_subdir == "mi_diagnostics"
    assert isinstance(
        instantiate(physics_ae.callbacks.binning),
        BinningDiagnosticsCallback,
    )
    assert "binning" not in default.callbacks

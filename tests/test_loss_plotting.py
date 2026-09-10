from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.algorithms.ae import AE
from src.evaluation.callbacks.losses import LossesCallback
from src.plot import scatter


def test_plot_lines_supports_smaller_yaxis_title(
    tmp_path: Path,
    monkeypatch,
) -> None:
    initial_figures = tuple(plt.get_fignums())
    ylabel_calls = []
    original_set_ylabel = Axes.set_ylabel

    def capture_ylabel(axis, ylabel, *args, **kwargs):
        ylabel_calls.append((ylabel, kwargs))
        return original_set_ylabel(axis, ylabel, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_ylabel", capture_ylabel)

    output_path = scatter.plot_lines(
        data={
            "Loss_mi * gamma": {1: 0.5, 2: 0.4},
            "Loss_reco": {1: 1.0, 2: 0.8},
        },
        xlabel="Epoch",
        ylabel="Loss_mi * gamma & Loss_reco",
        title="Unnormalized training losses",
        save_dir=tmp_path,
        filename="training_losses_unnormalized.png",
        right_axis_data={"Loss_total": {1: 1.5, 2: 1.2}},
        right_ylabel="Loss_total",
        ylabel_fontsize=14,
    )

    assert output_path.is_file()
    assert output_path.name == "training_losses_unnormalized.png"
    assert (
        "Loss_mi * gamma & Loss_reco",
        {"fontsize": 14},
    ) in ylabel_calls
    assert tuple(plt.get_fignums()) == initial_figures


def test_ae_outlog_includes_gamma_weighted_mi_loss() -> None:
    gamma_mi_loss = object()

    logged = AE.outlog(
        None,
        {
            "loss": object(),
            "loss/mean": object(),
            "loss/reco": object(),
            "loss/mi": object(),
            "loss/gamma_mi": gamma_mi_loss,
            "ascore/operational": object(),
        },
    )

    assert logged["loss_gamma_mi"] is gamma_mi_loss


def test_loss_callback_computes_step_aligned_paired_minima() -> None:
    metrics, final_step = LossesCallback._paired_minimum_metrics(
        mi_by_step={10: 0.4, 20: 0.1, 30: 0.2},
        reco_by_step={10: 0.9, 20: 0.8, 30: 0.3},
    )

    assert metrics == {
        "train/min_loss_mi": 0.1,
        "train/loss_reco_at_min_loss_mi": 0.8,
        "train/min_loss_reco": 0.3,
        "train/loss_mi_at_min_loss_reco": 0.2,
    }
    assert final_step == 30


def test_loss_callback_keeps_latest_value_for_duplicate_mlflow_step() -> None:
    history = [
        SimpleNamespace(step=10, timestamp=100, value=0.5),
        SimpleNamespace(step=10, timestamp=200, value=0.4),
        SimpleNamespace(step=20, timestamp=300, value=0.3),
    ]

    assert LossesCallback._history_by_step(history) == {10: 0.4, 20: 0.3}

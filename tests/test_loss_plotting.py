from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

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

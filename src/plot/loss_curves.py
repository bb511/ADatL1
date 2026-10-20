"""Training-loss figures of an AE run, shared by training and the redraw script.

``LossesCallback`` draws them at the end of training from the live MLflow run, and
``src/analysis/scripts/redo_loss_plots.py`` redraws them later from the local MLflow
file store. Both call the functions here, so the figures cannot drift apart. Only
numpy and matplotlib are needed (no torch, no mlflow).

Figures, all in ``<checkpoint run>/plots/losses/``:

* ``training_losses.png``: each history normalized to [0, 1].
* ``training_losses_unnormalized.png``: ``gamma * loss_mi`` and ``loss_reco`` on the
  left y-axis, ``loss`` (labelled ``Loss_total``) on the right y-axis.
* ``mi_vs_permuted_null.png``: MI loss next to its permutation noise floor and the
  analytic floor, for train and the normal validation split.
"""

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np

from src.plot import scatter
from src.plot.lossplotter import LOSS_COLORS


# MLflow metric name under ``train/`` -> legend label, in plotting order.
TRAINING_LOSSES = (
    ("loss_mi", "Loss_mi"),
    ("loss_reco", "Loss_reco"),
    ("loss", "Loss_total"),
)

# Legend label -> (MLflow metric key, colour, line opacity)
MI_NULL_SERIES = (
    ("Loss_mi (train)", "train/loss_mi", "blue", 1.0),
    ("Loss_mi_permuted (train)", "train/loss_mi_permuted", "blue", 0.45),
    ("Analytic floor (train)", "train/loss_mi_floor_analytic", "black", 0.45),
    ("Loss_mi (val)", "val/normal/loss_mi", "darkorange", 1.0),
    ("Loss_mi_permuted (val)", "val/normal/loss_mi_permuted", "darkorange", 0.45),
)

TRAINING_LOSSES_FILE = "training_losses.png"
UNNORMALIZED_FILE = "training_losses_unnormalized.png"
MI_NULL_FILE = "mi_vs_permuted_null.png"


def history_by_step(metric_history: Iterable) -> dict[int, float]:
    """Return a finite, step-aligned MLflow history with duplicates resolved.

    ``metric_history`` holds points with ``step``, ``timestamp`` and ``value``
    (MLflow ``Metric`` objects or equivalents). For a repeated step the point
    with the latest timestamp wins.
    """
    values_by_step = {}
    for point in sorted(
        metric_history,
        key=lambda item: (item.step, item.timestamp),
    ):
        values_by_step[int(point.step)] = float(point.value)

    values = np.asarray(list(values_by_step.values()), dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Loss histories must contain finite numeric values.")
    return values_by_step


def normalize(values: np.ndarray) -> np.ndarray:
    """Normalize a loss history to [0, 1], including constant histories."""
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Loss histories must contain finite numeric values.")

    minimum = values.min()
    span = values.max() - minimum
    if span == 0:
        return np.zeros_like(values)
    return (values - minimum) / span


def plot_training_losses(
    histories: Mapping[str, Mapping[int, float]],
    gamma: float,
    plot_folder: Path,
) -> list[Path]:
    """Write the normalized and the unnormalized training-loss figures.

    ``histories`` maps the ``train/`` metric names of ``TRAINING_LOSSES``
    (``loss_mi``, ``loss_reco``, ``loss``) to step-aligned histories; metrics left
    out are not plotted. The x-axis is the epoch number 1..N of each history.
    """
    gamma_title = f"{gamma:g}"
    raw_histories = {}
    normalized_histories = {}
    raw_colors = {}
    normalized_colors = {}
    for metric_name, label in TRAINING_LOSSES:
        if metric_name not in histories:
            continue

        values = np.asarray(list(histories[metric_name].values()), dtype=float)
        normalized = normalize(values)
        raw_label = "Loss_mi * gamma" if metric_name == "loss_mi" else label
        raw_values = values * gamma if metric_name == "loss_mi" else values
        raw_histories[raw_label] = {
            epoch: float(value) for epoch, value in enumerate(raw_values, start=1)
        }
        normalized_histories[label] = {
            epoch: float(value) for epoch, value in enumerate(normalized, start=1)
        }
        raw_colors[raw_label] = LOSS_COLORS[label]
        normalized_colors[label] = LOSS_COLORS[label]

    if not normalized_histories:
        raise ValueError("No training-loss history to plot.")

    raw_component_histories = {
        label: history
        for label, history in raw_histories.items()
        if label != "Loss_total"
    }
    raw_total_history = {
        label: history
        for label, history in raw_histories.items()
        if label == "Loss_total"
    }
    return [
        scatter.plot_lines(
            data=normalized_histories,
            xlabel="Epoch",
            ylabel="Normalized loss",
            title=f"Normalized training losses (gamma = {gamma_title})",
            save_dir=plot_folder,
            colors=normalized_colors,
            filename=TRAINING_LOSSES_FILE,
            alphas={"Loss_total": 0.8},
        ),
        scatter.plot_lines(
            data=raw_component_histories,
            xlabel="Epoch",
            ylabel="Loss_mi * gamma & Loss_reco",
            title=f"Unnormalized training losses (gamma = {gamma_title})",
            save_dir=plot_folder,
            colors=raw_colors,
            filename=UNNORMALIZED_FILE,
            right_axis_data=raw_total_history,
            right_ylabel="Loss_total",
            alphas={"Loss_total": 0.8},
            ylabel_fontsize=14,
        ),
    ]


def plot_mi_null(
    read_history: Callable[[str], Mapping[int, float] | None],
    plot_folder: Path,
) -> Path | None:
    """Plot MI next to its permutation noise floor; None if that was not logged.

    ``read_history(key)`` returns the step-aligned history of an MLflow metric key,
    or None when it is absent or unusable. Series that are missing are left out;
    without ``train/loss_mi_permuted`` (AE run with ``mi_num_permutations=0``) no
    figure is written.
    """
    if read_history("train/loss_mi_permuted") is None:
        return None

    data, colors, alphas = {}, {}, {}
    for label, key, color, alpha in MI_NULL_SERIES:
        history = read_history(key)
        if history is None:
            continue
        data[label] = {
            epoch: value for epoch, value in enumerate(history.values(), start=1)
        }
        colors[label] = color
        alphas[label] = alpha

    return scatter.plot_lines(
        data=data,
        xlabel="Epoch",
        ylabel="MI [bits]",
        title="MI loss vs. permutation noise floor",
        save_dir=plot_folder,
        colors=colors,
        filename=MI_NULL_FILE,
        alphas=alphas,
    )

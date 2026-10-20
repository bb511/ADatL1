# Simple scatter plot of x, y values.

from pathlib import Path
from pathvalidate import sanitize_filename

import matplotlib.pyplot as plt
import mplhep as hep


def plot(data: dict, xlabel: str, ylabel: str, title: str, save_dir: Path):
    """Plots the data as an xy scatter plot.

    Expects a dictionary with xy values.
    """
    plt.style.use(hep.style.CMS)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)

    x = list(data.keys())
    y = list(data.values())

    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}")

    ax.ticklabel_format(
        axis="x", style="sci", scilimits=(-2, 2), useMathText=True, useOffset=False
    )
    ax.ticklabel_format(
        axis="y", style="sci", scilimits=(-2, 2), useMathText=True, useOffset=False
    )
    ax.get_xaxis().get_offset_text().set_position((1.10, 1))
    ax.get_yaxis().get_offset_text().set_position((-0.12, 1))
    # hep.cms.label("Preliminary", data=False, loc=0, ax=ax)

    filename = sanitize_filename(f"{title}_{xlabel}_{ylabel}")
    filename = filename.replace(" ", "_")
    fig.savefig(save_dir / f"{filename}.jpg", bbox_inches="tight")
    fig.clear()
    plt.close(fig)


def plot_connected(data: dict, xlabel: str, ylabel: str, title: str, save_dir: Path):
    """Plots the data as an xy scatter plot with connected points.

    Expects a dictionary with xy values.
    """
    plt.style.use(hep.style.CMS)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)

    x = list(data.keys())
    y = list(data.values())

    ax.plot(x, y, marker="o", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}")

    ax.get_xaxis().get_offset_text().set_position((1.15, 1))
    ax.get_yaxis().get_offset_text().set_position((-0.12, 1))
    # hep.cms.label("Preliminary", data=False, loc=0, ax=ax)

    filename = sanitize_filename(f"{title}_{xlabel}_{ylabel}")
    filename = filename.replace(" ", "_")
    fig.savefig(save_dir / f"{filename}.jpg", bbox_inches="tight")
    fig.clear()
    plt.close(fig)


def plot_lines(
    data: dict[str, dict[int, float]],
    xlabel: str,
    ylabel: str,
    title: str,
    save_dir: Path,
    colors: dict[str, str] | None = None,
    filename: str | None = None,
    right_axis_data: dict[str, dict[int, float]] | None = None,
    right_ylabel: str | None = None,
    alphas: dict[str, float] | None = None,
    ylabel_fontsize: float | None = None,
    legend_max_columns: int = 3,
) -> Path:
    """Plot multiple labelled series as lines without point markers.

    ``data`` maps each legend label to an ``x: y`` mapping. ``colors`` optionally
    maps those same labels to Matplotlib colors. ``right_axis_data`` optionally maps
    lines to a secondary y-axis, labelled with ``right_ylabel``. ``alphas``
    optionally maps labels to line opacities. ``ylabel_fontsize`` can override the
    left-axis title size for plots with long labels. The legend sits outside the
    axes, centred below the x-axis title, with at most ``legend_max_columns``
    entries per row, so it never covers a curve. The plot is saved as a PNG and
    the resulting path is returned to the caller.
    """
    if not data and not right_axis_data:
        raise ValueError("Cannot plot an empty collection of lines.")
    if right_axis_data and right_ylabel is None:
        raise ValueError("A right y-axis label is required for secondary-axis data.")
    if legend_max_columns < 1:
        raise ValueError("legend_max_columns must be at least 1.")

    plt.style.use(hep.style.CMS)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)

    def plot_series(axis, series: dict[str, dict[int, float]]) -> None:
        for label, values in series.items():
            if not values:
                raise ValueError(f"Cannot plot empty data for {label!r}.")

            axis.plot(
                list(values.keys()),
                list(values.values()),
                color=None if colors is None else colors.get(label),
                label=label,
                linewidth=1.8,
                alpha=1.0 if alphas is None else alphas.get(label, 1.0),
            )

    plot_series(ax, data)
    right_ax = None
    if right_axis_data:
        right_ax = ax.twinx()
        plot_series(right_ax, right_axis_data)

    ax.set_xlabel(xlabel)
    if ylabel_fontsize is None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_title(title, pad=20)
    handles, labels = ax.get_legend_handles_labels()
    if right_ax is not None:
        right_ax.set_ylabel(right_ylabel)
        right_handles, right_labels = right_ax.get_legend_handles_labels()
        handles += right_handles
        labels += right_labels
    _legend_below_xaxis(fig, ax, handles, labels, legend_max_columns)

    if filename is None:
        filename = sanitize_filename(title).replace(" ", "_")
    if Path(filename).suffix.lower() != ".png":
        filename = f"{filename}.png"

    save_path = save_dir / filename
    fig.savefig(save_path, bbox_inches="tight")
    fig.clear()
    plt.close(fig)
    return save_path


def _legend_below_xaxis(fig, ax, handles, labels, max_columns: int):
    """Put the legend outside ``ax``, centred under its x-axis title.

    The figure is drawn once to measure the x-axis (tick labels and title), so
    the legend starts right below its lowest text for any figure size or style.
    ``savefig(bbox_inches="tight")`` then grows the image to include it.

    Entries are read row by row in the order given (Matplotlib fills columns
    first, so they are reordered), e.g. the train series in the first row and the
    validation series in the second. The font is one step smaller than the CMS
    default and the column gap halved, so three long labels per row stay about as
    wide as the axes.
    """
    columns = min(len(labels), max_columns)
    row_major = [
        index
        for column in range(columns)
        for index in range(column, len(labels), columns)
    ]
    fig.canvas.draw()
    xaxis_box = ax.xaxis.get_tightbbox(fig.canvas.get_renderer())
    _, xaxis_bottom = ax.transAxes.inverted().transform((0.0, xaxis_box.y0))
    return ax.legend(
        [handles[index] for index in row_major],
        [labels[index] for index in row_major],
        loc="upper center",
        bbox_to_anchor=(0.5, xaxis_bottom),
        ncol=columns,
        frameon=False,
        fontsize="x-small",
        columnspacing=1.0,
    )


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "logs" / "plots" / "scatter_manual_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_data = {1: 0.91, 2: 0.74, 3: 0.63, 4: 0.58, 5: 0.55}
    plot(
        sample_data,
        xlabel="Epoch",
        ylabel="Loss",
        title="Manual scatter",
        save_dir=output_dir,
    )
    plot_connected(
        sample_data,
        xlabel="Epoch",
        ylabel="Loss",
        title="Manual connected scatter",
        save_dir=output_dir,
    )
    plot_lines(
        {
            "Loss_mi": sample_data,
            "Loss_reco": {epoch: value * 0.8 for epoch, value in sample_data.items()},
            "Loss_total": {epoch: value * 0.9 for epoch, value in sample_data.items()},
        },
        xlabel="Epoch",
        ylabel="Normalized loss",
        title="Manual loss lines",
        save_dir=output_dir,
        colors={
            "Loss_mi": "blue",
            "Loss_reco": "red",
            "Loss_total": "darkgray",
        },
        filename="manual_loss_lines.png",
    )

    expected_paths = [
        output_dir / "Manual_scatter_Epoch_Loss.jpg",
        output_dir / "Manual_connected_scatter_Epoch_Loss.jpg",
        output_dir / "manual_loss_lines.png",
    ]
    for output_path in expected_paths:
        assert output_path.is_file(), f"Expected scatter plot at {output_path}"
    print(f"Manual scatter tests passed. Plots saved to {output_dir}")

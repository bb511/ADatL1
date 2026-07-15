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
) -> Path:
    """Plot multiple labelled series as lines without point markers.

    ``data`` maps each legend label to an ``x: y`` mapping. ``colors`` optionally
    maps those same labels to Matplotlib colors. The plot is saved as a PNG and the
    resulting path is returned to the caller.
    """
    if not data:
        raise ValueError("Cannot plot an empty collection of lines.")

    plt.style.use(hep.style.CMS)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    for label, values in data.items():
        if not values:
            raise ValueError(f"Cannot plot empty data for {label!r}.")

        ax.plot(
            list(values.keys()),
            list(values.values()),
            color=None if colors is None else colors.get(label),
            label=label,
            linewidth=1.8,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if filename is None:
        filename = sanitize_filename(title).replace(" ", "_")
    if Path(filename).suffix.lower() != ".png":
        filename = f"{filename}.png"

    save_path = save_dir / filename
    fig.savefig(save_path, bbox_inches="tight")
    fig.clear()
    plt.close(fig)
    return save_path


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
            "Loss_mi": "royalblue",
            "Loss_reco": "firebrick",
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

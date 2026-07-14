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

    expected_paths = [
        output_dir / "Manual_scatter_Epoch_Loss.jpg",
        output_dir / "Manual_connected_scatter_Epoch_Loss.jpg",
    ]
    for output_path in expected_paths:
        assert output_path.is_file(), f"Expected scatter plot at {output_path}"
    print(f"Manual scatter tests passed. Plots saved to {output_dir}")

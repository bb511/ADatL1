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

    fig, ax = plt.subplots(figsize=(6, 6), dpi=60)

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

    fig, ax = plt.subplots(figsize=(6, 6), dpi=60)

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

# Horizontal bar plots.
from pathlib import Path
from pathvalidate import sanitize_filename

import matplotlib.pyplot as plt
import mplhep as hep


def plot(data: dict, xlabel: str, save_dir: Path):
    """Plots the data in a horizontal bar plot.

    Expects a dictionary with data set names corresponding to number entries. The y-axis
    will be the data set names while the x-axis are the corresponding numbers.
    """
    plt.style.use(hep.style.CMS)

    dataset_names = list(data.keys())
    dataset_values = list(data.values())

    fig, ax = plt.subplots(figsize=(16, 24))

    ax.barh(dataset_names, dataset_values)
    ax.set_xlabel(xlabel)

    hep.cms.label("Preliminary", data=False, loc=0, ax=ax)
    plt.tight_layout()

    filename = sanitize_filename(xlabel)
    filename = filename.replace(" ", "_")
    fig.savefig(save_dir / f"{filename}.png")
    fig.clear()
    plt.close(fig)


def plot_yright(data: dict, ydata: dict, xlabel: str, ylabel: str, save_dir: Path):
    """Plots the data in a horizontal bar plot.

    Expects a dictionary with data set names corresponding to number entries. The y-axis
    will be the data set names while the x-axis are the corresponding numbers.

    This also puts values on the right of the y-axis, contained in ydata.
    """
    plt.style.use(hep.style.CMS)

    dataset_names = list(data.keys())
    dataset_values = list(data.values())
    dataset_yvals = list(ydata.values())

    fig, ax = plt.subplots(figsize=(24, 34))

    ax.barh(dataset_names, dataset_values)
    ax.set_xlabel(xlabel)

    hep.cms.label("Preliminary", data=False, loc=0, ax=ax)

    ax2 = ax.twinx()

    # Match the ticks using the same index positions
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(dataset_names)))
    ax2.set_yticklabels(dataset_yvals)
    ax2.set_ylabel(ylabel)

    plt.tight_layout()
    filename = sanitize_filename(xlabel)
    filename = filename.replace(" ", "_")
    fig.savefig(save_dir / f"{filename}.png")
    fig.clear()
    plt.close(fig)

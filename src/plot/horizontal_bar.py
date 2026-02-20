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

    fig, ax = plt.subplots(figsize=(8, 16), dpi=60)

    ax.barh(dataset_names, dataset_values)
    ax.set_xlabel(xlabel)

    hep.cms.label("Preliminary", data=False, loc=0, ax=ax)

    filename = xlabel.replace("\n", "__")
    filename = sanitize_filename(filename)
    filename = filename.replace(" ", "_")
    fig.savefig(save_dir / f"{filename}.jpg", bbox_inches="tight")
    fig.clear()
    plt.close(fig)


def plot_yright(
    data: dict,
    ydata: dict,
    xlabel: str,
    ylabel: str,
    save_dir: Path,
    percent: bool = False,
):
    """Plots the data in a horizontal bar plot.

    Expects a dictionary with data set names corresponding to number entries. The y-axis
    will be the data set names while the x-axis are the corresponding numbers.

    This also puts values on the right of the y-axis, contained in ydata.
    """
    plt.style.use(hep.style.CMS)

    dataset_names = list(data.keys())
    dataset_values = list(data.values())
    dataset_yvals = list(ydata.values())
    if percent:
        dataset_yvals = [f"{yval * 100:.2f}%" for yval in dataset_yvals]
    else:
        dataset_yvals = [round(yval, 4) for yval in dataset_yvals]

    fig, ax = plt.subplots(figsize=(8, 16), dpi=60)

    ax.barh(dataset_names, dataset_values)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(dataset_values) + 0.01)

    hep.cms.label("Preliminary", data=False, loc=0, ax=ax)

    ax2 = ax.twinx()

    # Match the ticks using the same index positions
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(dataset_names)))
    ax2.set_yticklabels(dataset_yvals)
    ax2.set_ylabel(ylabel)

    filename = xlabel.replace("\n", "__")
    filename = sanitize_filename(filename)
    filename = filename.replace(" ", "_")
    fig.savefig(save_dir / f"{filename}.jpg", bbox_inches="tight")
    fig.clear()
    plt.close(fig)

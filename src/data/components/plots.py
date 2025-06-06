# Plotting of the features that are stored in the converted h5s.

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def plot_hist(data: np.ndarray, object_type: str, feats: list, outdir: Path):
    """Plots data in 3d numpy array.

    The expected format of the data is nevents x nobjects x feats.
    Thus, this method plots a histogram of each feature of each object across all
    events in the given data numpy array.
    """
    outdir = outdir / f"{object_type}_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]

    for feat_nb, feat in enumerate(feats):
        plot_data = np.copy(data[..., feat_nb].flatten())

        fig = plt.figure(num=1, clear=True, tight_layout=True, figsize=(12, 10))
        ax = fig.add_subplot()
        ax.set_xmargin(0.1)
        ax.set_ymargin(0.1)
        ax.tick_params(axis='both', which='major', labelsize=22)

        plot_data = clip_hugedata(plot_data, ax)
        binnage = set_binnage(plot_data)
        counts, edges, bars = ax.hist(
            x=plot_data,
            bins=binnage,
            alpha=0.7,
            color="steelblue",
        )

        ax.set_xlabel(feat, fontsize=25)
        ax.set_ylabel("Counts", fontsize=25)
        ax.set_yscale("log")
        ax.text(
            0.85,
            0.95,
            f"N={int(sum(counts)):.2E}",
            transform=ax.transAxes,
            fontsize=15,
        )

        fig.savefig(outdir / f"{feat}.png")
        fig.clear()
        plt.close(fig)

    log.info(f"Plots for {object_type} saved to: {outdir}")

def clip_hugedata(plot_data: np.ndarray, ax: matplotlib.axes.Axes):
    """Clip data that has values which overflow when plotting."""
    if np.max(plot_data) > 1.0e15:
        plot_data = np.clip(plot_data, a_min=None, a_max=1.0e10)
        ax.text(
            0.05,
            0.95,
            f"Clipped: {1.0e10:.2E}",
            transform=ax.transAxes,
            fontsize=15
        )

    return plot_data

def set_binnage(plot_data: np.ndarray):
    """Sets the number of bins for the histogram depending on how data is distrib."""
    if np.min(plot_data) == np.max(plot_data):
        binnage = 10
    elif len(np.unique(plot_data)) in range(6, 30):
        binnage = int(np.max(plot_data) - np.min(plot_data))
        if binnage > 30:
            binnage = 30
    elif len(np.unique(plot_data)) <= 5:
        binnage = 20
    else:
        binnage = 'doane'

    return binnage

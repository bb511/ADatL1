# Overlaid histogram plot.
from pathlib import Path
from pathvalidate import sanitize_filename

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep


def plot_1d(
    x1: np.ndarray,
    x2: np.ndarray,
    obj_name: str,
    feat_name: str,
    save_dir: Path,
    label1: str = 'data1',
    label2: str = 'data2'
):
    """Plots 1d overalid histogram of data1 and data2.

    Expects that data1 and data2 are one dimensional tensors, containing a feature
    with the name feat_name.
    """
    plt.style.use(hep.style.CMS)


    bins = np.histogram_bin_edges(np.concatenate([x1, x2]), bins='doane')

    fig, ax = plt.subplots()

    # Use mplhep.histplot with precomputed counts
    c1, _ = np.histogram(x1, bins=bins)
    c2, _ = np.histogram(x2, bins=bins)

    # Optional: normalize to unity
    c1 = c1 / max(c1.sum(), 1)
    c2 = c2 / max(c2.sum(), 1)

    hep.histplot(c1, bins, ax=ax, label=label1, histtype="fill", color='C0', alpha=0.5)
    hep.histplot(c2, bins, ax=ax, label=label2, histtype="fill", color='C1', alpha=0.5)
    ax.legend()
    
    if check_feature_is_Et(feat_name):
        ax.set_yscale('log')
    else:
        ax.ticklabel_format(
            axis="y", style="sci", scilimits=(-2, 2), useMathText=True, useOffset=False
        )

    ax.set_title(obj_name)
    ax.set_xlabel(feat_name)
    ax.ticklabel_format(
        axis="x", style="sci", scilimits=(-2, 2), useMathText=True, useOffset=False
    )
    ax.get_xaxis().get_offset_text().set_position((1.10, 1))
    ax.get_yaxis().get_offset_text().set_position((-0.12, 1))

    filename = sanitize_filename(f"{obj_name}_{feat_name}")
    filename = filename.replace(" ", "_")
    fig.savefig(save_dir / f"{filename}.jpg", bbox_inches="tight")
    fig.clear()
    plt.close(fig)


def check_feature_is_Et(feat_name: str):
    is_et = 'Et' in feat_name or 'EtUnconstrained' in feat_name or 'ETTEM' in feat_name
    is_not_eta = not 'Eta' in feat_name

    return (is_et and is_not_eta)

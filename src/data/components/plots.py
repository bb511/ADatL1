# Plotting of the features that are stored in the converted h5s.

from pathlib import Path

import numpy as np
import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep

# Plot configuration.
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
hep.style.use("CMS")


def plot_hist(data: ak.Array, feat_name: str, outdir: Path):
    """Plots data in flat awkward array into a histogram."""
    # Flatten the data.
    data = ak.flatten(data, axis=None)
    # Skip if this is a padded field.
    if ak.all(ak.is_none(data)):
        return

    # Clip unreasonable large values.
    data = ak.where(data > 1e15, 1e15, data)

    fig, ax = plt.subplots()
    histogram = np.histogram(data, bins="doane")
    hep.histplot(*histogram, density=True, ax=ax)
    hep.cms.label("Preliminary", data=False, com=14)

    if check_feature_is_Et(feat_name):
        ax.set_yscale("log")
    else:
        ax.ticklabel_format(
            axis="y", style="sci", scilimits=(-2, 2), useMathText=True, useOffset=False
        )

    ax.set_xlabel(feat_name)
    ax.ticklabel_format(
        axis="x", style="sci", scilimits=(-2, 2), useMathText=True, useOffset=False
    )

    ax.get_xaxis().get_offset_text().set_position((1.10, 1))
    ax.get_yaxis().get_offset_text().set_position((-0.12, 1))

    fig.savefig(outdir / f"{feat_name}.png")
    fig.clear()
    plt.close(fig)


def check_feature_is_Et(feat_name: str):
    is_et = "Et" in feat_name or "EtUnconstrained" in feat_name or "ETTEM" in feat_name
    is_not_eta = not "Eta" in feat_name

    return is_et and is_not_eta

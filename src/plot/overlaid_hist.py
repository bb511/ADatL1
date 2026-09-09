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
    label1: str = "data1",
    label2: str = "data2",
):
    """Plots 1d overalid histogram of data1 and data2.

    Expects that data1 and data2 are one dimensional tensors, containing a feature
    with the name feat_name.
    """
    plt.style.use(hep.style.CMS)

    bins = np.histogram_bin_edges(np.concatenate([x1, x2]), bins="doane")

    fig, ax = plt.subplots()

    # Use mplhep.histplot with precomputed counts
    c1, _ = np.histogram(x1, bins=bins)
    c2, _ = np.histogram(x2, bins=bins)

    # Optional: normalize to unity
    c1 = c1 / max(c1.sum(), 1)
    c2 = c2 / max(c2.sum(), 1)

    hep.histplot(c1, bins, ax=ax, label=label1, histtype="fill", color="C0", alpha=0.5)
    hep.histplot(c2, bins, ax=ax, label=label2, histtype="fill", color="C1", alpha=0.5)
    ax.legend()

    if check_feature_is_Et(feat_name):
        ax.set_yscale("log")
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

    save_dir.mkdir(parents=True, exist_ok=True)

    filename = sanitize_filename(f"{obj_name}_{feat_name}")
    filename = filename.replace(" ", "_")
    fig.savefig(save_dir / f"{filename}.jpg", bbox_inches="tight")
    fig.clear()
    plt.close(fig)


def plot_streamed(
    counts1: np.ndarray,
    counts2: np.ndarray,
    edges: np.ndarray,
    obj_name: str,
    feat_name: str,
    save_dir: Path,
    label1: str = "data1",
    label2: str = "data2",
):
    """Plots a 1d overlaid histogram.

    The data is streamed into this histogram.
    """
    plt.style.use(hep.style.CMS)

    fig, ax = plt.subplots()

    counts1 = counts1 / max(counts1.sum(), 1)
    counts2 = counts2 / max(counts2.sum(), 1)

    hep.histplot(
        counts1, edges, ax=ax, label=label1, histtype="fill", color="C0", alpha=0.5
    )
    hep.histplot(
        counts2, edges, ax=ax, label=label2, histtype="fill", color="C1", alpha=0.5
    )
    ax.legend()

    if check_feature_is_Et(feat_name):
        ax.set_yscale("log")
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

    save_dir.mkdir(parents=True, exist_ok=True)

    filename = sanitize_filename(f"{obj_name}_{feat_name}").replace(" ", "_")
    fig.savefig(save_dir / f"{filename}.jpg", bbox_inches="tight")
    fig.clear()
    plt.close(fig)


def check_feature_is_Et(feat_name: str):
    is_et = "Et" in feat_name or "EtUnconstrained" in feat_name or "ETTEM" in feat_name
    is_not_eta = not "Eta" in feat_name

    return is_et and is_not_eta


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "logs" / "plots" / "overlaid_hist_manual_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=42)
    sample_a = rng.normal(loc=0.0, scale=1.0, size=1_000)
    sample_b = rng.normal(loc=0.4, scale=1.2, size=1_000)
    plot_1d(
        sample_a,
        sample_b,
        obj_name="jets",
        feat_name="Phi",
        save_dir=output_dir,
        label1="input",
        label2="reconstruction",
    )

    sample_edges = np.linspace(0.0, 100.0, 11)
    sample_counts_a = np.array([1, 4, 9, 16, 23, 19, 12, 7, 3, 1])
    sample_counts_b = np.array([1, 2, 6, 12, 19, 23, 17, 10, 5, 2])
    plot_streamed(
        sample_counts_a,
        sample_counts_b,
        sample_edges,
        obj_name="taus",
        feat_name="Et",
        save_dir=output_dir,
        label1="input",
        label2="reconstruction",
    )

    assert check_feature_is_Et("Et")
    assert check_feature_is_Et("EtUnconstrained")
    assert not check_feature_is_Et("Eta")

    expected_paths = [output_dir / "jets_Phi.jpg", output_dir / "taus_Et.jpg"]
    for output_path in expected_paths:
        assert output_path.is_file(), f"Expected overlaid histogram at {output_path}"
    print(f"Manual overlaid-histogram tests passed. Plots saved to {output_dir}")

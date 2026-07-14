# Plot a streamed histogram.
from pathlib import Path
from pathvalidate import sanitize_filename

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep


def plot_streamed(
    counts: np.ndarray,
    edges: np.ndarray,
    obj_name: str,
    feat_name: str,
    save_dir: Path,
    log: bool = False,
):
    """Plot a histogram from streamed bin counts."""

    plt.style.use(hep.style.CMS)

    fig, ax = plt.subplots()

    # normalize to unity
    counts = counts / max(counts.sum(), 1)

    hep.histplot(
        counts,
        edges,
        ax=ax,
        histtype="fill",
        color="C0",
        alpha=0.6,
    )

    if log:
        ax.set_yscale("log")
    else:
        ax.ticklabel_format(
            axis="y", style="sci", scilimits=(-2, 2), useMathText=True, useOffset=False
        )

    ax.set_title(obj_name)
    ax.set_xlabel(feat_name)
    ax.set_ylabel("counts")

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


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "logs" / "plots" / "histogram_manual_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_edges = np.linspace(0.0, 100.0, 11)
    sample_counts = np.array([1, 3, 8, 15, 22, 18, 12, 7, 3, 1])
    plot_streamed(
        counts=sample_counts,
        edges=sample_edges,
        obj_name="jets",
        feat_name="Et",
        save_dir=output_dir,
    )

    output_path = output_dir / "jets_Et.jpg"
    assert output_path.is_file(), f"Expected histogram at {output_path}"
    print(f"Manual histogram test passed. Plot saved to {output_path}")

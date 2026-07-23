# Histogram and categorical count plotting helpers.
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from pathvalidate import sanitize_filename


def plot_categorical_bin_counts(
    counts: Sequence[int] | np.ndarray,
    save_path: Path | str,
    *,
    title: str,
    expected_counts: Sequence[float] | np.ndarray | None = None,
    expected_label: str = "Expected from full training-set proportions",
    xlabel: str = "Bin ID",
    ylabel: str = "Number of events in minibatch",
) -> Path:
    """Save observed categorical counts and an optional expected-count reference."""
    observed = np.asarray(counts)
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("counts must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(observed)) or np.any(observed < 0):
        raise ValueError("counts must contain finite, non-negative values.")

    expected = None
    if expected_counts is not None:
        expected = np.asarray(expected_counts, dtype=float)
        if expected.shape != observed.shape:
            raise ValueError(
                "expected_counts must have the same shape as counts. "
                f"Got {expected.shape} and {observed.shape}."
            )
        if not np.all(np.isfinite(expected)) or np.any(expected < 0):
            raise ValueError(
                "expected_counts must contain finite, non-negative values."
            )

    output_path = _png_output_path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bin_ids = np.arange(observed.size)

    with plt.style.context(hep.style.CMS):
        fig, ax = plt.subplots()
        try:
            ax.bar(
                bin_ids,
                observed,
                color="C0",
                alpha=0.75,
                label="Observed minibatch counts",
            )
            if expected is not None:
                ax.plot(
                    bin_ids,
                    expected,
                    color="C1",
                    linestyle="--",
                    marker="o",
                    linewidth=2,
                    label=expected_label,
                )

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xticks(bin_ids)
            ax.set_xlim(-0.6, observed.size - 0.4)
            ax.set_ylim(bottom=0)
            ax.legend()
            fig.savefig(output_path, bbox_inches="tight")
        finally:
            plt.close(fig)

    return output_path


def plot_minibatch_scalar_histogram(
    values: Sequence[int | float] | np.ndarray,
    save_path: Path | str,
    *,
    title: str,
    xlabel: str,
    ylabel: str = "Number of minibatches",
    bins: str | int | Sequence[float] = "auto",
) -> Path:
    """Save a histogram for one scalar collected from each minibatch."""
    scalar_values = np.asarray(values)
    if scalar_values.ndim != 1 or scalar_values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(scalar_values)):
        raise ValueError("values must contain only finite scalars.")

    output_path = _png_output_path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.style.context(hep.style.CMS):
        fig, ax = plt.subplots()
        try:
            ax.hist(
                scalar_values,
                bins=bins,
                color="C0",
                alpha=0.75,
                edgecolor="black",
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            fig.savefig(output_path, bbox_inches="tight")
        finally:
            plt.close(fig)

    return output_path


def _png_output_path(save_path: Path | str) -> Path:
    output_path = Path(save_path)
    if output_path.suffix.lower() != ".png":
        raise ValueError(f"Plot output must be a PNG file. Got {output_path}.")
    return output_path


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

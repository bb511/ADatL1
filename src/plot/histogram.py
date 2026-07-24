# Histogram and categorical count plotting helpers.
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.ticker import MaxNLocator
from pathvalidate import sanitize_filename


def plot_categorical_bin_counts(
    counts: Sequence[int] | np.ndarray,
    save_path: Path | str,
    *,
    title: str,
    observed_label: str = "Observed minibatch counts",
    expected_counts: Sequence[float] | np.ndarray | None = None,
    expected_label: str = "Expected from full training-set proportions",
    xlabel: str = "Bin ID",
    ylabel: str = "Number of events in minibatch",
    metadata: Mapping[str, object] | None = None,
    y_axis_max: float | None = None,
    annotate_clipped_values: bool = False,
    integer_y_ticks: bool = False,
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
    if y_axis_max is not None:
        y_axis_max = float(y_axis_max)
        if not np.isfinite(y_axis_max) or y_axis_max <= 0:
            raise ValueError("y_axis_max must be finite and greater than zero.")

    output_path = _png_output_path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bin_ids = np.arange(observed.size)

    with plt.style.context(hep.style.CMS):
        fig, ax = plt.subplots(figsize=(16, 7))
        fig.subplots_adjust(left=0.10, right=0.71, bottom=0.16, top=0.88)
        try:
            ax.bar(
                bin_ids,
                observed,
                color="C0",
                alpha=0.75,
                label=observed_label,
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
            ax.set_xlabel(xlabel, fontsize=20, loc="center", labelpad=14)
            ax.set_ylabel(ylabel, fontsize=20, loc="center", labelpad=14)
            major_ticks = np.arange(0, observed.size, 5)
            ax.set_xticks(major_ticks)
            ax.set_xticks(bin_ids, minor=True)
            ax.set_xlim(-0.6, observed.size - 0.4)
            ax.tick_params(axis="x", labelsize=14, pad=8)

            if y_axis_max is None:
                maxima = [float(observed.max())]
                if expected is not None:
                    maxima.append(float(expected.max()))
                y_max = max(max(maxima) * 1.22, 1.0)
                ax.set_ylim(0, y_max)
            else:
                ax.set_ylim(0, y_axis_max)
                if annotate_clipped_values:
                    for bin_id, count in enumerate(observed):
                        if float(count) <= y_axis_max:
                            continue
                        ax.text(
                            bin_id,
                            y_axis_max * 0.96,
                            f"{float(count):g}",
                            ha="center",
                            va="top",
                            color="black",
                            fontsize=10,
                            fontweight="bold",
                            zorder=4,
                        )
            if integer_y_ticks:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            if metadata:
                rows = [[str(label), str(value)] for label, value in metadata.items()]
                table = ax.table(
                    cellText=rows,
                    cellLoc="left",
                    bbox=[1.04, 0.54, 0.50, 0.34],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                for row_idx in range(len(rows)):
                    table[(row_idx, 0)].set_text_props(weight="bold")

            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.03, 0.42),
                borderaxespad=0,
                fontsize=10,
            )
            fig.savefig(output_path, bbox_inches="tight")
        finally:
            plt.close(fig)

    return output_path


def plot_minibatch_scalar_histogram(
    values: Sequence[int | float] | np.ndarray,
    save_path: Path | str,
    *,
    title: str,
    xlabel: str = "Minibatch number",
    ylabel: str = "Number of unique FET.Et values",
) -> Path:
    """Plot one collected scalar per minibatch as contiguous histogram bars."""
    scalar_values = np.asarray(values)
    if scalar_values.ndim != 1 or scalar_values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(scalar_values)):
        raise ValueError("values must contain only finite scalars.")

    output_path = _png_output_path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    minibatch_ids = np.arange(scalar_values.size)

    with plt.style.context(hep.style.CMS):
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.subplots_adjust(left=0.13, right=0.96, bottom=0.16, top=0.88)
        try:
            ax.bar(
                minibatch_ids,
                scalar_values,
                color="C0",
                width=1.0,
                alpha=0.8,
                linewidth=0,
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel, fontsize=20, loc="center", labelpad=14)
            ax.set_ylabel(ylabel, fontsize=20, loc="center", labelpad=14)
            ax.set_xlim(
                -0.5,
                max(float(scalar_values.size - 1) + 0.5, 0.5),
            )
            value_min = float(scalar_values.min())
            value_max = float(scalar_values.max())
            y_buffer = max((value_max - value_min) * 0.08, 1.0)
            ax.set_ylim(
                max(0.0, value_min - y_buffer),
                value_max + y_buffer,
            )
            ax.grid(axis="both", alpha=0.25)
            fig.savefig(output_path, bbox_inches="tight")
        finally:
            plt.close(fig)

    return output_path


def plot_fixed_bin_widths(
    widths: Sequence[int | float] | np.ndarray,
    save_path: Path | str,
    *,
    title: str,
    xlabel: str = "Bin ID",
    ylabel: str = "Bin width ΔFET.Et",
) -> Path:
    """Plot the numerical width covered by every fitted fixed MI bin."""
    bin_widths = np.asarray(widths, dtype=float)
    if bin_widths.ndim != 1 or bin_widths.size == 0:
        raise ValueError("widths must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(bin_widths)) or np.any(bin_widths < 0):
        raise ValueError("widths must contain finite, non-negative values.")

    output_path = _png_output_path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bin_ids = np.arange(bin_widths.size)
    histogram_edges = np.arange(bin_widths.size + 1) - 0.5

    with plt.style.context(hep.style.CMS):
        fig, ax = plt.subplots(figsize=(16, 7))
        fig.subplots_adjust(left=0.11, right=0.96, bottom=0.16, top=0.88)
        try:
            ax.stairs(
                bin_widths,
                histogram_edges,
                fill=True,
                color="C0",
                alpha=0.8,
                linewidth=1.5,
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel, fontsize=20, loc="center", labelpad=14)
            ax.set_ylabel(ylabel, fontsize=20, loc="center", labelpad=14)
            ax.set_xticks(np.arange(0, bin_widths.size, 5))
            ax.set_xticks(bin_ids, minor=True)
            ax.set_xlim(-0.5, bin_widths.size - 0.5)
            ax.set_ylim(0, max(float(bin_widths.max()) * 1.08, 1e-12))
            ax.tick_params(axis="x", labelsize=14, pad=8)
            ax.grid(axis="y", alpha=0.25)
            fig.savefig(output_path, bbox_inches="tight")
        finally:
            plt.close(fig)

    return output_path


def plot_histogram_counts(
    counts: Sequence[int | float] | np.ndarray,
    edges: Sequence[int | float] | np.ndarray,
    save_path: Path | str,
    *,
    title: str,
    xlabel: str,
    ylabel: str = "Number of events",
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Plot precomputed, unnormalized continuous-histogram counts."""
    histogram_counts = np.asarray(counts)
    histogram_edges = np.asarray(edges, dtype=float)
    if histogram_counts.ndim != 1 or histogram_counts.size == 0:
        raise ValueError("counts must be a non-empty one-dimensional array.")
    if histogram_edges.shape != (histogram_counts.size + 1,):
        raise ValueError("edges must contain exactly one more value than counts.")
    if (
        not np.all(np.isfinite(histogram_counts))
        or np.any(histogram_counts < 0)
    ):
        raise ValueError("counts must contain finite, non-negative values.")
    if (
        not np.all(np.isfinite(histogram_edges))
        or np.any(np.diff(histogram_edges) <= 0)
    ):
        raise ValueError("edges must be finite and strictly increasing.")

    output_path = _png_output_path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.style.context(hep.style.CMS):
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.subplots_adjust(left=0.11, right=0.76, bottom=0.16, top=0.88)
        try:
            ax.stairs(
                histogram_counts,
                histogram_edges,
                fill=True,
                color="C0",
                alpha=0.8,
                linewidth=1.5,
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel, fontsize=20, loc="center", labelpad=14)
            ax.set_ylabel(ylabel, fontsize=20, loc="center", labelpad=14)
            ax.set_xlim(histogram_edges[0], histogram_edges[-1])
            ax.set_ylim(
                0,
                max(float(histogram_counts.max()) * 1.12, 1.0),
            )
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="y", alpha=0.25)

            if metadata:
                rows = [[str(label), str(value)] for label, value in metadata.items()]
                table = ax.table(
                    cellText=rows,
                    cellLoc="left",
                    bbox=[1.04, 0.36, 0.48, 0.52],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                for row_idx in range(len(rows)):
                    table[(row_idx, 0)].set_text_props(weight="bold")

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

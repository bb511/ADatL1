# Matrix plot.

from collections.abc import Iterable
from pathlib import Path
from pathvalidate import sanitize_filename
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import mplhep as hep
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot(
    data: dict[dict],
    value_name: str,
    save_dir: Path,
    cmap: str | colors.Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    filename: str | None = None,
    figure_scale: float = 1.0,
    outline_row: str | None = None,
    outline_linewidth: float = 3.0,
    text_highlight_max_abs: float | None = None,
    text_highlight_columns: Iterable[str] | None = None,
    text_highlight_color: str = "green",
):
    """Plot the data as a labelled matrix.

    Expects a dictionary where each key corresponds to a row, i.e., another dictionary.
    Each key of the latter dictionary corresponds to a column entry for that row.
    The column labels are expected to be the same for each row.

    ``cmap`` accepts a Matplotlib colormap name or object. For example, use
    ``"coolwarm"`` for a diverging red-blue color scheme.

    ``vmin`` and ``vmax`` optionally fix the color scale. ``filename`` overrides the
    default filename derived from ``value_name`` and may select a format by extension.
    ``figure_scale`` scales the complete figure, including the matrix cells.

    ``outline_row`` names one row label (exact match first, then case-insensitive)
    that is framed by a bold black border of ``outline_linewidth`` points. A label
    that is not among the rows is ignored, so callers can always pass it.
    ``text_highlight_max_abs`` prints every value of that row whose absolute value
    is at most this threshold in bold ``text_highlight_color``; ``None`` disables it.
    ``text_highlight_columns`` does the same for the entries of that row in the named
    columns, whatever their value (e.g. chosen from another matrix).
    """
    if figure_scale <= 0:
        raise ValueError(f"figure_scale must be greater than zero, got {figure_scale}.")

    plt.style.use(hep.style.CMS)

    rows = list(data.keys())
    cols = list(data[rows[0]].keys())

    mat = np.array([[data[r][c] for c in cols] for r in rows], dtype=float)
    n_rows, n_cols = mat.shape
    outline_index = _find_row(rows, outline_row)
    highlight_columns = {str(column) for column in text_highlight_columns or ()}

    cell_size = 0.72  # 20% larger than the previous 0.6-inch cells
    fig_size_max = 9.6
    fig_w = max(fig_size_max, n_cols * cell_size) * figure_scale
    fig_h = max(fig_size_max, n_rows * cell_size) * figure_scale

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # axis labels
    label_fontsize = max(8, min(14, int(180 / max(n_rows, n_cols))))
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(
        [str(c) for c in cols], rotation=90, fontsize=label_fontsize
    )
    ax.set_yticklabels([str(r) for r in rows], fontsize=label_fontsize)
    ax.set_title(value_name, pad=20)
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        top=False,
        right=False,
        bottom=False,
        left=False,
    )

    # Heuristic: scale font size with the grid size
    fontsize = max(6, min(12, int(180 / max(n_rows, n_cols))))
    fmt = "{:.2f}"

    norm = im.norm
    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            fontweight = "normal"
            if np.isnan(val):
                text = "NaN"
                txt_color = "black"
            elif i == outline_index and (
                str(cols[j]) in highlight_columns
                or (
                    text_highlight_max_abs is not None
                    and abs(val) <= text_highlight_max_abs
                )
            ):
                text = fmt.format(val)
                txt_color = text_highlight_color
                fontweight = "bold"
            else:
                text = fmt.format(val)
                red, green, blue, _ = im.cmap(norm(val))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                txt_color = "black" if luminance > 0.5 else "white"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=txt_color,
                fontweight=fontweight,
                clip_on=True,  # ensures nothing bleeds outside the axes
            )

    if outline_index is not None:
        # Drawn unclipped and above the spines so the border is equally thick on
        # all four sides, also where it coincides with the axes edge.
        ax.add_patch(
            patches.Rectangle(
                (-0.5, outline_index - 0.5),
                n_cols,
                1,
                fill=False,
                edgecolor="black",
                linewidth=outline_linewidth,
                joinstyle="miter",
                clip_on=False,
                zorder=5,
            )
        )

    # colorbar
    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes("right", size="5%", pad=0.15)
    colorbar = fig.colorbar(im, cax=colorbar_ax)
    colorbar.ax.minorticks_off()
    colorbar.ax.tick_params(which="both", length=0)

    if filename is None:
        filename = sanitize_filename(f"{value_name}")
        filename = f"{filename.replace(' ', '_')}.jpg"
    fig.savefig(save_dir / filename, bbox_inches="tight")
    fig.clear()
    plt.close(fig)


def _find_row(rows: list, label: str | None) -> int | None:
    """Index of ``label`` among ``rows``; exact match wins over case-insensitive."""
    if label is None:
        return None
    for index, row in enumerate(rows):
        if str(row) == label:
            return index
    lowered = label.lower()
    for index, row in enumerate(rows):
        if str(row).lower() == lowered:
            return index
    return None












if __name__ == "__main__":
    sample_data = {
        "FET.Et": {
            "FET.Et": 1.0,
            "jets.Et": 0.72,
            "muons.Et": -0.18,
            "taus.Et": -1,
        },
        "jets.Et": {
            "FET.Et": 0.72,
            "jets.Et": 1.0,
            "muons.Et": 0.31,
            "taus.Et": 0.44,
        },
        "muons.Et": {
            "FET.Et": -0.18,
            "jets.Et": 0.31,
            "muons.Et": 1.0,
            "taus.Et": -0.27,
        },
        "taus.Et": {
            "FET.Et": -1,
            "jets.Et": 0.44,
            "muons.Et": -0.27,
            "taus.Et": 1.0,
        },
    }

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "logs" / "plots" / "matrix_manual_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot(
        sample_data,
        value_name="Manual correlation matrix",
        save_dir=output_dir,
        outline_row="FET.Et",
        text_highlight_max_abs=0.1,
    )
    plot(
        sample_data,
        value_name="Manual correlation matrix coolwarm",
        save_dir=output_dir,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        filename="Manual_correlation_matrix_coolwarm.jpg",
        outline_row="FET.Et",
        text_highlight_max_abs=0.1,
    )

    expected_paths = [
        output_dir / "Manual_correlation_matrix.jpg",
        output_dir / "Manual_correlation_matrix_coolwarm.jpg",
    ]
    for output_path in expected_paths:
        assert output_path.is_file(), f"Expected matrix plot at {output_path}"
    print(f"Manual matrix tests passed. Plots saved to {output_dir}")

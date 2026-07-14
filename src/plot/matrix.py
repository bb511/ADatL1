# Matrix plot.

from pathlib import Path
from pathvalidate import sanitize_filename
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mplhep as hep


def plot(
    data: dict[dict],
    value_name: str,
    save_dir: Path,
    cmap: str | colors.Colormap = "viridis",
):
    """Plot the data as a labelled matrix.

    Expects a dictionary where each key corresponds to a row, i.e., another dictionary.
    Each key of the latter dictionary corresponds to a column entry for that row.
    The column labels are expected to be the same for each row.

    ``cmap`` accepts a Matplotlib colormap name or object. For example, use
    ``"coolwarm"`` for a diverging red-blue color scheme.
    """
    plt.style.use(hep.style.CMS)

    rows = list(data.keys())
    cols = list(data[rows[0]].keys())

    mat = np.array([[data[r][c] for c in cols] for r in rows], dtype=float)
    n_rows, n_cols = mat.shape

    cell_size = 0.6  # inches per cell (tune if desired)
    fig_size_max = 6
    fig_w = max(fig_size_max, n_cols * cell_size)
    fig_h = max(fig_size_max, n_rows * cell_size)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
    im = ax.imshow(mat, aspect="auto", cmap=cmap)

    # axis labels
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels([str(c) for c in cols], rotation=90)
    ax.set_yticklabels([str(r) for r in rows])
    ax.set_title(value_name, pad=20)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Heuristic: scale font size with the grid size
    fontsize = max(6, min(12, int(180 / max(n_rows, n_cols))))
    fmt = "{:.1f}"

    norm = im.norm
    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            if np.isnan(val):
                continue

            red, green, blue, _ = im.cmap(norm(val))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            txt_color = "black" if luminance > 0.5 else "white"

            ax.text(
                j,
                i,
                fmt.format(val),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=txt_color,
                clip_on=True,  # ensures nothing bleeds outside the axes
            )

    # colorbar
    fig.colorbar(im, ax=ax)

    filename = sanitize_filename(f"{value_name}")
    filename = filename.replace(" ", "_")
    fig.savefig(save_dir / f"{filename}.jpg", bbox_inches="tight")
    fig.clear()
    plt.close(fig)


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
    )
    plot(
        sample_data,
        value_name="Manual correlation matrix coolwarm",
        save_dir=output_dir,
        cmap="coolwarm",
    )

    expected_paths = [
        output_dir / "Manual_correlation_matrix.jpg",
        output_dir / "Manual_correlation_matrix_coolwarm.jpg",
    ]
    for output_path in expected_paths:
        assert output_path.is_file(), f"Expected matrix plot at {output_path}"
    print(f"Manual matrix tests passed. Plots saved to {output_dir}")

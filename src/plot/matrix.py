# Matrix plot.

from pathlib import Path
from pathvalidate import sanitize_filename
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mplhep as hep


def plot(data: dict[dict], value_name: str, save_dir: Path):
    """Plots the data as an xy scatter plot.

    Expects a dictionary where each key corresponds to a row, i.e., another dictionary.
    Each key of the latter dictionary corresponds to a column entry for that row.
    The column labels are expected to be the same for each row.
    """
    plt.style.use(hep.style.CMS)

    rows = list(data.keys())
    cols = list(data[rows[0]].keys())

    mat = np.array([[data[r][c] for c in cols] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(16, 16), dpi=120)
    im = ax.imshow(mat, aspect="auto", cmap="viridis")

    # axis labels
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels([str(c) for c in cols], rotation=90)
    ax.set_yticklabels([str(r) for r in rows])

    n_rows, n_cols = mat.shape

    # Heuristic: scale font size with the grid size
    fontsize = max(6, min(14, int(220 / max(n_rows, n_cols))))
    fmt = "{:.2g}" if max(n_rows, n_cols) > 20 else "{:.3g}"

    norm = im.norm
    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            if np.isnan(val):
                continue

            # Contrast-aware text color
            txt_color = "white" if norm(val) < 0.6 else "black"

            t = ax.text(
                j, i, fmt.format(val),
                ha="center", va="center",
                fontsize=fontsize,
                color=txt_color,
            )


    # colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{value_name}")

    filename = sanitize_filename(f"{value_name}")
    filename = filename.replace(" ", "_")
    fig.savefig(save_dir / f"{filename}.jpg", bbox_inches='tight')
    fig.clear()
    plt.close(fig)

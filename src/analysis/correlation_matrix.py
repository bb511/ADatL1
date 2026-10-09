from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.analysis.checkpointloader import CheckpointLoader
    from src.analysis.run_mi_hyperparameters import read_mi_hyperparameters, run_dir_of
except ImportError:
    from .checkpointloader import CheckpointLoader
    from .run_mi_hyperparameters import read_mi_hyperparameters, run_dir_of


REPO_ROOT = Path(__file__).resolve().parents[2]
# Same rule as src/plot/correlation_matrix.py (kept local: this module must not need
# matplotlib). Highlighted-row entries whose reconstructed |r| is at most this are
# printed in bold green.
DECORRELATED_ABS_THRESHOLD = 0.1
DECORRELATED_TEXT_COLOR = "#008000"
DEFAULT_MATRIX_DIR = (
    REPO_ROOT
    / "checkpoints"
    / "physics_ae_models"
    / "Bernoulli-MI_No_FET_Et_Bins_50_Gamma_0.1_Run_3"
    / "plots"
    / "test"
    / "last"
    / "correlation_matrix"
    / "normal"
)


@dataclass(frozen=True)
class CorrelationMatrixSpecs:
    input_path: str | Path
    reconstruction_path: str | Path
    label: str = "reconstruction_minus_input"
    # Row framed with a bold black border, as in src/plot/correlation_matrix.py. Its
    # entries are printed green where the reconstruction |r| <= 0.1.
    highlight_variable: str | None = "FET.Et"


class CorrelationMatrixPlotter:
    """Plot abs(reconstruction correlation) - abs(input correlation)."""

    def __init__(self, specs: CorrelationMatrixSpecs) -> None:
        self.specs = specs

    def plot(
        self,
        title: str | None = None,
        output_dir: str | Path | None = None,
        output_stem: str | None = None,
        subtitle: str | None = None,
    ) -> Path:
        """Write the delta CSV and PNG. ``subtitle`` defaults to the MI hyperparameters
        of the checkpoint run that holds ``input_path``."""
        input_corr = self._load_matrix(self.specs.input_path)
        reconstruction_corr = self._load_matrix(self.specs.reconstruction_path)
        delta_corr = self.compute_delta(input_corr, reconstruction_corr)

        save_dir = self._resolve_output_dir(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        stem = output_stem or f"{self.specs.label}_abs_correlation_delta"
        csv_path = save_dir / f"{stem}.csv"
        png_path = save_dir / f"{stem}.png"

        delta_corr.to_csv(csv_path)
        self._plot_heatmap(
            corr=delta_corr,
            save_path=png_path,
            title=title or "Absolute correlation change",
            decorrelation_reference=reconstruction_corr,
            subtitle=subtitle if subtitle is not None else self.mi_subtitle(),
        )

        print(f"Saved delta correlation matrix CSV to {csv_path}.")
        print(f"Saved delta correlation matrix plot to {png_path}.")
        return png_path

    @staticmethod
    def compute_delta(
        input_corr: pd.DataFrame,
        reconstruction_corr: pd.DataFrame,
    ) -> pd.DataFrame:
        CorrelationMatrixPlotter._validate_matrix(input_corr, "input")
        CorrelationMatrixPlotter._validate_matrix(reconstruction_corr, "reconstruction")

        if list(input_corr.index) != list(reconstruction_corr.index):
            raise ValueError("Input and reconstruction matrix row labels do not match.")

        if list(input_corr.columns) != list(reconstruction_corr.columns):
            raise ValueError(
                "Input and reconstruction matrix column labels do not match."
            )

        return reconstruction_corr.abs() - input_corr.abs()

    @staticmethod
    def _validate_matrix(corr: pd.DataFrame, name: str) -> None:
        if corr.empty:
            raise ValueError(f"{name} correlation matrix is empty.")

        if list(corr.index) != list(corr.columns):
            raise ValueError(
                f"{name} correlation matrix row and column labels do not match."
            )

    def _load_matrix(self, path: str | Path) -> pd.DataFrame:
        matrix = CheckpointLoader(path).load_matrix()
        return matrix.apply(pd.to_numeric, errors="raise")

    def mi_subtitle(self) -> str | None:
        """MI hyperparameters of the run that holds the input matrix, if any are on disk."""
        run_dir = run_dir_of(Path(self.specs.input_path).resolve())
        if run_dir is None:
            return None
        mi = read_mi_hyperparameters(run_dir)
        return mi.text() if mi.known else None

    def _highlight_index(self, labels: list) -> int | None:
        """Row index of the highlighted variable; exact match before case-insensitive."""
        target = self.specs.highlight_variable
        if target is None:
            return None
        names = [str(label) for label in labels]
        if target in names:
            return names.index(target)
        lowered = [name.lower() for name in names]
        if target.lower() in lowered:
            return lowered.index(target.lower())
        return None

    def _decorrelated_columns(
        self,
        labels: list,
        reference: pd.DataFrame | None,
    ) -> set[int]:
        """Column indices whose reconstructed |r| with the highlighted variable is small."""
        if reference is None:
            return set()
        row = self._highlight_index(list(reference.index))
        if row is None:
            return set()
        values = pd.to_numeric(reference.iloc[row], errors="coerce")
        small = {
            str(label)
            for label, value in values.items()
            if np.isfinite(value) and abs(value) <= DECORRELATED_ABS_THRESHOLD
        }
        return {index for index, label in enumerate(labels) if str(label) in small}

    def _resolve_output_dir(self, output_dir: str | Path | None) -> Path:
        if output_dir is not None:
            return Path(output_dir)

        input_parent = Path(self.specs.input_path).parent
        if input_parent.exists():
            return input_parent

        return REPO_ROOT / "logs" / "plots"

    def _plot_heatmap(
        self,
        corr: pd.DataFrame,
        save_path: Path,
        title: str,
        decorrelation_reference: pd.DataFrame | None = None,
        subtitle: str | None = None,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except ModuleNotFoundError:
            self._plot_heatmap_with_pillow(
                corr=corr,
                save_path=save_path,
                title=title,
                decorrelation_reference=decorrelation_reference,
                subtitle=subtitle,
            )
            return

        labels = list(corr.columns)
        highlight_index = self._highlight_index(labels)
        green_columns = self._decorrelated_columns(labels, decorrelation_reference)
        mat = corr.to_numpy(dtype=float)
        n = len(labels)
        fig_size = max(6, 0.9 * n)

        finite_values = mat[np.isfinite(mat)]
        max_abs = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
        color_limit = max(max_abs, 0.05)

        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=140)
        im = ax.imshow(
            mat,
            vmin=-color_limit,
            vmax=color_limit,
            cmap="coolwarm",
        )

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        if subtitle:
            ax.annotate(subtitle, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 6),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=10, color="#333333", annotation_clip=False)
            ax.set_title(title, pad=24)
        else:
            ax.set_title(title)

        for i in range(n):
            for j in range(n):
                value = mat[i, j]
                weight = "normal"
                if np.isnan(value):
                    text = "nan"
                    color = "black"
                elif i == highlight_index and j in green_columns:
                    text = f"{value:.2f}"
                    color = DECORRELATED_TEXT_COLOR
                    weight = "bold"
                else:
                    text = f"{value:.2f}"
                    color = "white" if abs(value) > 0.55 * color_limit else "black"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10,
                    fontweight=weight,
                )

        if highlight_index is not None:
            ax.add_patch(
                Rectangle(
                    (-0.5, highlight_index - 0.5),
                    n,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=3.0,
                    joinstyle="miter",
                    clip_on=False,
                    zorder=5,
                )
            )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("|corr after| - |corr before|")
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        fig.clear()
        plt.close(fig)

    def _plot_heatmap_with_pillow(
        self,
        corr: pd.DataFrame,
        save_path: Path,
        title: str,
        decorrelation_reference: pd.DataFrame | None = None,
        subtitle: str | None = None,
    ) -> None:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Plotting needs either matplotlib or Pillow installed."
            ) from exc

        labels = list(corr.columns)
        mat = corr.to_numpy(dtype=float)
        n = len(labels)
        highlight_index = self._highlight_index(labels)
        green_columns = self._decorrelated_columns(labels, decorrelation_reference)

        finite_values = mat[np.isfinite(mat)]
        max_abs = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
        color_limit = max(max_abs, 0.05)

        cell_size = 96
        left_margin = 210
        top_margin = 190
        right_margin = 120
        bottom_margin = 210
        width = left_margin + n * cell_size + right_margin
        height = top_margin + n * cell_size + bottom_margin

        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        draw.text((left_margin, 24), title, fill="black", font=font)
        if subtitle:
            draw.text((left_margin, 48), subtitle, fill=(51, 51, 51), font=font)

        for index, label in enumerate(labels):
            x = left_margin + index * cell_size + cell_size // 2
            y = top_margin - 12
            draw.text((x - 24, y), label, fill="black", font=font, anchor="mm")

            y_label = top_margin + index * cell_size + cell_size // 2
            draw.text((left_margin - 12, y_label), label, fill="black", font=font, anchor="rm")

        for i in range(n):
            for j in range(n):
                value = mat[i, j]
                x0 = left_margin + j * cell_size
                y0 = top_margin + i * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size

                color = self._delta_color(value, color_limit)
                draw.rectangle((x0, y0, x1, y1), fill=color, outline=(230, 230, 230))

                text = "nan" if np.isnan(value) else f"{value:.2f}"
                text_color = "white" if abs(value) > 0.55 * color_limit else "black"
                if i == highlight_index and j in green_columns and not np.isnan(value):
                    text_color = DECORRELATED_TEXT_COLOR
                draw.text(
                    ((x0 + x1) // 2, (y0 + y1) // 2),
                    text,
                    fill=text_color,
                    font=font,
                    anchor="mm",
                )

        if highlight_index is not None:
            y0 = top_margin + highlight_index * cell_size
            draw.rectangle(
                (left_margin, y0, left_margin + n * cell_size, y0 + cell_size),
                outline="black",
                width=5,
            )

        legend_x = left_margin + n * cell_size + 35
        legend_y = top_margin
        legend_height = n * cell_size
        for offset in range(legend_height):
            value = color_limit - 2 * color_limit * offset / max(legend_height - 1, 1)
            draw.line(
                ((legend_x, legend_y + offset), (legend_x + 24, legend_y + offset)),
                fill=self._delta_color(value, color_limit),
            )

        draw.text(
            (legend_x + 32, legend_y),
            f"{color_limit:.2f}",
            fill="black",
            font=font,
        )
        draw.text(
            (legend_x + 32, legend_y + legend_height - 12),
            f"{-color_limit:.2f}",
            fill="black",
            font=font,
        )
        draw.text(
            (left_margin, height - 48),
            "|corr after| - |corr before|",
            fill="black",
            font=font,
        )

        image.save(save_path)

    @staticmethod
    def _delta_color(value: float, color_limit: float) -> tuple[int, int, int]:
        if np.isnan(value):
            return (230, 230, 230)

        normalized = float(np.clip(value / color_limit, -1.0, 1.0))
        if normalized < 0:
            t = normalized + 1.0
            start = np.array([59, 76, 192])
            middle = np.array([245, 245, 245])
            color = start * (1.0 - t) + middle * t
        else:
            t = normalized
            middle = np.array([245, 245, 245])
            end = np.array([180, 4, 38])
            color = middle * (1.0 - t) + end * t

        return tuple(int(channel) for channel in color)


def _build_default_paths(matrix_dir: Path, method: str) -> tuple[Path, Path]:
    return (
        matrix_dir / f"input_{method}_correlation_matrix.csv",
        matrix_dir / f"reconstruction_{method}_correlation_matrix.csv",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot abs(reconstruction correlation matrix) - "
            "abs(input correlation matrix)."
        )
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=DEFAULT_MATRIX_DIR,
        help="Folder containing input/reconstruction correlation matrix CSV files.",
    )
    parser.add_argument("--input-csv", type=Path, help="Input correlation matrix CSV.")
    parser.add_argument(
        "--reconstruction-csv",
        type=Path,
        help="Reconstruction correlation matrix CSV.",
    )
    parser.add_argument("--method", default="pearson", help="Correlation method name.")
    parser.add_argument("--output-dir", type=Path, help="Directory for CSV and PNG.")
    parser.add_argument("--output-stem", help="Output filename stem without extension.")
    parser.add_argument("--title", help="Plot title.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.input_csv is None and args.reconstruction_csv is None:
        input_path, reconstruction_path = _build_default_paths(
            args.matrix_dir,
            args.method,
        )
    elif args.input_csv is not None and args.reconstruction_csv is not None:
        input_path = args.input_csv
        reconstruction_path = args.reconstruction_csv
    else:
        raise ValueError(
            "Pass both --input-csv and --reconstruction-csv, or neither to use "
            "--matrix-dir."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_stem = args.output_stem or f"abs_correlation_delta_{args.method}_{timestamp}"

    plotter = CorrelationMatrixPlotter(
        CorrelationMatrixSpecs(
            input_path=input_path,
            reconstruction_path=reconstruction_path,
        )
    )
    plotter.plot(
        title=args.title,
        output_dir=args.output_dir,
        output_stem=output_stem,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from checkpointloader import CheckpointLoader
except ImportError:
    from .checkpointloader import CheckpointLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIABLES_CSV = (
    REPO_ROOT
    / "checkpoints"
    / "physics_ae_models"
    / "Bernoulli-MI_No_FET_Et_Bins_50_Gamma_0.1_Run_3"
    / "plots"
    / "test"
    / "last"
    / "correlation_matrix"
    / "normal"
    / "input_variables.csv"
)
DEFAULT_MATRIX_DIR = DEFAULT_VARIABLES_CSV.parent


@dataclass(frozen=True)
class ValueCountSpecs:
    variables_csv: str | Path = DEFAULT_VARIABLES_CSV
    column: str = "FET.Et"
    space: str = "input"


class ValueCountPlotter:
    """Plot distinct values of one variable and how often each value occurs."""

    def __init__(self, specs: ValueCountSpecs) -> None:
        self.specs = specs

    def plot(
        self,
        title: str | None = None,
        output_dir: str | Path | None = None,
        output_stem: str | None = None,
    ) -> Path:
        counts = self.compute_counts(self._load_values())
        save_dir = self._resolve_output_dir(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        stem = output_stem or f"{self.specs.column.replace('.', '_')}_value_counts"
        csv_path = save_dir / f"{stem}.csv"
        png_path = save_dir / f"{stem}.png"

        counts.to_csv(csv_path, index=False)
        plot_title = title or (
            f"{self.specs.column} value counts "
            f"({len(counts)} distinct values, {int(counts['count'].sum())} entries)"
        )
        self._plot_counts(counts=counts, save_path=png_path, title=plot_title)

        print(f"Saved value counts CSV to {csv_path}.")
        print(f"Saved value counts plot to {png_path}.")
        return png_path

    @staticmethod
    def compute_counts(values: pd.Series) -> pd.DataFrame:
        values = pd.to_numeric(values.dropna(), errors="raise")
        counts = values.value_counts(sort=False).sort_index()
        return counts.rename_axis("value").reset_index(name="count")

    def _load_values(self) -> pd.Series:
        variables_path = Path(self.specs.variables_csv)
        if variables_path.name == "correlation_variables.csv":
            source_table = pd.read_csv(variables_path, header=[0, 1])
            spaces = set(source_table.columns.get_level_values(0))
            if self.specs.space not in spaces:
                raise KeyError(
                    f"Space {self.specs.space!r} not found in {variables_path}. "
                    f"Available spaces: {sorted(spaces)}"
                )
            df = source_table.xs(self.specs.space, axis=1, level=0)
        else:
            df = CheckpointLoader(variables_path).load_table()

        if self.specs.column not in df.columns:
            raise KeyError(
                f"Column {self.specs.column!r} not found in {self.specs.variables_csv}. "
                f"Available columns: {list(df.columns)}"
            )
        return df[self.specs.column]

    def _resolve_output_dir(self, output_dir: str | Path | None) -> Path:
        if output_dir is not None:
            return Path(output_dir)

        return REPO_ROOT / "logs" / "plots"

    def _plot_counts(self, counts: pd.DataFrame, save_path: Path, title: str) -> None:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            self._plot_counts_with_pillow(counts=counts, save_path=save_path, title=title)
            return

        values = counts["value"].to_numpy(dtype=float)
        count_values = counts["count"].to_numpy(dtype=int)
        n_values = len(counts)

        fig_width = max(8, min(28, 0.28 * n_values))
        fig, ax = plt.subplots(figsize=(fig_width, 5), dpi=140)
        ax.bar(np.arange(n_values), count_values, color="#4c78a8")
        ax.set_title(title)
        ax.set_xlabel(self.specs.column)
        ax.set_ylabel("count")
        ax.set_xticks(np.arange(n_values))
        ax.set_xticklabels([f"{value:g}" for value in values], rotation=90)
        ax.margins(x=0.01)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        fig.clear()
        plt.close(fig)

    def _plot_counts_with_pillow(
        self,
        counts: pd.DataFrame,
        save_path: Path,
        title: str,
    ) -> None:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Plotting needs either matplotlib or Pillow installed."
            ) from exc

        values = counts["value"].to_numpy(dtype=float)
        count_values = counts["count"].to_numpy(dtype=int)
        n_values = len(counts)
        max_count = int(count_values.max()) if n_values else 1

        left_margin = 90
        right_margin = 40
        top_margin = 80
        bottom_margin = 160
        bar_width = max(8, min(36, 900 // max(n_values, 1)))
        bar_gap = 3
        plot_width = n_values * (bar_width + bar_gap)
        plot_height = 420
        width = max(900, left_margin + plot_width + right_margin)
        height = top_margin + plot_height + bottom_margin

        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        x0 = left_margin
        y0 = top_margin
        x1 = left_margin + plot_width
        y1 = top_margin + plot_height

        draw.text((left_margin, 24), title, fill="black", font=font)
        draw.line((x0, y1, x1, y1), fill="black")
        draw.line((x0, y0, x0, y1), fill="black")
        draw.text((16, y0 + plot_height // 2), "count", fill="black", font=font)
        draw.text((x0, height - 28), self.specs.column, fill="black", font=font)

        for tick in range(0, 5):
            count = max_count * tick / 4
            y = y1 - int(plot_height * tick / 4)
            draw.line((x0 - 5, y, x0, y), fill="black")
            draw.text((x0 - 10, y), f"{count:.0f}", fill="black", font=font, anchor="rm")

        for index, (value, count) in enumerate(zip(values, count_values, strict=False)):
            bar_x0 = x0 + index * (bar_width + bar_gap)
            bar_x1 = bar_x0 + bar_width
            bar_height = int((count / max_count) * plot_height) if max_count else 0
            bar_y0 = y1 - bar_height
            draw.rectangle((bar_x0, bar_y0, bar_x1, y1), fill=(76, 120, 168))

            if n_values <= 80 or index % max(1, n_values // 60) == 0:
                label = f"{value:g}"
                draw.text(
                    (bar_x0 + bar_width // 2, y1 + 8),
                    label,
                    fill="black",
                    font=font,
                    anchor="ma",
                )

        image.save(save_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot distinct FET.Et values and counts per value."
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=DEFAULT_MATRIX_DIR,
        help="Folder containing input_variables.csv.",
    )
    parser.add_argument(
        "--variables-csv",
        type=Path,
        default=None,
        help="CSV containing event-level correlation variables.",
    )
    parser.add_argument("--space", default="input", help="Source-table space to use.")
    parser.add_argument("--column", default="FET.Et", help="Column to count.")
    parser.add_argument("--output-dir", type=Path, help="Directory for CSV and PNG.")
    parser.add_argument("--output-stem", help="Output filename stem without extension.")
    parser.add_argument("--title", help="Plot title.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_stem = args.output_stem or f"{args.column.replace('.', '_')}_counts_{timestamp}"
    variables_csv = args.variables_csv or args.matrix_dir / "input_variables.csv"

    plotter = ValueCountPlotter(
        ValueCountSpecs(
            variables_csv=variables_csv,
            column=args.column,
            space=args.space,
        )
    )
    plotter.plot(
        title=args.title,
        output_dir=args.output_dir,
        output_stem=output_stem,
    )


if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd


class CheckpointLoader:
    def __init__(self, path: str | Path, metric: str | None = None) -> None:
        self.path = Path(path)
        self.metric = metric
        print(
            f"Initialized CheckpointLoader with path={self.path}, "
            f"metric={self.metric!r}."
        )

        if not self.path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.path}")

    def load(self, column: int | None = None) -> pd.DataFrame | pd.Series:
        """Load a checkpoint CSV.

        Correlation matrix CSVs are stored with variable names in the first column.
        When ``column`` is omitted, this returns the full matrix as a DataFrame. If a
        column is supplied, the old metric-loader behaviour is preserved and the
        selected column is returned as a Series.
        """
        csv_path = self._csv_path()
        print(f"Loading CSV from {csv_path}.")

        if column is None:
            df = pd.read_csv(csv_path, index_col=0)
        else:
            df = pd.read_csv(csv_path)

        print(f"Loaded CSV with shape {df.shape}.")

        if column is None:
            return df

        if df.shape[1] < column + 1:
            raise ValueError(
                f"Expected at least {column + 1} columns in {csv_path}, "
                f"but found {df.shape[1]} column(s)."
            )

        values = df.iloc[:, column]
        print(f"Returning column {column} with {len(values)} values.")
        return values

    def load_matrix(self) -> pd.DataFrame:
        return self.load()

    def load_table(self) -> pd.DataFrame:
        csv_path = self._csv_path()
        print(f"Loading CSV table from {csv_path}.")
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV table with shape {df.shape}.")
        return df

    def _csv_path(self) -> Path:
        if self.path.is_file():
            return self.path

        if self.metric is None:
            raise ValueError(
                f"Path {self.path} is a directory. Pass metric as the CSV filename."
            )

        csv_path = self.path / self.metric
        if not csv_path.exists() and csv_path.suffix != ".csv":
            csv_path = csv_path.with_suffix(".csv")

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

        return csv_path

from pathlib import Path

import pandas as pd


ALLOWED_METRICS = {
    "ascore_operational",
    "bernoulli_prob_mean",
    "bernoulli_prob_std",
    "latent_mean",
    "latent_std",
    "loss",
    "loss_mean",
    "loss_reco",
    "loss_mi",
}


class DataLoader:
    def __init__(self, path: str | Path, metric: str) -> None:
        self.path = Path(path)
        self.metric = metric
        print(f"Initialized DataLoader with path={self.path}, metric='{self.metric}'.")

        if not self.path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.path}")

        if self.metric not in ALLOWED_METRICS:
            allowed = ", ".join(sorted(ALLOWED_METRICS))
            raise ValueError(
                f"Unknown metric '{self.metric}'. Allowed metrics are: {allowed}"
            )

    def load(self) -> pd.Series:
        path_to_file: Path = Path(self.path / "metrics/train" / self.metric)
        print(f"Loading CSV from {path_to_file}.")

        df = pd.read_csv(
            path_to_file,
            sep=r"\s+",
            header=None,
            names=["timestamp", "value", "step"],
        )
        print(f"Loaded CSV with shape {df.shape}.")

        if df.shape[1] < 2:
            raise ValueError(
                f"Expected at least two columns in {path_to_file}, "
                f"but found {df.shape[1]} column(s)."
            )

        values = df.iloc[:, 1]
        print(f"Returning second column with {len(values)} values.")
        return values

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

        if not self.path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.path}")

        if self.metric not in ALLOWED_METRICS:
            allowed = ", ".join(sorted(ALLOWED_METRICS))
            raise ValueError(
                f"Unknown metric '{self.metric}'. Allowed metrics are: {allowed}"
            )

    def load(self) -> pd.Series:
        df = pd.read_csv(self.path)

        if self.metric not in df.columns:
            available = ", ".join(map(str, df.columns))
            raise ValueError(
                f"Metric '{self.metric}' not found in {self.path}. "
                f"Available columns are: {available}"
            )

        return df
